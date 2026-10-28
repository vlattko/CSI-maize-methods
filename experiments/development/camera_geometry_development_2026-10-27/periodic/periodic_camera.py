"""Fixed first-harmonic RGB-to-NoIR affine coefficients from training ECC.

This is a servo-index-conditioned image surrogate. It is not a physical 3D
camera, measured servo telemetry, narrow-band registration or tissue mapping.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
import numpy as np
import camera_geometry as g

POLICY=dict(harmonics=1,ridge=.1,irls_iterations=6,huber_threshold_pixels=2.,
            teacher_min_iou=.45,minimum_teachers=20,
            minimum_train_median_iou=.65,inner_mean_delta=.02,
            minimum_positive_inner_folds=2,inner_folds=3,
            determinant_range=[.35,3.],maximum_condition=3.,
            diagonal_range=[.5,1.8],maximum_absolute_offdiagonal=.6,
            maximum_normalized_translation=.4)

def basis(index):
    t=2*np.pi*np.asarray(index)/40
    return np.stack([np.ones_like(t),np.sin(t),np.cos(t)],axis=-1)

def normalize(matrix,grid):
    w,h=grid
    scale=np.diag([w,h,1.]);m=np.vstack([matrix,[0,0,1]])
    return (np.linalg.inv(scale)@m@scale)[:2]

def denormalize(matrix,grid):
    w,h=grid
    scale=np.diag([w,h,1.]);m=np.vstack([matrix,[0,0,1]])
    return (scale@m@np.linalg.inv(scale))[:2]

def matrix_guard(matrix,grid):
    m=np.asarray(matrix,dtype=float);reasons=[]
    if m.shape!=(2,3) or not np.isfinite(m).all():
        return dict(valid=False,reasons=['nonfinite_or_wrong_shape'],determinant=None,condition=None)
    det=float(np.linalg.det(m[:,:2]));cond=float(np.linalg.cond(m[:,:2]))
    if not POLICY['determinant_range'][0]<=det<=POLICY['determinant_range'][1]:reasons.append('determinant')
    if cond>POLICY['maximum_condition']:reasons.append('conditioning')
    norm=normalize(m,grid)
    if not all(POLICY['diagonal_range'][0]<=norm[j,j]<=POLICY['diagonal_range'][1] for j in range(2)):
        reasons.append('diagonal_scale')
    if max(abs(norm[0,1]),abs(norm[1,0]))>POLICY['maximum_absolute_offdiagonal']:reasons.append('offdiagonal')
    if np.max(abs(norm[:,2]))>POLICY['maximum_normalized_translation']:reasons.append('translation')
    return dict(valid=not reasons,reasons=reasons,determinant=det,condition=cond)

def fit_coefficients(teachers,grid):
    if any(t['view'] in g.TEST for t in teachers):raise ValueError('heldout_teacher_not_allowed')
    if len(teachers)<6:raise ValueError('fewer_than_six_training_teachers')
    x=basis([t['view'] for t in teachers])
    if np.linalg.matrix_rank(x)<3:raise ValueError('degenerate_training_angles')
    y=np.array([normalize(t['matrix'],grid).ravel() for t in teachers])
    base=np.array([t['weight'] for t in teachers],float)
    weights=base.copy();penalty=np.diag([0,POLICY['ridge'],POLICY['ridge']])
    probes=np.array([[0,0,1],[1,0,1],[0,1,1],[1,1,1],[.5,.5,1]])
    coefficients=None
    for _ in range(POLICY['irls_iterations']):
        coefficients=np.linalg.solve(x.T@(weights[:,None]*x)+penalty,x.T@(weights[:,None]*y))
        # Robustness is measured as image displacement of five unit-grid
        # probes, in pixels, rather than mixing translations with scale units.
        error=(y-x@coefficients).reshape(-1,2,3)
        displacement=np.einsum('nij,kj->nki',error,probes)*np.array(grid)
        residual=np.sqrt(np.mean(np.sum(displacement**2,axis=-1),axis=1))
        robust=np.minimum(1,POLICY['huber_threshold_pixels']/np.maximum(residual,1e-12))
        weights=base*robust
    return dict(coefficients=coefficients.tolist(),basis=['1','sin(2*pi*i/40)','cos(2*pi*i/40)'],
                coefficient_coordinates='unit source/target width and height',grid=list(grid),
                teacher_indices=[t['view'] for t in teachers],weights=weights.tolist(),
                design_condition=float(np.linalg.cond(x.T@(weights[:,None]*x)+penalty)))

def predict(fit,index):
    return denormalize((basis(index)@np.asarray(fit['coefficients'])).reshape(2,3),fit['grid'])

def select_periodic(inner_deltas,n_teachers,train_median_iou,all_matrices_valid):
    reasons=[]
    if n_teachers<POLICY['minimum_teachers']:reasons.append('fewer_than_20_training_teachers')
    if train_median_iou is None or train_median_iou<POLICY['minimum_train_median_iou']:
        reasons.append('training_median_iou_below_0.65')
    if len(inner_deltas)!=3:reasons.append('incomplete_inner_validation')
    if not inner_deltas or np.mean(inner_deltas)<POLICY['inner_mean_delta']:reasons.append('inner_mean_gain_below_0.02')
    if sum(x>0 for x in inner_deltas)<POLICY['minimum_positive_inner_folds']:reasons.append('fewer_than_two_improving_inner_folds')
    if not all_matrices_valid:reasons.append('one_or_more_query_matrix_guards_failed')
    return dict(accepted=not reasons,reasons=reasons,inner_mean_delta=float(np.mean(inner_deltas)) if inner_deltas else None,
                uses_heldout=False)

def fit_periodic(rgb,noir,shared):
    """Only outer training indices enter teachers or inner selection."""
    if not shared.get('selected'):
        return dict(status='no_shared_training_fit',model=None,teachers=[],
                    selection=dict(accepted=False,reasons=['no_shared_training_fit'],uses_heldout=False))
    selected=shared['selected'];sign=selected['sign'];phase=selected['phase']
    pairs=g.valid_pairs(rgb,noir,sign,phase)
    grid=(pairs[0][0].shape[1],pairs[0][0].shape[0])
    teachers=[]
    for a,b,i in pairs:
        ecc=g.per_view_ecc(a,b);guard=matrix_guard(ecc['matrix'],grid)
        accepted=guard['valid'] and ecc['iou']>=POLICY['teacher_min_iou'] and ecc['solver_status']=='converged'
        teachers.append(dict(view=i,noir_view=(sign*i+phase)%40,**ecc,guard=guard,
                             accepted=bool(accepted),weight=max(ecc['iou'],.01)**2))
    valid=[t for t in teachers if t['accepted']]
    if len(valid)<6:
        return dict(status='insufficient_training_teachers',model=None,teachers=teachers,
                    selection=dict(accepted=False,reasons=['fewer_than_six_training_teachers'],uses_heldout=False))
    model=fit_coefficients(valid,grid);queries=[]
    for i in range(40):
        m=predict(model,i);queries.append(dict(view=i,matrix=m.tolist(),guard=matrix_guard(m,grid)))
    train_scores=[g.iou_warp(a,b,predict(model,i)) for a,b,i in pairs]
    inner=[]
    for fold in range(3):
        training=[t for t in valid if t['view']%3!=fold]
        fitting=[x for x in pairs if x[2]%3!=fold]
        validation=[x for x in pairs if x[2]%3==fold]
        if len(training)<6 or not validation:
            inner.append(dict(fold=fold,status='insufficient_inner_support',training_teachers=len(training),validation_pairs=len(validation)))
            continue
        inner_model=fit_coefficients(training,grid)
        shared_inner=g.fit_shared(fitting)
        pp=[];ss=[];bad=0
        for a,b,i in validation:
            m=predict(inner_model,i)
            if not matrix_guard(m,grid)['valid']:
                # Failed matrices count as zero conditional image overlap in
                # complexity selection; they are not silently dropped.
                pp.append(0.);bad+=1
            else:pp.append(g.iou_warp(a,b,m))
            ss.append(g.iou_warp(a,b,shared_inner['matrix']))
        inner.append(dict(fold=fold,status='executed',training_teachers=len(training),
            training_indices=[t['view'] for t in training],validation_indices=[x[2] for x in validation],
            validation_pairs=len(validation),invalid_matrices=bad,
            periodic_median_iou=float(np.median(pp)),shared_median_iou=float(np.median(ss)),
            median_paired_delta=float(np.median(np.array(pp)-np.array(ss)))))
    deltas=[x['median_paired_delta'] for x in inner if x['status']=='executed']
    selection=select_periodic(deltas,len(valid),float(np.median(train_scores)),all(q['guard']['valid'] for q in queries))
    return dict(status='fitted',model=model,teachers=teachers,queries=queries,
                n_accepted_teachers=len(valid),train_indices=[i for _,_,i in pairs],
                train_scores=train_scores,train_median_iou=float(np.median(train_scores)),
                inner_validation=inner,selection=selection,sign=sign,phase=phase)
