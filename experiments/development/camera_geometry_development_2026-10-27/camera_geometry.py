"""Training-angle-only camera surrogates. Matrices map RGB pixels to NoIR.

These are silhouette alignment surrogates, not physical camera calibration or
corresponding-tissue maps. No assay measurements enter this module.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import cv2
from scipy.optimize import least_squares

cv2.setNumThreads(2)
TRAIN=tuple(i for i in range(40) if i%4!=2)
TEST=tuple(range(2,40,4))
POLICY=dict(min_train_pairs=16, phase_step=4, refine_top=3,
            phase_improvement=.03, phase_runner_margin=.015,
            phase_strata_wins=2, min_mask_pixels=10, max_nfev=35,
            boundary_samples=64, robust_scale_pixels=1.5,
            affine_diagonal_bounds=[.6,1.6], affine_offdiagonal_bound=.3,
            normalized_translation_bound=.35)

def deployment_decision(fit):
    selected=fit.get('selected')
    reasons=[]
    if selected is None:
        reasons.append('no_training_fit')
    else:
        if selected['n_train']<20:
            reasons.append('fewer_than_20_training_pairs')
        if selected['median_iou']<.65:
            reasons.append('training_median_iou_below_0.65')
        m=np.asarray(selected['matrix'])
        if not np.isfinite(m).all() or np.linalg.det(m[:,:2])<=0:
            reasons.append('invalid_affine')
    return dict(deployment_abstain=bool(reasons),deployment_reasons=reasons,
                deployment_status='abstain' if reasons else 'accepted_camera_surrogate',
                deployment_gate_uses_heldout=False,
                deployment_policy=dict(min_training_pairs=20,min_training_median_iou=.65))

def rescale_matrix(matrix, source_fit_size, source_new_size,
                   target_fit_size, target_new_size):
    """Convert a fitted affine to arbitrary source/target pixel grids."""
    m=np.vstack([matrix,[0,0,1]])
    source=np.diag([source_fit_size[0]/source_new_size[0],
                    source_fit_size[1]/source_new_size[1],1])
    target=np.diag([target_new_size[0]/target_fit_size[0],
                    target_new_size[1]/target_fit_size[1],1])
    return (target@m@source)[:2]

def moment(a,b):
    if int(a.sum())<10 or int(b.sum())<10:
        raise ValueError('empty_or_tiny_foreground')
    def stats(mask):
        y,x=np.where(mask)
        return np.array([x.mean(),y.mean()]),np.maximum([x.std(),y.std()],1)
    ca,sa=stats(a); cb,sb=stats(b)
    scale=np.clip(sb/sa,.6,1.6)
    return np.array([[scale[0],0,cb[0]-scale[0]*ca[0]],
                     [0,scale[1],cb[1]-scale[1]*ca[1]]],np.float64)

def warp_target(target,matrix,source_shape):
    return cv2.warpAffine(target.astype(np.uint8),np.asarray(matrix,np.float32),
        (source_shape[1],source_shape[0]),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)>0

def iou_warp(source,target,matrix):
    warped=warp_target(target,matrix,source.shape)
    n=int(np.logical_or(source,warped).sum())
    return float(np.logical_and(source,warped).sum()/n) if n else 0.

def valid_pairs(rgb,noir,sign=1,phase=0,indices=TRAIN):
    return [(rgb[i],noir[(sign*i+phase)%40],i) for i in indices
            if i in rgb and (sign*i+phase)%40 in noir
            and rgb[i].sum()>=10 and noir[(sign*i+phase)%40].sum()>=10]

def median_moment(pairs):
    return np.median([moment(a,b) for a,b,_ in pairs],axis=0)

def _boundary(mask,n):
    edge=mask & ~cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    y,x=np.where(edge)
    take=np.linspace(0,len(x)-1,min(n,len(x))).round().astype(int)
    return np.column_stack([x[take],y[take]]).astype(np.float32)

def _signed_distance(mask):
    a=mask.astype(np.uint8)
    return cv2.distanceTransform(1-a,cv2.DIST_L2,3)-cv2.distanceTransform(a,cv2.DIST_L2,3)

def _sample_distance(field,points):
    h,w=field.shape
    x,y=points[:,0],points[:,1]
    xc=np.clip(x,0,w-1); yc=np.clip(y,0,h-1)
    sample=cv2.remap(field,xc.astype(np.float32)[None,:],yc.astype(np.float32)[None,:],
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]
    return sample+np.hypot(x-xc,y-yc)

def fit_shared(pairs,refine=True):
    if not pairs:
        raise ValueError('no_training_pairs')
    initial=median_moment(pairs)
    initial_scores=[iou_warp(a,b,initial) for a,b,_ in pairs]
    result=dict(matrix=initial.tolist(),moment_matrix=initial.tolist(),
                train_indices=[i for _,_,i in pairs],n_train=len(pairs),
                initial_median_iou=float(np.median(initial_scores)),
                train_scores=initial_scores,median_iou=float(np.median(initial_scores)),
                optimizer_status='moment_only',optimizer_nfev=0)
    if not refine:
        return result
    h,w=pairs[0][0].shape
    scale=np.array([[w,0,0],[0,h,0],[0,0,1.]])
    normalized=np.linalg.inv(scale)@np.vstack([initial,[0,0,1]])@scale
    params=normalized[:2].ravel()
    lower=[.6,-.3,-.35,-.3,.6,-.35]
    upper=[1.6,.3,.35,.3,1.6,.35]
    params=np.clip(params,np.array(lower)+1e-7,np.array(upper)-1e-7)
    samples=[(_boundary(a,POLICY['boundary_samples']),_boundary(b,POLICY['boundary_samples']),
              _signed_distance(a),_signed_distance(b)) for a,b,_ in pairs]
    def pixel_matrix(p):
        q=np.vstack([np.reshape(p,(2,3)),[0,0,1]])
        return (scale@q@np.linalg.inv(scale))[:2]
    def residual(p):
        m=pixel_matrix(p); inv=cv2.invertAffineTransform(m)
        values=[]
        for pa,pb,da,db in samples:
            qa=pa@m[:,:2].T+m[:,2]; qb=pb@inv[:,:2].T+inv[:,2]
            values.extend([_sample_distance(db,qa),_sample_distance(da,qb)])
        return np.concatenate(values)
    try:
        fit=least_squares(residual,params,bounds=(lower,upper),loss='soft_l1',
            f_scale=POLICY['robust_scale_pixels'],max_nfev=POLICY['max_nfev'],
            diff_step=1e-3,xtol=1e-5,ftol=1e-5,gtol=1e-5)
        m=pixel_matrix(fit.x)
        scores=[iou_warp(a,b,m) for a,b,_ in pairs]
        accepted=np.isfinite(m).all() and np.linalg.det(m[:,:2])>0 and np.median(scores)>result['median_iou']
        result.update(optimizer_status='refined' if accepted else 'moment_fallback',
                      optimizer_success=bool(fit.success),optimizer_message=str(fit.message),
                      optimizer_nfev=int(fit.nfev),optimizer_cost=float(fit.cost))
        if accepted:
            result.update(matrix=m.tolist(),train_scores=scores,median_iou=float(np.median(scores)))
    except (ValueError,np.linalg.LinAlgError,cv2.error) as exc:
        result.update(optimizer_status='optimizer_error_moment_fallback',optimizer_error=str(exc))
    return result

def fit_session(rgb,noir):
    """Never inspects outer test masks, including for mapping/eligibility."""
    candidates=[]
    for sign in (1,-1):
        for phase in range(0,40,POLICY['phase_step']):
            pairs=valid_pairs(rgb,noir,sign,phase)
            candidate=dict(sign=sign,phase=phase,n_train=len(pairs))
            if len(pairs)>=POLICY['min_train_pairs']:
                candidate.update(fit_shared(pairs,False))
            else:
                candidate.update(median_iou=None,status='insufficient_training_pairs')
            candidates.append(candidate)
    nominal=next(c for c in candidates if c['sign']==1 and c['phase']==0)
    if nominal['n_train']<POLICY['min_train_pairs']:
        return dict(status='insufficient_training_pairs',selected=None,nominal=None,
                    candidates=candidates,phase_gate=dict(accepted_alternate=False))
    ranked=sorted([c for c in candidates if c['median_iou'] is not None],
                  key=lambda c:(-c['median_iou'],c['sign']!=1,c['phase']))
    refine_ids={(c['sign'],c['phase']) for c in ranked[:POLICY['refine_top']]}
    refine_ids.add((1,0))
    for c in candidates:
        if (c['sign'],c['phase']) in refine_ids:
            c.update(fit_shared(valid_pairs(rgb,noir,c['sign'],c['phase']),True))
    ranked=sorted([c for c in candidates if c['median_iou'] is not None],
                  key=lambda c:(-c['median_iou'],c['sign']!=1,c['phase']))
    best=ranked[0]; runner=ranked[1]
    common=sorted(set(best['train_indices'])&set(nominal['train_indices']))
    best_scores=dict(zip(best['train_indices'],best['train_scores']))
    nominal_scores=dict(zip(nominal['train_indices'],nominal['train_scores']))
    delta=float(np.median([best_scores[i]-nominal_scores[i] for i in common]))
    wins=0; strata=[]
    for residue in (0,1,3):
        ds=[best_scores[i]-nominal_scores[i] for i in common if i%4==residue]
        med=float(np.median(ds)) if ds else None
        strata.append(dict(residue=residue,n=len(ds),delta=med))
        wins+=bool(med is not None and med>=POLICY['phase_improvement'])
    margin=float(best['median_iou']-runner['median_iou'])
    alternate=(best['sign'],best['phase'])!=(1,0)
    accepted=bool(alternate and delta>=POLICY['phase_improvement'] and
                  margin>=POLICY['phase_runner_margin'] and wins>=POLICY['phase_strata_wins'])
    return dict(status='fitted',nominal=nominal,selected=best if accepted else nominal,
                candidates=candidates,phase_gate=dict(accepted_alternate=accepted,
                best_sign=best['sign'],best_phase=best['phase'],matched_delta=delta,
                runner_margin=margin,strata=strata,reason='alternate_training_gate_passed' if accepted
                else 'nominal_retained_training_evidence_insufficient_or_nominal_best'))

def per_view_ecc(a,b,max_iterations=80):
    """Adaptive test-time comparator; never a training-only prediction."""
    initial=moment(a,b); matrix=initial.copy(); score=iou_warp(a,b,matrix)
    result=dict(matrix=matrix.tolist(),iou=score,solver_status='not_attempted',
                selection='moment',correlation=None)
    try:
        aa=cv2.GaussianBlur(a.astype(np.float32),(9,9),2)
        bb=cv2.GaussianBlur(b.astype(np.float32),(9,9),2)
        correlation,proposal=cv2.findTransformECC(aa,bb,initial.astype(np.float32),
            cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,max_iterations,1e-5),
            None,5)
        result.update(solver_status='converged',correlation=float(correlation))
        proposal_score=iou_warp(a,b,proposal)
        if np.isfinite(proposal).all() and np.linalg.det(proposal[:,:2])>0 and proposal_score>score:
            result.update(matrix=proposal.tolist(),iou=proposal_score,selection='ecc')
    except cv2.error as exc:
        result.update(solver_status='opencv_error',reason=str(exc))
    return result
