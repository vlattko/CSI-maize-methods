"""Conditional rigid turntable point models; no organ surfaces or calibration."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='2'
import itertools
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
_THREAD_LIMITER=threadpool_limits(limits=2)


def families():
    out=[]
    for kind,focal,elevation,roll in itertools.product(['perspective'],[.8,1.2,2.,4.],[-5,0,5],[0,3]):
        out.append(dict(family=f'p_f{focal:g}_e{elevation}_r{roll}',kind=kind,focal=focal,elevation=elevation,roll=roll))
    for elevation,roll in itertools.product([-5,0,5],[0,3]):
        out.append(dict(family=f'o_e{elevation}_r{roll}',kind='orthographic',focal=1.,elevation=elevation,roll=roll))
    return out


def project_point(point,views,family,offset=0.,phase_degrees=0.,strict=False):
    x,y,z=np.asarray(point,float);t=np.asarray(list(views),float)*np.pi/20+np.radians(phase_degrees)
    X=x*np.cos(t)+z*np.sin(t);Z=-x*np.sin(t)+z*np.cos(t)
    e=np.radians(family['elevation']);r=np.radians(family['roll']);up=np.cos(e)*y+np.sin(e)*Z
    depth=1+np.cos(e)*Z-np.sin(e)*y
    if family['kind']=='perspective':
        if strict and np.min(depth)<.15:raise ValueError('Point crosses the minimum positive camera-depth constraint')
        divisor=np.maximum(depth,.05)
    else:divisor=np.ones(len(t))
    u=family['focal']*(np.cos(r)*X+np.sin(r)*up)/divisor+offset
    v=family['focal']*(-np.sin(r)*X+np.cos(r)*up)/divisor
    return np.column_stack([u,v])


def depth_minimum(point,family):
    x,y,z=point;e=np.radians(family['elevation'])
    return float(1-abs(np.cos(e))*np.hypot(x,z)-np.sin(e)*y)


def fit_harmonic(views,xy,vertical=False):
    if any(v%4==2 for v in views):raise ValueError('Outer heldout views cannot fit a model')
    v=np.asarray(views);xy=np.asarray(xy);a=np.column_stack([np.ones(len(v)),np.cos(v*np.pi/20),np.sin(v*np.pi/20)])
    coef=np.linalg.lstsq(a,xy,rcond=None)[0]
    if not vertical:coef[1:,1]=0;coef[0,1]=np.mean(xy[:,1])
    residual=a@coef-xy
    return dict(kind='image_harmonic',vertical_harmonic=vertical,coefficient=coef.tolist(),training_rmse=float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))))


def fit_point(views,xy,family,offset=0.):
    views=list(views);xy=np.asarray(xy,float)
    if any(v%4==2 for v in views):raise ValueError('Outer heldout views cannot fit a point model')
    if len(views)<5:return dict(status='insufficient_training_observations',family=family)
    adjusted=xy-np.array([offset,0.])
    if family['kind']=='orthographic':
        basis=np.stack([project_point(p,views,family) for p in np.eye(3)],axis=-1).reshape(-1,3)
        p=np.linalg.lstsq(basis,adjusted.ravel(),rcond=None)[0];jac=basis;success=True;nfev=0
    else:
        h=fit_harmonic(views,adjusted,vertical=True);coef=np.asarray(h['coefficient']);initial=np.array([coef[1,0],coef[0,1],coef[2,0]])/family['focal']
        initial=np.clip(initial,[-.7,-.4,-.7],[.7,2.,.7])
        def residual(p):
            prediction=project_point(p,views,family,offset=offset)
            # Fixed-length barrier, with no effect for positive adequate depth.
            penalty=max(.15-depth_minimum(p,family),0.)*10
            return np.r_[(prediction-xy).ravel(),penalty]
        fit=least_squares(residual,initial,bounds=([-.85,-.5,-.85],[.85,2.5,.85]),loss='soft_l1',f_scale=.01,max_nfev=100,xtol=1e-9,ftol=1e-9,gtol=1e-9)
        p=fit.x;jac=fit.jac;success=bool(fit.success);nfev=fit.nfev
    predicted=project_point(p,views,family,offset=offset);rr=predicted-xy
    s=np.linalg.svd(jac,compute_uv=False);condition=float(s[0]/max(s[-1],1e-15));variance=float(np.sum(rr**2)/max(2*len(views)-3,1));covariance=np.linalg.pinv(jac.T@jac)*variance
    status='fitted' if success else 'optimizer_not_converged'
    if family['kind']=='perspective' and depth_minimum(p,family)<.149:status='inadequate_camera_depth'
    return dict(status=status,kind='kinematic_point',family=family,point=p.tolist(),offset=offset,
        distance_gauge=1.,training_rmse=float(np.sqrt(np.mean(np.sum(rr**2,axis=1)))),
        point_jacobian_condition=condition,point_parameter_covariance=covariance.tolist(),minimum_camera_depth=depth_minimum(p,family),
        optimizer_evaluations=nfev,parameter_order=['x','up','depth'],uncertainty='Conditional point-fit sensitivity; camera-family ambiguity is separate')


def predict_model(model,views):
    if model['kind']=='image_harmonic':
        v=np.asarray(views);a=np.column_stack([np.ones(len(v)),np.cos(v*np.pi/20),np.sin(v*np.pi/20)])
        return a@np.asarray(model['coefficient'])
    return project_point(model['point'],views,model['family'],offset=model['offset'])


def select_families(rows,tolerance=.01):
    valid=[r for r in rows if r.get('inner_case_median') is not None]
    if not valid:return dict(selected=None,retained=[],status='no_evaluable_inner_predictions')
    best=min(valid,key=lambda r:(r['inner_case_median'],r['family']));retained=[r['family'] for r in valid if r['inner_case_median']<=best['inner_case_median']+tolerance+1e-12]
    return dict(selected=best['family'],retained=retained,best_inner_median=best['inner_case_median'],tolerance=tolerance,
        status='family_selected_with_retained_alternatives',interpretation='Training-only numerical sensitivity; no uniquely calibrated camera implied')


def marginal_camera_information(rows,family):
    """Local camera sensitivity after removing each point's coordinate freedom.

    Camera parameters are log(f/image-width), elevation radians and roll radians.
    The common image offset/origin is held at its training-derived value. This
    conditional numerical information is not a physical calibration certificate.
    """
    information=np.zeros((3,3));step=1e-4
    for row in rows:
        views=row['views']
        if any(v%4==2 for v in views):raise ValueError('Outer angles cannot establish training conditioning')
        point=np.asarray(row['point'],float);offset=row.get('offset',0.);scale=row.get('scale',1.)
        jp=[]
        for j in range(3):
            d=np.zeros(3);d[j]=1e-5
            jp.append(((project_point(point+d,views,family,offset)-project_point(point-d,views,family,offset))/(2e-5*scale)).ravel())
        jp=np.column_stack(jp);jc=[]
        for parameter in ['focal','elevation','roll']:
            plus=dict(family);minus=dict(family)
            if parameter=='focal':plus[parameter]*=np.exp(step);minus[parameter]*=np.exp(-step)
            else:plus[parameter]+=np.degrees(step);minus[parameter]-=np.degrees(step)
            jc.append(((project_point(point,views,plus,offset)-project_point(point,views,minus,offset))/(2*step*scale)).ravel())
        jc=np.column_stack(jc);residual=jc-jp@np.linalg.lstsq(jp,jc,rcond=None)[0];information+=residual.T@residual
    values,vectors=np.linalg.eigh(information);values=np.maximum(values,0);rank=int(np.sum(values>max(values[-1]*1e-8,1e-12))) if len(rows) else 0
    return dict(information=information.tolist(),eigenvalues=values.tolist(),numerical_rank=rank,
        condition=float(values[-1]/values[0]) if rank==3 else None,parameter_order=['log_focal_over_width','elevation_radians','roll_radians'],
        point_nuisance_parameters_removed=True,training_offset_and_axis_held_fixed=True,points=len(rows),
        scope='Local model conditioning conditional on inferred correspondences, fixed turntable gauge and fixed training image origin; not calibrated uncertainty')
