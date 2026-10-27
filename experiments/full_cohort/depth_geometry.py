"""Camera-gauge operators for a bounded learned-depth turntable experiment."""
import numpy as np


def turntable(views,distance=4.,sign=1):
    result=[]
    for view in views:
        a=sign*int(view)*2*np.pi/40;c,s=np.cos(a),np.sin(a)
        e=np.eye(4);e[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]];e[:3,3]=[0,0,distance];result.append(e)
    return np.array(result)


def unproject(depth,mask,intrinsic,extrinsic,stride=2):
    y,x=np.where(mask&np.isfinite(depth)&(depth>0));y=y[::stride];x=x[::stride]
    rays=np.linalg.solve(intrinsic,np.stack([x,y,np.ones(len(x))]))
    return (extrinsic[:3,:3].T@(rays*depth[y,x]-extrinsic[:3,3,None])).T


def project(points,intrinsic,extrinsic,shape,radius=1):
    import cv2
    camera=extrinsic[:3,:3]@points.T+extrinsic[:3,3,None]
    finite=np.isfinite(camera).all(0)&(camera[2]>0);camera=camera[:,finite]
    screen=intrinsic@camera
    x=np.rint(screen[0]/camera[2]).astype(int);y=np.rint(screen[1]/camera[2]).astype(int)
    good=(x>=0)&(x<shape[1])&(y>=0)&(y<shape[0]);out=np.zeros(shape,np.uint8);out[y[good],x[good]]=1
    if radius:out=cv2.dilate(out,np.ones((2*radius+1,2*radius+1),np.uint8))
    return out.astype(bool)


def similarity(source,target):
    source=np.asarray(source,float);target=np.asarray(target,float)
    if source.shape!=target.shape or len(source)<3:raise ValueError('Three matching training camera centres required')
    a=source-source.mean(0);b=target-target.mean(0)
    if np.linalg.matrix_rank(a)<2 or np.linalg.matrix_rank(b)<2:raise ValueError('Degenerate training camera centres')
    u,s,v=np.linalg.svd(a.T@b);fix=np.eye(3);fix[-1,-1]=np.linalg.det(v.T@u.T)
    rotation=v.T@fix@u.T;scale=float(np.sum(s*np.diag(fix))/np.sum(a*a))
    if scale<=0 or not np.isfinite(scale):raise ValueError('Invalid camera similarity scale')
    shift=target.mean(0)-scale*(rotation@source.mean(0))
    return scale,rotation,shift
