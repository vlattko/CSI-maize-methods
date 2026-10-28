"""Visible skeleton evidence, without interpreting a path as an organ surface."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize


def extract_paths(mask,min_length=6,max_path_points=40):
    sk=skeletonize(mask);ys,xs=np.where(sk)
    if not len(xs):return []
    lookup=np.full(mask.shape,-1,int);lookup[ys,xs]=np.arange(len(xs));edges=[];weights=[]
    for dy,dx in [(0,1),(1,-1),(1,0),(1,1)]:
        ny=ys+dy;nx=xs+dx;ok=(ny>=0)&(ny<mask.shape[0])&(nx>=0)&(nx<mask.shape[1]);src=np.flatnonzero(ok);dst=lookup[ny[ok],nx[ok]];hit=dst>=0
        edges.extend(zip(src[hit],dst[hit]));weights.extend([np.hypot(dx,dy)]*int(hit.sum()))
    if not edges:return []
    edges=np.asarray(edges);g=csr_matrix((weights,(edges[:,0],edges[:,1])),shape=(len(xs),len(xs)));g=g+g.T;deg=np.diff(g.indptr)
    basal=xs[ys>=np.percentile(ys,95)];axis=float(np.median(basal));root=int(np.argmax(ys-.02*abs(xs-axis)))
    dist,previous=dijkstra(g,indices=root,return_predecessors=True);h=max(float(np.ptp(ys)+1),1);result=[]
    for tip in np.flatnonzero(deg==1):
        if tip==root or not np.isfinite(dist[tip]):continue
        path=[int(tip)];at=int(tip)
        while at!=root and previous[at]>=0:at=int(previous[at]);path.append(at)
        junction=next((i for i in path[1:] if deg[i]>=3),root);length=float(dist[tip]-dist[junction])
        if length<min_length:continue
        path=path[::-1];ix=np.unique(np.linspace(0,len(path)-1,min(max_path_points,len(path))).round().astype(int));pp=np.array(path)[ix]
        result.append(dict(candidate_id=len(result),terminal_x_px=int(xs[tip]),terminal_y_px=int(ys[tip]),
            root_x_px=int(xs[root]),root_y_px=int(ys[root]),mask_height_px=h,
            root_geodesic_over_height=float(dist[tip]/h),visible_branch_over_height=length/h,
            path_processing_px=[[int(xs[i]),int(ys[i])] for i in pp],
            censored=bool(xs[tip]<3 or xs[tip]>=mask.shape[1]-3 or ys[tip]<3 or ys[tip]>=mask.shape[0]-3),
            status='observed_skeleton_path_not_anatomical_mid_rib'))
    return result


def normalize_paths(paths,axis,base,height):
    return [dict(x=(c['terminal_x_px']-axis)/height,y=(base-c['terminal_y_px'])/height,
        root_length=c['root_geodesic_over_height']*c['mask_height_px']/height,
        original_candidate_id=c['candidate_id'],px=c['terminal_x_px'],py=c['terminal_y_px'],
        path_processing_px=c['path_processing_px']) for c in paths if not c['censored']]
