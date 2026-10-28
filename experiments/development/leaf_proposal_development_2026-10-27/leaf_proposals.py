"""Prompt geometry and accounting for unreviewed terminal-region proposals."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import numpy as np
import cv2
cv2.setNumThreads(2)

POLICY=dict(distal_branch_fraction=.8,positive_fractions=[.35,.65,.9],
            minimum_branch_pixels=6,negative_target_distance=6,
            maximum_negatives=3,maximum_track_nodes_per_frame=4,
            minimum_noir_match_gate_pixels=6,noir_match_height_fraction=.08,
            source_grid=[256,192],unreviewed=True)

def processing_prompt(prompt,parent,grid_shape=(192,256)):
    """PIL nearest-neighbour grid centres plus explicit exact-mask support."""
    h,w=parent.shape;gh,gw=grid_shape;scale=np.array([w/gw,h/gh])
    convert=lambda xy:(np.asarray(xy,float)+.5)*scale-.5
    before=[convert(xy).tolist() for xy in prompt['points']]
    refined=[];distances=[];inside=[]
    for xy,label in zip(before,prompt['labels']):
        xy=np.array(xy);x,y=np.round(xy).astype(int)
        supported=0<=x<w and 0<=y<h and parent[y,x]
        original=xy.copy()
        if not supported:
            radius=6;lo_x=max(0,x-radius);lo_y=max(0,y-radius)
            yy,xx=np.where(parent[lo_y:min(h,y+radius+1),lo_x:min(w,x+radius+1)])
            if len(xx):
                choices=np.column_stack([xx+lo_x,yy+lo_y])
                xy=choices[np.argmin(np.linalg.norm(choices-xy,axis=1))].astype(float)
                supported=True
            elif label==1:raise ValueError('processing_positive_cannot_be_supported_within_6pixels')
        refined.append(xy.tolist());distances.append(float(np.linalg.norm(xy-original)));inside.append(bool(supported))
    low=np.clip(convert(prompt['box'][:2]),[0,0],[w-1,h-1]);high=np.clip(convert(prompt['box'][2:]),[0,0],[w-1,h-1])
    return dict(box=[*low.tolist(),*high.tolist()],points=refined,labels=prompt['labels'],
        points_before_support_refinement=before,support_refinement_distance_pixels=distances,
        point_inside_exact_parent=inside,grid_conversion='pixel centers: (coordinate+0.5)*scale-0.5',
        refinement='nearest exact parent foreground within6processing pixels; unsupported positives abstain')

def make_prompt(path,terminal,branch_length,parent,other_terminals):
    points=np.asarray(path,dtype=float);terminal=np.asarray(terminal,float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<2:raise ValueError('invalid_path')
    if np.linalg.norm(points[0]-terminal)<np.linalg.norm(points[-1]-terminal):points=points[::-1]
    if np.linalg.norm(points[-1]-terminal)>2:raise ValueError('path_terminal_orientation_mismatch')
    h,w=parent.shape
    if not (2<=terminal[0]<w-2 and 2<=terminal[1]<h-2):raise ValueError('censored_terminal')
    if branch_length<POLICY['minimum_branch_pixels']:raise ValueError('distal_branch_too_short')
    distances=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    distinct=np.r_[True,np.diff(distances)>1e-8];points=points[distinct];distances=distances[distinct]
    total=distances[-1]
    if total<6:raise ValueError('path_too_short')
    distal=min(float(branch_length)*POLICY['distal_branch_fraction'],total)
    cut=total-distal
    at=lambda d:np.array([np.interp(d,distances,points[:,j]) for j in range(2)])
    trimmed=np.vstack([at(cut),points[distances>cut]])
    positive=[]
    for fraction in POLICY['positive_fractions']:
        xy=at(cut+fraction*distal);x,y=np.round(xy).astype(int)
        if not (0<=x<w and 0<=y<h and parent[y,x]):
            yy,xx=np.where(parent[max(0,y-3):min(h,y+4),max(0,x-3):min(w,x+4)])
            if not len(xx):raise ValueError('positive_outside_parent')
            choices=np.column_stack([xx+max(0,x-3),yy+max(0,y-3)])
            xy=choices[np.argmin(np.linalg.norm(choices-xy,axis=1))].astype(float)
        positive.append(xy.tolist())
    negative=[]
    for xy in sorted(other_terminals,key=lambda x:float(np.linalg.norm(np.asarray(x)-terminal))):
        xy=np.asarray(xy,float)
        if not (0<=xy[0]<w and 0<=xy[1]<h):continue
        if np.linalg.norm(xy-terminal)<POLICY['negative_target_distance']:continue
        if min(np.linalg.norm(xy-np.array(p)) for p in positive)<4:continue
        if any(np.linalg.norm(xy-np.array(p))<2 for p in negative):continue
        negative.append(xy.tolist())
        if len(negative)==POLICY['maximum_negatives']:break
    dt=cv2.distanceTransform(parent.astype(np.uint8),cv2.DIST_L2,3)
    sample=np.round(trimmed).astype(int);sample[:,0]=np.clip(sample[:,0],0,w-1);sample[:,1]=np.clip(sample[:,1],0,h-1)
    padding=max(3,float(np.quantile(dt[sample[:,1],sample[:,0]],.9))*1.4)
    low=np.maximum(trimmed.min(axis=0)-padding,[0,0]);high=np.minimum(trimmed.max(axis=0)+padding,[w-1,h-1])
    if np.any(high-low<2):raise ValueError('degenerate_local_box')
    return dict(box=[*low.tolist(),*high.tolist()],points=positive+negative,
                labels=[1]*len(positive)+[0]*len(negative),distal_path=trimmed.tolist(),
                terminal=terminal.tolist(),branch_length_pixels=float(branch_length),
                retained_distal_length_pixels=distal,path_total_length_pixels=float(total),
                interpretation='unreviewed_distal_terminal_region_not_anatomical_leaf')

def match_terminal(mapped,candidates,gate):
    ranked=sorted([(float(np.linalg.norm(np.array([c['terminal_x_px'],c['terminal_y_px']])-mapped)),c)
                   for c in candidates],key=lambda x:(x[0],x[1]['candidate_id']))
    distance=ranked[0][0] if ranked else None
    return dict(candidate=ranked[0][1] if ranked and distance<=gate else None,
                nearest_distance=distance,gate=float(gate),
                runner_distance=ranked[1][0] if len(ranked)>1 else None,
                ambiguous=bool(len(ranked)>1 and ranked[1][0]-ranked[0][0]<2))

def clip_statistics(raw,parent):
    raw=np.asarray(raw,bool);parent=np.asarray(parent,bool)
    clipped=raw&parent;n=int(raw.sum());cn=int(clipped.sum());pn=int(parent.sum())
    return clipped,dict(raw_area_pixels=n,clipped_area_pixels=cn,parent_area_pixels=pn,
        raw_outside_parent_pixels=n-cn,raw_outside_parent_fraction=(n-cn)/n if n else None,
        parent_coverage_fraction=cn/pn if pn else None,empty_clipped=cn==0)

def coco_rle(mask):
    flat=np.asarray(mask,dtype=np.uint8).ravel(order='F')
    counts=[];last=0;run=0
    for value in flat:
        if value==last:run+=1
        else:counts.append(run);run=1;last=int(value)
    counts.append(run)
    return dict(size=list(mask.shape),counts=counts)
