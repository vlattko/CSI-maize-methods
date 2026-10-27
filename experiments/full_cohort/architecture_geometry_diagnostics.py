"""Numerical adequacy and conditional perspective-family geometry diagnostics.

Conservative carving retains any voxel whose projected axis-aligned bounding
box intersects every TRAIN silhouette. This is a computable outer bound on
voxel/cone intersection, explicitly looser than exact polyhedral carving.
All alternatives are retained; held-out images do not select camera parameters.
"""
import argparse,itertools,json,math,time,hashlib
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from architecture_analysis import read,write,dump,sha,accepted,coverage,load_mask,resolve_input,KEYS

def project_points(points,view,camera):
    p=np.asarray(points);t=view*2*np.pi/40;e=np.radians(camera['elevation']);r=np.radians(camera['roll']);cr,sr=np.cos(r),np.sin(r)
    xx=np.cos(t)*p[:,0]+np.sin(t)*p[:,2];zz=-np.sin(t)*p[:,0]+np.cos(t)*p[:,2];up=np.cos(e)*p[:,1]+np.sin(e)*zz
    if camera['distance'] is None:return np.column_stack([camera['axis']+camera['scale']*(cr*xx+sr*up),camera['base']+camera['scale']*(sr*xx-cr*up)])
    cx,cy=camera['principal'];dx=(camera['axis']-cx)/camera['scale'];dy=(camera['base']-cy)/camera['scale'];xx=xx+cr*dx+sr*dy;up=up+sr*dx-cr*dy;d=camera['distance'];depth=d+np.cos(e)*zz-np.sin(e)*p[:,1]
    # The declared family keeps this domain strictly in front of the camera.
    if np.any(depth<=0):raise ValueError('Conditional camera domain crosses focal plane')
    return np.column_stack([cx+camera['scale']*d*(cr*xx+sr*up)/depth,cy+camera['scale']*d*(sr*xx-cr*up)/depth])

def cell_bounds(points,step,view,camera):
    lo=np.full((len(points),2),np.inf);hi=np.full((len(points),2),-np.inf)
    for signs in itertools.product([-1,1],repeat=3):
        q=project_points(points+np.asarray(signs)*step/2,view,camera);lo=np.minimum(lo,q);hi=np.maximum(hi,q)
    return np.floor(lo).astype(int),np.ceil(hi).astype(int)

def carve(masks,camera,resolution=64,operator='center',bounds=None):
    radius,ymin,ymax=bounds if bounds is not None else (.8,0,1.15)
    xx=np.linspace(-radius,radius,resolution,dtype='float32');yy=np.linspace(ymin,ymax,resolution,dtype='float32');x,y,z=np.meshgrid(xx,yy,xx,indexing='ij');grid=np.column_stack([x.ravel(),y.ravel(),z.ravel()]);alive=np.arange(len(grid));step=np.array([2*radius,ymax-ymin,2*radius])/(resolution-1)
    for view,mask in sorted(masks.items()):
        if not len(alive):break
        points=grid[alive]
        if operator=='center':
            xy=np.rint(project_points(points,view,camera)).astype(int);valid=(xy[:,0]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]>=0)&(xy[:,1]<mask.shape[0]);inside=np.zeros(len(points),bool);inside[valid]=mask[xy[valid,1],xy[valid,0]]
        else:
            lo,hi=cell_bounds(points,step,view,camera);valid=(hi[:,0]>=0)&(lo[:,0]<mask.shape[1])&(hi[:,1]>=0)&(lo[:,1]<mask.shape[0]);lo=np.clip(lo,[0,0],[mask.shape[1]-1,mask.shape[0]-1]);hi=np.clip(hi,[0,0],[mask.shape[1]-1,mask.shape[0]-1]);integral=cv2.integral(mask.astype('uint8'));counts=integral[hi[:,1]+1,hi[:,0]+1]-integral[lo[:,1],hi[:,0]+1]-integral[hi[:,1]+1,lo[:,0]]+integral[lo[:,1],lo[:,0]];inside=valid&(counts>0)
        alive=alive[inside]
    hull=np.zeros(len(grid),bool);hull[alive]=True;return hull.reshape(x.shape),grid

def render_cells(hull,grid,view,camera,shape,step=None):
    points=grid[hull.ravel()];out=np.zeros(shape,bool)
    if not len(points):return out
    step=np.array([1.6,1.15,1.6])/(hull.shape[0]-1) if step is None else np.asarray(step);lo,hi=cell_bounds(points,step,view,camera);valid=(hi[:,0]>=0)&(lo[:,0]<shape[1])&(hi[:,1]>=0)&(lo[:,1]<shape[0]);lo=lo[valid];hi=hi[valid];lo=np.clip(lo,[0,0],[shape[1]-1,shape[0]-1]);hi=np.clip(hi,[0,0],[shape[1]-1,shape[0]-1]);diff=np.zeros((shape[0]+1,shape[1]+1),np.int32)
    np.add.at(diff,(lo[:,1],lo[:,0]),1);np.add.at(diff,(hi[:,1]+1,lo[:,0]),-1);np.add.at(diff,(lo[:,1],hi[:,0]+1),-1);np.add.at(diff,(hi[:,1]+1,hi[:,0]+1),1)
    return np.cumsum(np.cumsum(diff,0),1)[:-1,:-1]>0

def scores(hull,grid,masks,camera,step=None):
    rows=[]
    for v,m in sorted(masks.items()):
        p=render_cells(hull,grid,v,camera,m.shape,step);intersection=(p&m).sum();rows.append(dict(view=v,split='heldout' if v%4==2 else 'train',iou=float(intersection/max((p|m).sum(),1)),precision=float(intersection/max(p.sum(),1)),recall=float(intersection/max(m.sum(),1))))
    return rows

def synthetic_masks(camera):
    # Continuous thin sheets are rasterized directly, independent of voxel grid.
    vertices=[];faces=[]
    for az,height,length in [(0,.3,.45),(1.9,.5,.5),(3.8,.7,.4)]:
        start=len(vertices)
        for along,across in [(0,-.008),(length,-.008),(length,.008),(0,.008)]:vertices.append([along*np.cos(az)-across*np.sin(az),height+.3*along,along*np.sin(az)+across*np.cos(az)])
        faces.extend([[start,start+1,start+2],[start,start+2,start+3]])
    vertices=np.array(vertices);masks={}
    for v in range(40):
        xy=np.rint(project_points(vertices,v,camera)).astype('int32');mask=np.zeros((120,160),np.uint8)
        for face in faces:cv2.fillConvexPoly(mask,xy[face],1)
        masks[v]=mask.astype(bool)
    return masks

def select_cases(groups,limit):
    eligible=[(k,g) for k,g in sorted(groups.items()) if k[-1]=='rgb' and coverage([r['view'] for r in g if accepted(r)])['strict_eligible']]
    # One candidate per treatment/genotype, choose most covered repeated label,
    # then balance treatment before visits. No geometry result enters selection.
    by={}
    for k,g in eligible:by.setdefault((k[0],k[2],k[3]),[]).append((k,g))
    chosen=[]
    for treatment in sorted(set(k[0] for k in by)):
        geno=sorted(set(k[1] for k in by if k[0]==treatment))
        for genotype in geno:
            ids=[k for k in by if k[:2]==(treatment,genotype)];best=min(ids,key=lambda k:(-len(by[k]),k[2]));chosen.append((best,by[best]))
    # Prefer at least one whole trajectory for each treatment before filling.
    treatments=sorted(set(k[0] for k,g in chosen));order=[]
    for rank in range(max([sum(k[0]==t for k,g in chosen) for t in treatments] or [0])):
        for treatment_index,treatment in enumerate(treatments):
            options=[(k,g) for k,g in chosen if k[0]==treatment]
            if rank<len(options):
                # Rotate genotype preference across treatments, independent of
                # any geometry result, while preserving complete trajectories.
                option=options[(rank+treatment_index)%len(options)]
                order.extend(sorted(option[1],key=lambda kg:int(kg[0][1])))
    return order[:limit]

def run(args):
    started=time.time();out=args.run/'architecture'/args.output_name/'geometry_diagnostics';out.mkdir(parents=True,exist_ok=True);source,maskroot,mode=resolve_input(args,out)
    sources={}
    for name in ['architecture_geometry_diagnostics.py','architecture_analysis.py']:
        body=(Path(__file__).parent/name).read_bytes();(out/('execution_'+name)).write_bytes(body);sources[name]=hashlib.sha256(body).hexdigest()
    groups={}
    for row in read(source):groups.setdefault(tuple(row[k] for k in KEYS),[]).append(row)
    cases=select_cases(groups,args.cases);results=[];case_rows=[]
    camera=dict(axis=80,base=113,scale=95,principal=[80,60],distance=None,roll=0,elevation=0)
    tasks=[('synthetic_continuous_thin_ribbons',{},synthetic_masks(camera),camera)]
    for key,group in cases:
        identity=dict(zip(KEYS,key));masks={int(r['view']):load_mask(r,maskroot,160) for r in group if accepted(r)};train={v:m for v,m in masks.items() if v%4!=2};axes=[];bases=[];heights=[]
        for m in train.values():
            yy,xx=np.where(m);axes.append(float(np.median(xx[yy>=np.percentile(yy,95)])));bases.append(float(yy.max()));heights.append(float(np.ptp(yy)+1))
        shape=next(iter(train.values())).shape;cam=dict(axis=float(np.median(axes)),base=float(np.median(bases)),scale=float(np.median(heights)),principal=[shape[1]/2,shape[0]/2],distance=None,roll=0,elevation=0);tasks.append((key[4],identity,masks,cam));case_rows.append({**identity,'train_views':len(train),'heldout_views':sum(v%4==2 for v in masks),'bottom_censored_train_views':sum(bool(np.any(m[-2:])) for m in train.values()),'axis_sd_train_px':float(np.std(axes)),'height_sd_train_px':float(np.std(heights)),'selection':'coverage_and_metadata_only'})
    for i,(label,identity,masks,camera) in enumerate(tasks):
        train={v:m for v,m in masks.items() if v%4!=2}
        alternatives=[(resolution,operator,None,0,0,'numerical_adequacy') for resolution in [64,96,128] for operator in ['center','conservative_cell_bbox']]
        if i>0:
            alternatives += [(96,'conservative_cell_bbox',distance,roll,elevation,'perspective_family') for distance,roll,elevation in itertools.product([2.,4.,8.],[0.,3.],[0.,5.])]
        for resolution,operator,distance,roll,elevation,experiment in alternatives:
            cam={**camera,'distance':distance,'roll':roll,'elevation':elevation};hull,grid=carve(train,cam,resolution,operator);metrics=scores(hull,grid,masks,cam)
            for row in metrics:results.append({**identity,'case':label,'experiment':experiment,'resolution':resolution,'operator':operator,'distance_over_height':distance if distance is not None else 'orthographic','roll_deg':roll,'elevation_deg':elevation,'occupied_voxels':int(hull.sum()),'voxel_volume_gauge':float(hull.sum()*1.6*1.6*1.15/(resolution-1)**3),**row})
        print(json.dumps(dict(stage='geometry_diagnostics',completed=i+1,total=len(tasks))),flush=True)
    write(out/'numerical_and_camera_scores.csv',results);write(out/'geometry_diagnostic_cases.csv',case_rows)
    # Selection within the camera family uses training median only. Report the
    # separately frozen model's held-out score, retain every alternative as well.
    selected=[]
    for label in sorted(set(r['case'] for r in results)):
        rr=[r for r in results if r['case']==label and r['experiment']=='perspective_family'];models={}
        for r in rr:models.setdefault((r['distance_over_height'],r['roll_deg'],r['elevation_deg']),[]).append(r)
        if not models:continue
        best=min(models,key=lambda k:(-np.median([r['iou'] for r in models[k] if r['split']=='train']),k));br=models[best];selected.append(dict(case=label,distance_over_height=best[0],roll_deg=best[1],elevation_deg=best[2],train_median_iou=float(np.median([r['iou'] for r in br if r['split']=='train'])),heldout_median_iou=float(np.median([r['iou'] for r in br if r['split']=='heldout'])),selection='max_training_median_iou_only; development_camera_family_not_external_validation'))
    write(out/'train_selected_camera_diagnostic.csv',selected);dump(out/'geometry_diagnostic_summary.json',dict(mode=mode,input_sha256=sha(source),execution_sources=sources,real_cases=len(cases),synthetic_cases=1,score_rows=len(results),elapsed_seconds=time.time()-started,claim='Voxel/camera observation-model adequacy diagnostics; no anatomy, organ count, physiology, or biological failure inferred.',conservative_operator='Projected voxel axis-aligned bounding-box intersection is an outer bound, not exact voxel/cone intersection. Predictions rasterize the same conservative cell boxes.',camera_selection='Training median IoU only within the predeclared perspective family; all alternatives retained. Heldout views are not selected on.'))
    print(json.dumps(dict(real_cases=len(cases),selected=selected,elapsed_seconds=time.time()-started),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--intake',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--cases',type=int,default=6);p.add_argument('--input-table',type=Path);p.add_argument('--records-snapshot',action='store_true');p.add_argument('--output-name',default='');run(p.parse_args())
