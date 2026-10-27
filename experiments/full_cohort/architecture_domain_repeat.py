"""Repeat the same conditional camera family with training-supported domain bounds."""
import argparse,itertools,json,math,time
from pathlib import Path
import numpy as np
from architecture_analysis import read,write,dump,accepted,load_mask,sha
from architecture_geometry_diagnostics import carve,scores

def run(args):
    started=time.time();out=args.run/'architecture'/'geometry_domain_repeat';out.mkdir(exist_ok=True);frames=read(args.run/'extraction/view_traits.csv');cases=read(args.run/'architecture/geometry_diagnostics/geometry_diagnostic_cases.csv');results=[];selected=[];domains=[]
    for name in ['architecture_domain_repeat.py','architecture_geometry_diagnostics.py','architecture_analysis.py']:(out/('execution_'+name)).write_bytes((Path(__file__).parent/name).read_bytes())
    for ci,case in enumerate(cases):
        rr=[r for r in frames if r['session']==case['session'] and r['camera']=='rgb' and accepted(r)];masks={int(r['view']):load_mask(r,args.run,160) for r in rr};train={v:m for v,m in masks.items() if v%4!=2};axes=[];bases=[];heights=[];bounds=[]
        for m in train.values():
            y,x=np.where(m);axes.append(float(np.median(x[y>=np.percentile(y,95)])));bases.append(float(y.max()));heights.append(float(np.ptp(y)+1));bounds.append((float(x.min()),float(x.max()),float(y.min()),float(y.max())))
        axis=float(np.median(axes));base=float(np.median(bases));scale=float(np.median(heights));radius=max(.8,math.ceil((max(max(abs(x0-axis),abs(x1-axis))/scale for x0,x1,y0,y1 in bounds)+.05)*20)/20);top=max(1.15,math.ceil((max((base-y0)/scale for x0,x1,y0,y1 in bounds)+.05)*20)/20);bottom=min(0,math.floor((min((base-y1)/scale for x0,x1,y0,y1 in bounds)-.02)*50)/50);domain=(radius,bottom,top);shape=next(iter(masks.values())).shape;domains.append({k:case[k] for k in ['treatment','measurement','genotype','replicate','session']}|dict(radius=radius,ymin=bottom,ymax=top,rule='TRAIN silhouette support plus .05height horizontal/top and .02height basal margin; no heldout mask used.'))
        modelrows={}
        for distance,roll,elevation in itertools.product([2.,4.,8.],[0.,3.],[0.,5.]):
            camera=dict(axis=axis,base=base,scale=scale,principal=[shape[1]/2,shape[0]/2],distance=distance,roll=roll,elevation=elevation);hull,grid=carve(train,camera,96,'conservative_cell_bbox',bounds=domain);step=np.array([2*radius,top-bottom,2*radius])/95;metrics=scores(hull,grid,masks,camera,step);key=(distance,roll,elevation);modelrows[key]=metrics
            for row in metrics:results.append(dict(session=case['session'],distance_over_height=distance,roll_deg=roll,elevation_deg=elevation,radius=radius,ymin=bottom,ymax=top,resolution=96,occupied_voxels=int(hull.sum()),**row))
        best=min(modelrows,key=lambda k:(-np.median([r['iou'] for r in modelrows[k] if r['split']=='train']),k));chosen=modelrows[best];selected.append({k:case[k] for k in ['treatment','measurement','genotype','replicate','session']}|dict(distance_over_height=best[0],roll_deg=best[1],elevation_deg=best[2],train_median_iou=float(np.median([r['iou'] for r in chosen if r['split']=='train'])),heldout_median_iou=float(np.median([r['iou'] for r in chosen if r['split']=='heldout'])),selection='TRAIN median only; retained domain repeat is exploratory adequacy check'))
        print(json.dumps(dict(completed=ci+1,total=len(cases))),flush=True)
    write(out/'camera_alternatives.csv',results);write(out/'train_selected_camera.csv',selected);write(out/'training_domains.csv',domains);dump(out/'domain_repeat_summary.json',dict(cases=len(cases),score_rows=len(results),median_case_train_iou=float(np.median([r['train_median_iou'] for r in selected])),median_case_heldout_iou=float(np.median([r['heldout_median_iou'] for r in selected])),heldout_case_range=[float(min(r['heldout_median_iou'] for r in selected)),float(max(r['heldout_median_iou'] for r in selected))],input_sha256=sha(args.run/'extraction/view_traits.csv'),script_sha256=sha(__file__),elapsed_seconds=time.time()-started,scope='Training-supported bounded-domain repeat after identifying undersized fixed-box priors. All camera alternatives retained, training-only selection. Exploratory observation-model adequacy; no anatomical surface validation.'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);run(p.parse_args())
