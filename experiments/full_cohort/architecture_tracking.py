"""Unassisted rotational terminal trajectories; candidate organ hypotheses only.

This tests the inherited harmonic-trajectory premise without transferring its
source-assisted coordinates. A terminal is a medial-axis observation, not an
anatomical tip annotation. Matching uses TRAIN views, all held-out compatibility
scores are opened after frozen models exist. No botanic rank is assigned.
"""
import argparse,itertools,json,math,time
from pathlib import Path
import numpy as np
from architecture_analysis import read,write,dump,sha,accepted,coverage,load_mask,mask_candidates,KEYS,resolve_input,MaskIntegrityError

def design(views):
    t=np.asarray(views)*2*np.pi/40;return np.column_stack([np.ones(len(t)),np.cos(t),np.sin(t)])
def predict(track,view):return np.array([float(design([view])[0]@np.array(track['x_coefficient'])),track['height']])

def fit_tracks(candidates,gate=.07,max_tracks=8):
    """Deterministic minimal-triple proposal, greedy disjoint TRAIN support.

    RANSAC-like proposals are exhaustive at three separated anchor views; this
    is an image-support estimator, not a claim of optimal organ segmentation.
    """
    views=sorted(candidates)
    if any(v%4==2 for v in views):raise ValueError('Held-out views cannot enter terminal-trajectory fitting')
    if len(views)<12:return []
    remaining={v:list(enumerate(candidates[v])) for v in views};tracks=[];minimum=max(8,math.ceil(.4*len(views)))
    for _ in range(max_tracks):
        proposals=[]
        for targets in [(0,13,27),(4,17,31),(8,21,35)]:
            anchors=[min(views,key=lambda v:min(abs(v-t),40-abs(v-t))) for t in targets]
            if len(set(anchors))<3:continue
            aa=design(anchors)
            if np.linalg.cond(aa)>10:continue
            choices=[remaining[v] for v in anchors]
            for combo in itertools.product(*choices):
                xy=np.array([[c['x'],c['y']] for i,c in combo])
                if np.ptp(xy[:,1])>gate*2:continue
                coef=np.linalg.solve(aa,xy[:,0]);height=float(np.mean(xy[:,1]));radius=float(np.hypot(coef[1],coef[2]))
                if radius>1.5 or abs(coef[0])>.5:continue
                proposals.append((coef,height))
        if not proposals:break
        bx=np.stack([q[0] for q in proposals]);by=np.array([q[1] for q in proposals]);pred=design(views)@bx.T;support=np.zeros(len(proposals),int);loss=np.zeros(len(proposals))
        for j,v in enumerate(views):
            if not remaining[v]:loss+=gate;continue
            xy=np.array([[c['x'],c['y']] for i,c in remaining[v]])
            distances=np.sqrt((pred[j,:,None]-xy[None,:,0])**2+(by[:,None]-xy[None,:,1])**2);best=distances.min(1);support+=best<gate;loss+=np.minimum(best,gate)
        best=int(np.lexsort((loss,-support))[0]);coef,height=proposals[best]
        if support[best]<minimum:break
        selected=[]
        for iteration in range(4):
            selected=[]
            for v in views:
                p=np.array([design([v])[0]@coef,height]);choices=remaining[v]
                if not choices:continue
                dd=np.array([np.linalg.norm(p-[c['x'],c['y']]) for i,c in choices]);index=int(dd.argmin())
                if dd[index]<gate:selected.append((v,choices[index][0],choices[index][1],float(dd[index])))
            if len(selected)<minimum:break
            a=design([s[0] for s in selected]);xy=np.array([[s[2]['x'],s[2]['y']] for s in selected]);coef=np.linalg.lstsq(a,xy[:,0],rcond=None)[0];height=float(np.mean(xy[:,1]))
        if len(selected)<minimum:break
        track=dict(track_id=len(tracks),x_coefficient=coef.tolist(),height=height,radius=float(np.hypot(coef[1],coef[2])),training_support=len(selected),training_total_views=len(views),training_rmse=float(np.sqrt(np.mean([s[3]**2 for s in selected]))),root_length=float(np.median([s[2].get('root_length',np.nan) for s in selected])),assignments=[dict(view=v,candidate_id=i,distance=d) for v,i,c,d in selected],status='unassisted_rotational_terminal_hypothesis')
        tracks.append(track)
        for v,i,c,d in selected:remaining[v]=[(j,q) for j,q in remaining[v] if j!=i]
    return tracks

def run(args):
    started=time.time();out=args.run/'architecture'/args.output_name/'tracking';out.mkdir(parents=True,exist_ok=True);source,maskroot,mode=resolve_input(args,out)
    sources={}
    for name in ['architecture_tracking.py','architecture_analysis.py']:
        data=(Path(__file__).parent/name).read_bytes();(out/('execution_'+name)).write_bytes(data)
        import hashlib
        sources[name]=hashlib.sha256(data).hexdigest()
    rows=read(source);groups={}
    for row in rows:
        if row['camera']=='rgb':groups.setdefault(tuple(row[k] for k in KEYS),[]).append(row)
    statuses=[];trackrows=[];evalrows=[];associationrows=[];models={};errors=[]
    for count,(key,group) in enumerate(sorted(groups.items()),1):
        identity=dict(zip(KEYS,key));good=[r for r in group if accepted(r)];views=[int(r['view']) for r in good];cv=coverage(views)
        if not cv['eligible']:statuses.append({**identity,'status':'insufficient_coverage',**cv});continue
        train=[r for r in good if int(r['view'])%4!=2];test=[r for r in good if int(r['view'])%4==2]
        if len(train)<15 or len(test)<3:statuses.append({**identity,'status':'insufficient_train_or_test',**cv});continue
        try:
            # Gauge is estimated from training images; no test mask enters here.
            raw={};axes=[];bases=[];heights=[]
            for row in train:
                mask=load_mask(row,maskroot);yy,xx=np.where(mask);axes.append(float(np.median(xx[yy>=np.percentile(yy,95)])));bases.append(float(yy.max()));heights.append(float(np.ptp(yy)+1));raw[int(row['view'])]=mask_candidates(mask)
            axis=float(np.median(axes));base=float(np.median(bases));scale=float(np.median(heights))
            def normalized(cc):return [dict(x=(c['terminal_x_px']-axis)/scale,y=(base-c['terminal_y_px'])/scale,root_length=c['root_geodesic_over_height']*c['mask_height_px']/scale) for c in cc if not c['censored']]
            trainc={v:normalized(cc) for v,cc in raw.items()};tracks=fit_tracks(trainc,gate=args.gate);models[key]=tracks
            # Models are persisted before held-out masks are opened.
            name=hashlib_name(key);dump(out/f'frozen_{name}.json',dict(identity=identity,axis=axis,base=base,height=scale,gate=args.gate,train_views=sorted(trainc),tracks=tracks))
            testc={int(row['view']):normalized(mask_candidates(load_mask(row,maskroot))) for row in test}
            for track in tracks:
                tr={k:v for k,v in track.items() if k not in ['assignments','x_coefficient']};trackrows.append({**identity,**tr,'x_intercept':track['x_coefficient'][0],'x_cos':track['x_coefficient'][1],'x_sin':track['x_coefficient'][2]})
                for a in track['assignments']:associationrows.append({**identity,'track_id':track['track_id'],**a})
                for v,cc in testc.items():
                    points=np.array([[c['x'],c['y']] for c in cc]);p=predict(track,v);wrong=predict(track,(v+10)%40);constant=np.array([track['x_coefficient'][0],track['height']])
                    distances=[float(np.linalg.norm(points-q,axis=1).min()) if len(points) else None for q in [p,wrong,constant]]
                    evalrows.append({**identity,'track_id':track['track_id'],'view':v,'candidate_count':len(cc),'nearest_terminal_distance':distances[0],'wrong_quarter_view_distance':distances[1],'constant_model_distance':distances[2],'compatible_at_gate':bool(distances[0] is not None and distances[0]<args.gate),'evaluation':'nearest_unannotated_terminal_compatibility_not_tracking_accuracy'})
            statuses.append({**identity,'status':'executed','track_hypotheses':len(tracks),'train_views':len(train),'heldout_views':len(test),**cv})
        except Exception as e:
            if isinstance(e,MaskIntegrityError):raise
            errors.append({**identity,'error':str(e)});statuses.append({**identity,'status':'failed','error':str(e),**cv})
        if count%20==0:print(json.dumps(dict(stage='tracking',groups=count,total=len(groups))),flush=True)
    cross=[]
    for key,early in models.items():
        for later,late in models.items():
            if (key[0],key[2],key[3])!=(later[0],later[2],later[3]) or int(later[1])!=int(key[1])+1:continue
            for a in early:
                costs=[abs(a['height']-b['height'])/.15+abs(a['radius']-b['radius'])/.2+abs(a['root_length']-b['root_length'])/.3 for b in late];den=math.exp(-2)+sum(math.exp(-min(c,700)) for c in costs)
                for b,c in zip(late,costs):cross.append(dict(treatment=key[0],genotype=key[2],replicate=key[3],earlier_visit=key[1],later_visit=later[1],earlier_track=a['track_id'],later_track=b['track_id'],cost=c,conditional_weight=math.exp(-min(c,700))/den,unmatched_weight=math.exp(-2)/den,interpretation='conditional_shape_map_no_selected_organ_identity'))
    write(out/'tracking_cases.csv',statuses);write(out/'rotational_track_hypotheses.csv',trackrows);write(out/'training_assignments.csv',associationrows);write(out/'heldout_terminal_compatibility.csv',evalrows);write(out/'cross_visit_track_ambiguity.csv',cross);write(out/'tracking_errors.csv',errors)
    valid=[r for r in evalrows if r['nearest_terminal_distance'] is not None];summary=dict(mode=mode,input_sha256=sha(source),execution_source_sha256=sources,source_changed_during_run=any(sha(Path(__file__).parent/name)!=value for name,value in sources.items()),rgb_camera_sessions=len(groups),executed_sessions=sum(r['status']=='executed' for r in statuses),track_hypotheses=len(trackrows),heldout_rows=len(evalrows),heldout_rows_with_terminal=len(valid),compatible_rows=sum(r['compatible_at_gate'] for r in evalrows),candidate_cross_visit_pair_rows=len(cross),selected_physical_correspondences=0,gate_height_fraction=args.gate,errors=len(errors),elapsed_seconds=time.time()-started)
    for metric in ['nearest_terminal_distance','wrong_quarter_view_distance','constant_model_distance']:summary['median_'+metric]=float(np.median([r[metric] for r in valid])) if valid else None
    dump(out/'tracking_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

def hashlib_name(key):
    import hashlib
    return hashlib.sha256('|'.join(key).encode()).hexdigest()[:20]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--intake',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--gate',type=float,default=.07);p.add_argument('--input-table',type=Path);p.add_argument('--records-snapshot',action='store_true');p.add_argument('--output-name',default='');run(p.parse_args())
