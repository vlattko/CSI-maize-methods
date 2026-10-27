"""Terminal-trajectory benchmark on rendered Pheno4D point geometry.

Known orthographic cameras, supplied foreground semantics, static registered
laser geometry and 3x3 point splats. Ground-truth leaf-tip IDs affect evaluation
ownership only, never silhouettes, skeleton extraction or trajectory fitting.
This isolates geometry/occlusion/association; it is not an RGB/SAM benchmark.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.optimize import linear_sum_assignment
from architecture_analysis import read,write,dump,sha,mask_candidates
from architecture_tracking import fit_tracks,predict

def render_points(xyz,labels,view,size=384,scale=300,axis=192,base=360):
    t=view*2*np.pi/40;horizontal=xyz[:,0]*np.cos(t)+xyz[:,1]*np.sin(t);depth=-xyz[:,0]*np.sin(t)+xyz[:,1]*np.cos(t);u=np.rint(axis+scale*horizontal).astype(int);v=np.rint(base-scale*xyz[:,2]).astype(int);owner=np.zeros((size,size),np.int16)
    uu=[];vv=[];dd=[];ll=[]
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            ux=u+dx;vy=v+dy;valid=(ux>=0)&(ux<size)&(vy>=0)&(vy<size);uu.append(ux[valid]);vv.append(vy[valid]);dd.append(depth[valid]);ll.append(labels[valid])
    if not any(len(x) for x in uu):return owner>0,owner
    u=np.concatenate(uu);v=np.concatenate(vv);d=np.concatenate(dd);l=np.concatenate(ll);index=v*size+u;order=np.lexsort((d,index));sorted_index=index[order];last=np.r_[sorted_index[1:]!=sorted_index[:-1],True];chosen=order[last];owner[v[chosen],u[chosen]]=l[chosen]
    return owner>0,owner

def owner_at(owner,x,y):
    x=int(x);y=int(y);values=owner[max(0,y-1):y+2,max(0,x-1):x+2].ravel();values=values[values>0]
    return int(np.bincount(values).argmax()) if len(values) else 0

def run(args):
    started=time.time();out=args.run/'architecture'/'external_pheno4d'/'rendered';out.mkdir(parents=True,exist_ok=True);manifest=json.loads((args.external/'download_manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed' or len(manifest['files'])!=14:raise ValueError('Expected completed prespecified14-scan subset')
    cases=[];evalrows=[];observability=[];trackrows=[];sourcehash={}
    for name in ['architecture_external_rendered.py','architecture_tracking.py','architecture_analysis.py']:
        body=(Path(__file__).parent/name).read_bytes();(out/('execution_'+name)).write_bytes(body);sourcehash[name]=hashlib.sha256(body).hexdigest()
    for fi,record in enumerate(manifest['files']):
        path=args.external/record['path']
        if sha(path)!=record['sha256']:raise ValueError('External source hash mismatch')
        cloud=np.loadtxt(path,dtype='float32');foreground=cloud[:,4]>0;points=cloud[foreground,:3];labels=cloud[foreground,4].astype(np.int16)
        low=np.percentile(points[:,2],2);basal=points[points[:,2]<=np.percentile(points[:,2],8)];origin=np.r_[np.median(basal[:,:2],axis=0),low];height=float(points[:,2].max()-low);xyz=(points-origin)/height
        # Uniform spatial bins reduce scanner sampling-density imbalance. One
        # observed point per .003H cube; no triangle/surface truth is invented.
        cells=np.floor(xyz/.003).astype(np.int32);_,indices=np.unique(cells,axis=0,return_index=True);xyz=xyz[indices];labels=labels[indices]
        masks={};owners={};raw={};axes=[];bases=[];heights=[]
        for v in range(40):
            masks[v],owners[v]=render_points(xyz,labels,v);raw[v]=mask_candidates(masks[v],min_length=12)
            if v%4!=2:
                yy,xx=np.where(masks[v]);axes.append(float(np.median(xx[yy>=np.percentile(yy,95)])));bases.append(float(yy.max()));heights.append(float(np.ptp(yy)+1))
        axis=float(np.median(axes));base=float(np.median(bases));scale=float(np.median(heights))
        def normalized(cc):return [dict(x=(c['terminal_x_px']-axis)/scale,y=(base-c['terminal_y_px'])/scale,root_length=c['root_geodesic_over_height']*c['mask_height_px']/scale,px=c['terminal_x_px'],py=c['terminal_y_px']) for c in cc if not c['censored']]
        candidates={v:normalized(cc) for v,cc in raw.items()};train={v:cc for v,cc in candidates.items() if v%4!=2};tracks=fit_tracks(train)
        stem=path.stem;plant=path.parent.name;identity=dict(plant=plant,scan=stem)
        # Freeze all scientific models before opening evaluation ownership IDs.
        dump(out/f'frozen_{stem}.json',dict(identity=identity,axis=axis,base=base,height=scale,tracks=tracks))
        ids=sorted(int(i) for i in np.unique(labels));counts=np.zeros((len(tracks),len(ids)),int)
        for ti,track in enumerate(tracks):
            for a in track['assignments']:
                c=train[a['view']][a['candidate_id']];label=owner_at(owners[a['view']],c['px'],c['py'])
                if label in ids:counts[ti,ids.index(label)]+=1
        mapping={}
        if len(tracks):
            left,right=linear_sum_assignment(-counts)
            mapping={int(i):ids[int(j)] for i,j in zip(left,right) if counts[i,j]>0}
        dump(out/f'frozen_scoring_map_{stem}.json',dict(scan=stem,train_views=sorted(train),mapping={str(k):v for k,v in mapping.items()},true_ids_for_scoring=ids,training_ownership_counts=counts.tolist(),input_frozen_model_sha256=sha(out/f'frozen_{stem}.json'),rule='One-to-one assignment maximizes TRAIN ownership count; IDs used only for scoring; no heldout labels enter mapping.'))
        visible_train={}
        for label in ids:
            train_terminal_views=0;test_terminal_views=0
            for v,cc in candidates.items():
                terminal_count=sum(owner_at(owners[v],c['px'],c['py'])==label for c in cc);visible_pixels=int((owners[v]==label).sum());observability.append({**identity,'view':v,'split':'heldout' if v%4==2 else 'train','true_leaf_id_for_scoring':label,'oracle_frontmost_pixels':visible_pixels,'oracle_terminal_candidates':terminal_count})
                if terminal_count and v%4!=2:train_terminal_views+=1
                if terminal_count and v%4==2:test_terminal_views+=1
            visible_train[label]=train_terminal_views
        for ti,track in enumerate(tracks):
            mapped=mapping.get(ti,0);purity=float(counts[ti,ids.index(mapped)]/max(track['training_support'],1)) if mapped else 0.
            trackrows.append({**identity,'track_id':ti,'training_support':track['training_support'],'mapped_id_evaluation_only':mapped,'training_oracle_purity':purity,'oracle_label_has_8_train_terminal_views':visible_train.get(mapped,0)>=8})
            for v in range(40):
                if v%4!=2:continue
                cc=candidates[v];p=predict(track,v);xy=np.array([[c['x'],c['y']] for c in cc]);distances=np.linalg.norm(xy-p,axis=1) if len(cc) else np.array([]);nearest=int(distances.argmin()) if len(cc) else None;label=owner_at(owners[v],cc[nearest]['px'],cc[nearest]['py']) if nearest is not None else 0;distance=float(distances[nearest]) if nearest is not None else None;has_target=bool(mapped and any(owner_at(owners[v],c['px'],c['py'])==mapped for c in cc));evalrows.append({**identity,'track_id':ti,'view':v,'mapped_id_evaluation_only':mapped,'nearest_terminal_id_evaluation_only':label,'nearest_terminal_distance':distance,'target_oracle_foreground_pixels':int((owners[v]==mapped).sum()) if mapped else None,'target_has_oracle_terminal':has_target,'compatible_at_gate':distance is not None and distance<.07,'correct_mapped_organ_and_gate':bool(mapped and label==mapped and distance is not None and distance<.07)})
        # One fixed held-out view per scan, black background and true ownership.
        view=2;owner=owners[view];image=np.zeros((*owner.shape,3),np.uint8);palette=np.array([[230,180,70],[70,200,170],[230,120,150],[130,160,240],[190,140,240],[180,210,90],[240,150,90]])
        for label in ids:image[owner==label]=palette[(label-1)%len(palette)]
        im=Image.fromarray(image);draw=ImageDraw.Draw(im)
        for ti,track in enumerate(tracks):
            p=predict(track,view);x=axis+scale*p[0];y=base-scale*p[1];draw.ellipse((x-4,y-4,x+4,y+4),outline='white',width=2);draw.text((x+5,y),str(ti),fill='white')
        draw.rectangle((0,0,384,28),fill='black');draw.text((5,5),f'{plant} {stem} view2; oracle colours / predicted terminals',fill='white');im.save(out/f'{stem}_heldout.png')
        cases.append({**identity,'source_sha256':record['sha256'],'foreground_points':int(foreground.sum()),'spatially_binned_points':len(xyz),'true_leaf_tip_instances':len(ids),'oracle_instances_with_8_train_terminal_views':sum(v>=8 for v in visible_train.values()),'track_hypotheses':len(tracks),'tracks_mapped_to_true_ids_for_scoring':len(mapping)})
        print(json.dumps(dict(stage='external_rendered',completed=fi+1,total=len(manifest['files']))),flush=True)
    write(out/'rendered_cases.csv',cases);write(out/'oracle_organ_observability.csv',observability);write(out/'rotational_track_oracle_purity.csv',trackrows);write(out/'heldout_oracle_organ_compatibility.csv',evalrows)
    target=[r for r in evalrows if r['target_has_oracle_terminal']];summary=dict(scans=len(cases),plants=len(set(r['plant'] for r in cases)),rendered_views=len(cases)*40,true_tip_instances=sum(r['true_leaf_tip_instances'] for r in cases),oracle_instances_with_8_train_terminal_views=sum(r['oracle_instances_with_8_train_terminal_views'] for r in cases),track_hypotheses=len(trackrows),tracks_train_purity_atleast_80percent=sum(r['training_oracle_purity']>=.8 for r in trackrows),heldout_track_rows=len(evalrows),heldout_rows_with_mapped_organ_terminal=len(target),correct_mapped_organ_and_gate_all=sum(r['correct_mapped_organ_and_gate'] for r in evalrows),correct_mapped_organ_and_gate_when_terminal_available=sum(r['correct_mapped_organ_and_gate'] for r in target),execution_source_sha256=sourcehash,input_manifest_sha256=sha(args.external/'download_manifest.json'),elapsed_seconds=time.time()-started,scope='Static real laser geometry rendered under known orthographic cameras with supplied foreground semantics and3x3 point splats. Leaf-tip labels used only for oracle scoring/colours after models frozen. Does not validate RGB/SAM segmentation, camera estimation or CSI physical leaf identity.')
    dump(out/'rendered_external_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--external',type=Path,required=True);p.add_argument('--run',type=Path,required=True);run(p.parse_args())
