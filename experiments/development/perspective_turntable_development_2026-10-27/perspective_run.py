"""Training-only camera-family selection and full-cohort point prediction."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='2'
import argparse,gzip,json,sys,time
from pathlib import Path
import numpy as np
from perspective_points import families,fit_point,fit_harmonic,predict_model,select_families,project_point

TRACK=Path(__file__).parent.parent/'longitudinal_tracking_development_2026-10-27';sys.path.insert(0,str(TRACK))
from tracking_development import read,write,dump,sha,log,gauge_from_masks
from tracking_graph_core import fit_cycles
from tracking_observations import extract_paths,normalize_paths
from architecture_analysis import load_mask,accepted


def observed_track(track,candidates,scale):
    views=[];xy=[]
    for a in track['assignments']:
        if a['view']%4==2:raise ValueError('Outer angle in training assignment')
        c=candidates[a['view']][a['candidate_id']];views.append(a['view']);xy.append([c['x']*scale,c['y']*scale])
    return dict(node_id=track['node_id'],views=views,xy=xy)


def prepare(args):
    source=args.full_run/'extraction/view_traits.csv';rows=read(source);groups={}
    for r in rows:
        if r['camera']=='rgb':groups.setdefault((r['treatment'],r['genotype'],r['replicate'],int(r['measurement'])),[]).append(r)
    index=json.loads((args.tracking/'index.json').read_text(encoding='utf-8'));cases=[]
    for gi,r in enumerate(index['candidates'],1):
        g=json.loads((args.tracking/r['path']).read_text(encoding='utf-8'));identity=g['identity']
        for session in g['sessions']:
            if not session['coverage']['eligible']:continue
            visit=session['visit'];key=(identity['treatment'],identity['genotype'],identity['replicate'],visit);good=[q for q in groups.get(key,[]) if accepted(q)]
            masks={int(q['view']):load_mask(q,args.full_run,width=256) for q in good};paths={v:extract_paths(m,min_length=6) for v,m in masks.items()}
            train_masks={v:m for v,m in masks.items() if v%4!=2};inner_masks={v:m for v,m in masks.items() if v%4 not in [1,2]}
            gauge=session.get('gauge') or gauge_from_masks(train_masks);scale=gauge['height']/256
            cc={v:normalize_paths(p,**gauge) for v,p in paths.items()};full=[]
            for n in session['nodes']:
                # Verify source-supported assignment coordinates survived re-extraction.
                for a in n['assignments']:
                    c=cc[a['view']][a['candidate_id']];original=n['angle_evidence'][a['view']]['observed_candidate']['terminal_pixel_processing']
                    if [c['px'],c['py']]!=original:raise ValueError('Frozen source candidate mismatch')
                full.append(observed_track(n,cc,scale))
            inner_gauge=gauge_from_masks(inner_masks);inner_scale=inner_gauge['height']/256
            ic={v:normalize_paths(p,**inner_gauge) for v,p in paths.items()};inner_train={v:c for v,c in ic.items() if v%4 not in [1,2]}
            inner_tracks=fit_cycles(inner_train)
            for i,t in enumerate(inner_tracks):t['node_id']=f"{g['candidate_id']}:v{visit}:inner{i}"
            inn=[observed_track(t,ic,inner_scale) for t in inner_tracks]
            token=sha(args.tracking/r['path'])[:16]+'_v'+str(visit)
            cases.append(dict(case_id=token,identity=dict(**identity,measurement=str(visit)),gauge=gauge,inner_gauge=inner_gauge,
                full_scale=scale,inner_scale=inner_scale,full_tracks=full,inner_tracks=inn,
                full_offset=float(np.median([n['x_coefficient'][0]*scale for n in session['nodes']])) if full else 0.,
                inner_offset=float(np.median([t['x_coefficient'][0]*inner_scale for t in inner_tracks])) if inn else 0.,
                candidates={str(v):[[c['x']*scale,c['y']*scale] for c in ccc] for v,ccc in cc.items()},
                inner_candidates={str(v):[[c['x']*inner_scale,c['y']*inner_scale] for c in ccc] for v,ccc in ic.items()},
                sources=[dict(view=int(q['view']),source=q['source'],source_sha256=q['source_sha256'],mask=q['mask'],mask_sha256=q['mask_sha256']) for q in good]))
        if gi%12==0:log(args.output,'observations_prepared',candidate_identities=gi,total=72,cases=len(cases))
    payload=dict(cases=cases,input_view_table_sha256=sha(source),tracking_index_sha256=sha(args.tracking/'index.json'),
        inner_training_rule='view%4 in[0,3]',inner_validation_rule='view%4==1',outer_validation_rule='view%4==2',
        inner_detector_refitted=True,inner_gauge_refitted=True)
    raw=json.dumps(payload,separators=(',',':'),allow_nan=False).encode();(args.output/'observations.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    log(args.output,'all_observations_prepared',cases=len(cases),full_nodes=sum(len(c['full_tracks']) for c in cases),inner_nodes=sum(len(c['inner_tracks']) for c in cases))
    return payload


def evaluate(model,case,split,model_name,node_id):
    views=list(range(1 if split=='inner' else 2,40,4));candidates=case['inner_candidates' if split=='inner' else 'candidates'];scale=case['inner_scale' if split=='inner' else 'full_scale'];pred=predict_model(model,views);rows=[]
    fitted=model.get('status','fitted')=='fitted'
    for v,p in zip(views,pred):
        cc=candidates.get(str(v));available=cc is not None;distance=float(np.min(np.linalg.norm(np.asarray(cc)-p,axis=1))/scale) if cc else None
        score=min(distance,.5) if distance is not None and fitted else .5 if available else None
        rows.append(dict(case_id=case['case_id'],node_id=node_id,model=model_name,split=split,view=v,mask_available=available,
            candidate_count=len(cc) if cc is not None else 0,nearest_terminal_distance=distance,clipped_error_for_camera_selection=score,
            compatible_at_point07=bool(distance is not None and distance<.07 and fitted),fit_status=model.get('status','fitted')))
    return rows


def fit_case(case,family,split):
    tracks=case['inner_tracks' if split=='inner' else 'full_tracks'];offset=case['inner_offset' if split=='inner' else 'full_offset'];models=[]
    for t in tracks:
        if split=='inner' and any(v%4 in [1,2] for v in t['views']):raise ValueError('Inner validation leakage')
        model=fit_point(t['views'],t['xy'],family,offset);models.append((t,model))
    return models


def select_camera(args,payload):
    allrows=[];familyrows=[];cam=families()
    for fi,f in enumerate(cam,1):
        case_scores=[];point_fits=0;failures=0
        for case in payload['cases']:
            rr=[]
            for t,m in fit_case(case,f,'inner'):
                rr+=evaluate(m,case,'inner',f['family'],t['node_id']);point_fits+=1;failures+=m['status']!='fitted'
            available=[r['clipped_error_for_camera_selection'] for r in rr if r['clipped_error_for_camera_selection'] is not None]
            score=float(np.sqrt(np.mean(np.square(available)))) if available else None
            allrows.append(dict(family=f['family'],case_id=case['case_id'],inner_nodes=len(case['inner_tracks']),planned_inner_rows=len(rr),
                available_inner_rows=len(available),inner_case_rmse=score))
            if score is not None:case_scores.append(score)
        familyrows.append(dict(**f,inner_case_median=float(np.median(case_scores)) if case_scores else None,inner_evaluable_cases=len(case_scores),point_fits=point_fits,fit_failures=failures))
        log(args.output,'inner_family_completed',completed=fi,total=len(cam),**familyrows[-1])
    choice=select_families(familyrows,.01);ortho=select_families([r for r in familyrows if r['kind']=='orthographic'],.01)
    write(args.output/'inner_family_cases.csv',allrows);write(args.output/'inner_family_summary.csv',familyrows)
    selection=dict(**choice,selected_orthographic=ortho['selected'],families=familyrows,
        selection_uses_outer_angles=False,selection_uses_original_full_training_associations=False,
        input_observation_cache_sha256=sha(args.output/'observations.json.gz'))
    dump(args.output/'frozen_camera_selection.json',selection);log(args.output,'camera_selection_frozen',selected=choice['selected'],retained=choice['retained'],selected_orthographic=ortho['selected'])
    return selection


def outer(args,payload,selection):
    camera={f['family']:f for f in families()};selected=selection['selected'];required=list(dict.fromkeys([selected,selection['selected_orthographic']]+selection['retained']));rows=[];modelrows=[];predictions=[];case_models={}
    for ci,case in enumerate(payload['cases'],1):
        models=[]
        for family_name in required:
            for t,m in fit_case(case,camera[family_name],'full'):
                models.append(dict(node_id=t['node_id'],model=m));rows+=evaluate(m,case,'outer',family_name,t['node_id'])
                modelrows.append(dict(case_id=case['case_id'],node_id=t['node_id'],family=family_name,**{k:v for k,v in m.items() if k not in ['family']}))
                if family_name==selected:
                    pp=predict_model(m,range(40));g=case['gauge']
                    for v,p in enumerate(pp):
                        pixel=[g['axis']+p[0]*256,g['base']-p[1]*256];predictions.append(dict(case_id=case['case_id'],node_id=t['node_id'],view=v,
                            split='outer' if v%4==2 else 'training',pixel_x=pixel[0],pixel_y=pixel[1],image_x_fraction=pixel[0]/256,image_y_fraction=pixel[1]/192,
                            point_x=m['point'][0],point_up=m['point'][1],point_depth=m['point'][2],family=selected,fit_status=m['status']))
        for t in case['full_tracks']:
            for vertical,name in [(False,'constant_height_harmonic'),(True,'unrestricted_2D_harmonic')]:
                m=fit_harmonic(t['views'],t['xy'],vertical);rows+=evaluate(m,case,'outer',name,t['node_id'])
        if case['full_tracks']:
            dump(args.output/'session_models'/f"{case['case_id']}.json",dict(identity=case['identity'],selected_family=selected,gauge=case['gauge'],models=models,
                sources=case['sources'],coordinate_units='Camera distance D=1 by convention; no physical calibration'))
        if ci%40==0:log(args.output,'outer_case_progress',completed=ci,total=len(payload['cases']))
    write(args.output/'outer_predictions.csv',rows);write(args.output/'conditional_point_family_fits.csv',modelrows);write(args.output/'selected_family_projection_40views.csv',predictions)
    aggregate=[]
    for method in sorted(set(r['model'] for r in rows)):
        rr=[r for r in rows if r['model']==method];available=[r for r in rr if r['nearest_terminal_distance'] is not None];casevals={}
        for r in rr:
            score=r['clipped_error_for_camera_selection']
            if score is not None:casevals.setdefault(r['case_id'],[]).append(score)
        case_rmse=[float(np.sqrt(np.mean(np.square(v)))) for v in casevals.values()]
        aggregate.append(dict(method=method,planned_track_view_rows=len(rr),candidate_available_rows=len(available),mask_available_rows=sum(r['mask_available'] for r in rr),
            compatible_rows=sum(r['compatible_at_point07'] for r in rr),median_nearest_terminal_distance=float(np.median([r['nearest_terminal_distance'] for r in available])) if available else None,
            median_case_clipped_rmse=float(np.median(case_rmse)) if case_rmse else None,case_count=len(case_rmse),fit_failed_rows=sum(r['fit_status']!='fitted' for r in rr)))
    write(args.output/'outer_summary.csv',aggregate)
    result=dict(reference_visit_slots=360,eligible_acquisitions=len(payload['cases']),full_training_nodes=sum(len(c['full_tracks']) for c in payload['cases']),
        acquisitions_with_full_nodes=sum(bool(c['full_tracks']) for c in payload['cases']),inner_training_nodes=sum(len(c['inner_tracks']) for c in payload['cases']),
        camera_family_count=30,selected_family=selected,selected_orthographic_family=selection['selected_orthographic'],retained_families=selection['retained'],
        outer_results=aggregate,source_camera_gauge='D=1; coordinates scaled to camera distance, no physical units',
        scope='Source-supported terminal prediction only. No anatomical correspondences, unique calibration, physiological endpoint or leaf surfaces validated.',
        frozen_selection_sha256=sha(args.output/'frozen_camera_selection.json'))
    dump(args.output/'summary.json',result);log(args.output,'completed',**result)


def synthetic(args):
    points=np.array([[.22,.5,.12],[-.3,.3,.1],[.1,.7,-.25],[-.1,.15,-.12]]);rr=[]
    for kind,focal,e,r,noise,phase in [('perspective',1.2,5,3,0,0),('perspective',.8,0,0,.002,90),('orthographic',1.,0,0,0,0),('orthographic',1.,-5,3,.002,171)]:
        truth=dict(kind=kind,focal=focal,elevation=e,roll=r);rng=np.random.default_rng(1);views=[v for v in range(40) if v%4 in [0,3] and v%7!=0];validation=[v for v in range(40) if v%4==1];data=[]
        for p in points:
            train=project_point(p,views,truth,phase_degrees=phase)+rng.normal(0,noise,(len(views),2));val=project_point(p,validation,truth,phase_degrees=phase)
            data.append((train,val))
        scores=[]
        for f in families():
            err=[]
            for train,val in data:
                m=fit_point(views,train,f);err.extend(np.linalg.norm(predict_model(m,validation)-val,axis=1))
            scores.append(dict(family=f['family'],inner_case_median=float(np.sqrt(np.mean(np.square(err))))))
        choice=select_families(scores,.01);rr.append(dict(true_kind=kind,true_focal=focal,true_elevation=e,true_roll=r,noise=noise,phase=phase,**choice))
    dump(args.output/'synthetic_camera_tests.json',dict(cases=rr,scope='Known numeric points and cameras; IDs/correspondences supplied by synthetic construction; occluded fitting angles retained in design'))
    log(args.output,'synthetic_camera_tests_completed',cases=len(rr))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--full-run',type=Path,required=True);p.add_argument('--tracking',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse-observations',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    hashes={}
    for f in ['perspective_points.py','perspective_run.py','test_perspective_points.py']:
        src=Path(__file__).parent/f;hashes[f]=sha(src);target=a.output/'execution_sources'/f;target.parent.mkdir(exist_ok=True);target.write_bytes(src.read_bytes())
    dump(a.output/'configuration.json',dict(families=families(),retention_tolerance_height=.01,thread_limit=2,execution_source_sha256=hashes,
        dependency_sha256={str(TRACK/f):sha(TRACK/f) for f in ['tracking_graph_core.py','tracking_observations.py','tracking_development.py']}))
    synthetic(a)
    payload=json.loads(gzip.decompress((a.output/'observations.json.gz').read_bytes())) if a.reuse_observations else prepare(a)
    selection=select_camera(a,payload);outer(a,payload,selection)
    if any(sha(Path(__file__).parent/f)!=h for f,h in hashes.items()):raise ValueError('Execution sources changed during run')


if __name__=='__main__':main()
