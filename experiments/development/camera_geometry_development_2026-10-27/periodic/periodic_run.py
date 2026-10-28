"""Training-ECC first-harmonic competitor; preserves the shared run intact."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
import argparse
import json
import shutil
import time
from collections import Counter
import numpy as np
import cv2
import camera_geometry as g
import periodic_camera as p
from camera_geometry_run import (atomic,sha,digest,csv_out,read_csv,load_masks,other_replicate,GRID,KEYS)
from camera_geometry_exports import load_records

def evaluate(record,fit,cameras,other_cameras):
    shared=record['fit']['selected'];identity=record['identity'];rows=[]
    parent_lookup={(e['rgb_view'],e['method']):e for e in record['evaluation']}
    for i in g.TEST:
        j=(shared['sign']*i+shared['phase'])%40 if shared else i
        periodic=p.predict(fit['model'],i) if fit['model'] else None
        guard=p.matrix_guard(periodic,GRID) if periodic is not None else dict(valid=False,reasons=['no_model'])
        periodic_selected=fit['selection']['accepted'] and guard['valid']
        matrix=periodic if periodic_selected else shared['matrix'] if shared else None
        for method,target_index,target_camera,m,valid in [
            ('periodic_candidate',j,cameras,periodic,guard['valid']),
            ('selected_causal',j,cameras,matrix,matrix is not None),
            ('phase_shared',j,cameras,shared['matrix'] if shared else None,shared is not None),
            ('mismatch_view_periodic',(j+20)%40,cameras,periodic,guard['valid']),
            ('mismatch_other_periodic',j,other_cameras,periodic,guard['valid'])]:
            row={**identity,'rgb_view':i,'noir_view':target_index,'method':method,'split':'test',
                 'status':'unavailable','adaptive_test_time_fit':False,
                 'periodic_session_selected':bool(fit['selection']['accepted'])}
            try:
                if not valid:raise ValueError('matrix_unavailable_or_guard_failed')
                if i not in cameras['rgb']:raise ValueError('rgb_test_mask_unavailable')
                if not target_camera or target_index not in target_camera['noir']:
                    raise ValueError('noir_test_mask_unavailable')
                a,b=cameras['rgb'][i],target_camera['noir'][target_index]
                row.update(status='executed',iou=g.iou_warp(a,b,m),matrix=np.asarray(m).tolist())
            except ValueError as exc:row['reason']=str(exc)
            rows.append(row)
        # Parent baseline is exactly the same nominal test pair when mapping is
        # nominal; otherwise fit an explicitly adaptive comparator to that pair.
        if (j==i and (i,'nominal_per_view_ecc') in parent_lookup):
            base=dict(parent_lookup[i,'nominal_per_view_ecc']);base['method']='adaptive_selected_pair_ecc'
        else:
            base={**identity,'rgb_view':i,'noir_view':j,'method':'adaptive_selected_pair_ecc',
                  'status':'unavailable','split':'test','adaptive_test_time_fit':True}
            if i in cameras['rgb'] and j in cameras['noir']:
                ecc=g.per_view_ecc(cameras['rgb'][i],cameras['noir'][j]);base.update(status='executed',**ecc)
            else:base['reason']='test_mask_unavailable'
        rows.append(base)
    return rows

def process(parent,frame_lookup,full_run,all_sessions):
    identity=parent['identity']
    cameras,inputs,failures=load_masks(identity['session'],frame_lookup,full_run)
    fit=p.fit_periodic(cameras['rgb'],cameras['noir'],parent['fit'])
    other=other_replicate(all_sessions,identity)
    other_cameras,other_inputs,other_failures=load_masks(other['session'],frame_lookup,full_run) if other else (None,[],[])
    metrics=evaluate(parent,fit,cameras,other_cameras)
    return dict(identity=identity,fit=fit,shared_fit=parent['fit'],
                shared_payload_sha256=digest(parent),inputs=inputs,input_failures=failures,
                other_session=other['session'] if other else None,
                other_input_digest=digest(other_inputs),other_input_failures=other_failures,
                evaluation=metrics)

def publish(output,records,config):
    metrics=[e for r in records for e in r['evaluation']]
    csv_out(output/'heldout_metrics.csv',metrics)
    sessions=[];lookups=[];teachers=[];validation=[]
    for r in records:
        fit=r['fit'];shared=r['shared_fit'];selected=shared['selected'];identity=r['identity']
        use_periodic=fit['selection']['accepted']
        deploy=g.deployment_decision(shared)
        if use_periodic:
            deploy.update(deployment_abstain=False,deployment_status='accepted_periodic_camera_surrogate',deployment_reasons=[])
        deploy['deployment_method']='first_harmonic' if use_periodic else 'shared_affine'
        method='first_harmonic' if use_periodic else 'shared_affine'
        session={**identity,**deploy,'fit_status':fit['status'],
            'selected_sign':selected['sign'] if selected else None,
            'selected_phase':selected['phase'] if selected else None,
            'phase_status':shared['phase_gate']['reason'] if 'reason' in shared['phase_gate'] else shared['status'],
            'periodic_accepted':use_periodic,'periodic_selection':fit['selection'],
            'periodic_coefficients':fit['model']['coefficients'] if fit['model'] else None,
            'periodic_basis':['1','sin(2*pi*i/40)','cos(2*pi*i/40)'],
            'n_training_teachers':fit.get('n_accepted_teachers',0),
            'nominal_matrix_rgb_to_noir':shared['nominal']['matrix'] if shared['nominal'] else None,
            'shared_matrix_rgb_to_noir':selected['matrix'] if selected else None,
            'source_grid_width':GRID[0],'source_grid_height':GRID[1],
            'target_grid_width':GRID[0],'target_grid_height':GRID[1],
            'training_only_selection':True,'physical_correspondence_verified':False}
        session['view_matrices_rgb_to_noir']={}
        session['view_matrices_noir_to_rgb']={}
        for i in range(40):
            candidate=p.predict(fit['model'],i) if fit['model'] else None
            guard=p.matrix_guard(candidate,GRID) if candidate is not None else dict(valid=False,reasons=['no_model'])
            matrix=candidate if use_periodic and guard['valid'] else np.array(selected['matrix']) if selected else None
            inverse=cv2.invertAffineTransform(matrix) if matrix is not None else None
            session['view_matrices_rgb_to_noir'][str(i)]=matrix.tolist() if matrix is not None else None
            session['view_matrices_noir_to_rgb'][str(i)]=inverse.tolist() if inverse is not None else None
            lookups.append({**identity,**deploy,'method':method,'rgb_view':i,
                'noir_view':(selected['sign']*i+selected['phase'])%40 if selected else None,
                'matrix_rgb_to_noir':matrix.tolist() if matrix is not None else None,
                'matrix_noir_to_rgb':inverse.tolist() if inverse is not None else None,
                'periodic_candidate_matrix_rgb_to_noir':candidate.tolist() if candidate is not None else None,
                'periodic_candidate_guard':guard,'split':'test' if i in g.TEST else 'train',
                'source_grid_width':GRID[0],'source_grid_height':GRID[1],
                'target_grid_width':GRID[0],'target_grid_height':GRID[1]})
        for m,label in [('periodic_candidate','periodic'),('selected_causal','selected'),
                        ('phase_shared','shared'),('adaptive_selected_pair_ecc','adaptive_ecc')]:
            ok=[x for x in r['evaluation'] if x['method']==m and x['status']=='executed']
            session[label+'_heldout_n']=len(ok)
            session[label+'_heldout_median_iou']=float(np.median([x['iou'] for x in ok])) if ok else None
        sessions.append(session)
        teachers.extend({**identity,**t} for t in fit.get('teachers',[]))
        validation.extend({**identity,**v} for v in fit.get('inner_validation',[]))
    csv_out(output/'alignment_lookup.csv',lookups)
    csv_out(output/'session_camera_lookup.csv',sessions)
    csv_out(output/'training_ecc_teachers.csv',teachers)
    csv_out(output/'inner_angle_validation.csv',validation)
    atomic(output/'session_camera_lookup.json',dict(status='completed',schema=1,
        configuration_signature=config['signature'],rows=sessions,
        matrix_direction='RGB pixel to NoIR pixel',coordinate_grid=GRID,
        query='Read view_matrices_rgb_to_noir[str(rgb_index)] or corresponding inverse',
        abstention='Training-only; shared fallback when first-harmonic fails inner validation; both may abstain'))
    methods={}
    for method in sorted({e['method'] for e in metrics}):
        ok=[e for e in metrics if e['method']==method and e['status']=='executed']
        means=[]
        for session in sorted({e['session'] for e in ok}):
            means.append(np.median([e['iou'] for e in ok if e['session']==session]))
        methods[method]=dict(planned_rows=len(records)*10,executed_rows=len(ok),sessions=len(means),
             pooled_median_iou=float(np.median([e['iou'] for e in ok])) if ok else None,
             median_session_median_iou=float(np.median(means)) if means else None,
             failures=dict(Counter(e.get('reason','') for e in metrics if e['method']==method and e['status']!='executed')))
    contrasts=[]
    for first,second in [('periodic_candidate','phase_shared'),('selected_causal','phase_shared'),
                         ('adaptive_selected_pair_ecc','selected_causal'),
                         ('periodic_candidate','mismatch_view_periodic'),('periodic_candidate','mismatch_other_periodic')]:
        aa={(e['session'],e['rgb_view']):e['iou'] for e in metrics if e['method']==first and e['status']=='executed'}
        bb={(e['session'],e['rgb_view']):e['iou'] for e in metrics if e['method']==second and e['status']=='executed'}
        keys=sorted(set(aa)&set(bb));ds=[aa[k]-bb[k] for k in keys]
        contrasts.append(dict(first=first,second=second,matched_rows=len(keys),
                matched_sessions=len({k[0] for k in keys}),median_paired_delta=float(np.median(ds)) if ds else None))
    summary=dict(status='completed',schema=1,signature=config['signature'],acquisitions=len(records),
        fitted=sum(r['fit']['model'] is not None for r in records),
        accepted_periodic=sum(r['fit']['selection']['accepted'] for r in records),
        deployment_status=dict(Counter(s['deployment_status'] for s in sessions)),
        methods=methods,contrasts=contrasts,teacher_status=dict(Counter(
            'accepted' if t['accepted'] else 'rejected' for r in records for t in r['fit'].get('teachers',[]))),
        outputs={x.name:sha(x) for x in output.glob('*.csv')},
        limitations=['fixed first harmonic is a nominal-index surrogate, not observed servo telemetry',
                    'teacher ECC uses training masks only; test ECC is a separate adaptive comparator',
                    'inner selection compares only outer training angles; heldout never gates deployment'])
    atomic(output/'summary.json',summary);return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shared',type=Path,required=True)
    parser.add_argument('--full-run',type=Path,required=True);parser.add_argument('--intake',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    upstream,parents=load_records(args.shared)
    shared_summary=json.loads((args.shared/'summary.json').read_text(encoding='utf-8'))
    if shared_summary['status']!='completed':raise ValueError('shared_run_not_complete')
    output=args.output;output.mkdir(parents=True,exist_ok=True);(output/'records').mkdir(exist_ok=True)
    sources={str(path.relative_to(Path(__file__).parent.parent)).replace('\\','/'):sha(path)
         for path in list(Path(__file__).parent.glob('*.py'))+
             [Path(g.__file__),Path(__file__).parent.parent/'camera_geometry_run.py',Path(__file__).parent.parent/'camera_geometry_exports.py']}
    config=dict(schema=1,policy=p.POLICY,shared_signature=upstream['signature'],
        shared_summary_sha256=sha(args.shared/'summary.json'),cohort=upstream['cohort'],
        full_view_traits_sha256=sha(args.full_run/'extraction/view_traits.csv'),sources=sources,
        threads=2,CPU_only=True,training_indices=g.TRAIN,test_indices=g.TEST,
        mode=upstream['mode'])
    config['signature']=digest(config);atomic(output/'configuration.json',config)
    snapshot=output/'source_snapshot'/config['signature'];snapshot.mkdir(parents=True,exist_ok=True)
    for rel in sources:
        source=Path(__file__).parent.parent/rel;dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
    frames=read_csv(args.full_run/'extraction/view_traits.csv')
    lookup={(r['session'],r['camera'].lower(),int(r['view'])):r for r in frames}
    sessions=read_csv(args.intake/'session_decode_qc.csv')
    records=[];started=time.monotonic()
    for n,parent in enumerate(sorted(parents,key=lambda r:r['identity']['session'])):
        name=digest(parent['identity']['session'])[:24]+'.json';path=output/'records'/name;record=None
        if path.exists():
            old=json.loads(path.read_text(encoding='utf-8'));saved=old.pop('payload_sha256',None)
            if old['signature']==config['signature'] and saved==digest(old):record=old
        if record is None:
            record=process(parent,lookup,args.full_run,sessions)
            record['signature']=config['signature'];record['payload_sha256']=digest(record);atomic(path,record)
        records.append(record)
        progress=dict(status='running',completed=n+1,planned=len(parents),
            elapsed_seconds=time.monotonic()-started,last_session=parent['identity']['session'])
        atomic(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    summary=publish(output,records,config)
    atomic(output/'progress.json',dict(status='completed',completed=len(records),planned=len(records),signature=config['signature']))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
