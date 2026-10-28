"""Execute mask-only shared-camera development without changing upstream data."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import csv
import hashlib
import io
import json
import platform
import shutil
import time
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
from PIL import Image
import scipy
import camera_geometry as g

cv2.setNumThreads(2)
GRID=(256,192)
KEYS=('treatment','measurement','genotype','replicate')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,
         separators=(',',':')).encode()).hexdigest()

def atomic(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def csv_out(path,rows):
    path=Path(path); tmp=path.with_suffix('.csv.tmp')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    os.replace(tmp,path)

def yes(value):
    return str(value).lower() in ('true','1')

def load_masks(session,frame_lookup,full_run):
    cameras={c:{} for c in ('rgb','noir')}; inputs=[]; failures=[]
    for camera in cameras:
        for i in range(40):
            row=frame_lookup.get((session,camera,i))
            record=dict(camera=camera,view=i,split='test' if i in g.TEST else 'train')
            if row:
                record.update({k:row.get(k) for k in ('source','source_sha256','mask','mask_sha256',
                                      'configuration_signature','qc_pass','qc_issues','mask_origin')})
            try:
                if row is None:
                    raise ValueError('extraction_record_unavailable')
                if not yes(row['qc_pass']):
                    raise ValueError('mask_qc_rejected:'+row['qc_issues'])
                path=full_run/row['mask']; raw=path.read_bytes()
                actual=hashlib.sha256(raw).hexdigest()
                if actual!=row['mask_sha256']:
                    raise ValueError('mask_hash_mismatch')
                with Image.open(io.BytesIO(raw)) as im:
                    if im.size!=(int(row['processing_width']),int(row['processing_height'])):
                        raise ValueError('mask_processing_dimensions_mismatch')
                    mask=np.asarray(im.convert('L').resize(GRID,Image.Resampling.NEAREST))>0
                if mask.sum()<g.POLICY['min_mask_pixels']:
                    raise ValueError('mask_too_small_at_256_grid')
                cameras[camera][i]=mask
                record.update(status='usable',actual_mask_sha256=actual,grid_pixels=int(mask.sum()))
            except (ValueError,OSError) as exc:
                record.update(status='unavailable',reason=str(exc))
                failures.append(record.copy())
            inputs.append(record)
    return cameras,inputs,failures

def other_replicate(sessions,anchor):
    choices=sorted([s for s in sessions if all(s[k]==anchor[k] for k in
            ('treatment','measurement','genotype')) and s['replicate']!=anchor['replicate']],
            key=lambda s:s['session'])
    later=[s for s in choices if s['session']>anchor['session']]
    return (later or choices)[0] if choices else None

def evaluate(session,fit,cameras,other_cameras,other_session):
    rows=[]
    for i in g.TEST:
        selected=fit.get('selected'); nominal=fit.get('nominal')
        selected_j=(selected['sign']*i+selected['phase'])%40 if selected else i
        specs=[('nominal_fixed_moment',i,nominal.get('moment_matrix') if nominal else None,None),
               ('nominal_shared',i,nominal.get('matrix') if nominal else None,None),
               ('selected_shared',selected_j,selected.get('matrix') if selected else None,None),
               ('nominal_per_view_moment',i,None,'moment'),
               ('nominal_per_view_ecc',i,None,'ecc'),
               ('mismatch_view_fixed',(i+20)%40,nominal.get('matrix') if nominal else None,None),
               ('mismatch_view_per_view_ecc',(i+20)%40,None,'ecc'),
               ('mismatch_other_fixed',i,nominal.get('matrix') if nominal else None,None),
               ('mismatch_other_per_view_ecc',i,None,'ecc')]
        for method,j,matrix,adaptive in specs:
            other=method.startswith('mismatch_other')
            row={**{k:session[k] for k in KEYS},'session':session['session'],
                 'rgb_view':i,'noir_view':j,'method':method,'split':'test',
                 'noir_session':other_session['session'] if other and other_session else session['session'],
                 'adaptive_test_time_fit':bool(adaptive),'status':'unavailable'}
            try:
                if i not in cameras['rgb']:
                    raise ValueError('rgb_test_mask_unavailable')
                target=other_cameras['noir'] if other and other_cameras else cameras['noir'] if not other else {}
                if j not in target:
                    raise ValueError('noir_test_mask_unavailable')
                a,b=cameras['rgb'][i],target[j]
                if adaptive=='moment':
                    matrix=g.moment(a,b); row['solver_status']='moment'
                elif adaptive=='ecc':
                    e=g.per_view_ecc(a,b); matrix=e['matrix']
                    row.update({k:v for k,v in e.items() if k not in ('matrix','iou')})
                elif matrix is None:
                    raise ValueError('shared_training_fit_unavailable')
                m=np.asarray(matrix); warped=g.warp_target(b,m,a.shape)
                ai=cv2.erode(a.astype(np.uint8),np.ones((3,3),np.uint8))>0
                bi=g.warp_target(cv2.erode(b.astype(np.uint8),np.ones((3,3),np.uint8)),m,a.shape)
                common=ai&bi
                row.update(status='executed',iou=g.iou_warp(a,b,m),
                    common_interior_pixels=int(common.sum()),rgb_pixels=int(a.sum()),
                    noir_warped_pixels=int(warped.sum()),matrix=m.tolist(),
                    determinant=float(np.linalg.det(m[:,:2])))
            except (ValueError,np.linalg.LinAlgError,cv2.error) as exc:
                row['reason']=str(exc)
            rows.append(row)
    return rows

def session_record(session,sessions,lookup,full_run):
    cameras,inputs,failures=load_masks(session['session'],lookup,full_run)
    fit=g.fit_session(cameras['rgb'],cameras['noir'])
    other=other_replicate(sessions,session)
    other_cameras,other_inputs,other_failures=load_masks(other['session'],lookup,full_run) if other else (None,[],[])
    rows=evaluate(session,fit,cameras,other_cameras,other)
    return dict(identity={k:session[k] for k in (*KEYS,'session')},fit=fit,
                inputs=inputs,input_failures=failures,other_identity=other,
                other_inputs=other_inputs,other_input_failures=other_failures,evaluation=rows)

def summarize(records):
    rows=[r for record in records for r in record['evaluation']]
    methods={}
    for method in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==method]
        ok=[r for r in rr if r['status']=='executed']
        sessions={r['session'] for r in ok}
        per_session=[np.median([r['iou'] for r in ok if r['session']==s]) for s in sessions]
        methods[method]=dict(planned_rows=len(rr),executed_rows=len(ok),sessions=len(sessions),
            pooled_median_iou=float(np.median([r['iou'] for r in ok])) if ok else None,
            median_session_median_iou=float(np.median(per_session)) if ok else None,
            common_interior_10px_rows=sum(r['common_interior_pixels']>=10 for r in ok),
            failures=dict(Counter(r.get('reason','') for r in rr if r['status']!='executed')),
            solver_status=dict(Counter(r.get('solver_status','fixed') for r in ok)))
    contrasts=[]
    for a,b in [('nominal_shared','nominal_fixed_moment'),('selected_shared','nominal_shared'),
                ('nominal_per_view_ecc','nominal_shared'),('nominal_shared','mismatch_view_fixed'),
                ('nominal_shared','mismatch_other_fixed')]:
        aa={(r['session'],r['rgb_view']):r for r in rows if r['method']==a and r['status']=='executed'}
        bb={(r['session'],r['rgb_view']):r for r in rows if r['method']==b and r['status']=='executed'}
        keys=sorted(set(aa)&set(bb)); ds=[aa[k]['iou']-bb[k]['iou'] for k in keys]
        session_deltas=[]
        for session in sorted({k[0] for k in keys}):
            session_deltas.append(np.median([aa[k]['iou']-bb[k]['iou'] for k in keys if k[0]==session]))
        contrasts.append(dict(first=a,second=b,matched_rows=len(ds),
             matched_sessions=len(session_deltas),median_paired_delta=float(np.median(ds)) if ds else None,
             median_session_paired_delta=float(np.median(session_deltas)) if ds else None))
    return dict(acquisitions=len(records),fitted_acquisitions=sum(r['fit']['status']=='fitted' for r in records),
                fit_status=dict(Counter(r['fit']['status'] for r in records)),
                alternate_mapping_acquisitions=sum(r['fit']['phase_gate']['accepted_alternate'] for r in records),
                input_status=dict(Counter(i['status'] for r in records for i in r['inputs'])),
                methods=methods,contrasts=contrasts,phase_selections=dict(Counter(
                    f"sign={r['fit']['selected']['sign']};phase={r['fit']['selected']['phase']}"
                    for r in records if r['fit']['selected'])))

def publish(output,records,configuration,elapsed):
    evaluations=[row for r in records for row in r['evaluation']]
    csv_out(output/'heldout_metrics.csv',evaluations)
    candidates=[]; lookups=[]; fits=[]; unavailable=[]
    for r in records:
        identity=r['identity']; fit=r['fit']; selected=fit['selected']
        fits.append({**identity,'status':fit['status'],'n_nominal_train':fit['nominal']['n_train'] if fit['nominal'] else 0,
                     'alternate_selected':fit['phase_gate']['accepted_alternate'],
                     'selected_sign':selected['sign'] if selected else None,
                     'selected_phase':selected['phase'] if selected else None,
                     'matrix':selected['matrix'] if selected else None,'phase_gate':fit['phase_gate']})
        candidates.extend({**identity,**c} for c in fit['candidates'])
        unavailable.extend({**identity,**i} for i in r['input_failures'])
        for i in range(40):
            lookups.append({**identity,'rgb_view':i,'noir_view':(selected['sign']*i+selected['phase'])%40 if selected else None,
                'split':'test' if i in g.TEST else 'train','status':'fitted' if selected else fit['status'],
                'sign':selected['sign'] if selected else None,'phase':selected['phase'] if selected else None,
                'matrix_rgb_to_noir':selected['matrix'] if selected else None,
                'inverse_matrix_noir_to_rgb':cv2.invertAffineTransform(np.array(selected['matrix'])).tolist() if selected else None,
                'source_grid_width':GRID[0],'source_grid_height':GRID[1],
                'target_grid_width':GRID[0],'target_grid_height':GRID[1],
                'training_indices':selected['train_indices'] if selected else [],
                'training_only':True,'physical_correspondence_verified':False})
    csv_out(output/'session_fits.csv',fits); csv_out(output/'mapping_candidates.csv',candidates)
    csv_out(output/'alignment_lookup.csv',lookups); csv_out(output/'input_unavailable.csv',unavailable)
    atomic(output/'alignment_lookup.json',dict(schema=1,configuration_signature=configuration['signature'],
            grid=GRID,matrix_direction='RGB_pixel_to_NoIR_pixel',
            source_sampling='warp NoIR onto RGB with WARP_INVERSE_MAP',rows=lookups))
    summary=summarize(records); summary.update(status='completed',elapsed_seconds=elapsed,
             signature=configuration['signature'],mode=configuration['mode'],
             training_indices=g.TRAIN,test_indices=g.TEST,policy=g.POLICY,
             outputs={p.name:sha(p) for p in output.glob('*.csv')},
             limitations=['silhouette camera surrogate, not physical calibration',
                 'coarse phase/sign is training-gated, not true angle or organ matching',
                 'test-time ECC is adaptive and has access to each test mask',
                 'full masks are read for evaluation but only training masks reach fit_session'])
    atomic(output/'summary.json',summary)
    return summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--intake',type=Path,required=True)
    parser.add_argument('--full-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['pilot','full'],required=True)
    args=parser.parse_args(); output=args.output/args.mode
    output.mkdir(parents=True,exist_ok=True); (output/'records').mkdir(exist_ok=True)
    frames=read_csv(args.full_run/'extraction/view_traits.csv')
    lookup={}
    for row in frames:
        key=(row['session'],row['camera'].lower(),int(row['view']))
        if key in lookup: raise ValueError('duplicate_extraction_frame_key')
        lookup[key]=row
    sessions=[s for s in read_csv(args.intake/'session_decode_qc.csv') if yes(s['known_reference_identity'])]
    if len(sessions)!=360: raise ValueError('expected_360_reference_acquisitions')
    if args.mode=='pilot':
        chosen=[]
        for treatment in ('C','P','S'):
            for visit in ('1','2','3','4','5'):
                pool=sorted([s for s in sessions if s['treatment']==treatment and s['measurement']==visit],key=lambda s:s['session'])
                usable=lambda s:sum(all((s['session'],c,i) in lookup and yes(lookup[s['session'],c,i]['qc_pass'])
                                       for c in ('rgb','noir')) for i in g.TRAIN)
                candidates=[s for s in pool if usable(s)>=24]
                chosen.append((candidates or pool)[0])
        selected_sessions=chosen
    else: selected_sessions=sorted(sessions,key=lambda s:s['session'])
    source={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    config=dict(schema=1,mode=args.mode,grid=GRID,training_indices=g.TRAIN,test_indices=g.TEST,
        policy=g.POLICY,source_hashes=source,
        view_traits_sha256=sha(args.full_run/'extraction/view_traits.csv'),
        intake_session_qc_sha256=sha(args.intake/'session_decode_qc.csv'),
        extraction_configs={p.name:sha(p) for p in (args.full_run/'extraction').glob('configuration_shard_*.json')},
        cohort=[s['session'] for s in selected_sessions],
        python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,scipy=scipy.__version__,
        threads=2,CPU_only=True,source_pixels_rehashed=False,
        input_policy='source hashes bound to extraction and intake; mask bytes verified; source originals rehashed for illustrations')
    config['signature']=digest(config)
    atomic(output/'configuration.json',config)
    snapshot=output/'source_snapshot'/config['signature']; snapshot.mkdir(parents=True,exist_ok=True)
    for name in source: shutil.copyfile(Path(__file__).parent/name,snapshot/name)
    records=[]; started=time.monotonic()
    for n,session in enumerate(selected_sessions):
        name=hashlib.sha256(session['session'].encode()).hexdigest()[:24]+'.json'
        path=output/'records'/name
        record=None
        if path.exists():
            old=json.loads(path.read_text(encoding='utf-8'))
            saved=old.pop('payload_sha256',None)
            if old.get('signature')==config['signature'] and saved==digest(old):
                record=old
        if record is None:
            step=time.monotonic()
            try:
                record=session_record(session,sessions,lookup,args.full_run)
            except Exception as exc:
                # A fatal software/input error is retained and aborts; do not
                # publish a falsely completed all-cohort table.
                atomic(output/'fatal_error.json',dict(session=session['session'],
                       error=repr(exc),completed=len(records),signature=config['signature']))
                raise
            record.update(signature=config['signature'],elapsed_seconds=time.monotonic()-step)
            record['payload_sha256']=digest(record); atomic(path,record)
        records.append(record)
        atomic(output/'progress.json',dict(status='running',completed=n+1,planned=len(selected_sessions),
               elapsed_seconds=time.monotonic()-started,last_session=session['session']))
        print(json.dumps(dict(completed=n+1,planned=len(selected_sessions),session=session['session'],
               status=record['fit']['status'],alternate=record['fit']['phase_gate']['accepted_alternate'],
               elapsed_seconds=round(time.monotonic()-started,2))),flush=True)
    summary=publish(output,records,config,time.monotonic()-started)
    atomic(output/'progress.json',dict(status='completed',completed=len(records),planned=len(selected_sessions),
                                    signature=config['signature']))
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
