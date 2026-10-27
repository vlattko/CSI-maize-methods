"""Matched common-support DN controls from immutable completed registrations."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
CHANNELS=('rgb_R','rgb_G','rgb_B','noir_R','noir_G','noir_B')
IDENTITY=('treatment','measurement','genotype','replicate')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def truth(value):return str(value).lower() in ('true','1')


def atomic_csv(path,rows):
    temporary=path.with_suffix('.csv.tmp')
    with temporary.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.replace(temporary,path)


def atomic_json(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)


def quantile(values,p):
    values=sorted(values);index=(len(values)-1)*p
    lo=math.floor(index);hi=math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def frame_statistics(six,channel_valid,common):
    import numpy as np
    rows=[]
    for i,channel in enumerate(CHANNELS):
        valid=common & channel_valid[...,i] & np.isfinite(six[...,i]) & (six[...,i]>1) & (six[...,i]<254)
        values=six[...,i][valid]
        n=len(values);eligible=int(common.sum())
        rows.append(dict(channel=channel,eligible_pixels=eligible,valid_pixels=n,
                         valid_fraction=n/eligible if eligible else None,qualified=n>=40,
                         median=float(np.median(values)) if n>=40 else None,
                         iqr=float(np.quantile(values,.75)-np.quantile(values,.25)) if n>=40 else None,
                         failure_reason='' if n>=40 else 'fewer_than_40_valid_common_pixels'))
    return rows


def aggregate(sessions,views):
    grouped={};seen=set();canonical={s['session']:s for s in sessions}
    for row in views:
        key=(row['session'],row['channel'],int(row['view']))
        if key in seen:raise ValueError('Duplicate DN quarter/channel')
        seen.add(key)
        if key[0] not in canonical or key[1] not in CHANNELS or key[2] not in (0,10,20,30):
            raise ValueError('Unknown DN key')
        if any(str(row[k])!=str(canonical[key[0]][k]) for k in IDENTITY):raise ValueError('DN identity mismatch')
        if truth(row['qualified']):
            if int(row['valid_pixels'])<40:raise ValueError('DN count below qualification')
            value=float(row['median'])
            if not math.isfinite(value):raise ValueError('Nonfinite DN median')
            grouped.setdefault(key[:2],[]).append(value)
    output=[]
    for session in sessions:
        row={k:session[k] for k in (*IDENTITY,'session')}
        eligible_channels=0
        for channel in CHANNELS:
            values=grouped.get((session['session'],channel),[])
            n=len(values);eligible=n>=2;eligible_channels+=eligible
            prefix='paired_dn__'+channel
            row[prefix+'__median']=statistics.median(values) if eligible else None
            row[prefix+'__iqr']=quantile(values,.75)-quantile(values,.25) if eligible else None
            row[prefix+'__n_views']=n
            row[prefix+'__eligible']=eligible
            row[prefix+'__missing_reason']='' if eligible else 'no_qualified_quarters' if n==0 else 'fewer_than_two_qualified_quarters'
        row['paired_dn_n_eligible_channels']=eligible_channels
        row['paired_dn_all_6_eligible']=eligible_channels==6
        output.append(row)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('intake','run','repo'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--watch',action='store_true')
    args=p.parse_args()
    intake=args.intake.resolve();run=args.run.resolve();repo=args.repo.resolve()
    root=run/'pairing/full';summary_path=root/'summary.json'
    while True:
        summary=json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
        if summary.get('status')=='completed':break
        if not args.watch:raise ValueError('Completed full pairing required')
        time.sleep(30)
    if summary.get('mode')!='full' or summary.get('planned_anchor_pairs')!=1440:
        raise ValueError('Cannot publish pilot or partial DN controls as full cohort')
    config=json.loads((root/'configuration.json').read_text(encoding='utf-8'))
    methods=repo/'src/csi_maize/_methods'
    for name in ('channel_fusion_core.py','multimodal_leaf_core.py','bootstrap.py'):
        if sha(methods/name)!=config['scientific_source_sha256'][name]:raise ValueError('Scientific operator source changed')
    sys.path.insert(0,str(methods))
    from channel_fusion_core import warp_to_rgb,interior
    import cv2
    import numpy as np
    from PIL import Image
    cv2.setNumThreads(1)
    sessions=[r for r in read(intake/'session_decode_qc.csv') if truth(r['known_reference_identity'])]
    if len(sessions)!=360:raise ValueError('All360 canonical sessions required')
    canonical={s['session']:s for s in sessions}
    rows=[];failures=[];verified_records={}
    def load(binding):
        row=binding['record'];source=intake/row['source'];mask_path=run/row['mask']
        if sha(source)!=binding['actual_source_sha256'] or sha(source)!=row['source_sha256']:
            raise ValueError('Changed original source')
        if sha(mask_path)!=binding['actual_mask_sha256'] or sha(mask_path)!=row['mask_sha256']:
            raise ValueError('Changed extraction mask')
        with Image.open(source) as im:
            im.load();image=np.asarray(im.convert('RGB').resize((512,385),Image.Resampling.LANCZOS))
        with Image.open(mask_path) as im:
            if im.size!=(int(row['processing_width']),int(row['processing_height'])):raise ValueError('Mask dimension mismatch')
            mask=np.asarray(im.convert('L').resize((512,385),Image.Resampling.NEAREST))>0
        return image,mask
    for path in sorted((root/'records').glob('*.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        expected=record.pop('payload_sha256')
        actual=hashlib.sha256(json.dumps(record,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
        if actual!=expected:raise ValueError('Numerical payload hash mismatch')
        verified_records[path.name]=expected
        base={k:record[k] for k in (*IDENTITY,'session','view')}
        if base['session'] not in canonical or any(str(base[k])!=str(canonical[base['session']][k]) for k in IDENTITY):
            raise ValueError('Wrong DN/canonical join')
        nominal=[e for e in record['evaluations'] if e['pair_kind']=='nominal' and e['method']=='ECC']
        reason='nominal_ECC_unavailable' if not nominal else 'cross_support_gate_failed' if not nominal[0]['cross_support_usable'] else ''
        output=[]
        if not reason:
            try:
                a,am=load(record['source_bindings']['rgb']);b,bm=load(record['source_bindings']['nominal'])
                matrix=np.asarray(nominal[0]['source_to_target'],np.float32)
                wb,field=warp_to_rgb(b.astype(np.float32),matrix,am.shape)
                wn,_=warp_to_rgb(interior(bm).astype(np.uint8),matrix,am.shape,True)
                common=interior(am)&(wn>0)&field
                validity,_=warp_to_rgb(((b>1)&(b<254)).astype(np.float32),matrix,am.shape)
                six=np.concatenate([a.astype(np.float32),wb],axis=-1)
                channel_valid=np.concatenate([(a>1)&(a<254),validity>.999],axis=-1)
                output=frame_statistics(six,channel_valid,common)
            except Exception as exc:
                reason=type(exc).__name__+':'+str(exc)
                failures.append(dict(base,reason=reason))
        if not output:
            output=[dict(channel=c,eligible_pixels=0,valid_pixels=0,valid_fraction=None,
                         qualified=False,median=None,iqr=None,failure_reason=reason) for c in CHANNELS]
        rows.extend(dict(base,**r) for r in output)
    if len(verified_records)!=1440 or len(rows)!=1440*6:raise ValueError('Incomplete fixed-quarter DN controls')
    output=aggregate(sessions,rows)
    atomic_csv(root/'camera_dn_common_views.csv',rows)
    atomic_csv(root/'session_camera_dn.csv',output)
    manifest=dict(status='completed',expected_rows=360,rows=len(output),
                  session_camera_dn_sha256=sha(root/'session_camera_dn.csv'),
                  feature_columns=['paired_dn__'+c+'__'+s for c in CHANNELS for s in ('median','iqr')],
                  availability_columns=['paired_dn__'+c+'__'+s for c in CHANNELS for s in ('n_views','eligible','missing_reason')],
                  aggregation_policy=dict(pair_kind='nominal',method='ECC',support='exact gated eroded whole-mask common support',
                    channel_guard='1<DN<254 on unregistered 512x385 JPEG grid; all NoIR affine contributors valid',
                    minimum_valid_pixels_per_quarter=40,minimum_qualified_quarters_per_channel=2,
                    summary='median and linear IQR of qualified per-quarter DN medians',
                    no_all_six_complete_requirement=True,targets_included=False,
                    note='Common support matches cross-index family; unimodal index families use separate camera interiors.'),
                  failures=failures,eligible_sessions_per_channel={c:sum(r['paired_dn__'+c+'__eligible'] for r in output) for c in CHANNELS},
                  inputs_sha256=dict(configuration=sha(root/'configuration.json'),summary=sha(summary_path)),
                  numerical_record_payload_sha256=verified_records,source_sha256=sha(__file__))
    atomic_json(root/'session_camera_dn_manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='numerical_record_payload_sha256'},indent=2))


if __name__=='__main__':main()
