"""Independent integrity, split and complete-denominator audit of both runs."""
import argparse
import csv
import hashlib
import json
import shutil
import os
from pathlib import Path
from collections import Counter
from functools import lru_cache

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'

TRAIN=set(i for i in range(40) if i%4!=2)
TEST=set(range(2,40,4))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(condition,message):
    if not condition:raise AssertionError(message)

def verify_run(folder,periodic=False):
    config=read(folder/'configuration.json');signature=config.pop('signature')
    require(digest(config)==signature,'configuration_signature')
    config['signature']=signature
    expected=set(config['cohort']);records=[];masks={}
    for path in sorted((folder/'records').glob('*.json')):
        record=read(path);payload=record.pop('payload_sha256')
        require(payload==digest(record),'record_payload:'+str(path))
        require(record['signature']==signature,'record_signature')
        require(len(record['evaluation'])==(60 if periodic else 90),'all_planned_test_method_rows')
        require(len(record['inputs'])==80,'all_80_input_slots')
        for frame in record['inputs']:
            require(frame['split']==('test' if frame['view'] in TEST else 'train'),'input_split')
            if frame['status']=='usable':
                masks[frame['mask']]=frame['actual_mask_sha256']
        if periodic:
            fit=record['fit']
            for t in fit.get('teachers',[]):
                require(t['view'] in TRAIN and t['noir_view'] in TRAIN,'heldout_teacher_leakage')
            if fit.get('model'):
                require(set(fit['model']['teacher_indices'])<=TRAIN,'heldout_regression_leakage')
            for inner in fit.get('inner_validation',[]):
                require(set(inner.get('training_indices',[])).isdisjoint(inner.get('validation_indices',[])),'inner_overlap')
                require(set(inner.get('training_indices',[]))|set(inner.get('validation_indices',[]))<=TRAIN,'outer_leak_in_inner')
        else:
            fit=record['fit']
            require(len(fit['candidates'])==20,'all_mapping_candidates')
            for c in fit['candidates']:
                require(set(c.get('train_indices',[]))<=TRAIN,'heldout_fit_leakage')
                require(all((c['sign']*i+c['phase'])%40 in TRAIN for i in TRAIN),'phase_training_set_leakage')
                require(all((c['sign']*i+c['phase'])%40 in TEST for i in TEST),'phase_test_set_leakage')
        records.append(record)
    require({r['identity']['session'] for r in records}==expected,'full_cohort_record_set')
    require(len(records)==len(expected),'unique_acquisition_records')
    lookup=read(folder/'session_camera_lookup.json')
    require(lookup['status']=='completed' and len(lookup['rows'])==len(expected),'session_contract')
    require({r['session'] for r in lookup['rows']}==expected,'contract_all_identities')
    flat=rows(folder/'alignment_lookup.csv')
    require(len(flat)==len(expected)*40,'lookup_all_40_indices')
    require(len({(r['session'],int(r['rgb_view'])) for r in flat})==len(expected)*40,'lookup_unique_indices')
    metrics=rows(folder/'heldout_metrics.csv')
    require(len(metrics)==len(expected)*(60 if periodic else 90),'all_evaluation_rows')
    summary=read(folder/'summary.json');require(summary['status']=='completed','summary_completed')
    for filename,expected_hash in summary['outputs'].items():
        require(sha(folder/filename)==expected_hash,'output_hash:'+filename)
    source=config['sources'] if periodic else config['source_hashes']
    snapshot=folder/'source_snapshot'/signature
    for rel,h in source.items():require(sha(snapshot/rel)==h,'cached_source_hash:'+rel)
    return dict(signature=signature,acquisitions=len(records),input_slots=len(records)*80,
                lookup_rows=len(flat),heldout_method_rows=len(metrics),unique_mask_hashes=len(masks),
                split_integrity=True,source_snapshot_verified=True,output_hashes_verified=True),masks

def recompute_pixel_sample(output,full_run,frame_rows):
    """Recompute actual held-out overlap independently of fitted operators."""
    import numpy as np
    import cv2
    from PIL import Image
    cv2.setNumThreads(2)
    lookup={(r['session'],r['camera'].lower(),int(r['view'])):r for r in frame_rows}
    @lru_cache(maxsize=256)
    def mask(session,camera,view):
        row=lookup[session,camera,view]
        with Image.open(full_run/row['mask']) as im:
            return np.asarray(im.convert('L').resize((256,192),Image.Resampling.NEAREST))>0
    checked=0;max_error=0.;by_branch={}
    for branch in ('full','periodic/full'):
        count=0
        for path in sorted((output/branch/'records').glob('*.json')):
            record=read(path)
            reference='selected_causal' if branch.startswith('periodic') else 'nominal_shared'
            available=[r['rgb_view'] for r in record['evaluation'] if r['status']=='executed' and r['method']==reference]
            if not available:continue
            view=min(available)
            for row in record['evaluation']:
                if row['rgb_view']!=view or row['status']!='executed':continue
                session=row['session']
                other=record.get('other_session') if branch.startswith('periodic') else row.get('noir_session')
                target_session=other if row['method'].startswith('mismatch_other') else session
                a=mask(session,'rgb',view);b=mask(target_session,'noir',int(row['noir_view']))
                matrix=np.asarray(row['matrix'],np.float32)
                warped=cv2.warpAffine(b.astype(np.uint8),matrix,(256,192),
                    flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)>0
                union=np.count_nonzero(a|warped)
                measured=np.count_nonzero(a&warped)/union if union else 0.
                error=abs(measured-row['iou']);max_error=max(max_error,error)
                require(error<=1e-12,'independent_pixel_iou:'+str(path)+':'+row['method'])
                checked+=1;count+=1
        by_branch[branch]=count
    return dict(recomputed_rows=checked,rows_by_branch=by_branch,max_absolute_error=max_error,
                sample='first available withheld reference index in every fitted acquisition; all executed methods at that index')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--intake',type=Path,required=True);p.add_argument('--full-run',type=Path,required=True)
    a=p.parse_args();out=a.output
    checks,masks=verify_run(out/'full')
    periodic_checks,periodic_masks=verify_run(out/'periodic/full',True)
    require(masks==periodic_masks,'shared_and_periodic_mask_inputs_identical')
    for rel,h in masks.items():require(sha(a.full_run/rel)==h,'actual_mask_bytes:'+rel)
    extraction=rows(a.full_run/'extraction/view_traits.csv')
    pixel_check=recompute_pixel_sample(out,a.full_run,extraction)
    # The explicit source-key discovery is bounded to declared table headers.
    manifest_rows=rows(a.intake/'jpeg_decode_manifest.csv')
    path_field=next(k for k in ('relative_path','source','path','relative') if k in manifest_rows[0])
    decoded={r[path_field]:r for r in manifest_rows}
    source_bound=0
    for frame in extraction:
        require(frame['source'] in decoded,'source_not_in_intake_manifest')
        require(frame['source_sha256']==decoded[frame['source']]['sha256'],'source_hash_binding')
        source_bound+=1
    provenance=out/'upstream_provenance';provenance.mkdir(exist_ok=True)
    provenance_hashes={}
    sources=(list((a.full_run/'extraction').glob('configuration_shard_*.json'))+
             [a.intake/'intake_summary.json',a.full_run/'extraction/summary.json'])
    for source in sources:
        name=source.name if source.parent.name=='extraction' else 'intake_'+source.name
        shutil.copyfile(source,provenance/name);provenance_hashes[name]=sha(source)
    result=dict(status='verified_completed_full_data_camera_layers',schema=1,
        shared=checks,periodic=periodic_checks,actual_mask_hashes_reverified=len(masks),
        original_source_hash_bindings_verified=source_bound,
        independent_pixel_check=pixel_check,
        source_pixels_rehashed=False,source_verification_scope='Existing intake decode/source digest plus extraction binding; examples rehash original pixels separately',
        upstream_provenance_hashes=provenance_hashes,verification_source_sha256=sha(__file__))
    path=out/'verification.json';tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2),encoding='utf-8');tmp.replace(path)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
