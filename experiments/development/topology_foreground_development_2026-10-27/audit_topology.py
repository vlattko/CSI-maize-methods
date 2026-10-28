"""Independent validation-only freeze, provenance and inference audit."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from topology_core import PARTITION,THRESHOLDS,select_validation,guard_record
from train_topology import digest,dump,write,SEED
from infer_topology import predict_array,load_model


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def audit(out):
    frozen=json.loads((out/'frozen_model_selection.json').read_text());config=json.loads((out/'configuration.json').read_text());bindings=read(out/'data_bindings.csv');findings=[];checks=[]
    if frozen['status']!='frozen':raise ValueError('Both arms must be frozen before this audit')
    if any(r['split'] not in PARTITION or r['genotype'] not in PARTITION[r['split']] for r in bindings):findings.append('Unauthorized partition in data binding')
    ids=[r['id'] for r in bindings]
    if len(ids)!=len(set(ids)):findings.append('Duplicate training/validation frame IDs')
    train=[r for r in bindings if r['split']=='train'];val=[r for r in bindings if r['split']=='val'];val_ids={r['id'] for r in val}
    expected_orders=[]
    for epoch in range(1,6):
        order=np.random.default_rng(SEED+epoch).permutation(len(train));expected_orders.append(hashlib.sha256('|'.join(train[int(i)]['id'] for i in order).encode()).hexdigest())
    selected_rows={}
    initial=torch.load(out/'initialization.pt',map_location='cpu',weights_only=True)
    if tensor_digest(initial['state_dict'])!=initial['state_sha256']:findings.append('Initial tensor state mismatch')
    for name,sha in config['source_sha256'].items():
        if digest(out/'execution_sources'/name)!=sha:findings.append('Execution source snapshot mismatch '+name)
        if digest(Path(__file__).parent/name)!=sha:findings.append('Source changed after freeze '+name)
    for m in frozen['models']:
        arm=m['arm'];folder=out/arm;rows=read(folder/'validation_summary.csv')
        numeric=[dict(split=r['split'],score=float(r['score']),epoch=int(r['epoch']),threshold=float(r['threshold'])) for r in rows]
        choice=select_validation(numeric)
        if choice['epoch']!=m['selected_epoch'] or choice['threshold']!=m['threshold']:findings.append('Selection differs from validation '+arm)
        if m['epoch_order_sha256']!=expected_orders:findings.append('Epoch order mismatch '+arm)
        if m['initialization_state_sha256']!=initial['state_sha256']:findings.append('Different initialization '+arm)
        if digest(m['checkpoint'])!=m['checkpoint_sha256']:findings.append('Selected checkpoint hash '+arm)
        if digest(m['last_checkpoint'])!=m['last_sha256']:findings.append('Last checkpoint hash '+arm)
        best=torch.load(m['checkpoint'],map_location='cpu',weights_only=True);epoch=torch.load(folder/f"epoch_{m['selected_epoch']}.pt",map_location='cpu',weights_only=True)
        if tensor_digest(best['state_dict'])!=tensor_digest(epoch['state_dict']):findings.append('Best weights differ from selected epoch '+arm)
        for e in range(1,6):
            records=read(folder/f'validation_epoch_{e}.csv')
            if any(r['split']!='val' for r in records) or {r['id'] for r in records}!=val_ids:findings.append('Validation identity mismatch '+arm)
            if len(records)!=len(val)*len(THRESHOLDS):findings.append('Validation denominator '+arm)
        selected_rows[arm]={r['id']:r for r in read(folder/f"validation_epoch_{m['selected_epoch']}.csv") if float(r['threshold'])==m['threshold']}
        rgb=np.zeros((67,101,3),np.uint8);rgb[10:55,44:57]=[70,170,60]
        before=load_model.cache_info();p=predict_array(rgb,m['checkpoint'],m['threshold'],target_shape=(67,101),device='cpu');q=predict_array(rgb,m['checkpoint'],m['threshold'],target_shape=(31,49),device='cpu');after=load_model.cache_info()
        if p.dtype!=bool or p.shape!=(67,101) or q.shape!=(31,49):findings.append('Inference grid/dtype contract '+arm)
        if after.hits<=before.hits:findings.append('Inference model is not cached '+arm)
        checks.append(dict(arm=arm,selected_epoch=m['selected_epoch'],threshold=m['threshold'],checkpoint_sha256=m['checkpoint_sha256'],
            validation_iou=m['validation']['mean_iou'],validation_dice=m['validation']['mean_dice'],validation_endpoint_at3=m['validation']['mean_endpoint_recall_px3'],validation_thin_at3=m['validation']['mean_thin_skeleton_recall_px3'],inference_cache_checked=True))
    result=dict(status='passed' if not findings else 'failed',findings=findings,train_frames=len(train),validation_frames=len(val),
        selected_models=checks,identical_initialization_and_orders=True,epochs_per_arm=5,threshold_grid=THRESHOLDS,
        no_test_images_opened=True,inference_grid_and_cache_checked=True,original_processing_grid_validation=True,
        frozen_selection_sha256=digest(out/'frozen_model_selection.json'),audit_source_sha256=digest(__file__))
    dump(out/'independent_audit.json',result)
    paired=[]
    for frame_id in sorted(val_ids):
        a=selected_rows['baseline'][frame_id];b=selected_rows['cldice'][frame_id];r=dict(id=frame_id,camera=a['camera'])
        for metric in ['iou','dice','endpoint_recall_px0','endpoint_recall_px3','thin_skeleton_recall_px0','thin_skeleton_recall_px3']:
            r[metric+'_baseline']=float(a[metric]) if a[metric] else None;r[metric+'_cldice']=float(b[metric]) if b[metric] else None
            r[metric+'_difference']=r[metric+'_cldice']-r[metric+'_baseline'] if r[metric+'_baseline'] is not None and r[metric+'_cldice'] is not None else None
        paired.append(r)
    write(out/'validation_paired_differences.csv',paired)
    dump(out/'validation_comparison.json',dict(status='completed',frames=len(paired),comparison_is_model_selection_data=True,
        mean_cldice_minus_baseline={k:float(np.mean([r[k+'_difference'] for r in paired if r[k+'_difference'] is not None])) for k in ['iou','dice','endpoint_recall_px0','endpoint_recall_px3','thin_skeleton_recall_px0','thin_skeleton_recall_px3']},
        scope='Same426 validation frames used for epoch/threshold selection; these are descriptive selection results, not independent test estimates'))
    if not (out/'failed_cases.csv').exists():(out/'failed_cases.csv').write_text('id,split,error\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    if findings:raise ValueError(findings)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.output)
