"""Resumable all-frame SAM foreground extraction and explicit cohort aggregation.

Every identified, fully decodable source frame is retained. Operational QC is
not anatomical accuracy. The pinned public scientific operators are unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path,PureWindowsPath
import shutil
import sys
import time

KEYS=('treatment','measurement','genotype','replicate')
TRAITS=('silhouette_area_px','area_fraction','visible_height_px','silhouette_width_px',
        'height_fraction','width_fraction','perimeter_px','convex_hull_area_px','solidity',
        'eccentricity','skeleton_length_px','skeleton_endpoint_proxy','skeleton_junction_proxy',
        'centroid_x_fraction','centroid_y_fraction','bottom_censored','top_censored','side_censored',
        *[c+s for c in 'rgb' for s in ('_median','_iqr','_chromaticity_median')],
        'normalized_excess_green','green_red_normalized_difference','blue_red_normalized_difference',
        'hue_median','saturation_median','yellow_hue_fraction','saturation_clip_fraction',
        *[f'stratum_{i}_width_fraction' for i in range(1,6)])


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def write_csv(path,rows,fields=None):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    temporary.replace(path)


def truth(value):return value is True or str(value).lower()=='true'


def key(row):return tuple(str(row[k]) for k in KEYS)


def aggregate(master,frames,min_views=20):
    """Keep all reference samples; qualify each camera on count and quadrants."""
    import numpy as np
    references={key(r):r for r in master}
    if len(references)!=len(master):raise ValueError('Duplicate reference identity')
    groups=defaultdict(list);seen=set()
    for r in frames:
        sample=key(r)
        if sample not in references:raise ValueError('Frame has no reference identity')
        cam=r['camera'];view=int(r['view'])
        if cam not in ('rgb','noir') or not 0<=view<40:raise ValueError('Unexpected camera/view')
        unique=(sample,cam,view)
        if unique in seen:raise ValueError('Duplicate camera/view')
        seen.add(unique);groups[(sample,cam)].append(r)
    result=[]
    for reference in master:
        sample=key(reference);out={k:str(reference[k]) for k in KEYS}
        for cam in ('rgb','noir'):
            rows=groups[(sample,cam)]
            good=[r for r in rows if truth(r.get('qc_pass',False))]
            n=len(good);quadrants=len({int(r['view'])//10 for r in good})
            out[cam+'_decodable_views']=len(rows)
            out[cam+'_usable_views']=n;out[cam+'_quadrants']=quadrants
            out[cam+'_eligible']=n>=min_views and quadrants==4
            out[cam+'_eligibility_reason']='accepted' if out[cam+'_eligible'] else 'insufficient_accepted_views_or_angular_coverage'
            for trait in TRAITS:
                values=[]
                for r in good:
                    value=r.get(trait)
                    if value is not None and value!='' and math.isfinite(float(value)):values.append(float(value))
                stats=(float(np.median(values)),float(np.max(values)),float(np.percentile(values,75)-np.percentile(values,25))) if values else (None,None,None)
                for suffix,value in zip(('median','max','iqr'),stats):out[f'{cam}__{trait}__{suffix}']=value
        out['paired_eligible']=out['rgb_eligible'] and out['noir_eligible']
        result.append(out)
    return result


def inputs(args):
    intake=args.intake.resolve();run=args.run.resolve();repo=args.repo.resolve()
    original=Path(json.loads((intake/'intake_summary.json').read_text(encoding='utf-8'))['raw_root']).resolve()
    if run.is_relative_to(original) or run.is_relative_to(intake):raise ValueError('Run must be separate from raw archive and intake evidence')
    source=read_csv(intake/'jpeg_decode_manifest.csv')
    sessions=read_csv(intake/'session_decode_qc.csv')
    identified={r['session']:r for r in sessions if truth(r['known_reference_identity'])}
    tasks=[]
    for r in source:
        relative=Path(r['path'])
        if relative.is_absolute() or PureWindowsPath(r['path']).drive or '..' in relative.parts or not relative.parts or relative.parts[0]!='IMAGES':
            raise ValueError('Invalid source identity in intake inventory')
        session=Path(r['path']).parent.parent.as_posix()
        if not truth(r['decode_ok']) or session not in identified:continue
        p=Path(r['path']);view=int(p.stem.rsplit('_',1)[1])
        if not 0<=view<40:raise ValueError('Unsupported index')
        tasks.append(dict(r,session=session,camera=p.parent.name,view=view,
                          **{k:identified[session][k] for k in KEYS}))
    tasks.sort(key=lambda r:r['path'])
    if len({r['path'] for r in tasks})!=len(tasks):raise ValueError('Duplicate source path')
    methods=repo/'src/csi_maize/_methods'
    scientific_names=['traits.py','sam_traits.py','sam_segmentation.py','bootstrap.py','common.py']
    config=dict(schema_version=1,intake=str(intake),run=str(run),repo=str(repo),
                image_manifest_sha256=sha(intake/'jpeg_decode_manifest.csv'),
                reference_sha256=sha(intake/'research_audit/reference_master.csv'),
                tasks=len(tasks),task_digest=hashlib.sha256(json.dumps([(r['path'],r['sha256']) for r in tasks]).encode()).hexdigest(),
                operator_hashes={name:sha(methods/name) for name in scientific_names},
                mode='box_only_float32',processing_max_dimension=1024,minimum_views=20,
                required_quadrants=4,shards=args.shards)
    return tasks,config


def extract(args):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    tasks,config=inputs(args);root=args.run.resolve();out=root/'extraction'
    out.mkdir(parents=True,exist_ok=True)
    for name in ('records','masks','previews','sources'):(out/name).mkdir(exist_ok=True)
    methods=args.repo.resolve()/'src/csi_maize/_methods'
    sys.path.insert(0,str(methods))
    from traits import segment,measure_mask
    from sam_segmentation import make_prompts,MODEL_ID,REVISION
    from sam_traits import quality_checks
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import Sam2Model,Sam2Processor
    torch.set_num_threads(3)
    checkpoint=args.models.resolve()/'sam2.1-hiera-small'
    config.update(model=MODEL_ID,revision=REVISION,
                  environment=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,cuda=torch.version.cuda,
                                   device=args.device,gpu=torch.cuda.get_device_name(args.device),batch_size=args.batch_size),
                  assets={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()},
                  extraction_source_sha256=sha(__file__))
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    config['signature']=signature
    configpath=out/f'configuration_shard_{args.shard}.json'
    if configpath.exists():
        existing=json.loads(configpath.read_text(encoding='utf-8'))
        if existing!=config:raise ValueError('Configuration/source changed; use a new run')
    else:atomic_json(configpath,config)
    (out/'sources'/f'extraction_{config["extraction_source_sha256"]}.py').write_bytes(Path(__file__).read_bytes())
    selected=tasks[args.shard::args.shards]
    progresspath=out/f'progress_shard_{args.shard}.json'
    diagnostic_config=json.loads((args.intake/'segmentation_triage/configuration.json').read_text(encoding='utf-8'))
    if any(diagnostic_config['source_hashes'][name]!=value for name,value in config['operator_hashes'].items()):
        raise ValueError('Diagnostic operators differ from current released operators')
    if diagnostic_config['asset_hashes']!=config['assets']:raise ValueError('Diagnostic assets differ')
    cached_diagnostic={r['source']:r for r in read_csv(args.intake/'segmentation_triage/view_diagnostics.csv')}
    source_root=args.intake.resolve()
    expected_traitset=set(TRAITS)
    pending=[];complete=0;reused=0;started=time.monotonic()
    for task in selected:
        identity=hashlib.sha256(task['path'].encode()).hexdigest()[:24]
        target=out/'records'/f'{identity}.json'
        if target.exists():
            row=json.loads(target.read_text(encoding='utf-8'))
            if row.get('configuration_signature')!=signature or row['source_sha256']!=task['sha256']:
                raise ValueError('Cached frame source/config mismatch')
            if sha(source_root/task['path'])!=task['sha256']:raise ValueError('Original changed on resume')
            if row.get('mask') and sha(root/row['mask'])!=row['mask_sha256']:raise ValueError('Cached mask changed')
            complete+=1;reused+=1
        else:pending.append((task,identity,target))
    print(json.dumps(dict(stage='extraction_start',shard=args.shard,device=args.device,total=len(selected),resume_frames=reused,pending=len(pending))),flush=True)
    model,loading=Sam2Model.from_pretrained(checkpoint,local_files_only=True,output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError(str(loading))
    model=model.to(args.device).eval();processor=Sam2Processor.from_pretrained(checkpoint,local_files_only=True)
    def prepare(item):
        task,identity,target=item;path=source_root/task['path'];blob=path.read_bytes()
        digest=hashlib.sha256(blob).hexdigest()
        if digest!=task['sha256']:raise ValueError('Source changed: '+task['path'])
        row={k:str(task[k]) for k in KEYS}
        row.update(session=task['session'],camera=task['camera'],view=task['view'],source=task['path'],
                   source_sha256=digest,configuration_signature=signature,device=args.device,shard=args.shard,
                   qc_pass=False,qc_issues='',mask_origin='sam_inference',mask='',mask_sha256='')
        with Image.open(io.BytesIO(blob)) as original:
            native=original.size;original.draft('RGB',(1024,1024));original.load()
            image=original.convert('RGB');image.thumbnail((1024,1024));image=np.asarray(image)
        row.update(processing_width=image.shape[1],processing_height=image.shape[0],native_width=native[0],native_height=native[1])
        try:
            seed=segment(image,task['camera']);box,points,labels=make_prompts(seed,image)
            if seed.mean()>.15:raise ValueError('Seed occupies >15% of frame')
            row.update(box_xyxy=box,seed_area_px=int(seed.sum()))
            return row,image,seed,box,target,identity
        except Exception as exc:
            row['qc_issues']='seed_error: '+str(exc);atomic_json(target,row)
            return None
    def finish(item,mask,score,origin):
        row,image,seed,box,target,identity=item
        row.update(quality_checks(image,seed,mask,float(score)))
        row['mask_origin']=origin
        name=Path(row['session']).name+f'__{row["camera"]}__{row["view"]:03d}'
        maskpath=out/'masks'/f'{name}.png'
        Image.fromarray(mask.astype(np.uint8)*255).save(maskpath)
        row['mask']=maskpath.relative_to(root).as_posix();row['mask_sha256']=sha(maskpath)
        if mask.sum()>=10:
            values,_=measure_mask(image,mask,row['camera'])
            if set(values)!=expected_traitset:raise ValueError('Scientific trait schema changed')
            row.update(values)
        if row['view']%10==0:
            canvas=image.copy();canvas[~mask]=0
            preview=out/'previews'/f'{name}.jpg';Image.fromarray(canvas).save(preview,quality=90)
            row['preview']=preview.relative_to(root).as_posix()
        atomic_json(target,row)
    for start in range(0,len(pending),args.batch_size):
        batch=pending[start:start+args.batch_size]
        with ThreadPoolExecutor(max_workers=3) as pool:items=list(pool.map(prepare,batch))
        infer=[]
        for item in items:
            if item is None:continue
            row,image,seed,box,target,identity=item
            cached=cached_diagnostic.get(row['source'])
            if cached and cached.get('mask'):
                if cached['source_sha256']!=row['source_sha256']:raise ValueError('Diagnostic source mismatch')
                mask=np.asarray(Image.open(source_root/cached['mask']))>0
                if mask.shape!=seed.shape:raise ValueError('Diagnostic processing shape mismatch')
                finish(item,mask,float(cached['sam_score']),'intake_box_diagnostic')
            else:infer.append(item)
        if infer:
            data=processor(images=[Image.fromarray(x[1]) for x in infer],input_boxes=[[x[3]] for x in infer],return_tensors='pt').to(args.device)
            with torch.inference_mode():prediction=model(**data,multimask_output=False)
            masks=processor.post_process_masks(prediction.pred_masks.cpu(),data['original_sizes'])
            scores=prediction.iou_scores.detach().cpu().numpy().reshape(-1)
            for item,tensor,score in zip(infer,masks,scores):finish(item,tensor[0,0].numpy().astype(bool),float(score),'sam_inference')
        complete+=len(batch)
        progress=dict(status='running',shard=args.shard,device=args.device,completed=complete,total=len(selected),
                      elapsed_seconds=round(time.monotonic()-started,1),signature=signature)
        atomic_json(progresspath,progress)
        if (start//args.batch_size)%10==0:print(json.dumps(progress),flush=True)
    progress=dict(status='completed',shard=args.shard,device=args.device,completed=len(selected),total=len(selected),
                  elapsed_seconds=round(time.monotonic()-started,1),signature=signature)
    atomic_json(progresspath,progress);print(json.dumps(progress),flush=True)


def summarize(args):
    tasks,_=inputs(args);root=args.run.resolve();out=root/'extraction'
    rows=[]
    for task in tasks:
        identity=hashlib.sha256(task['path'].encode()).hexdigest()[:24]
        path=out/'records'/f'{identity}.json'
        if not path.exists():raise ValueError('Extraction incomplete; no final cohort table published')
        row=json.loads(path.read_text(encoding='utf-8'))
        if row['source_sha256']!=task['sha256']:raise ValueError('Frame source mismatch')
        rows.append(row)
    for shard in range(args.shards):
        p=json.loads((out/f'progress_shard_{shard}.json').read_text(encoding='utf-8'))
        if p['status']!='completed':raise ValueError('Incomplete shard')
    master=read_csv(args.intake/'research_audit/reference_master.csv')
    serial=[]
    for row in rows:
        serial.append({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    write_csv(out/'view_traits.csv',serial)
    sessions=aggregate(master,rows);write_csv(out/'session_traits.csv',sessions)
    for threshold in (30,40):write_csv(out/f'session_traits_min{threshold}.csv',aggregate(master,rows,threshold))
    summary=dict(status='completed',frames=len(rows),reference_samples=len(sessions),
                 frames_qc_pass=sum(truth(r['qc_pass']) for r in rows),
                 rgb_eligible=sum(r['rgb_eligible'] for r in sessions),
                 noir_eligible=sum(r['noir_eligible'] for r in sessions),
                 paired_eligible=sum(r['paired_eligible'] for r in sessions),
                 minimum_accepted_views=20,required_quadrants=4,manual_annotation_validation=False,
                 frame_table_sha256=sha(out/'view_traits.csv'),session_table_sha256=sha(out/'session_traits.csv'))
    atomic_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['extract','summarize'])
    for name in ('intake','run','repo'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--models',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=2);p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    if not 0<=args.shard<args.shards:raise ValueError('Invalid shard')
    if args.command=='extract':
        if args.models is None:raise ValueError('--models required')
        extract(args)
    else:summarize(args)


if __name__=='__main__':main()
