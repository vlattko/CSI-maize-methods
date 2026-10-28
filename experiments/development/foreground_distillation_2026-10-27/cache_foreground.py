"""Resume model-bound whole-cohort foreground masks for the interactive viewer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import foreground_core as core


def model_card_weights(card_path,card):
    """Resolve portable checkpoint names beside their hash-bound model card."""
    weights=Path(card['weights'])
    if not weights.is_absolute():
        weights=Path(card_path).resolve().parent/weights
    weights=weights.resolve()
    if core.digest(weights)!=card['weights_sha256']:
        raise ValueError('Published checkpoint hash changed')
    return weights


def restrict_cache_rows(rows,protocol_ready,development_only):
    if development_only:
        return [r for r in rows if r['genotype']!='Os7798/2']
    if not protocol_ready and any(r['genotype']=='Os7798/2' for r in rows):
        raise ValueError('Held-out student previews are locked until model choices freeze; use --development-only')
    return rows


def coverage_rows(references,records,expected_views):
    from collections import defaultdict
    grouped=defaultdict(list)
    keys=('treatment','measurement','genotype','replicate')
    for record in records:
        grouped[tuple(str(record[k]) for k in keys)+(record['camera'],)].append(record)
    rows=[]
    for ref in references:
        for camera in ('rgb','noir'):
            values=grouped.get(tuple(str(ref[k]) for k in keys)+(camera,),[])
            rows.append(dict(**{k:ref[k] for k in keys},camera=camera,planned_views=expected_views,
                available_student_frames=len(values),nonempty_student_frames=sum(bool(r['nonempty']) for r in values),
                teacher_accepted_frames=sum(str(r['teacher_accepted']).lower()=='true' for r in values)))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','intake','teacher-run','output','model-card'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--decode-workers',type=int,default=4)
    p.add_argument('--quarters-only',action='store_true')
    p.add_argument('--development-only',action='store_true')
    p.add_argument('--limit',type=int)
    p.add_argument('--device',type=int,choices=[0,1],default=1)
    args=p.parse_args()
    wall_start=time.perf_counter()
    if not 1<=args.batch_size<=8 or not 1<=args.decode_workers<=4:
        raise ValueError('Cache is bounded to batch1..8 and decode workers1..4')
    card=json.loads(args.model_card.read_text())
    weights=model_card_weights(args.model_card,card)
    tag=card['name']
    imgsz=int(card.get('imgsz',640))
    if imgsz not in (640,960,1280):
        raise ValueError('Unsupported explicit model-card inference size')
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in tag):
        raise ValueError('Unsafe model-card cache name')
    destination=args.output/'student_cache'/tag
    if args.quarters_only:
        destination=destination/'quarters'
    if args.development_only:
        destination=destination/'development'
    if args.limit:
        destination=destination/f'limited_{args.limit}'
    records_dir=destination/'records'; masks_dir=destination/'masks'
    for path in (records_dir,masks_dir):
        path.mkdir(parents=True,exist_ok=True)
    table=args.teacher_run/'extraction/view_traits.csv'
    summary=json.loads((args.teacher_run/'extraction/summary.json').read_text())
    if summary['status']!='completed' or core.digest(table)!=summary['frame_table_sha256']:
        raise ValueError('Completed, hash-bound teacher inventory is required')
    rows=core.read_csv(table)
    if args.quarters_only:
        rows=[r for r in rows if int(r['view']) in core.QUARTERS]
    if args.limit:
        rows=rows[:args.limit]
    gate=args.output/'additional_model_test_gate.json'
    ready=(args.output/'evaluation/final_protocol_ready.json').exists() and (not gate.exists() or json.loads(gate.read_text()).get('status')=='frozen')
    rows=restrict_cache_rows(rows,ready,args.development_only)
    operator=dict(weights_sha256=card['weights_sha256'],confidence=card['confidence'],imgsz=imgsz,
        decode='original source; same PIL draft/thumbnail maximum1024',mask='retina all-instance plant_foreground union',
        teacher_table_sha256=core.digest(table),source_sha256=core.digest(__file__),core_sha256=core.digest(Path(core.__file__)),
        batch_size=args.batch_size,quarters_only=args.quarters_only,limit=args.limit,development_only=args.development_only,
        device=args.device,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    signature=hashlib.sha256(json.dumps(operator,sort_keys=True).encode()).hexdigest()
    specification=destination/'operator.json'
    if specification.exists() and json.loads(specification.read_text())['signature']!=signature:
        raise ValueError('Student cache operator changed; do not mix model outputs')
    core.write_json(specification,dict(signature=signature,**operator))
    shutil.copyfile(__file__,destination/'execution_cache_foreground.py')
    shutil.copyfile(Path(core.__file__),destination/'execution_foreground_core.py')
    core.configure_offline(args.repo,args.output)
    from ultralytics import YOLO
    import torch
    model=YOLO(str(weights))
    model.predict(np.zeros((769,1024,3),np.uint8),device=args.device,imgsz=imgsz,conf=card['confidence'],retina_masks=True,max_det=100,verbose=False)
    torch.cuda.reset_peak_memory_stats(args.device)
    completed=[]; pending=[]
    for row in rows:
        identifier=hashlib.sha256(row['source'].encode()).hexdigest()[:24]
        path=records_dir/(identifier+'.json')
        if path.exists():
            record=json.loads(path.read_text())
            if record['signature']!=signature or record['source_sha256']!=row['source_sha256'] or core.digest(destination/record['student_mask'])!=record['student_mask_sha256']:
                raise ValueError('Student cache record or mask changed')
            completed.append(record)
        else:
            pending.append((identifier,row))
    start=time.perf_counter()
    def decode(item):
        identifier,row=item
        source=args.intake/row['source']; begin=time.perf_counter()
        if core.digest(source)!=row['source_sha256']:
            raise ValueError('Raw source digest changed before student inference')
        rgb=core.processing_image(source)
        return identifier,row,np.ascontiguousarray(rgb[:,:,::-1]),(time.perf_counter()-begin)*1000
    with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
        for begin in range(0,len(pending),args.batch_size):
            batch=list(pool.map(decode,pending[begin:begin+args.batch_size]))
            torch.cuda.synchronize(args.device); predict_start=time.perf_counter()
            results=model.predict([x[2] for x in batch],device=args.device,imgsz=imgsz,conf=card['confidence'],retina_masks=True,max_det=100,verbose=False)
            torch.cuda.synchronize(args.device); batch_ms=(time.perf_counter()-predict_start)*1000
            for (identifier,row,image,decode_ms),result in zip(batch,results):
                mask=core.prediction_union(result,image.shape[:2],card['confidence'])
                path=masks_dir/(identifier+'.png')
                Image.fromarray(mask.astype(np.uint8)*255).save(path)
                record=dict(signature=signature,**{k:row[k] for k in ('treatment','measurement','genotype','replicate','session','camera','view','source','source_sha256')},
                    id=identifier,teacher_accepted=row['qc_pass'],teacher_mask=row['mask'],
                    student_mask=str(path.relative_to(destination)).replace(chr(92),'/'),student_mask_sha256=core.digest(path),
                    foreground_pixels=int(mask.sum()),nonempty=bool(mask.any()),processing_width=mask.shape[1],processing_height=mask.shape[0],
                    decode_and_source_hash_ms=decode_ms,batch_predict_ms=batch_ms,batch_size=len(batch))
                core.write_json(records_dir/(identifier+'.json'),record); completed.append(record)
            core.write_json(destination/'status.json',dict(status='running',model=tag,expected_frames=len(rows),cached_frames=len(completed),
                elapsed_seconds=time.perf_counter()-start,operator_signature=signature))
            if len(completed)%200<args.batch_size:
                print('student cache',len(completed),'of',len(rows),flush=True)
    completed.sort(key=lambda r:r['source'])
    core.write_csv(destination/'frames.csv',completed)
    references=core.read_csv(args.intake/'research_audit/reference_master.csv')
    coverage=coverage_rows(references,completed,4 if args.quarters_only else 40)
    for row in coverage:
        row['inference_scope']='heldout_excluded' if args.development_only and row['genotype']=='Os7798/2' else 'selected_cache_scope'
    core.write_csv(destination/'acquisition_coverage.csv',coverage)
    core.write_json(destination/'status.json',dict(status='completed',model=tag,expected_frames=len(rows),cached_frames=len(completed),
        nonempty=sum(r['nonempty'] for r in completed),frames_sha256=core.digest(destination/'frames.csv'),
        reference_records=len(references),coverage_sha256=core.digest(destination/'acquisition_coverage.csv'),
        development_only=args.development_only,heldout_excluded=args.development_only,
        elapsed_seconds_this_resume=time.perf_counter()-start,operator_signature=signature,
        elapsed_seconds_entire_run=time.perf_counter()-wall_start,device=args.device,
        GPU=torch.cuda.get_device_name(args.device),peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
        mask_root=str(destination),inference_gating='none; every available source including teacher-rejected frames is processed',
        limitations='UI masks are whole-plant predictions; training/validation frames in this cache are not independent evaluation records'))
    print('completed',destination,flush=True)


if __name__=='__main__':
    main()
