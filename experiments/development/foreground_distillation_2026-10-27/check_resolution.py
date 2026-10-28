"""Fixed-weight validation-only resolution diagnostic for teacher terminal preservation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import foreground_core as core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','intake','teacher-run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    card=json.loads((args.output/'deployment/pilot_joint/model_card.json').read_text())
    if core.digest(card['weights'])!=card['weights_sha256']:
        raise ValueError('Pilot model changed')
    inventory=core.read_csv(args.output/'quarter_frames.csv')
    grouped=defaultdict(list)
    for r in inventory:
        if r['split']=='val' and r['teacher_accepted'].lower()=='true':
            grouped[(r['treatment'],r['measurement'],r['camera'])].append(r)
    selected=[]
    for _,rows in sorted(grouped.items()):
        selected.extend(sorted(rows,key=lambda r:(int(r['view']),int(r['replicate']),r['id']))[:2])
    destination=args.output/'validation_resolution'; destination.mkdir(exist_ok=True)
    core.write_json(destination/'protocol.json',dict(status='declared_before_diagnostic',sample_ids=[r['id'] for r in selected],
        sample_policy='two accepted fixed-quarter validation frames per treatment/visit/camera; ordered by view,replicate,id',
        genotype='Os6251/15',sizes=[640,960,1280],confidence=card['confidence'],weights_sha256=card['weights_sha256'],
        checkpoint_training='five-epoch joint quarter pilot at640',test_used=False,concurrent_training=True))
    core.configure_offline(args.repo,args.output)
    from ultralytics import YOLO
    import torch
    model=YOLO(card['weights'])
    prepared=[]
    for row in selected:
        rgb=core.processing_image(args.intake/row['source'])
        truth=np.array(Image.open(args.teacher_run/row['mask']))>0
        prepared.append((row,np.ascontiguousarray(rgb[:,:,::-1]),truth,core.terminal_geometry(truth)))
    records=[]; summaries=[]
    for size in (640,960,1280):
        model.predict(prepared[0][1],device=1,imgsz=size,conf=card['confidence'],retina_masks=True,max_det=100,verbose=False)
        for row,image,truth,geometry in prepared:
            torch.cuda.synchronize(1); start=time.perf_counter()
            result=model.predict(image,device=1,imgsz=size,conf=card['confidence'],retina_masks=True,max_det=100,verbose=False)[0]
            torch.cuda.synchronize(1); elapsed=(time.perf_counter()-start)*1000
            mask=core.prediction_union(result,truth.shape,card['confidence'])
            records.append(dict(id=row['id'],camera=row['camera'],treatment=row['treatment'],measurement=row['measurement'],
                imgsz=size,predict_ms=elapsed,**core.mask_metrics(truth,mask),**core.terminal_metrics(truth,mask,geometry)))
        subset=[r for r in records if r['imgsz']==size]
        summary=dict(imgsz=size,n=len(subset),nonempty=sum(r['predicted_pixels']>0 for r in subset),
                     predict_ms_median=float(np.median([r['predict_ms'] for r in subset])))
        for metric in ('iou','dice','endpoint_recall_px0','endpoint_recall_px3','endpoint_recall_px5','thin_skeleton_recall_px0','thin_skeleton_recall_px3'):
            values=[r[metric] for r in subset if r[metric] is not None]
            summary['mean_'+metric]=float(np.mean(values)) if values else None
        summaries.append(summary)
        core.write_json(destination/'summary.json',dict(status='running' if size!=1280 else 'completed',summaries=summaries,
            limits='Validation-only teacher preservation on60balanced frames; same640trained weights and fixed confidence; timings overlap training'))
        print(json.dumps(summary),flush=True)
    core.write_csv(destination/'frame_metrics.csv',records)


if __name__=='__main__':
    main()
