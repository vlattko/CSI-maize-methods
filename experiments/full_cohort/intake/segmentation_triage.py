"""Stratified diagnostic replay of published SAM operators on the full archive.

One lowest-numbered fully decodable replicate per treatment/visit/genotype,
four fixed views and both cameras. This is a challenge audit, not an accuracy
benchmark: no independently annotated masks are supplied.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intake',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--mode',choices=['box','prompted'],default='box')
    p.add_argument('--output-name',default='segmentation_triage')
    args=p.parse_args()
    root=args.intake.resolve()
    if Path(args.output_name).name!=args.output_name or args.output_name in ('.','..'):
        raise ValueError('Output name must be a contained directory name')
    output=root/args.output_name
    if output.exists():
        raise ValueError('Refusing to overwrite a previous diagnostic run')
    output.mkdir()
    (output/'masks').mkdir();(output/'black_previews').mkdir()
    methods=args.repo.resolve()/'src/csi_maize/_methods'
    sys.path.insert(0,str(methods))
    # All scientific operators are imported unchanged from the public release.
    from traits import load_image, segment
    from sam_segmentation import make_prompts, MODEL_ID, REVISION
    from sam_traits import quality_checks
    import numpy as np
    import torch
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise ValueError('CUDA required for this declared replay')
    source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in methods.glob('*.py')}
    rows=read(root/'session_decode_qc.csv')
    chosen={}
    for r in sorted(rows,key=lambda r:(r['session'],int(r['replicate'] or 0))):
        if r['known_reference_identity']!='True' or r['strict_complete_40_pairs']!='True':continue
        key=(r['treatment'],r['measurement'],r['genotype'])
        if key not in chosen or int(r['replicate'])<int(chosen[key]['replicate']):chosen[key]=r
    sessions=[chosen[k] for k in sorted(chosen)]
    tasks=[(r,cam,view) for r in sessions for cam in ('rgb','noir') for view in (0,10,20,30)]
    decoded={r['path']:r['sha256'] for r in read(root/'jpeg_decode_manifest.csv')}
    checkpoint=args.models.resolve()/'sam2.1-hiera-small'
    assets={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in checkpoint.iterdir() if f.is_file()}
    config=dict(model=MODEL_ID,revision=REVISION,
                selection='lowest-numbered complete replicate per treatment/visit/genotype',
                fixed_views=[0,10,20,30],cameras=['rgb','noir'],sessions=len(sessions),
                planned_images=len(tasks),torch=torch.__version__,source_hashes=source_hashes,
                asset_hashes=assets,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                manual_annotation_validation=False,mode=args.mode)
    (output/'configuration.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (output/'execution_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in config.items() if k not in ('source_hashes','asset_hashes')}),flush=True)
    model,info=Sam2Model.from_pretrained(checkpoint,local_files_only=True,output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError(str(info))
    model=model.to('cuda:0').eval()
    processor=Sam2Processor.from_pretrained(checkpoint,local_files_only=True)
    fields=['session','treatment','measurement','genotype','replicate','camera','view','source',
            'source_sha256','processing_width','processing_height','seed_fraction','mask_fraction',
            'mask_pixels','mask_border_touch','qc_pass','qc_issues','sam_score','seed_coverage',
            'mask_seed_area_ratio','dark_pixel_fraction','box_xyxy','points_xy','point_labels','mask','preview']
    allrows=[];started=time.monotonic()
    with (output/'view_diagnostics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for start in range(0,len(tasks),args.batch_size):
            ready=[];batchrows=[]
            for r,cam,view in tasks[start:start+args.batch_size]:
                relative=f"{r['session']}/{cam}/image_{cam}_{view:03d}.jpg"
                raw=root/relative
                digest=hashlib.sha256(raw.read_bytes()).hexdigest()
                if digest!=decoded[relative]:raise ValueError('Original bytes changed after intake: '+relative)
                row={k:r[k] for k in ('session','treatment','measurement','genotype','replicate')}
                row.update(camera=cam,view=view,source=relative,source_sha256=digest,qc_pass=False)
                try:
                    image,_=load_image(raw);seed=segment(image,cam)
                    box,points,labels=make_prompts(seed,image)
                    if seed.mean()>0.15:raise ValueError('Seed occupies >15% of frame')
                    row.update(box_xyxy=json.dumps(box),
                               points_xy=json.dumps(points) if args.mode=='prompted' else '[]',
                               point_labels=json.dumps(labels) if args.mode=='prompted' else '[]')
                    ready.append((row,image,seed,box,points,labels))
                except Exception as exc:row['qc_issues']='seed_error: '+str(exc);batchrows.append(row)
            if ready:
                kwargs=dict(input_boxes=[[t[3]] for t in ready])
                if args.mode=='prompted':
                    kwargs.update(input_points=[[t[4]] for t in ready],input_labels=[[t[5]] for t in ready])
                inputs=processor(images=[Image.fromarray(t[1]) for t in ready],
                                 **kwargs,return_tensors='pt').to('cuda:0')
                with torch.inference_mode():pred=model(**inputs,multimask_output=False)
                masks=processor.post_process_masks(pred.pred_masks.cpu(),inputs['original_sizes'])
                scores=pred.iou_scores.detach().cpu().numpy().reshape(-1)
                for (row,image,seed,box,points,labels),tensor,score in zip(ready,masks,scores):
                    mask=tensor[0,0].numpy().astype(bool)
                    row.update(quality_checks(image,seed,mask,float(score)))
                    stem=Path(row['session']).name+f"__{row['camera']}__{row['view']:03d}"
                    target=output/'masks'/f'{stem}.png'
                    Image.fromarray(mask.astype(np.uint8)*255).save(target)
                    preview=output/'black_previews'/f'{stem}.jpg'
                    canvas=image.copy();canvas[~mask]=0
                    Image.fromarray(canvas).save(preview,quality=92)
                    row.update(processing_width=image.shape[1],processing_height=image.shape[0],
                               seed_fraction=float(seed.mean()),mask_fraction=float(mask.mean()),
                               mask_pixels=int(mask.sum()),
                               mask_border_touch=bool(mask[0,:].any() or mask[-1,:].any() or mask[:,0].any() or mask[:,-1].any()),
                               mask=target.relative_to(root).as_posix(),preview=preview.relative_to(root).as_posix())
                    batchrows.append(row)
            writer.writerows(batchrows);stream.flush();allrows.extend(batchrows)
            progress=dict(status='running',processed=len(allrows),planned=len(tasks),
                          qc_pass=sum(r['qc_pass'] for r in allrows),elapsed_seconds=round(time.monotonic()-started,1))
            (output/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
            print(json.dumps(progress),flush=True)
    summary=dict(status='completed',sessions=len(sessions),images=len(allrows),
                 qc_pass=sum(r['qc_pass'] for r in allrows),manual_annotation_validation=False,
                 limitations=['Automated seed/model agreement is not mask accuracy.',
                              'Sparse views do not establish full-rotation QC.',
                              'Complete-case selection excludes partial and damaged sessions.',
                              'Stress-induced brown or detached tissue can be absent from a colour-seeded box.',
                              'Masks label foreground, not individual leaves or persistent identities.'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    progress['status']='completed'
    (output/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
