"""Local SAM2.1 bounded pilot; outputs are unreviewed region proposals."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import hashlib
import io
import json
import shutil
import time
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import cv2
import torch
import transformers
from transformers import Sam2Model,Sam2Processor
from leaf_plan import sha,digest,read,atomic
from leaf_proposals import clip_statistics

torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(2)

def prepare(frame,intake,full_run,output,token):
    row=frame['source_record'];source=intake/row['source'];blob=source.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=row['source_sha256']:raise ValueError('source_hash_mismatch')
    with Image.open(io.BytesIO(blob)) as original:
        native=original.size;original.draft('RGB',(1024,1024));original.load()
        image=original.convert('RGB');image.thumbnail((1024,1024));pixels=np.asarray(image)
    if image.size!=(int(row['processing_width']),int(row['processing_height'])):
        raise ValueError('image_processing_dimensions_mismatch')
    maskpath=full_run/row['mask']
    if sha(maskpath)!=row['mask_sha256']:raise ValueError('parent_mask_hash_mismatch')
    with Image.open(maskpath) as im:parent=np.asarray(im.convert('L'))>0
    if parent.shape!=pixels.shape[:2]:raise ValueError('parent_processing_grid_mismatch')
    display=pixels.copy();display[~parent]=0
    imagepath=output/'review/images'/f'{token}.png';imagepath.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(display).save(imagepath)
    return image,parent,dict(source_sha256=row['source_sha256'],parent_mask_sha256=row['mask_sha256'],
        prepared_rgb_pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
        prepared_size=list(image.size),native_size=list(native),
        review_image=imagepath.relative_to(output).as_posix(),review_image_sha256=sha(imagepath))

def supports(mask,prompt):
    result={}
    for label,name in [(1,'positive_hit_fraction'),(0,'negative_excluded_fraction')]:
        pts=[xy for xy,l in zip(prompt['points'],prompt['labels']) if l==label]
        hits=[]
        for x,y in pts:
            ix=min(mask.shape[1]-1,max(0,round(x)));iy=min(mask.shape[0]-1,max(0,round(y)))
            hits.append(bool(mask[iy,ix]))
        result[name]=float(np.mean(hits if label else [not x for x in hits])) if hits else None
    return result

def infer_frame(frame,model,processor,intake,full_run,output,token):
    image,parent,inputs=prepare(frame,intake,full_run,output,token)
    record=dict(identity=frame['identity'],camera=frame['camera'],view=frame['view'],
        frame_id=token,plan_status=frame['status'],input=inputs,source_record=frame['source_record'],
        graph_path=frame['graph_path'],graph_sha256=frame['graph_sha256'],
        planned_proposals=frame['proposals'],prompt_failures=frame['failures'],
        proposals=[],failures=[],status='not_inferred_no_ready_prompt',
        region_status='unreviewed_region_predictions_not_anatomical_ground_truth')
    candidates=frame['proposals']
    if not candidates:return record
    boxes=[c['prompt_processing']['box'] for c in candidates]
    first=processor(images=image,input_boxes=[boxes],return_tensors='pt')
    with torch.inference_mode():embeddings=model.get_image_embeddings(first['pixel_values'].to('cuda:0'))
    for mode in ('box_only','distal_points_negatives'):
        try:
            arguments=dict(image_embeddings=embeddings,input_boxes=first['input_boxes'].to('cuda:0'),multimask_output=True)
            if mode=='distal_points_negatives':
                points=[];labels=[]
                for candidate in candidates:
                    prompt=candidate['prompt_processing'];n=len(prompt['points'])
                    points.append(prompt['points']+[[0.,0.]]*(6-n));labels.append(prompt['labels']+[-1]*(6-n))
                encoded=processor(images=image,input_boxes=[boxes],input_points=[points],input_labels=[labels],return_tensors='pt')
                arguments.update(input_points=encoded['input_points'].to('cuda:0'),input_labels=encoded['input_labels'].to('cuda:0'))
            with torch.inference_mode():prediction=model(**arguments)
            if not torch.isfinite(prediction.pred_masks).all():raise ValueError('nonfinite_model_mask_logits')
            masks=processor.post_process_masks(prediction.pred_masks.cpu(),first['original_sizes'])[0].numpy().astype(bool)
            scores=prediction.iou_scores.detach().cpu().numpy()[0]
            if len(masks)!=len(candidates) or masks.shape[1]!=3:raise ValueError('unexpected_multimask_output_shape')
            for ci,candidate in enumerate(candidates):
                for alternative in range(3):
                    raw=masks[ci,alternative];clipped,stats=clip_statistics(raw,parent)
                    proposal_id=digest([token,candidate['candidate_id'],mode,alternative])[:24]
                    paths={}
                    for kind,mask in [('raw',raw),('clipped',clipped)]:
                        path=output/f'masks/{kind}/{proposal_id}.png';path.parent.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(mask.astype(np.uint8)*255).save(path)
                        paths[kind+'_mask']=path.relative_to(output).as_posix();paths[kind+'_mask_sha256']=sha(path)
                    n_components=cv2.connectedComponents(clipped.astype(np.uint8),connectivity=8)[0]-1
                    score=float(scores[ci,alternative])
                    record['proposals'].append(dict(proposal_id=proposal_id,candidate_id=candidate['candidate_id'],
                        track_id=candidate['track_id'],mode=mode,alternative=alternative,
                        model_score=score if np.isfinite(score) else None,score_is_anatomical_accuracy=False,
                        prompt_processing=candidate['prompt_processing'],prompt_grid=candidate['prompt_grid'],
                        prompt_origin=candidate['prompt_origin'],duplicate_local_terminal_match=candidate['duplicate_local_terminal_match'],
                        region_review_status='unreviewed',**paths,**stats,**supports(clipped,candidate['prompt_processing']),
                        clipped_connected_components=int(n_components),
                        status='empty_clipped_region' if not clipped.any() else 'unreviewed_region_proposal'))
            del prediction,masks
        except (ValueError,RuntimeError) as exc:
            record['failures'].append(dict(mode=mode,error=repr(exc),planned_candidates=len(candidates)))
            if isinstance(exc,torch.OutOfMemoryError):torch.cuda.empty_cache()
    record['status']='completed' if not record['failures'] else 'completed_with_inference_failures'
    del embeddings
    torch.cuda.synchronize()
    record['peak_gpu_allocated_bytes']=int(torch.cuda.max_memory_allocated())
    return record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--intake',type=Path,required=True);parser.add_argument('--full-run',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plan=read(args.plan);signature=plan.pop('signature')
    if digest(plan)!=signature:raise ValueError('pilot_plan_digest_mismatch')
    plan['signature']=signature
    if plan['actual_frame_tasks']!=60 or not plan['pilot']:raise ValueError('expected_fixed_60_frame_pilot')
    output=args.output/'pilot';output.mkdir(parents=True,exist_ok=True);(output/'records').mkdir(exist_ok=True)
    source=Path(__file__).parent
    config=dict(schema=1,pilot_plan_sha256=sha(args.plan),plan_signature=signature,
        source_hashes={p.name:sha(p) for p in source.glob('*.py')},
        model_id='facebook/sam2.1-hiera-small',local_checkpoint=str(args.model.resolve()),
        model_assets={p.name:sha(p) for p in args.model.iterdir() if p.is_file()},
        local_files_only=True,model_weights_trained=False,device='cuda:0',visible_physical_gpu='0',
        gpu_name=torch.cuda.get_device_name(0),gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,
        float32=True,image_embedding_reuse_per_frame=True,multimask_alternatives=3,threads=2,
        variants=['box_only','distal_points_negatives'],planned_frames=60,
        expected_proposal_alternatives=plan['ready_prompts']*2*3,
        review_status='unreviewed_regions_only_not_training_labels')
    config['signature']=digest(config);atomic(output/'configuration.json',config)
    snapshot=output/'source_snapshot'/config['signature'];snapshot.mkdir(parents=True,exist_ok=True)
    for name in config['source_hashes']:shutil.copyfile(source/name,snapshot/name)
    model,loading=Sam2Model.from_pretrained(args.model,local_files_only=True,output_loading_info=True)
    atomic(output/'model_loading.json',loading)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('model_loading_keys_not_exact')
    model=model.to('cuda:0',dtype=torch.float32).eval()
    processor=Sam2Processor.from_pretrained(args.model,local_files_only=True)
    records=[];started=time.monotonic()
    for n,frame in enumerate(plan['frames']):
        token=digest([frame['identity']['session'],frame['camera'],frame['view']])[:24]
        path=output/'records'/f'{token}.json';record=None
        if path.exists():
            old=read(path);payload=old.pop('payload_sha256',None)
            if old.get('signature')==config['signature'] and digest(old)==payload:
                for proposal in old['proposals']:
                    for kind in ('raw','clipped'):
                        if sha(output/proposal[kind+'_mask'])!=proposal[kind+'_mask_sha256']:raise ValueError('cached_region_mask_changed')
                record=old
        if record is None:
            step=time.monotonic()
            try:record=infer_frame(frame,model,processor,args.intake,args.full_run,output,token)
            except (ValueError,OSError) as exc:
                record=dict(identity=frame['identity'],camera=frame['camera'],view=frame['view'],frame_id=token,
                    status='failed_frame',error=repr(exc),proposals=[],failures=[dict(error=repr(exc))])
            record.update(signature=config['signature'],elapsed_seconds=time.monotonic()-step)
            record['payload_sha256']=digest(record);atomic(path,record)
        records.append(record)
        progress=dict(status='running',completed=n+1,planned=60,regions=sum(len(r['proposals']) for r in records),
               elapsed_seconds=time.monotonic()-started,last_frame=token)
        atomic(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    summary=dict(status='completed',signature=config['signature'],frames=len(records),
        frame_status=dict(Counter(r['status'] for r in records)),
        proposal_alternatives=sum(len(r['proposals']) for r in records),
        planned_proposal_alternatives=config['expected_proposal_alternatives'],
        inference_failures=sum(len(r['failures']) for r in records),
        peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),elapsed_seconds=time.monotonic()-started,
        outputs_review_only=True,anatomical_labels_created=False)
    atomic(output/'summary.json',summary);atomic(output/'progress.json',dict(status='completed',completed=60,planned=60,signature=config['signature']))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
