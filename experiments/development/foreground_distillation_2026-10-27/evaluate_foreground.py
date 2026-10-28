"""Full-mask teacher agreement, coverage and timing on frozen genotype partitions."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import importlib.util
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image
import foreground_core as core

EXECUTION_SOURCE=Path(__file__).read_bytes()
EXECUTION_SHA256=__import__('hashlib').sha256(EXECUTION_SOURCE).hexdigest()

CONFIDENCES=(.05,.10,.25,.50)


def verify_frame_inputs(frames,intake,teacher_run):
    for row in frames:
        if core.digest(Path(intake)/row['source'])!=row['source_sha256'] or core.digest(Path(teacher_run)/row['mask'])!=row['mask_sha256']:
            raise ValueError('Evaluation source image or original teacher-mask bytes changed')


def prepare_cached_frame(row,intake,teacher_run,cache):
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    identifier=row['id']+'_'+row['source_sha256'][:12]+'_'+row['mask_sha256'][:12]
    metadata=cache/(identifier+'.json')
    signature=core.digest(Path(core.__file__))
    if metadata.exists():
        entry=json.loads(metadata.read_text())
        if entry['operator_signature']!=signature or core.digest(entry['image'])!=entry['image_sha256'] or core.digest(entry['geometry'])!=entry['geometry_sha256']:
            raise ValueError('Evaluation processing cache changed; refusing a corrupt resume')
        return entry
    start=time.perf_counter(); image=core.processing_image(Path(intake)/row['source'])
    raw_decode_ms=(time.perf_counter()-start)*1000
    truth=np.array(Image.open(Path(teacher_run)/row['mask']))>0
    if image.shape[:2]!=truth.shape:
        raise ValueError('Evaluation processing image and original teacher mask differ in shape')
    geometry=core.terminal_geometry(truth)
    image_path=cache/(identifier+'.npy'); geometry_path=cache/(identifier+'.npz')
    np.save(image_path,image,allow_pickle=False)
    np.savez_compressed(geometry_path,truth=truth,**geometry)
    entry=dict(id=row['id'],image=str(image_path),geometry=str(geometry_path),
        image_sha256=core.digest(image_path),geometry_sha256=core.digest(geometry_path),
        operator_signature=signature,source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],raw_decode_ms=raw_decode_ms)
    core.write_json(metadata,entry)
    return entry


def load_cached_frame(entry):
    rgb=np.load(entry['image'],mmap_mode='r',allow_pickle=False)
    with np.load(entry['geometry'],allow_pickle=False) as data:
        truth=data['truth']
        geometry={key:data[key] for key in ('skeleton','endpoints','thin')}
        geometry['excluded_boundary_endpoints']=int(data['excluded_boundary_endpoints'])
    return rgb,truth,geometry


def topology_specs(path):
    gate=json.loads(Path(path).read_text())
    if gate['status']!='frozen' or {m['arm'] for m in gate['models']}!={'baseline','cldice'}:
        raise ValueError('Both topology arms must be frozen before common evaluation')
    module=Path(gate['inference_module']); selection=Path(gate['frozen_selection'])
    config_path=selection.parent/'configuration.json'
    if core.digest(module)!=gate['inference_module_sha256'] or core.digest(selection)!=gate['frozen_selection_sha256'] or core.digest(config_path)!=gate['configuration_sha256']:
        raise ValueError('Frozen topology inference or configuration changed')
    config=json.loads(config_path.read_text())
    for name in ('infer_topology.py','topology_core.py'):
        if core.digest(module.parent/name)!=config['source_sha256'][name]:
            raise ValueError('Frozen topology numerical source changed')
    specs={}
    for model in gate['models']:
        if core.digest(model['checkpoint'])!=model['checkpoint_sha256']:
            raise ValueError('Frozen topology checkpoint changed')
        specs['unet_'+model['arm']]=dict(path=model['checkpoint'],sha256=model['checkpoint_sha256'],
            threshold=float(model['threshold']),backend='topology',inference_module=str(module),
            inference_module_sha256=gate['inference_module_sha256'],imgsz=640,
            kind='dense original-SAM whole-plant U-Net; frozen probability threshold; no expert leaf truth')
    return specs


def validate_selected_models(current,selected):
    if set(current)!=set(selected):
        raise ValueError('Compared model set changed after validation selection')
    for name,spec in current.items():
        old=selected[name]
        if any(spec.get(key,default)!=old.get(key,default) for key,default in
               [('sha256',None),('imgsz',640),('backend','yolo'),('threshold',None),('inference_module_sha256',None)]):
            raise ValueError('A model or inference operator changed after validation selection')


def summarize(records):
    groups=defaultdict(list)
    for r in records:
        for group in ('overall','camera','condition'):
            key=(r['model'],r['split'],r['confidence'],r['teacher_accepted'],group,
                 r['camera'] if group!='overall' else 'all',
                 r['treatment'] if group=='condition' else 'all',
                 r['measurement'] if group=='condition' else 'all')
            groups[key].append(r)
    result=[]
    for key,rows in groups.items():
        model,split,conf,accepted,group,camera,treatment,visit=key
        item=dict(model=model,split=split,confidence=conf,teacher_accepted=accepted,
            group=group,camera=camera,treatment=treatment,measurement=visit,n=len(rows),
            nonempty=sum(r['predicted_pixels']>0 for r in rows),
            nonempty_fraction=float(np.mean([r['predicted_pixels']>0 for r in rows])),
            median_iou=float(np.median([r['iou'] for r in rows])),mean_iou=float(np.mean([r['iou'] for r in rows])),
            median_dice=float(np.median([r['dice'] for r in rows])),
            mean_precision=float(np.mean([r['precision'] for r in rows])),mean_recall=float(np.mean([r['recall'] for r in rows])),
            iou_below_05=sum(r['iou']<.5 for r in rows),
            median_predict_ms=float(np.median([r['predict_ms'] for r in rows])),
            median_cached_decode_ms=float(np.median([r['decode_ms'] for r in rows])))
        for metric in ('endpoint_recall_px0','endpoint_recall_px3','endpoint_recall_px5','thin_skeleton_recall_px0','thin_skeleton_recall_px3','skeleton_recall_px0'):
            values=[r[metric] for r in rows if r[metric] is not None]
            item['mean_'+metric]=float(np.mean(values)) if values else None
            item[metric+'_n']=len(values)
        result.append(item)
    return result


def models_for(args):
    runs={}
    modes=('joint',) if args.phase.startswith('all40') else ('rgb','joint')
    for mode in modes:
        phase='full' if args.phase=='comparison' else args.phase
        folder=args.output/'training'/f'{mode}_{phase}_e{args.epochs}'
        manifest=json.loads((folder/'run_manifest.json').read_text())
        if core.digest(manifest['best_checkpoint'])!=manifest['best_sha256']:
            raise ValueError('Trained checkpoint changed')
        name='joint_all40' if args.phase.startswith('all40') else mode
        runs[name]=dict(path=manifest['best_checkpoint'],sha256=manifest['best_sha256'],kind='SAM teacher distilled whole plant')
    if args.phase.startswith('all40'):
        manifest=json.loads((args.output/'training/joint_full_e25/run_manifest.json').read_text())
        runs['joint_quarter']=dict(path=manifest['best_checkpoint'],sha256=manifest['best_sha256'],kind='fixed-quarter25epoch joint control')
    if args.phase=='comparison':
        choice=json.loads((args.output/'all40_selection.json').read_text())
        runs['joint_all40']=dict(path=choice['checkpoint'],sha256=choice['sha256'],kind='all40joint selected using validation only')
        runs['joint_all40_960']=dict(path=choice['checkpoint'],sha256=choice['sha256'],imgsz=960,
            kind='same selected all40checkpoint; declared960inference resolution sensitivity')
        gate=args.output/'additional_model_test_gate.json'
        if gate.exists():
            runs.update(topology_specs(gate))
    if args.phase in ('full','comparison'):
        runs['generic_pretrained']=dict(path=str(args.output/'yolo26n-seg.pt'),
            sha256=core.digest(args.output/'yolo26n-seg.pt'),kind='generic COCO potted-plant class58 mask union; out-of-domain baseline')
        previous=args.repo/'prototype/research_v2/outputs/multimodal_leaf/2026-10-24/training'
        for camera in ('rgb','noir'):
            path=previous/f'{camera}_n_weak_e30/weights/best.pt'
            if path.exists():
                target=args.output/'baselines'/f'legacy_weak_{camera}.pt'
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    shutil.copyfile(path,target)
                if core.digest(target)!=core.digest(path):
                    raise ValueError('Legacy baseline copy hash mismatch')
                runs['legacy_weak_'+camera]=dict(path=str(target),sha256=core.digest(path),
                    kind='prior weak instance model union; training overlap assessed separately; no true leaf accuracy')
    return runs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','intake','teacher-run','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--phase',choices=['pilot','full','all40','all40_extension','comparison'],required=True)
    p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--split',choices=['val','test'],required=True)
    p.add_argument('--all-angles',action='store_true')
    args=p.parse_args()
    if args.phase=='pilot' and args.split!='val':
        raise ValueError('Pilot test evaluation is forbidden')
    if args.split=='test' and not (args.output/'evaluation/final_protocol_ready.json').exists():
        raise ValueError('Test is locked until all-angle training and validation-only choices are frozen')
    external_gate=args.output/'additional_model_test_gate.json'
    if args.split=='test' and external_gate.exists() and json.loads(external_gate.read_text()).get('status')!='frozen':
        raise ValueError('Test is locked until the parallel model competitor is frozen')
    if args.phase=='comparison' and args.split=='val' and external_gate.exists():
        while json.loads(external_gate.read_text()).get('status')!='frozen':
            print('waiting for both topology competitors to freeze before common validation',flush=True)
            time.sleep(10)
    destination=args.output/'evaluation'/f'{args.phase}_{"validation" if args.split=="val" else "test"}'
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'execution_evaluate_foreground.py').write_bytes(EXECUTION_SOURCE)
    shutil.copyfile(Path(core.__file__),destination/'execution_foreground_core.py')
    core.configure_offline(args.repo,args.output)
    from ultralytics import YOLO
    import torch
    input_table=args.output/'all_views/frame_inventory.csv' if args.all_angles else args.output/'quarter_frames.csv'
    preparation_path=args.output/'all_views/preparation.json' if args.all_angles else args.output/'preparation.json'
    preparation=json.loads(preparation_path.read_text())
    if preparation['status']!='completed' or core.digest(input_table)!=preparation['frame_table_sha256']:
        raise ValueError('Evaluation frame inventory differs from completed preparation')
    frames=core.read_csv(input_table)
    frames=[r for r in frames if r['split']==args.split]
    if not frames or any(r['genotype']!=('Os6251/15' if args.split=='val' else 'Os7798/2') for r in frames):
        raise ValueError('Evaluation violates the frozen genotype partition')
    verify_frame_inputs(frames,args.intake,args.teacher_run)
    processing_cache=args.output/'evaluation/processing_cache'/core.digest(Path(core.__file__))[:12]
    def prepare(row):
        return prepare_cached_frame(row,args.intake,args.teacher_run,processing_cache)
    with ThreadPoolExecutor(max_workers=4) as pool:
        prepared={entry['id']:entry for entry in pool.map(prepare,frames)}
    print('verified and cached evaluation inputs',len(prepared),flush=True)
    runs=models_for(args)
    selected={}
    if args.split=='test':
        validation=json.loads((args.output/'evaluation/comparison_validation/selection.json').read_text())
        selected=validation['thresholds']
        validate_selected_models(runs,validation['models'])
        val_manifest=json.loads((args.output/'evaluation/comparison_validation/manifest.json').read_text())
        protocol=json.loads((args.output/'evaluation/final_protocol_ready.json').read_text())
        if val_manifest['evaluation_source_sha256']!=EXECUTION_SHA256 or core.digest(args.output/'evaluation/comparison_validation/selection.json')!=protocol['confidence_selection_sha256'] or core.digest(args.output/'evaluation_protocol.json')!=protocol['protocol_sha256'] or core.digest(args.output/'all40_selection.json')!=protocol['all40_selection_sha256']:
            raise ValueError('The comparison protocol or evaluator changed after validation freeze')
    records=[]
    for name,spec in runs.items():
        checkpoint=Path(spec['path'])
        if core.digest(checkpoint)!=spec['sha256']:
            raise ValueError('Baseline or learned checkpoint hash changed')
        topology=spec.get('backend')=='topology'
        if topology:
            module_path=Path(spec['inference_module'])
            sys.path.insert(0,str(module_path.parent))
            module_spec=importlib.util.spec_from_file_location('foreground_topology_inference',module_path)
            module=importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name]=module; module_spec.loader.exec_module(module)
            model=None
        else:
            model=YOLO(str(checkpoint))
        inference_size=spec.get('imgsz',640)
        spec['imgsz']=inference_size
        predict_options={'classes':[58]} if name=='generic_pretrained' else {}
        # Warm-up is excluded from reported per-frame times and explicitly recorded.
        warm=np.zeros((769,1024,3),np.uint8)
        warm_start=time.perf_counter()
        if topology:
            module.predict_array(warm,str(checkpoint),spec['threshold'],target_shape=warm.shape[:2],device='cuda:1')
        else:
            model.predict(warm,device=1,imgsz=inference_size,conf=.01,retina_masks=True,max_det=100,verbose=False,**predict_options)
        torch.cuda.synchronize(1)
        spec['warmup_seconds']=time.perf_counter()-warm_start
        masks_dir=destination/'masks'/name
        masks_dir.mkdir(parents=True,exist_ok=True)
        for i,r in enumerate(frames):
            if name.startswith('legacy_weak_') and name.rsplit('_',1)[1]!=r['camera']:
                continue
            decode_start=time.perf_counter()
            rgb,truth,geometry=load_cached_frame(prepared[r['id']])
            bgr=np.ascontiguousarray(rgb[:,:,::-1])
            decode_ms=(time.perf_counter()-decode_start)*1000
            torch.cuda.synchronize(1); start=time.perf_counter()
            if topology:
                dense_prediction=module.predict_array(rgb,str(checkpoint),spec['threshold'],target_shape=truth.shape,device='cuda:1')
                result=None
            else:
                forward_conf=.01 if args.split=='val' else float(selected[name])
                result=model.predict(bgr,device=1,imgsz=inference_size,conf=forward_conf,retina_masks=True,max_det=100,verbose=False,**predict_options)[0]
            torch.cuda.synchronize(1); predict_ms=(time.perf_counter()-start)*1000
            choices=(spec['threshold'],) if topology else (CONFIDENCES if args.split=='val' else (float(selected[name]),))
            previous=None; terminal=None
            for confidence in choices:
                materialize_start=time.perf_counter()
                prediction=np.asarray(dense_prediction,dtype=bool) if topology else core.prediction_union(result,truth.shape,confidence)
                materialize_ms=(time.perf_counter()-materialize_start)*1000
                if previous is None or not np.array_equal(previous,prediction):
                    terminal=core.terminal_metrics(truth,prediction,geometry)
                    previous=prediction
                row=dict(model=name,split=args.split,confidence=confidence,imgsz=inference_size,
                    threshold_kind='foreground_probability' if topology else 'detector_confidence',
                    **{k:r[k] for k in ('id','treatment','measurement','genotype','replicate','camera','view','source','teacher_accepted')},
                    **core.mask_metrics(truth,prediction),**terminal,decode_ms=decode_ms,
                    initial_raw_decode_ms=prepared[r['id']]['raw_decode_ms'],predict_ms=predict_ms,
                    mask_materialize_ms=materialize_ms,foreground_ms=predict_ms+materialize_ms)
                records.append(row)
                if args.split=='test':
                    Image.fromarray(prediction.astype(np.uint8)*255).save(masks_dir/(r['id']+'.png'))
            if (i+1)%50==0:
                print(name,args.split,i+1,'of',len(frames),flush=True)
                core.write_json(destination/'progress.json',dict(model=name,frames=i+1,total=len(frames)))
        if model is not None:
            del model
        if topology:
            module.load_model.cache_clear()
        torch.cuda.empty_cache()
    if args.split=='val':
        thresholds={}
        for name in runs:
            candidates=[r for r in records if r['model']==name and r['teacher_accepted'].lower()=='true'
                        and (name!='rgb' or r['camera']=='rgb')]
            thresholds[name]=core.select_threshold(candidates)
        selection=dict(thresholds=thresholds,policy='maximize mean accepted-teacher validation IoU; RGB-only uses RGB validation, joint/generic use both cameras; ties choose higher confidence; no test rows',
            validation_genotypes=sorted({r['genotype'] for r in frames}),models=runs)
        core.write_json(destination/'selection.json',selection)
        if args.phase=='pilot':
            evidence={}
            for name in ('rgb','joint'):
                rows=[r for r in records if r['model']==name and r['confidence']==thresholds[name]
                      and r['teacher_accepted'].lower()=='true' and (name=='joint' or r['camera']=='rgb')]
                evidence[name]=dict(median_iou=float(np.median([r['iou'] for r in rows])),
                    nonempty_fraction=float(np.mean([r['predicted_pixels']>0 for r in rows])),n=len(rows))
            viable=all(e['median_iou']>=.45 and e['nonempty_fraction']>=.80 for e in evidence.values())
            core.write_json(destination/'viability.json',dict(viable=viable,evidence=evidence,
                rule='each pilot median teacher IoU>=0.45 and nonempty fraction>=0.80; RGB model on accepted RGB validation and joint model on both accepted cameras'))
    core.write_csv(destination/'frame_metrics.csv',records)
    summaries=summarize(records)
    core.write_csv(destination/'group_metrics.csv',summaries)
    core.write_json(destination/'manifest.json',dict(status='completed',phase=args.phase,split=args.split,
        evaluated_frames=len(frames),evaluated_genotypes=sorted({r['genotype'] for r in frames}),
        all_angles=args.all_angles,
        frame_inventory_sha256=core.digest(input_table),preparation_sha256=core.digest(preparation_path),
        source_and_teacher_masks_verified=len(frames),
        evaluation_processing_cache='lossless original processing1024 RGB and unchanged teacher geometry; model-independent',
        additional_model_test_gate_sha256=core.digest(external_gate) if external_gate.exists() else None,
        models=runs,records=len(records),frame_metrics_sha256=core.digest(destination/'frame_metrics.csv'),
        group_metrics_sha256=core.digest(destination/'group_metrics.csv'),
        evaluation_source_sha256=EXECUTION_SHA256,core_source_sha256=core.digest(Path(core.__file__)),
        limits='Agreement to automatic teacher masks, not expert truth; QC-rejected teacher masks are diagnostic only; generic baseline is out of domain'))
    print('completed evaluation',destination,flush=True)


if __name__=='__main__':
    main()
