"""Offline whole-plant foreground inference; saves a mask and black-background image."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import foreground_core as core


def inference_size(card,requested):
    value=requested if requested is not None else card.get('imgsz',640)
    if value not in (640,960,1280):
        raise ValueError('This operator supports explicit image sizes640,960,1280')
    return int(value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--model-card',type=Path)
    p.add_argument('--sha256')
    p.add_argument('--confidence',type=float)
    p.add_argument('--source',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='1')
    p.add_argument('--imgsz',type=int,choices=[640,960,1280])
    args=p.parse_args()
    if args.model_card:
        card=json.loads(args.model_card.read_text())
        expected=card['weights_sha256']
        confidence=card['confidence'] if args.confidence is None else args.confidence
    else:
        card={}; expected=args.sha256; confidence=args.confidence
    imgsz=inference_size(card,args.imgsz)
    if not expected or confidence is None or not 0<=confidence<=1:
        raise ValueError('Use a hash-bound model card or supply --sha256 and --confidence')
    if core.digest(args.weights)!=expected:
        raise ValueError('Model hash differs from declared checkpoint')
    args.output.mkdir(parents=True,exist_ok=True)
    core.configure_offline(args.repo,args.output)
    bound=args.output/'bound_foreground_model.pt'
    if bound.exists() and core.digest(bound)!=expected:
        raise ValueError('Output directory already binds a different model')
    if not bound.exists():
        shutil.copyfile(args.weights,bound)
    from ultralytics import YOLO
    import torch
    start=time.perf_counter(); model=YOLO(str(bound)); load_seconds=time.perf_counter()-start
    model.predict(np.zeros((769,1024,3),np.uint8),device=args.device,imgsz=imgsz,
                  conf=confidence,retina_masks=True,max_det=100,verbose=False)
    torch.set_num_threads(2)  # the upstream CPU device selector resets its default thread count
    rows=[]
    for source in args.source:
        frame_start=time.perf_counter()
        source_hash=core.digest(source)
        identifier=hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]
        decode_start=time.perf_counter(); rgb=core.processing_image(source)
        decode_ms=(time.perf_counter()-decode_start)*1000
        bgr=np.ascontiguousarray(rgb[:,:,::-1])
        if args.device!='cpu':
            torch.cuda.synchronize(int(args.device))
        predict_start=time.perf_counter()
        result=model.predict(bgr,device=args.device,imgsz=imgsz,conf=confidence,
                             retina_masks=True,max_det=100,verbose=False)[0]
        if args.device!='cpu':
            torch.cuda.synchronize(int(args.device))
        predict_ms=(time.perf_counter()-predict_start)*1000
        mask=core.prediction_union(result,rgb.shape[:2],confidence)
        foreground=rgb.copy(); foreground[~mask]=0
        mask_path=args.output/(identifier+'_mask.png')
        image_path=args.output/(identifier+'_foreground.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
        Image.fromarray(foreground).save(image_path,compress_level=1)
        rows.append(dict(source=str(source),source_sha256=source_hash,mask=str(mask_path),foreground=str(image_path),
            width=rgb.shape[1],height=rgb.shape[0],foreground_pixels=int(mask.sum()),nonempty=bool(mask.any()),
            decode_ms=decode_ms,predict_ms=predict_ms,
            end_to_end_ms=(time.perf_counter()-frame_start)*1000))
    core.write_json(args.output/'inference_manifest.json',dict(status='completed',weights_sha256=expected,
        confidence=confidence,device=args.device,imgsz=imgsz,max_decode_dimension=1024,
        mask_union='all predicted plant_foreground masks above confidence; no identity tracking',
        model_load_seconds=load_seconds,warmup_included_in_frame_times=False,rows=rows,
        torch_cpu_threads=torch.get_num_threads(),
        inference_source_sha256=core.digest(__file__),core_source_sha256=core.digest(Path(core.__file__)),
        limits='Whole-plant teacher-distilled foreground. No expert accuracy or individual-leaf tracking claim.'))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    main()
