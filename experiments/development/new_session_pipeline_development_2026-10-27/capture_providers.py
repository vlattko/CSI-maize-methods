"""Explicit foreground providers; confidence is never reused as a SAM score."""
import importlib,json,shutil,sys
from pathlib import Path
import cv2
import numpy as np
from capture_core import digest,provider_parameters


class UNetProvider:
    def __init__(self,card_path,device='cpu',pixel_threshold=None,module_path=None):
        card_path=Path(card_path).resolve();card=json.loads(card_path.read_text(encoding='utf-8'));p=provider_parameters('unet',card,pixel_threshold=pixel_threshold)
        checkpoint=Path(p['checkpoint']);checkpoint=checkpoint if checkpoint.is_absolute() else card_path.parent/checkpoint
        if digest(checkpoint)!=p['checkpoint_sha256']:raise ValueError('U-Net checkpoint hash mismatch')
        module_path=Path(module_path) if module_path else Path(__file__).parent.parent/'topology_foreground_development_2026-10-27/infer_topology.py'
        sys.path.insert(0,str(module_path.resolve().parent));module=importlib.import_module(module_path.stem)
        if Path(module.__file__).resolve()!=module_path.resolve():raise ValueError('Unexpected inference module import')
        self.predict_array=module.predict_array;self.checkpoint=str(checkpoint);self.threshold=p['threshold'];self.device=device
        self.provenance=dict(p,checkpoint=str(checkpoint.resolve()),model_card=str(card_path),model_card_sha256=digest(card_path),
            inference_module=str(module_path.resolve()),inference_module_sha256=digest(module_path),
            inference_core_sha256=digest(module_path.with_name('topology_core.py')),threshold_overridden=pixel_threshold is not None,
            device=device,scope='Whole-plant foreground prediction; no individual-leaf or human accuracy claim')
    def predict(self,image,frame):return self.predict_array(image,self.checkpoint,self.threshold,target_shape=image.shape[:2],device=self.device)


class YOLOProvider:
    def __init__(self,card_path,output,device='cpu',confidence=None,repo=None):
        card_path=Path(card_path).resolve();card=json.loads(card_path.read_text(encoding='utf-8'));p=provider_parameters('yolo',card,detection_confidence=confidence)
        checkpoint=Path(p['checkpoint']);checkpoint=checkpoint if checkpoint.is_absolute() else card_path.parent/checkpoint
        if digest(checkpoint)!=p['checkpoint_sha256']:raise ValueError('YOLO checkpoint hash mismatch')
        self.output=Path(output);self.checkpoint=checkpoint;self.expected=p['checkpoint_sha256'];self.model=None
        self.device=device.replace('cuda:','');self.confidence=p['threshold'];self.imgsz=card.get('imgsz',640)
        classes=card.get('classes',{})
        self.class_ids=card.get('class_ids') or [int(k) for k,v in classes.items() if 'plant' in str(v).lower() or 'foreground' in str(v).lower()]
        if not self.class_ids:raise ValueError('YOLO card must explicitly identify plant/foreground class IDs')
        from csi_maize import ultralytics_offline as adapter
        adapter_file=Path(adapter.__file__).resolve()
        if repo and not adapter_file.is_relative_to((Path(repo)/'src').resolve()):raise ValueError('YOLO adapter does not come from the requested repository')
        self.provenance=dict(p,checkpoint=str(checkpoint.resolve()),model_card=str(card_path),model_card_sha256=digest(card_path),device=device,
            adapter_file=str(adapter_file),adapter_sha256=digest(adapter_file),class_ids=self.class_ids,imgsz=self.imgsz,
            offline_policy='Deferred offline initialization at first inference',threshold_overridden=confidence is not None)
    def predict(self,image,frame):
        if self.model is None:
            runtime=self.output/'provider_runtime';runtime.mkdir(parents=True,exist_ok=True);bound=runtime/'bound_foreground.pt';shutil.copyfile(self.checkpoint,bound)
            from csi_maize.ultralytics_offline import configure_ultralytics,require_checkpoint
            require_checkpoint(bound,base=runtime,expected_sha256=self.expected);self.provenance['offline_policy']=configure_ultralytics(runtime,require_amp=False)
            from ultralytics import YOLO
            self.model=YOLO(str(bound))
        result=self.model.predict(np.ascontiguousarray(image[:,:,::-1]),device=self.device,imgsz=self.imgsz,conf=self.confidence,
            classes=self.class_ids,retina_masks=True,max_det=100,half=False,verbose=False)[0]
        mask=np.zeros(image.shape[:2],bool)
        if result.masks is not None:
            for data,score in zip(result.masks.data.detach().cpu().numpy(),result.boxes.conf.detach().cpu().numpy()):
                if score>=self.confidence:
                    if data.shape!=mask.shape:data=cv2.resize(data,(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_NEAREST)
                    mask|=data>.5
        return mask
