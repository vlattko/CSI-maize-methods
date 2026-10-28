"""Cached FP32 foreground inference. No SAM mask or teacher information is input."""
import argparse
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from topology_core import TinyUNet,INPUT_SHAPE


@lru_cache(maxsize=4)
def load_model(checkpoint,device='cpu'):
    device='cuda:'+str(device) if str(device).isdigit() else str(device)
    data=torch.load(str(checkpoint),map_location=device,weights_only=True)
    model=TinyUNet(base=int(data['architecture']['base'])).to(device);model.load_state_dict(data['state_dict']);model.eval()
    return model,data,device


def predict_array(rgb_uint8,checkpoint,threshold,target_shape=None,device='cpu'):
    rgb=np.asarray(rgb_uint8)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Expected unmasked RGB uint8 image')
    model,metadata,device=load_model(str(Path(checkpoint).resolve()),str(device))
    h,w=metadata.get('input_shape',INPUT_SHAPE);resized=np.asarray(Image.fromarray(rgb).resize((w,h),Image.Resampling.BILINEAR)).copy()
    x=torch.from_numpy(resized.transpose(2,0,1)).unsqueeze(0).to(device,dtype=torch.float32)/255
    shape=tuple(target_shape) if target_shape is not None else rgb.shape[:2]
    with torch.inference_mode():
        p=model(x).sigmoid();p=F.interpolate(p,size=shape,mode='bilinear',align_corners=False)
    threshold=metadata.get('chosen_threshold',.5) if threshold is None else float(threshold)
    return p[0,0].cpu().numpy()>=threshold


def predict_mask(image_path,checkpoint,threshold,target_shape=None,device='cpu'):
    with Image.open(image_path) as im:rgb=np.asarray(im.convert('RGB')).copy()
    return predict_array(rgb,checkpoint,threshold,target_shape,device)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--threshold',type=float);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=predict_mask(a.image,a.checkpoint,a.threshold,device=a.device);a.output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray((result*255).astype('uint8')).save(a.output)
