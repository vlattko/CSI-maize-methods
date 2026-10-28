"""MIT-licensed dense foreground model and original soft-clDice implementation.

The morphological loss follows Shit etal., CVPR2021, Algorithms1–2. No claim
of botanical topology or anatomical leaf segmentation follows from this loss.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='2'
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
cv2.setNumThreads(1);torch.set_num_threads(2)

PARTITION={'train':['B73','Mo17','Os024445','Os622448'],'val':['Os6251/15']}
THRESHOLDS=[.35,.5,.65]
INPUT_SHAPE=(481,640)


class Block(nn.Module):
    def __init__(self,cin,cout):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(4,cout),nn.ReLU(),nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(4,cout),nn.ReLU())
    def forward(self,x):return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self,base=8):
        super().__init__();self.base=base
        self.enc=nn.ModuleList([Block(3,base),Block(base,base*2),Block(base*2,base*4),Block(base*4,base*8)])
        self.bottom=Block(base*8,base*16)
        self.dec=nn.ModuleList([Block(base*24,base*8),Block(base*12,base*4),Block(base*6,base*2),Block(base*3,base)])
        self.head=nn.Conv2d(base,1,1)
    def forward(self,x):
        skips=[]
        for enc in self.enc: x=enc(x);skips.append(x);x=F.max_pool2d(x,2)
        x=self.bottom(x)
        for dec,skip in zip(self.dec,reversed(skips)):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False);x=dec(torch.cat([x,skip],dim=1))
        return self.head(x)


def soft_erode(x):return torch.minimum(-F.max_pool2d(-x,(3,1),stride=1,padding=(1,0)),-F.max_pool2d(-x,(1,3),stride=1,padding=(0,1)))
def soft_dilate(x):return F.max_pool2d(x,3,stride=1,padding=1)
def soft_skeleton(x,iterations=10):
    opened=soft_dilate(soft_erode(x));s=F.relu(x-opened)
    for _ in range(iterations):
        x=soft_erode(x);delta=F.relu(x-soft_dilate(soft_erode(x)));s=s+(1-s)*delta
    return s


def soft_cldice(probability,target,iterations=10,eps=1e-6):
    sp=soft_skeleton(probability,iterations);st=soft_skeleton(target,iterations);axes=(1,2,3)
    precision=((sp*target).sum(axes)+eps)/(sp.sum(axes)+eps)
    sensitivity=((st*probability).sum(axes)+eps)/(st.sum(axes)+eps)
    return (1-(2*precision*sensitivity+eps)/(precision+sensitivity+eps)).mean()


def segmentation_loss(logits,target,topology=False):
    p=logits.sigmoid();axes=(1,2,3);dice=1-((2*(p*target).sum(axes)+1e-6)/(p.sum(axes)+target.sum(axes)+1e-6)).mean()
    loss=.5*F.binary_cross_entropy_with_logits(logits,target)+.5*dice
    return loss+.25*soft_cldice(p,target,iterations=10) if topology else loss


def guard_record(row):
    split=row['split']
    if split not in PARTITION:raise ValueError('Only training and validation records may be opened')
    if row['genotype'] not in PARTITION[split]:raise ValueError('Frozen genotype partition violated')
    return str(row['teacher_accepted']).lower() in ['true','1']


def select_validation(records):
    if not records or any(r['split']!='val' for r in records):raise ValueError('Selection is validation-only')
    return max(records,key=lambda r:(r['score'],-r['epoch'],r['threshold']))


def teacher_geometry(mask):
    mask=mask.astype(bool);sk=skeletonize(mask);degree=ndi.convolve(sk.astype('uint8'),np.ones((3,3),np.uint8),mode='constant')-sk
    endpoints=sk&(degree==1);interior=endpoints.copy();interior[:2]=False;interior[-2:]=False;interior[:,:2]=False;interior[:,-2:]=False
    thickness=ndi.distance_transform_edt(mask);thin=sk&(thickness<=2.5)
    return dict(endpoint_coords=np.where(interior),thin_coords=np.where(thin),skeleton_coords=np.where(sk),
        teacher_pixels=int(mask.sum()),teacher_endpoints=int(interior.sum()),teacher_thin_pixels=int(thin.sum()),excluded_border_endpoints=int(endpoints.sum()-interior.sum()),
        teacher_components=int(cv2.connectedComponents(mask.astype('uint8'),connectivity=8)[0]-1))


def agreement_metrics(truth,prediction,geometry):
    truth=truth.astype(bool);prediction=prediction.astype(bool)
    if truth.shape!=prediction.shape:raise ValueError('Original processing grids must agree')
    tp=int((truth&prediction).sum());nt=int(truth.sum());npred=int(prediction.sum());union=nt+npred-tp
    out=dict(iou=tp/union if union else 1.,dice=2*tp/(nt+npred) if nt+npred else 1.,teacher_pixels=nt,predicted_pixels=npred,
        teacher_interior_endpoints=geometry['teacher_endpoints'],teacher_thin_skeleton_pixels=geometry['teacher_thin_pixels'],
        excluded_border_endpoints=geometry['excluded_border_endpoints'])
    for tol in [0,3,5]:
        if tol:
            yy,xx=np.mgrid[-tol:tol+1,-tol:tol+1];disk=(xx**2+yy**2<=tol**2).astype('uint8');support=cv2.dilate(prediction.astype('uint8'),disk)>0
        else:support=prediction
        for name,key in [('endpoint_recall','endpoint_coords'),('thin_skeleton_recall','thin_coords'),('skeleton_recall','skeleton_coords')]:
            coords=geometry[key];out[f'{name}_px{tol}']=float(np.mean(support[coords])) if len(coords[0]) else None
    out['predicted_components']=int(cv2.connectedComponents(prediction.astype('uint8'),connectivity=8)[0]-1);out['teacher_components']=geometry['teacher_components']
    return out
