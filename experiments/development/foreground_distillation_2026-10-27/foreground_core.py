"""Trust boundaries and numerical operators for whole-plant teacher distillation."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image

SEED = 20261027
QUARTERS = (0, 10, 20, 30)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf8')
    for attempt in range(40):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def genotype_splits(rows):
    genotypes = sorted({r['genotype'] for r in rows})
    if len(genotypes) != 6:
        raise ValueError('Exactly six full-cohort genotypes are required')
    return {g: ('train' if i < 4 else 'val' if i == 4 else 'test') for i,g in enumerate(genotypes)}


def processing_image(path):
    with Image.open(path) as source:
        source.draft('RGB', (1024, 1024)); source.load()
        result = source.convert('RGB'); result.thumbnail((1024, 1024))
    return np.array(result)


def mask_metrics(truth, prediction):
    if truth.shape != prediction.shape:
        raise ValueError('Teacher and prediction processing geometry differ')
    truth = truth.astype(bool); prediction = prediction.astype(bool)
    tp = int(np.count_nonzero(truth & prediction))
    nt = int(np.count_nonzero(truth)); npred = int(np.count_nonzero(prediction))
    union = nt + npred - tp
    return dict(iou=tp/union if union else 1., dice=2*tp/(nt+npred) if nt+npred else 1.,
                precision=tp/npred if npred else 0., recall=tp/nt if nt else 0.,
                teacher_pixels=nt, predicted_pixels=npred, area_ratio=npred/nt if nt else None)


def terminal_geometry(truth):
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    skeleton=skeletonize(truth.astype(bool))
    degree=ndi.convolve(skeleton.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-skeleton
    endpoints=skeleton & (degree==1)
    interior=endpoints.copy()
    interior[:2,:]=False; interior[-2:,:]=False; interior[:,:2]=False; interior[:,-2:]=False
    thickness=ndi.distance_transform_edt(truth)
    return dict(skeleton=skeleton,endpoints=interior,thin=skeleton & (thickness<=2.5),
                excluded_boundary_endpoints=int(endpoints.sum()-interior.sum()))


def terminal_metrics(truth,prediction,geometry=None):
    geometry=terminal_geometry(truth) if geometry is None else geometry
    if prediction.shape!=truth.shape:
        raise ValueError('Teacher and prediction processing geometry differ')
    endpoints=geometry['endpoints']; thin=geometry['thin']; skeleton=geometry['skeleton']
    values=dict(teacher_interior_endpoints=int(endpoints.sum()),
                excluded_boundary_endpoints=geometry['excluded_boundary_endpoints'],
                teacher_thin_skeleton_pixels=int(thin.sum()))
    supported={0:prediction.astype(bool)}
    for tolerance in (3,5):
        yy,xx=np.mgrid[-tolerance:tolerance+1,-tolerance:tolerance+1]
        disk=(xx*xx+yy*yy<=tolerance*tolerance).astype(np.uint8)
        supported[tolerance]=cv2.dilate(prediction.astype(np.uint8),disk)>0
    for tolerance in (0,3,5):
        values[f'endpoint_recall_px{tolerance}']=float(np.mean(supported[tolerance][endpoints])) if endpoints.any() else None
    for tolerance in (0,3):
        values[f'thin_skeleton_recall_px{tolerance}']=float(np.mean(supported[tolerance][thin])) if thin.any() else None
    values['skeleton_recall_px0']=float(np.mean(supported[0][skeleton])) if skeleton.any() else None
    return values


def mask_polygon(mask):
    binary = (mask > 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    if hierarchy is None:
        raise ValueError('Teacher mask is empty')
    outer = [c[:,0,:].astype(np.float64) for i,c in enumerate(contours)
             if hierarchy[0,i,3] < 0 and len(c) >= 3]
    if not outer:
        raise ValueError('Teacher mask has no polygon-representable component')
    outer.sort(key=lambda x: abs(cv2.contourArea(x.astype(np.float32))), reverse=True)
    path = outer[0]
    for part in outer[1:]:
        best = (float('inf'),0,0)
        for start in range(0,len(path),256):
            distances = ((path[start:start+256,None,:]-part[None,:,:])**2).sum(axis=2)
            i,j = np.unravel_index(np.argmin(distances),distances.shape)
            value = float(distances[i,j])
            if value < best[0]:
                best = (value,start+int(i),int(j))
        _,i,j = best
        loop = np.concatenate([part[j:],part[:j+1]])
        path = np.concatenate([path[:i+1],loop,path[i:i+1],path[i+1:]])
    h,w = mask.shape
    normalized = path / np.array([w,h])
    info = dict(external_components=len(outer), holes=int(np.count_nonzero(hierarchy[0,:,3]>=0)),
                polygon_vertices=len(path), omitted_degenerate_contours=sum(len(c)<3 for c in contours))
    return normalized,info


def raster_polygon(polygon, shape):
    h,w = shape
    points = np.rint(np.asarray(polygon)*[w,h]).astype(np.int32)
    result = np.zeros(shape,np.uint8)
    cv2.fillPoly(result,[points],1)
    return result.astype(bool)


def select_threshold(records):
    if not records or any(r['split'] != 'val' for r in records):
        raise ValueError('Confidence selection is restricted to nonempty validation records')
    choices = sorted({float(r['confidence']) for r in records})
    means = {c:np.mean([float(r['iou']) for r in records if float(r['confidence'])==c]) for c in choices}
    return max(choices,key=lambda c:(means[c],c))


def configure_offline(repo, output):
    """Reuse checked adapter with a disposable output-owned numerical workspace."""
    repo = Path(repo); output = Path(output)
    sys.path.insert(0,str(repo/'CSI-maize-methods/src'))
    sys.path.append(str(repo/'prototype/research_v2/yolo_deps'))
    base = output
    base.mkdir(parents=True,exist_ok=True)
    os.environ.update(YOLO_CONFIG_DIR=str(base/'outputs/multimodal_leaf/2026-10-24/runtime_config'),
                      MPLCONFIGDIR=str(base/'outputs/multimodal_leaf/2026-10-24/matplotlib_config'),
                      YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false',
                      OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    from csi_maize.ultralytics_offline import configure_ultralytics, require_checkpoint
    sources = {'yolo26n.pt':repo/'prototype/weights/yolo26n.pt',
               'yolo26n-seg.pt':repo/'prototype/research_v2/yolo26n-seg.pt'}
    for name,source in sources.items():
        target = base/name
        if not target.exists():
            shutil.copyfile(source,target)
        require_checkpoint(target,base=base,model_id='yolo26n-amp-check' if name=='yolo26n.pt' else 'yolo26n-seg')
    policy = configure_ultralytics(base,require_amp=True)
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(SEED)
    return base,policy


def prediction_union(result, shape, confidence):
    output = np.zeros(shape,bool)
    if result.masks is None or result.boxes is None:
        return output
    masks = result.masks.data.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    for mask,score in zip(masks,scores):
        if score < confidence:
            continue
        if mask.shape != shape:
            mask = cv2.resize(mask,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
        output |= mask > .5
    return output
