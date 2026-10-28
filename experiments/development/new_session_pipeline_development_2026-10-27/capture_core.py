"""Capture identities, safe inventory and model-independent structural checks."""
import hashlib,json,math,re
from pathlib import Path,PureWindowsPath
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def relative_path(value):
    p=Path(str(value))
    if p.is_absolute() or PureWindowsPath(str(value)).drive or '..' in p.parts:raise ValueError('Capture paths must stay relative to the input folder')
    return p


def validate_session(raw):
    s=dict(raw);sample=s.get('sample_id')
    if not isinstance(sample,str) or not sample.strip():raise ValueError('An explicit nonempty sample_id is required')
    angles=s.get('angles_degrees');steps=s.get('steps',len(angles) if angles is not None else None)
    if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=360:raise ValueError('Declare steps1..360 or an explicit angle list')
    if angles is None:angles=[360*i/steps for i in range(steps)];angle_source='nominal_uniform_index'
    else:
        if len(angles)!=steps or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in angles):raise ValueError('One finite angle is required per fixed slot')
        angles=[float(v)%360 for v in angles];angle_source='user_supplied_unverified'
        if len(set(round(v,9) for v in angles))!=steps:raise ValueError('Duplicate rotation angles')
    cameras=s.get('cameras',{'rgb':'rgb','noir':'noir'});normalized={}
    for camera in ['rgb','noir']:
        cfg=cameras.get(camera,camera);cfg={'directory':cfg} if isinstance(cfg,str) else dict(cfg)
        directory=relative_path(cfg.get('directory',camera));pattern=cfg.get('index_regex',r'(?P<index>\d+)$');compiled=re.compile(pattern)
        if 'index' not in compiled.groupindex:raise ValueError('index_regex requires a named index group')
        normalized[camera]=dict(directory=directory.as_posix(),index_regex=pattern)
    a=Path(normalized['rgb']['directory']);b=Path(normalized['noir']['directory'])
    if a==b or a.is_relative_to(b) or b.is_relative_to(a):raise ValueError('RGB and NoIR camera directories must be separate and nonoverlapping')
    uid=s.get('plant_uid')
    if uid is not None and (not isinstance(uid,str) or not uid.strip()):raise ValueError('plant_uid must be explicitly provided as a nonempty string or null')
    s.update(sample_id=sample,steps=steps,angles_degrees=angles,angle_source=angle_source,cameras=normalized,plant_uid=uid,
        physical_identity_status='user_supplied_not_independently_verified' if uid else 'not_provided')
    return s


def safe_output(raw,output):
    raw=Path(raw).resolve();output=Path(output).resolve()
    if output.is_relative_to(raw) or raw.is_relative_to(output):raise ValueError('Output must be separate from raw capture paths')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise ValueError('Use a new or empty output directory')
    return output


def inventory(root,session):
    root=Path(root).resolve();rows=[]
    a=(root/session['cameras']['rgb']['directory']).resolve();b=(root/session['cameras']['noir']['directory']).resolve()
    if a==b or a.is_relative_to(b) or b.is_relative_to(a):raise ValueError('Resolved RGB/NoIR camera directories overlap or alias the same location')
    for camera,cfg in session['cameras'].items():
        folder=(root/relative_path(cfg['directory'])).resolve()
        if not folder.is_relative_to(root):raise ValueError('Camera directory escapes capture root')
        found={};pattern=re.compile(cfg['index_regex'])
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix.lower() not in ['.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff']:continue
                if not path.resolve().is_relative_to(root):raise ValueError('Source file escapes capture root')
                match=pattern.search(path.stem)
                if not match:raise ValueError('Image filename has no declared frame index: '+path.name)
                index=int(match.group('index'))
                if not 0<=index<session['steps']:raise ValueError('Frame index outside declared slots: '+path.name)
                if index in found:raise ValueError('Duplicate camera/index: '+camera+'/'+str(index))
                found[index]=path
        for view in range(session['steps']):
            path=found.get(view);rows.append(dict(sample_id=session['sample_id'],camera=camera,view=view,angle_degrees=session['angles_degrees'][view],
                present=path is not None,source=path.relative_to(root).as_posix() if path else None,_path=str(path) if path else None))
    return rows


def provider_parameters(provider,card,pixel_threshold=None,detection_confidence=None):
    if provider=='unet':
        if detection_confidence is not None:raise ValueError('U-Net uses a foreground probability threshold, not detection confidence')
        threshold=card.get('threshold') if pixel_threshold is None else pixel_threshold;kind='foreground_probability'
        checkpoint=card.get('checkpoint');expected=card.get('checkpoint_sha256')
    elif provider=='yolo':
        if pixel_threshold is not None:raise ValueError('YOLO model-card confidence must not be used as a pixel threshold')
        threshold=card.get('confidence') if detection_confidence is None else detection_confidence;kind='detection_confidence'
        checkpoint=card.get('weights');expected=card.get('weights_sha256')
    else:raise ValueError('Unsupported inference provider')
    if isinstance(threshold,bool) or threshold is None or not 0<=float(threshold)<=1:raise ValueError('A declared threshold in[0,1] is required')
    return dict(provider=provider,threshold_kind=kind,threshold=float(threshold),checkpoint=checkpoint,checkpoint_sha256=expected,
        mask_decoder_threshold=.5 if provider=='yolo' else None,quality_score_interpretation='Provider threshold is not a SAM score or structural-usability estimate')


def structural_usability(mask):
    mask=np.asarray(mask)
    if mask.ndim!=2 or mask.dtype!=bool:raise ValueError('Provider must return a Boolean2D mask')
    area=int(mask.sum());fraction=float(mask.mean());issues=[]
    if area<10:issues.append('fewer_than10_foreground_pixels')
    if fraction>=.9:issues.append('foreground_covers_at_least90percent')
    if area and np.ptp(np.where(mask)[0])+1<3:issues.append('foreground_height_below3pixels')
    return dict(usable=not issues,issues=issues,foreground_pixels=area,area_fraction=fraction,
        interpretation='Engineering structural criteria; no human segmentation or anatomical validation')


def tracking_support(session):
    if session['steps']!=40:return dict(supported=False,reason='Current cyclic tracker requires40slots; traits/report retain the declared rotation unchanged')
    a=np.asarray(session['angles_degrees']);relative=(a-a[0])%360;expected=np.arange(40)*9
    if not np.allclose(relative,expected,atol=1e-6,rtol=0):return dict(supported=False,reason='Current tracker assumes uniform40-index rotation; nonuniform supplied angles are preserved without coercion')
    return dict(supported=True,reason='Conditional uniform40-index rotational model',angle_origin_degrees=float(a[0]),
        coordinate_scope='Training-image-height nominal orthographic gauge, without metric scale or anatomical organ IDs')
