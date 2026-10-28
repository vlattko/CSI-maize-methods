"""One new capture → source-bound masks,39 traits, conditional tracks and report."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='2'
import argparse,csv,datetime,hashlib,importlib.util,io,json,math,shutil,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from capture_core import validate_session,inventory,safe_output,structural_usability,tracking_support,digest
from report_capture import make_report


def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.startswith('_')}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def table(path,rows):
    rows=[clean(r) for r in rows];fields=list(dict.fromkeys(k for r in rows for k in r));Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def public_traits(repo=None):
    if repo:folder=Path(repo).resolve()/'src/csi_maize/_methods'
    else:
        spec=importlib.util.find_spec('csi_maize')
        if not spec:raise ValueError('Install the public methods package or supply --repo')
        folder=Path(next(iter(spec.submodule_search_locations)))/'_methods'
    path=folder/'traits.py';sys.path.insert(0,str(folder));spec=importlib.util.spec_from_file_location('capture_public_traits',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,dict(traits_file=str(path),traits_sha256=digest(path),support_sha256={n:digest(folder/n) for n in ['common.py','bootstrap.py']})


def bind_requested_repo(repo):
    if repo:
        source=(Path(repo).resolve()/'src');package=source/'csi_maize'
        if not (package/'__init__.py').is_file():raise ValueError('--repo must contain src/csi_maize')
        loaded=sys.modules.get('csi_maize')
        if loaded and not Path(loaded.__file__).resolve().is_relative_to(source):raise ValueError('Requested repository differs from an already imported csi_maize package')
        sys.path.insert(0,str(source))


def camera_summary(rows,session):
    result={}
    for camera in ['rgb','noir']:
        rr=[r for r in rows if r['camera']==camera];good=[r for r in rr if r.get('structural_usability',{}).get('usable') and r.get('traits')];angles=[r['angle_degrees'] for r in good];q=len({int(a%360//90) for a in angles})
        minimum=max(8,math.ceil(session['steps']/2));eligible=len(good)>=minimum and q==4;traits={}
        if good:
            for key in good[0]['traits']:
                v=np.array([r['traits'][key] for r in good],float);traits[key]=dict(median=float(np.median(v)),maximum=float(v.max()),iqr=float(np.percentile(v,75)-np.percentile(v,25)))
        result[camera]=dict(expected_views=session['steps'],present_views=sum(r['present'] for r in rr),decoded_views=sum(r.get('decode_ok',False) for r in rr),
            usable_views=len(good),angular_quadrants=q,minimum_summary_views=minimum,summary_eligible=eligible,
            qualification='At least half declared views (minimum8) and four angular quadrants; engineering coverage, not anatomical validation',traits=traits if eligible else {},
            available_but_ineligible_traits=traits if not eligible else {})
    return result


def track_camera(camera,rows,masks,session):
    support=tracking_support(session)
    if not support['supported']:return dict(status='abstained',reason=support['reason'],tracks=[])
    sibling=Path(__file__).parent.parent/'longitudinal_tracking_development_2026-10-27';sys.path.insert(0,str(sibling))
    from tracking_graph_core import fit_cycles,predict
    from tracking_observations import extract_paths,normalize_paths
    good=[r for r in rows if r['camera']==camera and r.get('structural_usability',{}).get('usable')];by={r['view']:r for r in good}
    if len(good)<20 or len({r['view']//10 for r in good})<4:return dict(status='abstained',reason='Insufficient usable40-view angular coverage',tracks=[])
    small={v:np.asarray(Image.fromarray(masks[(camera,v)]).resize((256,round(masks[(camera,v)].shape[0]*256/masks[(camera,v)].shape[1])),Image.Resampling.NEAREST))>0 for v in by}
    small={v:m for v,m in small.items() if m.any()}
    axes=[];bases=[];heights=[]
    for v,m in small.items():
        if v%4==2:continue
        yy,xx=np.where(m);axes.append(float(np.median(xx[yy>=np.percentile(yy,95)])));bases.append(float(yy.max()));heights.append(float(np.ptp(yy)+1))
    if len(axes)<15:return dict(status='abstained',reason='Insufficient training angles',tracks=[])
    gauge=dict(axis=float(np.median(axes)),base=float(np.median(bases)),height=float(np.median(heights)));paths={v:extract_paths(m,min_length=6) for v,m in small.items()};candidates={v:normalize_paths(p,**gauge) for v,p in paths.items()}
    tracks=fit_cycles({v:c for v,c in candidates.items() if v%4!=2})
    for t in tracks:
        t['point']=dict(x=t['x_coefficient'][1],up=t['height'],depth=t['x_coefficient'][2]);evidence=[]
        for v in range(40):
            p=predict(t,v);r=by.get(v);cc=candidates.get(v,[]);distance=[float(np.linalg.norm(p-[c['x'],c['y']])) for c in cc];i=int(np.argmin(distance)) if distance else None
            width=small[v].shape[1] if v in small else 256;height=small[v].shape[0] if v in small else 192
            e=dict(view=v,split='heldout' if v%4==2 else 'train',status='no_usable_source' if not r else 'terminal_unobserved',
                predicted_image_fraction=[(gauge['axis']+gauge['height']*p[0])/width,(gauge['base']-gauge['height']*p[1])/height],
                nearest_terminal_distance=min(distance) if distance else None,source_record=f'../records/{camera}_{v:03d}.json')
            if i is not None and distance[i]<.07:e.update(status='compatible_terminal_image_support',observed_path_fraction=[[x/width,y/height] for x,y in cc[i]['path_processing_px']])
            evidence.append(e)
        t['angle_evidence']=evidence;t['interpretation']='Local rotational terminal hypothesis, not anatomical leaf identity'
    return dict(status='executed',reason=support['reason'],gauge=gauge,tracks=tracks,physical_identity_status=session['physical_identity_status'],
        dependency_sha256={f:digest(sibling/f) for f in ['tracking_graph_core.py','tracking_observations.py']})


def run_pipeline(raw,session,output,provider,repo=None):
    raw=Path(raw).resolve();session=validate_session(session);output=safe_output(raw,output);output.mkdir(parents=True,exist_ok=True);start=time.time()
    bind_requested_repo(repo);operators,operator_provenance=public_traits(repo);planned=inventory(raw,session);dump(output/'session.json',session);table(output/'inventory.csv',planned)
    snapshot=output/'execution_sources';snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,snapshot/p.name)
    methods=Path(operator_provenance['traits_file']).parent;(snapshot/'public_methods').mkdir()
    for name in ['traits.py','common.py','bootstrap.py']:shutil.copyfile(methods/name,snapshot/'public_methods'/name)
    code={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')};provenance=dict(pipeline_source_sha256=code,**operator_provenance,provider=provider.provenance,raw_root=str(raw),
        processing_policy='Strict full native decode first; unchanged public load_image processing1024 from the same source bytes',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    dump(output/'configuration.json',dict(session=session,provenance=provenance));rows=[];masks={};previews={};traitset=None
    for i,original in enumerate(planned,1):
        r=dict(original);r.update(status='missing_source',decode_ok=False,traits=None)
        if r['present']:
            try:
                blob=Path(r['_path']).read_bytes();r['source_sha256']=hashlib.sha256(blob).hexdigest()
                with Image.open(io.BytesIO(blob)) as im:im.load();r.update(native_width=im.width,native_height=im.height)
                image,native=operators.load_image(io.BytesIO(blob),width=1024);r.update(decode_ok=True,processing_width=image.shape[1],processing_height=image.shape[0],status='decoded')
                mask=np.asarray(provider.predict(image,r))
                if mask.shape!=image.shape[:2] or mask.dtype!=bool:raise ValueError('Provider mask shape/dtype differs from processing grid')
                r['structural_usability']=structural_usability(mask);r['provider_threshold_kind']=provider.provenance.get('threshold_kind');r['provider_threshold']=provider.provenance.get('threshold')
                relative=f'masks/{r["camera"]}_{r["view"]:03d}.png';path=output/relative;path.parent.mkdir(exist_ok=True);Image.fromarray((mask*255).astype('uint8')).save(path);r.update(mask=relative,mask_sha256=digest(path),status='inferred')
                if mask.sum()>=10:
                    values,_=operators.measure_mask(image,mask,r['camera'])
                    if len(values)!=39:raise ValueError('Public39-trait schema changed')
                    if traitset is not None and set(values)!=traitset:raise ValueError('Trait schema varies between frames')
                    traitset=set(values);r['traits']=values
                masks[(r['camera'],r['view'])]=mask;previews[r['camera']+':'+str(r['view'])]=(image,mask)
            except Exception as e:r.update(status='inference_or_decode_failed',error=repr(e))
        r=clean(r);rows.append(r);dump(output/f'records/{r["camera"]}_{r["view"]:03d}.json',r)
        if i%10==0:print(json.dumps(dict(stage='capture_frames',completed=i,total=len(planned))),flush=True)
    changed=[r['source'] for r in rows if r.get('source_sha256') and digest(raw/r['source'])!=r['source_sha256']]
    groups={}
    for r in rows:
        if r.get('source_sha256'):groups.setdefault(r['source_sha256'],[]).append(dict(camera=r['camera'],view=r['view'],source=r['source']))
    duplicate_warnings=[dict(source_sha256=h,frames=rr,interpretation='Identical bytes may be legitimate static/black frames or copied camera inputs; inspect sources') for h,rr in groups.items() if len({r['camera'] for r in rr})>1]
    summaries=camera_summary(rows,session);tracking={}
    for camera in ['rgb','noir']:
        try:tracking[camera]=track_camera(camera,rows,masks,session)
        except Exception as e:tracking[camera]=dict(status='failed',reason=repr(e),tracks=[],interpretation='Numerical tracking failure; no biological interpretation')
    for c,t in tracking.items():dump(output/f'tracking/{c}.json',t)
    flattened=[{k:v for k,v in r.items() if k!='traits'}|{k:v for k,v in (r['traits'] or {}).items()} for r in rows];table(output/'frames.csv',flattened)
    failures=[r for r in rows if r['status'] in ['missing_source','inference_or_decode_failed']];table(output/'failures.csv',failures)
    result=dict(status='completed' if not changed else 'completed_with_source_changes',session=session,planned_frames=len(rows),decoded_frames=sum(r['decode_ok'] for r in rows),
        inferred_frames=sum(r['status']=='inferred' for r in rows),missing_frames=sum(not r['present'] for r in rows),failed_frames=sum(r['status']=='inference_or_decode_failed' for r in rows),
        trait_count=len(traitset) if traitset else 0,camera_summaries=summaries,provider=provider.provenance,provenance=provenance,source_changed_during_run=changed,
        record_index={r['camera']+':'+str(r['view']):f'records/{r["camera"]}_{r["view"]:03d}.json' for r in rows},tracking=tracking,
        duplicate_source_hash_warnings=duplicate_warnings,
        assays=dict(status='user_supplied',predicted=False,values=session['assays']) if 'assays' in session else dict(status='not_provided',predicted=False),
        elapsed_seconds=time.time()-start,limits=['No study registry or workbook required.','Structural usability is independent of provider confidence.','No metric dimensions, anatomical leaf identity or assay values are inferred.'])
    dump(output/'capture.json',result);make_report(output,result,rows,tracking,previews)
    dump(output/'output_manifest.json',dict(status=result['status'],files=[dict(path=p.relative_to(output).as_posix(),sha256=digest(p)) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='output_manifest.json']))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--session',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--provider',choices=['unet','yolo'],default='unet');p.add_argument('--model-card',type=Path,required=True);p.add_argument('--provider-module',type=Path);p.add_argument('--pixel-threshold',type=float);p.add_argument('--confidence',type=float);p.add_argument('--device',default='cpu');p.add_argument('--repo',type=Path);a=p.parse_args()
    from capture_providers import UNetProvider,YOLOProvider
    safe_output(a.input,a.output);bind_requested_repo(a.repo);session=json.loads(a.session.read_text(encoding='utf-8'));validate_session(session);inventory(a.input,validate_session(session))
    if a.provider=='unet':
        if a.confidence is not None:raise ValueError('Use --pixel-threshold for U-Net, not --confidence')
        provider=UNetProvider(a.model_card,a.device,a.pixel_threshold,a.provider_module)
    else:
        if a.pixel_threshold is not None:raise ValueError('YOLO confidence is not a pixel threshold')
        provider=YOLOProvider(a.model_card,a.output,a.device,a.confidence,a.repo)
    result=run_pipeline(a.input,session,a.output,provider,a.repo);print(json.dumps({k:result[k] for k in ['status','planned_frames','decoded_frames','inferred_frames','missing_frames','failed_frames','trait_count','elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
