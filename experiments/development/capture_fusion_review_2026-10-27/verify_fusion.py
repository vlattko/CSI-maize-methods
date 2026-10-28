"""Independent, read-only audit of capture/fusion bindings, grids and saved maps."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,base64,csv,hashlib,importlib.util,io,json,math,re,sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
cv2.setNumThreads(2)

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def contained(root,name):
    target=(root/name).resolve()
    if not target.is_relative_to(root.resolve()):raise ValueError('Escaping path')
    return target

def verify_manifest(root):
    manifest=json.loads((root/'output_manifest.json').read_text());assert manifest['status']=='completed'
    names=[r['path'] for r in manifest['files']];assert len(names)==len(set(names))
    for row in manifest['files']:assert sha(contained(root,row['path']))==row['sha256'],row['path']
    return len(names)

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def equal_stat(actual,stored):
    if actual is None:return stored==''
    if isinstance(actual,bool):return stored==str(actual)
    return math.isclose(float(actual),float(stored),rel_tol=1e-12,abs_tol=1e-12)

def mask(record,capture,size):
    with Image.open(contained(capture,record['mask'])) as im:return np.asarray(im.convert('L').resize(size,Image.Resampling.NEAREST))>0

def image(record,raw):
    with Image.open(contained(raw,record['source'])) as im:im.load();return np.asarray(im.convert('RGB').resize((512,385),Image.Resampling.LANCZOS))

def verify_case(name,capture,fusion,operators):
    channel,spatial=operators
    files=verify_manifest(capture)+verify_manifest(fusion)
    configuration=json.loads((capture/'configuration.json').read_text());raw=Path(configuration['provenance']['raw_root'])
    result=json.loads((capture/'capture.json').read_text());summary=json.loads((fusion/'summary.json').read_text());fit=json.loads((fusion/'camera_fit.json').read_text())
    assert result['status']=='completed' and summary['status']=='completed'
    for module in (channel,spatial):assert sha(module.__file__)==summary['provenance']['public_operators'][Path(module.__file__).name]
    assert result['session']['plant_uid'] is None
    index=result['record_index'];assert set(index)=={f'{camera}:{v}' for camera in ('rgb','noir') for v in range(40)}
    records={token:json.loads(contained(capture,path).read_text()) for token,path in index.items()}
    sources=0;mask_hashes=0
    for token,record in records.items():
        assert token==record['camera']+':'+str(record['view']);assert record['sample_id']==result['session']['sample_id']
        assert record['status']=='inferred';assert sha(contained(raw,record['source']))==record['source_sha256'];sources+=1
        assert sha(contained(capture,record['mask']))==record['mask_sha256'];mask_hashes+=1
        assert len(record['traits'])==39
    provider=configuration['provenance']['provider'];assert sha(Path(provider['checkpoint']))==provider['checkpoint_sha256'];assert sha(Path(provider['model_card']))==provider['model_card_sha256']
    if provider['provider']=='yolo':
        assert provider['threshold_kind']=='detection_confidence' and provider['threshold']==.05 and provider['imgsz']==960
        assert provider['checkpoint_sha256']=='c2a32e054ce99cd7fd3d66cb5cc2d06fbf75525af9a9bb4979bf2dc189732296'
        assert sha(Path(provider['adapter_file']))==provider['adapter_sha256']
    else:assert provider['threshold_kind']=='foreground_probability' and provider['threshold']==.35
    if name!='b73':
        assert not result['session'].get('metadata') and not result['session'].get('assays')
    selected=fit['shared']['selected'];train=selected['train_indices'];assert not any(v%4==2 for v in train)
    assert not any(v%4==2 for v in fit['periodic'].get('train_indices',[]))
    assert not any(r['view']%4==2 for r in fit['periodic'].get('teachers',[]))
    for inner in fit['periodic'].get('inner_validation',[]):
        assert not set(inner.get('training_indices',[]))&set(inner.get('validation_indices',[]))
    periodic=summary['camera_model']=='first_harmonic';abstained=summary['camera_decision']['deployment_abstain']
    if periodic:assert fit['periodic']['selection']['accepted'] and all(q['guard']['valid'] for q in fit['periodic']['queries'])
    pairs=read(fusion/'pairs.csv');stats=read(fusion/'index_statistics.csv');assert len(pairs)==40 and {int(r['view']) for r in pairs}==set(range(40));assert len(stats)==1080
    stats_by={(int(r['view']),r['index']):r for r in stats};assert len(stats_by)==1080
    definitions=spatial.index_definitions();maps_checked=0;stats_checked=0;replayed_views=[];representative=(0,2,20)
    for row in pairs:
        view=int(row['view']);nv=int(row['noir_view']);assert nv==(selected['sign']*view+selected['phase'])%40
        assert row['split']==('heldout' if view%4==2 else 'train')
        matrix=np.asarray(json.loads(row['matrix_rgb_to_noir']))
        fit_matrix=np.asarray(fit['periodic']['queries'][view]['matrix'] if periodic else selected['matrix'])
        scale=np.diag([2,385/192,1]);expected=(scale@np.vstack([fit_matrix,[0,0,1]])@np.linalg.inv(scale))[:2]
        np.testing.assert_allclose(matrix,expected,rtol=0,atol=1e-12)
        rm=mask(records['rgb:'+str(view)],capture,(512,385));nm=mask(records['noir:'+str(nv)],capture,(512,385))
        registered,field=channel.warp_to_rgb(nm.astype(np.uint8),matrix,rm.shape,True)
        union=int((rm|(registered>0)).sum());iou=float((rm&(registered>0)).sum()/union) if union else 0
        assert math.isclose(iou,float(row['map_grid_iou_512']),abs_tol=1e-12)
        interior,field=channel.warp_to_rgb(channel.interior(nm).astype(np.uint8),matrix,rm.shape,True)
        supports=dict(rgb=channel.interior(rm),noir=(interior>0)&field);common=supports['rgb']&supports['noir']
        assert int(common.sum())==int(row['common_pixels'])
        fraction=float(common.sum()/max(1,min(int(supports['rgb'].sum()),int(supports['noir'].sum()))))
        assert math.isclose(fraction,float(row['common_fraction']),abs_tol=1e-12)
        accepted=row['cross_formulas_usable']=='True';supports['cross']=common if accepted else np.zeros_like(common)
        if accepted:assert not abstained and iou>=.65 and common.sum()>=40 and fraction>=.35
        path=fusion/f'maps/view_{view:03d}.npz'
        if abstained:
            assert not path.exists() and not accepted
            for d in definitions:
                statistic=stats_by[(view,d['id'])];assert statistic['qualified']=='False' and int(statistic['valid_pixels'])==0
            continue
        with np.load(path) as archive:
            assert set(archive.files)=={d['id'] for d in definitions}
            for definition in definitions:
                values=archive[definition['id']];assert values.shape==(385,512) and values.dtype==np.float32
                assert not np.isfinite(values[~supports[definition['support']]]).any()
                expected_stats=spatial.index_statistics(values,supports[definition['support']]);stored=stats_by[(view,definition['id'])]
                assert all(equal_stat(value,stored[key]) for key,value in expected_stats.items()),(view,definition['id'])
                stats_checked+=1
            if view in representative:
                rgb=image(records['rgb:'+str(view)],raw);noir=image(records['noir:'+str(nv)],raw)
                moved,_=channel.warp_to_rgb(noir.astype(np.float32),matrix,rm.shape)
                validity,_=channel.warp_to_rgb(((noir>1)&(noir<254)).astype(np.float32),matrix,rm.shape)
                valid=np.concatenate([(rgb>1)&(rgb<254),validity>=1-1e-7],axis=-1)
                six=np.concatenate([rgb.astype(np.float32),moved],axis=-1);replay=spatial.spatial_indices(six,supports,valid)
                for definition in definitions:np.testing.assert_allclose(archive[definition['id']],replay[definition['id']],atol=0,rtol=0,equal_nan=True)
                replayed_views.append(view)
        maps_checked+=1
    assert sum(r['cross_formulas_usable']=='True' for r in pairs)==summary['usable_cross_pairs']
    page=(fusion/'report.html').read_text();assert 'https://' not in page and 'http://' not in page
    payload=re.search(r'<script id="payload" type="application/json">(.*?)</script>',page,re.S);gallery=json.loads(payload[1]);assert len(gallery)==40 and {r['view'] for r in gallery}==set(range(40))
    for row in gallery:
        data=base64.b64decode(row['image'].split(',',1)[1]);assert hashlib.sha256(data).hexdigest()==sha(fusion/f"previews/view_{row['view']:03d}.png")
    if (fusion/'presentation.json').exists():
        presentation=json.loads((fusion/'presentation.json').read_text());assert sha(fusion/'presentation/original.html')==presentation['original_html_sha256'] and sha(fusion/'report.html')==presentation['formatted_html_sha256']
    return dict(case=name,status='passed',capture_manifest_sha256=sha(capture/'output_manifest.json'),fusion_manifest_sha256=sha(fusion/'output_manifest.json'),operator_sha256={Path(m.__file__).name:sha(m.__file__) for m in operators},manifest_files_verified=files,source_hashes_verified=sources,mask_hashes_verified=mask_hashes,traits_per_frame=39,pairs=40,descriptor_rows=1080,maps_verified=maps_checked,all_saved_map_statistics_replayed=stats_checked,full_DN_formula_replay_views=replayed_views,full_DN_replayed_formula_maps=27*len(replayed_views),camera_model=summary['camera_model'],deployment_abstained=abstained,usable_cross_pairs=summary['usable_cross_pairs'],qualified_descriptor_rows=summary['qualified_descriptor_rows'],provider=provider['provider'],threshold_kind=provider['threshold_kind'],threshold=provider['threshold'],heldout_median_iou_512=summary['heldout_median_iou_512'],offline_report_gallery_frames=len(gallery),limits='Operational/provenance and numerical consistency checks, not accuracy or tissue correspondence.')

def operators_for(fusion,folders):
    required=json.loads((fusion/'summary.json').read_text())['provenance']['public_operators'];modules=[]
    for name in ('channel_fusion_core.py','spatial_fusion_core.py'):
        candidates=[folder/name for folder in folders if (folder/name).exists() and sha(folder/name)==required[name]]
        if not candidates:raise ValueError('Exact recorded scientific operator unavailable: '+name)
        spec=importlib.util.spec_from_file_location('fusion_review_'+required[name],candidates[0]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
    return modules

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fusion-root',type=Path,required=True);parser.add_argument('--capture-root',type=Path,required=True);parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--alternate-repo',type=Path);parser.add_argument('--yolo-directory',default='yolo_end_to_end');args=parser.parse_args()
    folders=[args.repo/'src/csi_maize/_methods']
    if args.alternate_repo:folders.append(args.alternate_repo/'src/csi_maize/_methods')
    folders.append(args.fusion_root/'execution_sources/public');sys.path.insert(0,str(folders[0]))
    out=args.fusion_root/'verification.json'
    if out.exists():raise ValueError('Verification output already exists')
    rows=[]
    for name in ('b73','unjoined_033402','unjoined_131906'):
        fusion=args.fusion_root/'final'/name;rows.append(verify_case(name,args.capture_root/name,fusion,operators_for(fusion,folders)))
    end=args.fusion_root/args.yolo_directory;manifest=json.loads((end/'digitization_manifest.json').read_text());assert manifest['status']=='completed'
    for stage in manifest['stages']:assert sha(end/stage['stage']/'output_manifest.json')==stage['output_manifest_sha256']
    rows.append(verify_case('yolo_unjoined_033402',end/'capture',end/'fusion',operators_for(end/'fusion',folders)))
    result=dict(status='passed',verifier_sha256=sha(__file__),cases=rows,scope='All320 frame bindings across four executions/source/mask hashes and traits counts; all160 pair keys and4320 descriptor rows; all saved map statistics, plus all27 formulas on indices0,2,20 wherever deployment is accepted. Source inputs are not model test labels.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
