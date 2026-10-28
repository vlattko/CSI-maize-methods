"""Build a private file-local annotation bundle from the verified pilot."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import base64
import hashlib
import io
import json
import shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
cv2.setNumThreads(2)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def data_uri(path,mime='image/png'):return f'data:{mime};base64,'+base64.b64encode(Path(path).read_bytes()).decode()

def mask_data(path,ow,oh,color):
    blob=Path(path).read_bytes()
    with Image.open(io.BytesIO(blob)) as im:mask=np.asarray(im.convert('L'))>0
    h,w=mask.shape
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    polygons=[]
    for contour in contours:
        approximation=cv2.approxPolyDP(contour,.75,True)[:,0]
        if len(approximation)<3 or abs(cv2.contourArea(approximation))<.5:continue
        polygons.append([[(float(x)+.5)*ow/w,(float(y)+.5)*oh/h] for x,y in approximation])
    rgba=np.zeros((h,w,4),np.uint8);rgba[mask]=[*color,110]
    buffer=io.BytesIO();Image.fromarray(rgba).save(buffer,format='PNG')
    return dict(binary='data:image/png;base64,'+base64.b64encode(blob).decode(),
                overlay='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(),
                polygons=polygons,width=w,height=h)

def write_html(path,dataset,source):
    shell=(source/'workbench_shell.html').read_text(encoding='utf-8')
    before,after=shell.split('<!-- DATA_AND_SCRIPTS -->')
    with Path(path).open('w',encoding='utf-8',newline='\n') as f:
        f.write(before);f.write('<script id="dataset" type="application/json">')
        # Source metadata are generated IDs/paths and base64; escape literal
        # angle brackets so untrusted source strings cannot terminate a script.
        encoder=json.JSONEncoder(ensure_ascii=True,separators=(',',':'))
        for chunk in encoder.iterencode(dataset):f.write(chunk.replace('<','\\u003c'))
        f.write('</script><script>');f.write((source/'review_core.js').read_text(encoding='utf-8'))
        f.write('</script><script>');f.write((source/'review_app.js').read_text(encoding='utf-8'));f.write('</script>');f.write(after)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proposals',type=Path,required=True)
    parser.add_argument('--intake',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--standalone',action='store_true');args=parser.parse_args();out=args.output;source=Path(__file__).parent
    out.mkdir(parents=True,exist_ok=True);originals=out/'assets/originals';originals.mkdir(parents=True,exist_ok=True)
    frames=read(args.proposals/'pilot/frame_index.json')['frames'];regions=read(args.proposals/'pilot/regions.json')['regions']
    if len(frames)!=60 or len(regions)!=516:raise ValueError('expected_verified_60_frame_516_alternative_pilot')
    config=read(args.proposals/'pilot/configuration.json');verification=read(args.proposals/'verification.json')
    if verification['status']!='verified_unreviewed_region_pilot':raise ValueError('upstream_not_verified')
    dataset=dict(schema=1,label='CSI maize region annotation review',proposal_configuration_signature=config['signature'],
        coordinate_system='Native original-image continuous pixel-edge coordinates. Model-mask contour pixel centres are mapped to this grid.',
        proposal_geometry_resolution='Processing-resolution model masks; editable on full original source grid',
        defaults=dict(review_status='unreviewed',extent='unknown',all_visible_leaves_reviewed=False,verified_empty=False),
        cross_date_identity_verified=False,frames=[])
    assets=[];bindings=[]
    for f in frames:
        record=read(args.proposals/'pilot/records'/f"{f['frame_id']}.json");row=record['source_record'];original=args.intake/row['source']
        actual=sha(original)
        if actual!=row['source_sha256']:raise ValueError('original_source_hash_mismatch')
        asset_path=f"assets/originals/{f['frame_id']}.jpg";destination=out/asset_path
        if destination.exists():
            if sha(destination)!=actual:raise ValueError('existing_native_asset_changed')
        else:shutil.copyfile(original,destination)
        with Image.open(original) as im:ow,oh=im.size
        if [ow,oh]!=record['input']['native_size']:raise ValueError('native_dimension_binding')
        frame={k:f[k] for k in ('frame_id','treatment','measurement','genotype','replicate','session','camera','view','source_sha256')}
        frame.update(native_width=ow,native_height=oh,source=row['source'],asset=asset_path,asset_path=asset_path,
                     foreground_data=data_uri(args.proposals/'pilot'/f['image']),candidates=[])
        subset=[p for p in regions if p['frame_id']==f['frame_id']]
        mask_bindings=[]
        for cid in sorted({p['candidate_id'] for p in subset}):
            candidate=dict(candidate_id=cid,track_identity_unverified=True,proposals=[])
            for p in sorted([p for p in subset if p['candidate_id']==cid],key=lambda p:(p['mode'],p['alternative'])):
                raw_path=args.proposals/'pilot'/p['raw_mask'];clip_path=args.proposals/'pilot'/p['clipped_mask']
                if sha(raw_path)!=p['raw_mask_sha256'] or sha(clip_path)!=p['clipped_mask_sha256']:raise ValueError('upstream_proposal_mask_changed')
                raw=mask_data(raw_path,ow,oh,(245,120,45));clip=mask_data(clip_path,ow,oh,(35,155,230))
                proposal=dict(proposal_id=p['proposal_id'],mode=p['mode'],alternative=p['alternative'],score=p['model_score'],
                    raw_sha256=p['raw_mask_sha256'],clipped_sha256=p['clipped_mask_sha256'],prompt=p['prompt_processing'],
                    prompt_origin=p['prompt_origin'],processing_width=clip['width'],processing_height=clip['height'],
                    mask_raw_data=raw['binary'],mask_clipped_data=clip['binary'],overlay_raw=raw['overlay'],overlay_clipped=clip['overlay'],
                    polygons_raw=raw['polygons'],polygons_clipped=clip['polygons'],
                    outside_parent_fraction=p['raw_outside_parent_fraction'],parent_coverage_fraction=p['parent_coverage_fraction'])
                candidate['proposals'].append(proposal)
                mask_bindings.append(dict(proposal_id=p['proposal_id'],candidate_id=cid,raw_sha256=p['raw_mask_sha256'],clipped_sha256=p['clipped_mask_sha256']))
            frame['candidates'].append(candidate)
        dataset['frames'].append(frame)
        bindings.append(dict(frame_id=f['frame_id'],source_sha256=actual,native_width=ow,native_height=oh,
                             identity={k:frame[k] for k in ('treatment','measurement','genotype','replicate','session','camera','view')},
                             proposal_masks=mask_bindings))
        assets.append(dict(path=asset_path,sha256=actual,bytes=destination.stat().st_size,frame_id=f['frame_id'],source=row['source'],
                           native_width=ow,native_height=oh))
    dataset['dataset_id']=digest(dict(configuration_signature=config['signature'],source_and_proposal_bindings=bindings))
    write_html(out/'leaf_review_workbench.html',dataset,source)
    if args.standalone:
        for frame in dataset['frames']:frame['asset']=data_uri(out/frame['asset_path'],'image/jpeg')
        write_html(out/'leaf_review_workbench_standalone.html',dataset,source)
        for frame in dataset['frames']:frame['asset']=frame['asset_path']
    # Untouched predictions stay explicitly separate from future reviewed GT.
    shutil.copyfile(args.proposals/'pilot/regions.json',out/'unreviewed_predictions.json')
    (out/'source_bindings.json').write_text(json.dumps(dict(dataset_id=dataset['dataset_id'],frames=bindings),indent=2),encoding='utf-8')
    source_snapshot=out/'source_snapshot'/digest({p.name:sha(p) for p in source.iterdir() if p.is_file()})
    source_snapshot.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        if p.is_file():shutil.copyfile(p,source_snapshot/p.name)
    manifests=dict(schema=1,status='completed',dataset_id=dataset['dataset_id'],private_source_bundle=True,
       primary_html='leaf_review_workbench.html',optional_standalone='leaf_review_workbench_standalone.html' if args.standalone else None,
       server_required=False,fetch_xhr_required=False,external_network_dependencies=[],
       source_images=60,proposal_alternatives=516,logical_region_requests=sum(len(f['candidates']) for f in dataset['frames']),
       all_visible_leaves_reviewed_default=False,verified_empty_default=False,auto_accepted_regions=0,
       coordinate_system=dataset['coordinate_system'],native_assets=assets,
       embedded_model_masks=dict(raw=516,parent_clipped=516),
       proposal_configuration_signature=config['signature'],proposal_verification_sha256=sha(args.proposals/'verification.json'),
       source_snapshot=source_snapshot.relative_to(out).as_posix(),
       files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size,optional=p.name.endswith('_standalone.html'))
              for p in [out/'leaf_review_workbench.html',out/'unreviewed_predictions.json',out/'source_bindings.json']+
                     ([out/'leaf_review_workbench_standalone.html'] if args.standalone else [])})
    (out/'asset_manifest.json').write_text(json.dumps(manifests,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='completed',dataset_id=dataset['dataset_id'],native_assets=len(assets),
          native_asset_bytes=sum(a['bytes'] for a in assets),files=manifests['files'])),flush=True)

if __name__=='__main__':main()
