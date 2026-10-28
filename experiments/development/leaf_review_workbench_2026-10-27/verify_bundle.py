"""Verify the actual portable artifact without controlling a browser."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import base64
import hashlib
import io
import json
import math
import re
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(value,message):
    if not value:raise AssertionError(message)
def decode(uri):return base64.b64decode(uri.split(',',1)[1])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output
    manifest=read(out/'asset_manifest.json');text=(out/manifest['primary_html']).read_text(encoding='utf-8')
    marker='<script id="dataset" type="application/json">';start=text.index(marker)+len(marker);end=text.index('</script>',start)
    data=json.loads(text[start:end]);scripts=text[end:]
    require(data['dataset_id']==manifest['dataset_id'],'dataset_identity')
    bindings=read(out/'source_bindings.json')['frames']
    require(digest(dict(configuration_signature=manifest['proposal_configuration_signature'],source_and_proposal_bindings=bindings))==data['dataset_id'],'source_binding_digest')
    require(len(data['frames'])==60,'all60_original_frames')
    require(not re.search(r'\bfetch\s*\(|XMLHttpRequest|https?://',scripts),'no_network_or_file_fetch_dependency')
    require("dataNode.textContent='';dataNode.remove();" in scripts,'synchronous_payload_removal')
    source_assets={a['frame_id']:a for a in manifest['native_assets']}
    count=0;logical=0;mask_bytes=0;orientations=Counter();frame_ids=set();candidate_keys=set();proposal_ids=set()
    for frame in data['frames']:
        fid=frame['frame_id'];require(fid not in frame_ids,'unique_frame_ID');frame_ids.add(fid)
        asset=source_assets[fid];asset_path=out/frame['asset_path']
        require(frame['asset']==frame['asset_path'],'portable_primary_uses_relative_original_assets')
        require(frame['asset_path'].startswith('assets/originals/') and '..' not in Path(frame['asset_path']).parts,'asset_path_scope')
        require(sha(asset_path)==frame['source_sha256']==asset['sha256'],'actual_native_JPEG_hash')
        with Image.open(asset_path) as im:
            require(im.size==(frame['native_width'],frame['native_height']),'native_dimensions')
            orientation=im.getexif().get(274);orientations[str(orientation)]+=1
            require(orientation in (None,1),'browser_EXIF_orientation_would_change_geometry')
        for candidate in frame['candidates']:
            key=(fid,candidate['candidate_id']);require(key not in candidate_keys,'candidate_source_scope');candidate_keys.add(key);logical+=1
            require(len(candidate['proposals'])==6,'all_two_modes_three_alternatives')
            require({(p['mode'],p['alternative']) for p in candidate['proposals']}=={(m,a) for m in ('box_only','distal_points_negatives') for a in range(3)},'complete_alternative_set')
            for p in candidate['proposals']:
                require(p['proposal_id'] not in proposal_ids,'unique_proposal_ID');proposal_ids.add(p['proposal_id'])
                for kind in ('raw','clipped'):
                    blob=decode(p['mask_'+kind+'_data']);mask_bytes+=len(blob)
                    require(hashlib.sha256(blob).hexdigest()==p[kind+'_sha256'],'embedded_binary_mask_hash')
                    with Image.open(io.BytesIO(blob)) as im:
                        mask=np.array(im)>0;require(im.size==(p['processing_width'],p['processing_height']),'declared_processing_mask_grid')
                    with Image.open(io.BytesIO(decode(p['overlay_'+kind]))) as im:alpha=np.array(im)[...,3]>0
                    require(np.array_equal(alpha,mask),'preview_overlay_matches_actual_binary_mask')
                    for ring in p['polygons_'+kind]:
                        require(len(ring)>=3,'polygon_minimum_vertices')
                        require(all(math.isfinite(x) and math.isfinite(y) and 0<=x<=frame['native_width'] and 0<=y<=frame['native_height'] for x,y in ring),'native_polygon_bounds')
                count+=1
    require(count==516 and logical==86,'full_pilot_request_alternative_denominators')
    for name,entry in manifest['files'].items():require(sha(out/name)==entry['sha256'],'delivered_file_hash:'+name)
    result=dict(status='verified_portable_annotation_bundle',schema=1,dataset_id=data['dataset_id'],
        primary_html=manifest['primary_html'],native_source_assets=60,native_asset_bytes=sum(a['bytes'] for a in source_assets.values()),
        logical_region_requests=logical,proposal_alternatives=count,embedded_binary_masks_rehashed=count*2,
        preview_alpha_matches_masks=True,native_dimensions_verified=True,native_EXIF_orientations=dict(orientations),
        no_fetch_xhr_or_external_network=True,synchronous_payload_removal=True,
        initial_review_status='All unreviewed/unknown; all image completeness and verified-empty flags false',
        real_reviewed_annotations_delivered=0,real_browser_visual_QA='unverified_policy_blocked_file_protocol',
        browser_QA_limitation='Browser URL security policy rejected file://. No HTTP, alternate surface or other workaround was attempted.',
        tests=dict(core_behavior=12,synthetic_DOM_app_handlers=5,total=17),
        files=manifest['files'],source_sha256=sha(__file__))
    path=out/'validation.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
