"""Independent source, coordinate, clipping and review-export checks."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def require(value,message):
    if not value:raise AssertionError(message)
def mask(path):
    with Image.open(path) as im:return np.asarray(im)>0

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--intake',type=Path,required=True);parser.add_argument('--full-run',type=Path,required=True)
    args=parser.parse_args();folder=args.output/'pilot';config=read(folder/'configuration.json')
    signature=config.pop('signature');require(digest(config)==signature,'configuration_digest');config['signature']=signature
    require(sha(args.output/'pilot_plan.json')==config['pilot_plan_sha256'],'plan_file_binding')
    plan=read(args.output/'pilot_plan.json');plan_sig=plan.pop('signature');require(digest(plan)==plan_sig,'plan_payload')
    require(plan_sig==config['plan_signature'],'plan_configuration_binding')
    require(len(plan['frames'])==60 and plan['actual_strata']==15,'fixed_pilot_denominator')
    frozen=folder/'source_snapshot'/signature
    for name,h in config['source_hashes'].items():require(sha(frozen/name)==h,'cached_source_hash:'+name)
    loading=read(folder/'model_loading.json');require(not any(loading.values()),'exact_model_loading')
    with (args.full_run/'extraction/view_traits.csv').open(encoding='utf-8-sig',newline='') as f:table=list(csv.DictReader(f))
    lookup={(r['session'],r['camera'],int(r['view'])):r for r in table}
    records=[];count=0;positives=0;outside=0;clipped_count=0;failures=0;by_camera=Counter();proposal_index={}
    for path in sorted((folder/'records').glob('*.json')):
        record=read(path);payload=record.pop('payload_sha256');require(digest(record)==payload,'record_payload')
        require(record['signature']==signature,'record_signature')
        source=lookup[record['identity']['session'],record['camera'],record['view']]
        require(all(str(record['identity'][k])==source[k] for k in ('treatment','measurement','genotype','replicate')),'four_key_identity')
        require(sha(args.intake/source['source'])==source['source_sha256'],'actual_source_bytes')
        parent=mask(args.full_run/source['mask']);require(sha(args.full_run/source['mask'])==source['mask_sha256'],'parent_hash')
        if record.get('input'):
            require(sha(folder/record['input']['review_image'])==record['input']['review_image_sha256'],'review_image_hash')
        for candidate in record.get('planned_proposals',[]):
            for xy,label in zip(candidate['prompt_processing']['points'],candidate['prompt_processing']['labels']):
                if label==1:
                    x,y=map(round,xy);positives+=1
                    supported=0<=x<parent.shape[1] and 0<=y<parent.shape[0] and parent[y,x]
                    outside+=not supported
        for p in record['proposals']:
            raw=mask(folder/p['raw_mask']);clip=mask(folder/p['clipped_mask'])
            require(raw.shape==parent.shape==clip.shape,'mask_coordinate_grid')
            require(sha(folder/p['raw_mask'])==p['raw_mask_sha256'],'raw_hash')
            require(sha(folder/p['clipped_mask'])==p['clipped_mask_sha256'],'clipped_hash')
            require(np.array_equal(raw&parent,clip),'exact_parent_intersection')
            require(int(raw.sum())==p['raw_area_pixels'] and int(clip.sum())==p['clipped_area_pixels'],'mask_area_accounting')
            require(int(raw.sum()-clip.sum())==p['raw_outside_parent_pixels'],'outside_accounting')
            require(p['region_review_status']=='unreviewed' and not p['score_is_anatomical_accuracy'],'unreviewed_status')
            require(p['mode'] in ('box_only','distal_points_negatives') and p['alternative'] in (0,1,2),'all_alternatives')
            proposal_index[p['proposal_id']]=p;count+=1;clipped_count+=int(clip.sum());by_camera[record['camera']]+=1
        failures+=len(record['failures']);records.append(record)
    require(len(records)==60 and len({(r['identity']['session'],r['camera'],r['view']) for r in records})==60,'all_unique60_frames')
    require(count==config['expected_proposal_alternatives']==516,'all516_proposal_alternatives')
    require(outside==0,'positive_prompt_outside_exact_processing_parent')
    coco=read(folder/'review_predictions_coco.json');decoded=0
    for ann in coco['annotations']:
        rle=ann['segmentation'];counts=rle['counts'];n=rle['size'][0]*rle['size'][1]
        require(sum(counts)==n,'RLE_total_pixels')
        flat=np.zeros(n,bool);offset=0
        for i,length in enumerate(counts):
            if i%2:flat[offset:offset+length]=True
            offset+=length
        pixels=flat.reshape(rle['size'],order='F');proposal=proposal_index[ann['proposal_id']]
        require(np.array_equal(pixels,mask(folder/proposal['clipped_mask'])),'COCO_RLE_exact_mask')
        require(ann['review_required'] and not ann['anatomical_identity_verified'],'COCO_review_status')
        decoded+=1
    export=read(folder/'review_export_manifest.json')
    for name,h in export['outputs'].items():require(sha(folder/name)==h,'review_output_hash:'+name)
    require(export['automatically_accepted_training_labels']==0,'no_pseudo_truth_labels')
    archive=args.output/'pilot_coordinate_v1';old_plan=read(args.output/'pilot_plan_coordinate_v1.json')
    require([(x['identity'],x['camera'],x['view']) for x in old_plan['frames']]==
            [(x['identity'],x['camera'],x['view']) for x in plan['frames']],'same_bounded_pilot_after_coordinate_fix')
    result=dict(status='verified_unreviewed_region_pilot',schema=1,configuration_signature=signature,
        all_frames=60,all_strata=15,ready_prompt_regions=plan['ready_prompts'],planned_prompt_failures=plan['failed_prompts'],
        raw_and_clipped_alternative_pairs=count,alternatives_by_camera=dict(by_camera),
        source_images_rehashed=60,parent_masks_rehashed=60,all_output_masks_rehashed=count*2,
        positive_processing_points=positives,positive_points_outside_parent=outside,
        exact_parent_clipping=True,independent_coco_masks_decoded=decoded,inference_failures=failures,
        coordinate_v1_preserved=str(archive),coordinate_fix_scope='same60frames/same86regions; pixel-center conversion and exact-parent positive refinement',
        automatically_accepted_training_labels=0,source_sha256=sha(__file__))
    target=args.output/'verification.json';temp=target.with_suffix('.json.tmp');temp.write_text(json.dumps(result,indent=2),encoding='utf-8');temp.replace(target)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
