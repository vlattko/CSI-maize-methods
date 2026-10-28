"""Stratified visual-QA sheets and an explicit extension decision."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import json
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def support_audit(plan,full_run):
    count=0;outside=0
    for f in plan['frames']:
        with Image.open(full_run/f['source_record']['mask']) as im:parent=np.array(im)>0
        for c in f['proposals']:
            for (x,y),label in zip(c['prompt_processing']['points'],c['prompt_processing']['labels']):
                if label==1:
                    count+=1;ix,iy=round(x),round(y)
                    outside+=not (0<=ix<parent.shape[1] and 0<=iy<parent.shape[0] and parent[iy,ix])
    return dict(positive_points=count,outside_exact_parent=outside)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full-run',type=Path,required=True);args=parser.parse_args();out=args.output;folder=out/'pilot'
    frames=read(folder/'frame_index.json')['frames'];regions=read(folder/'regions.json')['regions'];byid={r['proposal_id']:r for r in regions}
    qa=folder/'qa';qa.mkdir(exist_ok=True);sheets=[]
    for view in (0,20):
        canvas=Image.new('RGB',(6*260,5*300),(0,0,0));draw=ImageDraw.Draw(canvas)
        for ti,treatment in enumerate(('C','P','S')):
            for vi,visit in enumerate(('1','2','3','4','5')):
                for ci,camera in enumerate(('rgb','noir')):
                    f=next(x for x in frames if x['treatment']==treatment and x['measurement']==visit and x['camera']==camera and x['view']==view)
                    with Image.open(folder/f['image']) as im:image=np.array(im.convert('RGB'))
                    with Image.open(args.full_run/f['parent_mask']) as im:parent=np.array(im)>0
                    yy,xx=np.where(parent)
                    chosen=f['selected_region_ids']['distal_points_negatives']
                    for n,proposal in enumerate(chosen):
                        with Image.open(folder/byid[proposal]['clipped_mask']) as im:m=np.array(im)>0
                        color=np.array([(45,210,135),(240,150,45),(80,160,245),(235,75,150)][n%4])
                        image[m]=(image[m]*.2+color*.8).astype(np.uint8)
                    if len(xx):
                        pad=15;box=(max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),
                                    min(image.shape[1],int(xx.max())+pad+1),min(image.shape[0],int(yy.max())+pad+1))
                        crop=Image.fromarray(image).crop(box);crop.thumbnail((250,255),Image.Resampling.LANCZOS)
                    else:crop=Image.new('RGB',(250,255),'black')
                    x=(ti*2+ci)*260;y=vi*300
                    canvas.paste(crop,(x+(260-crop.width)//2,y+38+(255-crop.height)//2))
                    draw.text((x+5,y+5),f'{treatment}{visit} {camera} / index {view}',fill='white')
                    draw.text((x+5,y+20),f'{len(chosen)} unreviewed regions',fill='white')
        path=qa/f'all_strata_view_{view}.png';canvas.save(path);sheets.append(dict(path=path.relative_to(folder).as_posix(),sha256=sha(path)))
    old=read(out/'pilot_plan_coordinate_v1.json');new=read(out/'pilot_plan.json')
    coordinate=dict(v1=support_audit(old,args.full_run),v2=support_audit(new,args.full_run),
        same_case_frames=[(f['identity'],f['camera'],f['view']) for f in old['frames']]==[(f['identity'],f['camera'],f['view']) for f in new['frames']],
        v1_plan_sha256=sha(out/'pilot_plan_coordinate_v1.json'),v2_plan_sha256=sha(out/'pilot_plan.json'),
        preserved_v1='pilot_coordinate_v1',fix='PIL grid pixel centers plus exact processing-mask support refinement; same bounded60frames')
    paired=[]
    for frame in frames:
        a={byid[x]['candidate_id']:byid[x] for x in frame['selected_region_ids']['box_only']}
        b={byid[x]['candidate_id']:byid[x] for x in frame['selected_region_ids']['distal_points_negatives']}
        for candidate in set(a)&set(b):
            with Image.open(folder/a[candidate]['clipped_mask']) as im:am=np.array(im)>0
            with Image.open(folder/b[candidate]['clipped_mask']) as im:bm=np.array(im)>0
            union=(am|bm).sum();paired.append(float((am&bm).sum()/union) if union else 0.)
    export=read(folder/'review_export_manifest.json');verification=read(out/'verification.json')
    result=dict(status='completed_verified_unreviewed_annotation_bootstrap',schema=1,
        frame_index='pilot/frame_index.json',all_alternatives='pilot/regions.json',gallery='pilot/review_gallery.html',
        review_predictions='pilot/review_predictions_coco.json',
        acquisitions=15,camera_frames=60,angles=[0,20],ready_track_region_requests=86,
        alternatives=516,review_predictions_count=86,unreviewed_status=True,
        proposal_units='Observed distal terminal regions, often partial blades or short basal branches; not authenticated complete anatomical leaves',
        independent_verification=verification,coordinate_revision=coordinate,
        paired_prompt_ablation_regions=len(paired),median_best_mask_ablation_iou=float(np.median(paired)),
        frame_and_region_operational_metrics=export['aggregates'],visual_qa_contact_sheets=sheets,
        next_stage=dict(recommendation='Human annotation review with explicit region taxonomy and optional mask extension',
            full_cohort_as_automatic_leaf_training_labels='not_supported_by_this_pilot',
            full_cohort_as_unreviewed_region_review_candidates='technically_feasible_but_does_not_resolve_leaf_extent_or_identity',
            automatically_accepted_training_labels=0,
            reasons=['Distal prompts intentionally avoid shared stem and often cover only part of a visible blade',
                     'Some selected tracks point to short basal regions; terminal support alone does not establish organ class',
                     'Median foreground union coverage is low and additional points do not uniformly expand useful support',
                     'No independent expert region or organ labels were available for accuracy estimation']),
        tests=dict(passed=7,source='test_leaf_proposals.py'),
        inference_signature=read(folder/'configuration.json')['signature'],source_sha256=sha(__file__))
    target=out/'development_findings.json';tmp=target.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2),encoding='utf-8');tmp.replace(target)
    print(json.dumps(dict(status=result['status'],coordinate_revision=coordinate,paired_prompt_regions=len(paired),
         median_ablation_iou=result['median_best_mask_ablation_iou'])),flush=True)

if __name__=='__main__':main()
