"""Review gallery and COCO-style exports; no anatomical training labels."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import base64
import csv
import html
import json
import shutil
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from leaf_plan import read,sha,digest,atomic
from leaf_proposals import coco_rle

COLORS=[(40,210,120),(240,130,30),(90,160,250),(235,70,160)]

def mask_at(folder,proposal):
    with Image.open(folder/proposal['clipped_mask']) as im:return np.asarray(im)>0

def selected(proposals,mode):
    result=[]
    for candidate in sorted({r['candidate_id'] for r in proposals if r['mode']==mode}):
        choices=[r for r in proposals if r['candidate_id']==candidate and r['mode']==mode]
        result.append(max(choices,key=lambda r:((r['model_score'] if r['model_score'] is not None else -1),-r['alternative'])))
    return result

def frame_metrics(folder,record,parent):
    rows=[];pn=int(parent.sum())
    for mode in ('box_only','distal_points_negatives'):
        choices=selected(record['proposals'],mode);masks=[mask_at(folder,p) for p in choices]
        count=np.sum(masks,axis=0) if masks else np.zeros(parent.shape,int)
        covered=int(((count>0)&parent).sum());multiple=int(((count>1)&parent).sum())
        pairwise=[]
        for i in range(len(masks)):
            for j in range(i):
                union=(masks[i]|masks[j]).sum();pairwise.append(float((masks[i]&masks[j]).sum()/union) if union else 0.)
        rows.append(dict(**record['identity'],camera=record['camera'],view=record['view'],frame_id=record['frame_id'],
            mode=mode,selected_region_count=len(choices),parent_area_pixels=pn,
            covered_parent_pixels=covered,uncovered_parent_pixels=pn-covered,
            parent_coverage_fraction=covered/pn if pn else None,
            multiply_proposed_parent_pixels=multiple,parent_overlap_fraction=multiple/pn if pn else None,
            median_pairwise_region_iou=float(np.median(pairwise)) if pairwise else None,
            empty_selected_regions=sum(p['empty_clipped'] for p in choices),
            whole_parent_like_regions=sum(p['parent_coverage_fraction']>=.8 for p in choices),
            metrics_are_anatomical_accuracy=False))
    return rows

def gallery_card(folder,record,metrics,parent):
    if not record.get('input'):return None
    with Image.open(folder/record['input']['review_image']) as im:image=np.array(im.convert('RGB'))
    w,h=image.shape[1],image.shape[0];panels=[]
    colors={candidate:COLORS[i%len(COLORS)] for i,candidate in enumerate(sorted({x['candidate_id'] for x in record.get('planned_proposals',[])}))}
    original=Image.fromarray(image);draw=ImageDraw.Draw(original)
    for n,candidate in enumerate(record.get('planned_proposals',[])):
        color=colors[candidate['candidate_id']];prompt=candidate['prompt_processing']
        draw.rectangle(prompt['box'],outline=color,width=3)
        for (x,y),label in zip(prompt['points'],prompt['labels']):
            if label:draw.ellipse((x-5,y-5,x+5,y+5),fill=color)
            else:
                draw.line((x-5,y-5,x+5,y+5),fill=(255,50,50),width=3)
                draw.line((x-5,y+5,x+5,y-5),fill=(255,50,50),width=3)
    panels.append(original)
    for mode in ('box_only','distal_points_negatives'):
        for kind in ('raw','clipped'):
            canvas=image.copy()
            for proposal in selected(record['proposals'],mode):
                with Image.open(folder/proposal[kind+'_mask']) as im:mask=np.array(im)>0
                color=np.array(colors[proposal['candidate_id']])
                canvas[mask]=(canvas[mask]*.35+color*.65).astype(np.uint8)
                if kind=='raw':canvas[mask&~parent]=(240,45,45)
            panels.append(Image.fromarray(canvas))
    count=np.zeros((h,w),int)
    for proposal in selected(record['proposals'],'distal_points_negatives'):count+=mask_at(folder,proposal)
    coverage=np.zeros_like(image);coverage[parent]=(65,65,65);coverage[count==1]=(45,210,135);coverage[count>1]=(255,255,255)
    panels.append(Image.fromarray(coverage))
    scale=.5;pw=round(w*scale);ph=round(h*scale)
    canvas=Image.new('RGB',(pw*3,(ph+64)*2),(0,0,0));draw=ImageDraw.Draw(canvas)
    titles=['Local box and distal prompts','Box raw (outside parent red)','Box parent-clipped',
            'Points raw (outside parent red)','Points parent-clipped','Points coverage: gray gaps / white overlap']
    for i,panel in enumerate(panels):
        col=i%3;row=i//3;oy=row*(ph+64)
        canvas.paste(panel.resize((pw,ph),Image.Resampling.LANCZOS),(pw*col,oy+64))
        draw.text((pw*col+8,oy+8),titles[i],fill='white')
        if i in (2,4):
            m=metrics[0 if i==2 else 1];draw.text((pw*col+8,oy+27),f"parent coverage {m['parent_coverage_fraction']:.3f}; overlap {m['parent_overlap_fraction']:.3f}",fill='white')
    path=folder/'gallery'/f"{record['frame_id']}.png";path.parent.mkdir(exist_ok=True);canvas.save(path)
    return path

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full-run',type=Path,required=True)
    args=parser.parse_args();folder=args.output/'pilot';config=read(folder/'configuration.json');records=[]
    for path in sorted((folder/'records').glob('*.json')):
        record=read(path);expected=record.pop('payload_sha256')
        if digest(record)!=expected or record['signature']!=config['signature']:raise ValueError('record_digest_or_configuration_mismatch')
        records.append(record)
    if len(records)!=60:raise ValueError('expected_all60_planned_frames')
    regions=[];metrics=[];frames=[];coco=dict(info=dict(description='Unreviewed track-supported region predictions; not anatomical ground truth'),
        images=[],annotations=[],categories=[dict(id=1,name='unreviewed_terminal_region')])
    cards=[]
    for record in sorted(records,key=lambda r:(r['identity']['treatment'],int(r['identity']['measurement']),r['camera'],r['view'])):
        identity=record['identity']
        for proposal in record['proposals']:
            for kind in ('raw','clipped'):
                if sha(folder/proposal[kind+'_mask'])!=proposal[kind+'_mask_sha256']:raise ValueError('proposal_mask_changed')
            regions.append({**identity,'frame_id':record['frame_id'],'camera':record['camera'],'view':record['view'],
                           'source':record['source_record']['source'],'source_sha256':record['source_record']['source_sha256'],**proposal})
        if record.get('input'):
            with Image.open(args.full_run/record['source_record']['mask']) as im:parent=np.asarray(im)>0
            fm=frame_metrics(folder,record,parent);metrics.extend(fm);card=gallery_card(folder,record,fm,parent)
            frame_index=dict(**identity,frame_id=record['frame_id'],camera=record['camera'],view=record['view'],
                image=record['input']['review_image'],width=record['input']['prepared_size'][0],height=record['input']['prepared_size'][1],
                source_sha256=record['source_record']['source_sha256'],parent_mask=record['source_record']['mask'],
                parent_mask_sha256=record['source_record']['mask_sha256'],status=record['status'],
                selected_region_ids={mode:[p['proposal_id'] for p in selected(record['proposals'],mode)] for mode in ('box_only','distal_points_negatives')},
                gallery=card.relative_to(folder).as_posix() if card else None,metrics=fm)
            frames.append(frame_index)
            image_id=len(coco['images'])+1
            coco['images'].append(dict(id=image_id,file_name=record['input']['review_image'],width=frame_index['width'],height=frame_index['height'],
                                      source_sha256=frame_index['source_sha256'],**identity,camera=record['camera'],view=record['view']))
            for proposal in selected(record['proposals'],'distal_points_negatives'):
                mask=mask_at(folder,proposal);yy,xx=np.where(mask)
                if not len(xx):continue
                coco['annotations'].append(dict(id=len(coco['annotations'])+1,image_id=image_id,category_id=1,
                    segmentation=coco_rle(mask),bbox=[int(xx.min()),int(yy.min()),int(np.ptp(xx)+1),int(np.ptp(yy)+1)],
                    area=int(mask.sum()),iscrowd=0,score=proposal['model_score'],candidate_id=proposal['candidate_id'],
                    proposal_id=proposal['proposal_id'],review_required=True,anatomical_identity_verified=False))
            if card:
                label=f"{identity['treatment']} visit {identity['measurement']} / {identity['genotype']} replicate {identity['replicate']} / {record['camera']} index {record['view']}"
                encoded=base64.b64encode(card.read_bytes()).decode()
                alternatives=[]
                for proposal in record['proposals']:
                    thumb=np.zeros_like(np.asarray(Image.open(folder/record['input']['review_image']).convert('RGB')))
                    raw=np.array(Image.open(folder/proposal['raw_mask']))>0
                    clipped=mask_at(folder,proposal);thumb[clipped]=(45,210,135);thumb[raw&~parent]=(240,45,45)
                    import io
                    buffer=io.BytesIO();Image.fromarray(thumb).resize((256,192),Image.Resampling.NEAREST).save(buffer,format='PNG')
                    encoded_alt=base64.b64encode(buffer.getvalue()).decode()
                    caption=f"{proposal['candidate_id']} / {proposal['mode']} alt {proposal['alternative']}; score {proposal['model_score']:.3f}; parent fraction {proposal['parent_coverage_fraction']:.3f}"
                    alternatives.append(f"<figure style='display:inline-block;width:260px;margin:6px;vertical-align:top'><img src='data:image/png;base64,{encoded_alt}'><figcaption>{html.escape(caption)}</figcaption></figure>")
                detail=f"<details><summary>Review all {len(record['proposals'])} alternatives: green clipped region, red outside-parent prediction</summary>{''.join(alternatives)}</details>"
                cards.append(f"<article data-camera='{record['camera']}'><h2>{html.escape(label)}</h2><p>{record['status']}; {len(record['proposals'])} multimask alternatives. Unreviewed regions.</p><img loading='lazy' src='data:image/png;base64,{encoded}'>{detail}</article>")
    atomic(folder/'regions.json',dict(schema=1,status='completed',review_status='unreviewed_regions_only',configuration_signature=config['signature'],regions=regions))
    atomic(folder/'frame_index.json',dict(schema=1,status='completed',configuration_signature=config['signature'],frames=frames))
    atomic(folder/'review_predictions_coco.json',coco)
    write_csv(folder/'regions.csv',regions);write_csv(folder/'frame_overlap_coverage.csv',metrics)
    aggregate={}
    for camera in ('rgb','noir'):
        for mode in ('box_only','distal_points_negatives'):
            rr=[r for r in regions if r['camera']==camera and r['mode']==mode]
            best=[r for record in records if record['camera']==camera for r in selected(record['proposals'],mode)]
            mm=[r for r in metrics if r['camera']==camera and r['mode']==mode]
            aggregate[camera+'__'+mode]=dict(alternatives=len(rr),selected_regions=len(best),frames_with_metrics=len(mm),
                empty_selected=sum(r['empty_clipped'] for r in best),whole_parent_like_selected=sum(r['parent_coverage_fraction']>=.8 for r in best),
                median_region_parent_fraction=float(np.median([r['parent_coverage_fraction'] for r in best])) if best else None,
                median_selected_positive_hit_fraction=float(np.median([r['positive_hit_fraction'] for r in best])) if best else None,
                median_selected_outside_parent_fraction=float(np.median([r['raw_outside_parent_fraction'] for r in best if r['raw_outside_parent_fraction'] is not None])) if best else None,
                median_frame_parent_coverage=float(np.median([r['parent_coverage_fraction'] for r in mm])) if mm else None,
                median_frame_parent_overlap=float(np.median([r['parent_overlap_fraction'] for r in mm])) if mm else None)
    document=f"""<!doctype html><meta charset='utf-8'><title>Track-supported region proposal review</title><style>body{{background:#171a1e;color:#eee;font:16px Arial;margin:24px}}h1{{font-size:27px}}h2{{font-size:17px}}article{{background:#252a30;margin:18px 0;padding:14px;border-radius:8px}}img{{width:100%;height:auto}}button{{padding:8px 18px;margin:6px}}p{{line-height:1.5;color:#ccc}}</style><h1>Track-supported region proposal review</h1><p>60 planned RGB/NoIR camera frames from15 treatment-by-visit strata. Distal path prompts compare a local box with positive/negative point prompts. Every region is unreviewed; track IDs and model scores do not authenticate leaves. Black backgrounds show the accepted parent foreground. Red crosses mark negative prompts. No expert anatomical accuracy is estimated.</p><button onclick=\"filter('all')\">All</button><button onclick=\"filter('rgb')\">RGB</button><button onclick=\"filter('noir')\">NoIR</button>{''.join(cards)}<script>function filter(c){{document.querySelectorAll('article').forEach(x=>x.hidden=c!='all'&&x.dataset.camera!=c)}}</script>"""
    (folder/'review_gallery.html').write_text(document,encoding='utf-8')
    source=Path(__file__);cached=folder/'postprocessing_source'/sha(source);cached.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,cached/source.name)
    atomic(folder/'review_export_manifest.json',dict(status='completed',configuration_signature=config['signature'],
        frames=len(frames),all_multimask_regions=len(regions),review_coco_annotations=len(coco['annotations']),
        source_sha256=sha(source),aggregates=aggregate,frame_status=dict(Counter(r['status'] for r in records)),
        outputs={p.name:sha(p) for p in [folder/'regions.json',folder/'regions.csv',folder/'frame_index.json',
                   folder/'review_predictions_coco.json',folder/'frame_overlap_coverage.csv',folder/'review_gallery.html']},
        anatomical_ground_truth=False,automatically_accepted_training_labels=0))
    print(json.dumps(dict(status='completed',frames=len(frames),regions=len(regions),review_annotations=len(coco['annotations']),aggregates=aggregate)),flush=True)

if __name__=='__main__':main()
