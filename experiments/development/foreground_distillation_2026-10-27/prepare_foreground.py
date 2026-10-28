"""Export fixed-quarter whole-plant SAM teacher labels without touching source operators."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import foreground_core as core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo','intake','teacher-run','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    out = args.output; out.mkdir(parents=True,exist_ok=True)
    summary_path = args.teacher_run/'extraction/summary.json'
    summary = json.loads(summary_path.read_text())
    if summary['status'] != 'completed' or summary['reference_samples'] != 360:
        raise ValueError('Completed full-cohort teacher extraction is required')
    table = args.teacher_run/'extraction/view_traits.csv'
    if core.digest(table) != summary['frame_table_sha256']:
        raise ValueError('Teacher frame-table hash mismatch')
    records = core.read_csv(table)
    references = core.read_csv(args.intake/'research_audit/reference_master.csv')
    split_map = core.genotype_splits(references)
    manifest = dict(status='preparing',teacher_table_sha256=core.digest(table),
        teacher_summary_sha256=core.digest(summary_path),genotype_splits=split_map,
        training_views=list(core.QUARTERS),seed=core.SEED,label='plant_foreground',
        teacher='SAM2.1 operationally accepted whole-plant masks; no expert truth',
        polygon_policy='all representable external contours connected by zero-width bridges; holes filled; full-mask roundtrip audited',
        source_sha256={name:core.digest(Path(__file__).parent/name) for name in ('prepare_foreground.py','foreground_core.py')})
    core.write_json(out/'preparation_in_progress.json',manifest)
    images = out/'dataset/images/quarter'; images.mkdir(parents=True,exist_ok=True)
    labels = out/'dataset/labels/quarter'; labels.mkdir(parents=True,exist_ok=True)
    rows = []; seen = set(); start=time.time()
    for r in records:
        if int(r['view']) not in core.QUARTERS:
            continue
        identifier = hashlib.sha256(r['source'].encode()).hexdigest()[:24]
        if identifier in seen:
            raise ValueError('Duplicate quarter source')
        seen.add(identifier)
        source = args.intake/r['source']; mask_path=args.teacher_run/r['mask']
        if core.digest(source) != r['source_sha256'] or core.digest(mask_path) != r['mask_sha256']:
            raise ValueError('Source or teacher-mask bytes changed')
        image_path = images/(identifier+'.png')
        if image_path.exists():
            image = np.array(Image.open(image_path).convert('RGB'))
        else:
            image = core.processing_image(source)
            Image.fromarray(image).save(image_path,compress_level=1)
        mask = np.array(Image.open(mask_path)) > 0
        expected_shape = (int(r['processing_height']),int(r['processing_width']))
        if image.shape[:2] != mask.shape or mask.shape != expected_shape:
            raise ValueError('Processing operator geometry differs from saved teacher')
        accepted = r['qc_pass'].lower()=='true'
        info = dict(external_components=0,holes=0,polygon_vertices=0,omitted_degenerate_contours=0)
        roundtrip = None
        label_path = labels/(identifier+'.txt')
        if accepted:
            polygon,info = core.mask_polygon(mask)
            label_path.write_text('0 '+' '.join(f'{v:.8f}' for v in polygon.ravel())+'\n',encoding='utf8')
            roundtrip = core.mask_metrics(mask,core.raster_polygon(polygon,mask.shape))['iou']
        elif label_path.exists():
            raise ValueError('A teacher-rejected frame has an exported training label')
        rows.append(dict(id=identifier,split=split_map[r['genotype']],
            **{k:r[k] for k in ('treatment','measurement','genotype','replicate','session','camera','view','source','source_sha256','mask','mask_sha256','qc_issues','configuration_signature')},
            teacher_accepted=accepted,image=str(image_path),image_sha256=core.digest(image_path),
            label=str(label_path) if accepted else '',label_sha256=core.digest(label_path) if accepted else '',
            width=mask.shape[1],height=mask.shape[0],teacher_pixels=int(mask.sum()),
            polygon_roundtrip_iou=roundtrip,**info))
        if len(rows)%100==0:
            core.write_json(out/'prepare_progress.json',dict(frames=len(rows),elapsed_seconds=time.time()-start))
            print('prepared',len(rows),flush=True)
    core.write_csv(out/'quarter_frames.csv',rows)
    expected=[]
    indexed={(r['treatment'],str(r['measurement']),r['genotype'],str(r['replicate']),r['camera'],str(r['view'])):r for r in rows}
    for ref in references:
        for camera in ('rgb','noir'):
            for view in core.QUARTERS:
                key=(ref['treatment'],str(ref['measurement']),ref['genotype'],str(ref['replicate']),camera,str(view))
                r=indexed.get(key)
                expected.append(dict(**{k:ref[k] for k in ('treatment','measurement','genotype','replicate')},
                    camera=camera,view=view,split=split_map[ref['genotype']],
                    status=('teacher_accepted' if r['teacher_accepted'] else 'teacher_rejected') if r else 'missing_or_undecodable',
                    id=r['id'] if r else ''))
    core.write_csv(out/'planned_quarter_coverage.csv',expected)
    for mode in ('rgb','joint'):
        for split in ('train','val','test'):
            paths=[r['image'] for r in rows if r['teacher_accepted'] and r['split']==split and (mode=='joint' or r['camera']=='rgb')]
            (out/'dataset'/f'{mode}_{split}.txt').write_text('\n'.join(paths)+'\n',encoding='utf8')
        (out/'dataset'/f'{mode}.yaml').write_text(
            f"path: {str(out/'dataset').replace(chr(92),'/')}\ntrain: {mode}_train.txt\nval: {mode}_val.txt\ntest: {mode}_test.txt\nnames:\n  0: plant_foreground\n",encoding='utf8')
    accepted_roundtrips=[r['polygon_roundtrip_iou'] for r in rows if r['teacher_accepted']]
    manifest.update(status='completed',reference_acquisitions=len(references),planned_slots=len(expected),
        available_quarter_frames=len(rows),teacher_accepted=sum(r['teacher_accepted'] for r in rows),
        teacher_rejected=sum(not r['teacher_accepted'] for r in rows),
        frame_table_sha256=core.digest(out/'quarter_frames.csv'),coverage_table_sha256=core.digest(out/'planned_quarter_coverage.csv'),
        counts={str(k):v for k,v in Counter((r['split'],r['camera'],r['teacher_accepted']) for r in rows).items()},
        polygon_roundtrip_iou=dict(minimum=float(np.min(accepted_roundtrips)),median=float(np.median(accepted_roundtrips)),
            percentile_05=float(np.quantile(accepted_roundtrips,.05)),below_095=sum(x<.95 for x in accepted_roundtrips)),
        elapsed_seconds=time.time()-start)
    core.write_json(out/'preparation.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    main()
