"""Resumable all-angle teacher export for the bounded joint foreground model."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import foreground_core as core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','intake','teacher-run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args(); cv2.setNumThreads(1)
    if not 1<=args.workers<=4:
        raise ValueError('Preparation is limited to1..4CPUworkers')
    output=args.output/'all_views'; output.mkdir(parents=True,exist_ok=True)
    images=output/'dataset/images/all'; images.mkdir(parents=True,exist_ok=True)
    labels=output/'dataset/labels/all'; labels.mkdir(parents=True,exist_ok=True)
    cache=output/'records'; cache.mkdir(parents=True,exist_ok=True)
    quarter=json.loads((args.output/'preparation.json').read_text())
    frames=core.read_csv(args.teacher_run/'extraction/view_traits.csv')
    if core.digest(args.teacher_run/'extraction/view_traits.csv')!=quarter['teacher_table_sha256']:
        raise ValueError('Full teacher frame table changed')
    splits=quarter['genotype_splits']
    operator=dict(kind='all40wholeplant',source_sha256=core.digest(__file__),core_sha256=core.digest(Path(core.__file__)),
        export='same PIL processing1024 decode; aspect-preserving LANCZOS maximum640; JPEG quality95 subsampling0',
        teacher_table_sha256=quarter['teacher_table_sha256'],splits=splits,
        accepted_only_training=True,mask_geometry='original processing1024; normalized polygon coordinates')
    signature=hashlib.sha256(json.dumps(operator,sort_keys=True).encode()).hexdigest()
    spec=output/'operator.json'
    if spec.exists() and json.loads(spec.read_text())['signature']!=signature:
        raise ValueError('All-angle export operator changed; do not mix caches')
    core.write_json(spec,dict(signature=signature,**operator))
    start=time.perf_counter()

    def process(r):
        identifier=hashlib.sha256(r['source'].encode()).hexdigest()[:24]
        stored=cache/(identifier+'.json')
        if stored.exists():
            record=json.loads(stored.read_text())
            if record['signature']!=signature or record['source_sha256']!=r['source_sha256'] or record['mask_sha256']!=r['mask_sha256']:
                raise ValueError('All-angle cache provenance changed')
            if core.digest(record['image'])!=record['image_sha256']:
                raise ValueError('All-angle image cache changed')
            return record
        source=args.intake/r['source']; mask_path=args.teacher_run/r['mask']
        if core.digest(source)!=r['source_sha256'] or core.digest(mask_path)!=r['mask_sha256']:
            raise ValueError('Raw source or SAM teacher-mask digest mismatch')
        rgb=core.processing_image(source); mask=np.array(Image.open(mask_path))>0
        if rgb.shape[:2]!=mask.shape:
            raise ValueError('Teacher processing geometry mismatch')
        image=Image.fromarray(rgb); image.thumbnail((640,640),Image.Resampling.LANCZOS)
        path=images/(identifier+'.jpg'); image.save(path,quality=95,subsampling=0)
        accepted=r['qc_pass'].lower()=='true'
        label=labels/(identifier+'.txt'); roundtrip=None
        if accepted:
            polygon,_=core.mask_polygon(mask)
            label.write_text('0 '+' '.join(f'{v:.8f}' for v in polygon.ravel())+'\n',encoding='utf8')
            roundtrip=core.mask_metrics(mask,core.raster_polygon(polygon,mask.shape))['iou']
        record=dict(id=identifier,signature=signature,split=splits[r['genotype']],
            **{k:r[k] for k in ('treatment','measurement','genotype','replicate','session','camera','view','source','source_sha256','mask','mask_sha256','qc_issues','configuration_signature')},
            teacher_accepted=accepted,image=str(path),image_sha256=core.digest(path),
            label=str(label) if accepted else '',label_sha256=core.digest(label) if accepted else '',
            width=mask.shape[1],height=mask.shape[0],export_width=image.width,export_height=image.height,
            teacher_pixels=int(mask.sum()),polygon_roundtrip_iou=roundtrip)
        core.write_json(stored,record)
        return record

    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(process,frames):
            rows.append(record)
            if len(rows)%250==0:
                core.write_json(output/'progress.json',dict(status='running',frames=len(rows),total=len(frames),elapsed_seconds=time.perf_counter()-start))
                print('all-angle prepared',len(rows),'of',len(frames),flush=True)
    core.write_csv(output/'frame_inventory.csv',rows)
    for split in ('train','val','test'):
        paths=[r['image'] for r in rows if r['split']==split and r['teacher_accepted']]
        (output/'dataset'/f'joint_{split}.txt').write_text('\n'.join(paths)+'\n',encoding='utf8')
    (output/'dataset/joint.yaml').write_text(f'path: {(output/"dataset").as_posix()}\ntrain: joint_train.txt\nval: joint_val.txt\ntest: joint_test.txt\nnames:\n  0: plant_foreground\n',encoding='utf8')
    summary=dict(status='completed',frames=len(rows),teacher_accepted=sum(r['teacher_accepted'] for r in rows),
        teacher_rejected=sum(not r['teacher_accepted'] for r in rows),genotype_splits=splits,
        frame_table_sha256=core.digest(output/'frame_inventory.csv'),operator_signature=signature,
        counts={str(k):v for k,v in Counter((r['split'],r['camera'],r['teacher_accepted']) for r in rows).items()},
        elapsed_seconds=time.perf_counter()-start)
    core.write_json(output/'preparation.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
