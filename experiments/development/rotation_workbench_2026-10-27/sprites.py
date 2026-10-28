"""Generate hash-bound all-angle original and alpha-mask inspection sprites.

Every camera/sample has forty fixed positions. Missing views never shift later
indices. Reduced inspection pixels are separate from quantitative trait data.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import os
import sys
import time
from PIL import Image,ImageDraw

KEYS=('treatment','measurement','genotype','replicate')


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    for attempt in range(30):
        try:temporary.replace(path);return
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (5,32,33) or attempt==29:raise
            time.sleep(min(.05*(attempt+1),1.))


def tile_box(view,width,height):
    if not 0<=view<40:raise ValueError('Expected a fixed index in0..39')
    x=(view%8)*width;y=(view//8)*height
    return x,y,x+width,y+height


def mask_alpha(mask):
    mask=mask.convert('L')
    result=Image.new('LA',mask.size,(255,0));result.putalpha(mask);return result


def slot_rows(rows):
    slots=[None]*40
    for row in rows:
        view=int(row['view'])
        if not 0<=view<40 or slots[view] is not None:raise ValueError('Duplicate or invalid view index')
        slots[view]=row
    return slots


def contained(root,relative):
    path=(root/relative).resolve()
    if path==root.resolve() or not path.is_relative_to(root.resolve()):raise ValueError('Source path escapes its declared root')
    return path


def render_slot(task,*,raw,run,out,width,height,signature):
    key,camera,rows=task;identity=hashlib.sha256(json.dumps([*key,camera]).encode()).hexdigest()[:24]
    cache=out/'sprites/records'/(identity+'.json')
    digest=hashlib.sha256(json.dumps([(r.get('source'),r.get('source_sha256'),r.get('mask'),r.get('mask_sha256'),r['view'],r.get('qc_pass')) for r in rows],sort_keys=True).encode()).hexdigest()
    if cache.exists():
        stored=json.loads(cache.read_text(encoding='utf-8'))
        if stored['signature']!=signature or stored['input_digest']!=digest:raise ValueError('Changed sprite source/configuration; use a new output')
        for name in ('original','mask'):
            if sha(out/stored[name])!=stored[name+'_sha256']:raise ValueError('Cached inspection asset changed')
        return stored
    original=Image.new('RGB',(width*8,height*5),'black')
    masks=Image.new('LA',(width*8,height*5),(255,0));frames=[]
    for view,row in enumerate(slot_rows(rows)):
        box=tile_box(view,width,height)
        meta=dict(view=view,present=row is not None,qc_pass=False,qc_issues='source_missing_or_undecodable',source='',source_sha256='',mask_path='',mask_sha256='')
        if row is not None:
            source=contained(raw,row['source']);blob=source.read_bytes()
            if hashlib.sha256(blob).hexdigest()!=row['source_sha256']:raise ValueError('Original image changed: '+row['source'])
            with Image.open(io.BytesIO(blob)) as im:
                im.draft('RGB',(1024,1024));im.load();im=im.convert('RGB');im.thumbnail((1024,1024))
                expected=(int(row['processing_width']),int(row['processing_height']))
                if im.size!=expected:raise ValueError('Processing canvas does not match recorded source')
                original.paste(im.resize((width,height),Image.Resampling.LANCZOS),box[:2])
            meta.update(qc_pass=row['qc_pass'].lower()=='true',qc_issues=row.get('qc_issues',''),source=row['source'],source_sha256=row['source_sha256'],mask_path=row.get('mask',''),mask_sha256=row.get('mask_sha256',''),processing_width=int(row['processing_width']),processing_height=int(row['processing_height']))
            for field in ('area_fraction','height_fraction','width_fraction','bottom_censored','top_censored','side_censored'):
                value=row.get(field)
                meta[field]=float(value) if value not in ('',None) else None
            if row.get('mask'):
                path=contained(run,row['mask'])
                if sha(path)!=row['mask_sha256']:raise ValueError('Recorded mask changed: '+row['mask'])
                with Image.open(path) as mask:
                    if mask.size!=expected:raise ValueError('Mask canvas differs from source')
                    masks.paste(mask_alpha(mask.resize((width,height),Image.Resampling.NEAREST)),box[:2])
        else:
            draw=ImageDraw.Draw(original);draw.text((box[0]+12,box[1]+height//2),'Missing index '+str(view),fill='#aaaaaa')
        frames.append(meta)
    image_path=out/'sprites'/(identity+'.webp');mask_path=out/'sprites'/(identity+'_mask.png')
    original.save(image_path,format='WEBP',quality=78,method=4)
    masks.save(mask_path,format='PNG',optimize=True)
    value=dict(id=identity,key=dict(zip(KEYS,key)),camera=camera,signature=signature,input_digest=digest,
               original=image_path.relative_to(out).as_posix(),mask=mask_path.relative_to(out).as_posix(),
               original_sha256=sha(image_path),mask_sha256=sha(mask_path),frames=frames)
    atomic_json(cache,value);return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('intake','run','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--width',type=int,default=384);parser.add_argument('--height',type=int,default=288)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--max-slots',type=int)
    args=parser.parse_args();intake=args.intake.resolve();run=args.run.resolve();out=args.output.resolve()
    raw=Path(json.loads((intake/'intake_summary.json').read_text(encoding='utf-8'))['raw_root']).resolve()
    if out.is_relative_to(raw) or out.is_relative_to(run) or out.is_relative_to(intake):raise ValueError('Derived assets need a separate output folder')
    summary=json.loads((run/'extraction/summary.json').read_text(encoding='utf-8'))
    if summary['status']!='completed':raise ValueError('Final extraction is required')
    table=run/'extraction/view_traits.csv'
    if sha(table)!=summary['frame_table_sha256']:raise ValueError('Final frame table changed')
    references=read(intake/'research_audit/reference_master.csv');frames=read(table)
    reference_keys=[tuple(r[k] for k in KEYS) for r in references]
    if len(set(reference_keys))!=360:raise ValueError('Expected360unique reference identities')
    groups=defaultdict(list)
    for row in frames:groups[(tuple(row[k] for k in KEYS),row['camera'])].append(row)
    tasks=[(key,camera,groups.get((key,camera),[])) for key in reference_keys for camera in ('rgb','noir')]
    if args.max_slots:tasks=tasks[:args.max_slots]
    config=dict(frame_table_sha256=sha(table),reference_sha256=sha(intake/'research_audit/reference_master.csv'),
        script_sha256=sha(__file__),frame_width=args.width,frame_height=args.height,columns=8,rows=5,
        preview_format='WebPquality78; nearest-neighbour lossless alpha masks',sampled_slots=len(tasks),raw=str(raw))
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    (out/'sprites/records').mkdir(parents=True,exist_ok=True)
    saved=out/'sprite_configuration.json'
    if saved.exists() and json.loads(saved.read_text(encoding='utf-8'))!=config:raise ValueError('Changed producer/input; use a new output')
    atomic_json(saved,config);(out/'sprites_execution_source.py').write_bytes(Path(__file__).read_bytes())
    start=time.time();completed=[];pool=ThreadPoolExecutor(max_workers=args.workers)
    futures=[pool.submit(render_slot,t,raw=raw,run=run,out=out,width=args.width,height=args.height,signature=signature) for t in tasks]
    try:
        for future in as_completed(futures):
            completed.append(future.result())
            if len(completed)%10==0 or len(completed)==len(tasks):
                progress=dict(status='running',camera_slots=len(completed),expected_slots=len(tasks),source_frames=sum(f['present'] for s in completed for f in s['frames']),elapsed_seconds=round(time.time()-start,1))
                atomic_json(out/'sprites_progress.json',progress);print(json.dumps(progress),flush=True)
    except Exception as error:
        for future in futures:future.cancel()
        atomic_json(out/'sprites_progress.json',dict(status='failed',camera_slots=len(completed),error=repr(error)))
        raise
    finally:pool.shutdown(wait=True,cancel_futures=True)
    completed.sort(key=lambda s:([s['key'][k] for k in KEYS],s['camera']))
    manifest=dict(status='completed',signature=signature,frame_width=args.width,frame_height=args.height,columns=8,rows=5,reference_rows=len(references),camera_slots=len(completed),source_frames=sum(f['present'] for s in completed for f in s['frames']),slots=completed,elapsed_seconds=time.time()-start)
    atomic_json(out/'sprites/sprite_manifest.json',manifest)
    atomic_json(out/'sprites_progress.json',{k:v for k,v in manifest.items() if k!='slots'})
    print(json.dumps({k:v for k,v in manifest.items() if k!='slots'},indent=2),flush=True)


if __name__=='__main__':main()
