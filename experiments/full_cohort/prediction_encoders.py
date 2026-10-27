"""Stream frozen local DINOv2-small quarter-view features from accepted masks."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageOps
import prediction as p

QUARTERS = (0,10,20,30)
REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'


def make_crop(image,mask,variant='recorded'):
    if image.shape[:2] != mask.shape:
        raise ValueError('Mask/image dimensions differ')
    yy,xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError('Empty foreground mask')
    masked = image.copy()
    masked[~mask] = 0
    if variant == 'blue_triplicate':
        masked = np.repeat(masked[:,:,2:3],3,axis=2)
    elif variant != 'recorded':
        raise ValueError('Unknown camera ablation')
    # At least one processing pixel and 5% of each bbox extent on either side.
    px = max(1,int(np.ceil((xx.max()-xx.min()+1)*.05)))
    py = max(1,int(np.ceil((yy.max()-yy.min()+1)*.05)))
    box = [max(0,int(xx.min())-px), max(0,int(yy.min())-py),
           min(image.shape[1],int(xx.max())+1+px),
           min(image.shape[0],int(yy.max())+1+py)]
    crop = Image.fromarray(masked).crop(box)
    return ImageOps.pad(crop,(224,224),method=Image.Resampling.BICUBIC,
                        color=(0,0,0),centering=(.5,.5)),box


def aggregate(rows,records):
    groups = defaultdict(list)
    for record in records:
        camera = record['camera']
        name = camera if record['variant']=='recorded' else 'noir_blue'
        groups[(p.key(record),name)].append(record['feature'])
    arrays = {name:np.full((len(rows),384),np.nan,np.float32)
              for name in ('rgb','noir','noir_blue')}
    counts = {name:np.zeros(len(rows),int) for name in arrays}
    for i,row in enumerate(rows):
        for name in arrays:
            values = groups.get((p.key(row),name),[])
            counts[name][i] = len(values)
            if values:
                arrays[name][i] = np.mean(values,axis=0)
    return arrays,counts


def load_processing_image(path):
    # Same decoded/downsampled operator as released traits.load_image(width=1024).
    with Image.open(path) as source:
        source.draft('RGB',(1024,1024))
        source.load()
        image = source.convert('RGB')
        image.thumbnail((1024,1024))
    return np.array(image)


def atomic_npz(path,**values):
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**values)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('intake','run','repo'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--batch-size',type=int,default=4)
    args = parser.parse_args()
    if args.batch_size not in range(1,5):
        raise ValueError('GPU sharing requires batch size1..4')
    dest = args.run/'prediction'/'encoder'
    cache = dest/'views'
    errors = dest/'errors'
    for directory in (dest,cache,errors):
        directory.mkdir(parents=True,exist_ok=True)
    references = p.read_keyed(args.intake/'research_audit'/'reference_master.csv')
    if len(references)!=360:
        raise ValueError('Expected all360 reference records')
    assets = {name:p.digest(args.model/name) for name in
              ('config.json','preprocessor_config.json','model.safetensors')}
    operator = dict(source_sha256=p.digest(__file__),assets=assets,revision=REVISION,
        quarter_views=QUARTERS,preprocessing='released1024decode; accepted mask; black '
        'background; bbox plus max(1pixel,ceil5%extent) each side clipped to image; '
        'aspect-preserving bicubic224square black padding; ImageNet normalization',
        feature='384-D L2-normalized CLS; unweighted arithmetic view mean; no renormalization',
        variants=['rgb_recorded','noir_recorded','noir_blue_triplicate'],
        torch_dtype='float32',augmentation='none',device='cuda:0',
        per_process_memory_fraction=.12,cpu_threads=2,batch_size=args.batch_size,
        limits=['Frozen RGB-pretrained generic visual features are not physiological measurements.',
        'Recorded NoIR and blue-triplicate images are outside normal RGB pretraining.',
        'Accepted automatic masks are not independent accuracy references.',
        'Only fixed quarter views are encoded; this is not an all-view geometric reconstruction.'])
    signature = hashlib.sha256(json.dumps(operator,sort_keys=True).encode()).hexdigest()
    manifest = dest/'operator.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['signature'] != signature:
            raise ValueError('Encoder operator/assets changed; cannot mix caches')
    else:
        p.save_json(manifest,dict(signature=signature,**operator))
        (dest/'execution_source.py').write_bytes(Path(__file__).read_bytes())
    import torch
    from transformers import AutoImageProcessor,AutoModel
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.cuda.set_per_process_memory_fraction(.12,device=0)
    torch.manual_seed(20261027)
    processor = AutoImageProcessor.from_pretrained(args.model,use_fast=False,local_files_only=True)
    model,loading = AutoModel.from_pretrained(args.model,local_files_only=True,
                        output_loading_info=True,attn_implementation='eager')
    if any(loading[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError(str(loading))
    model = model.to('cuda:0').eval()
    model.requires_grad_(False)
    seen = {}
    cached = {}
    for path in cache.glob('*.npz'):
        with np.load(path) as data:
            meta = json.loads(str(data['metadata']))
            if meta['operator_signature'] != signature:
                raise ValueError('Incompatible view encoder cache')
            cached[(meta['source'],meta['variant'])] = dict(meta,feature=data['feature'].copy())
    started = time.time()
    batches = []

    def flush():
        nonlocal batches
        if not batches:
            return
        inputs = processor(images=[v[0] for v in batches],do_resize=False,
                           do_center_crop=False,return_tensors='pt').to('cuda:0')
        with torch.inference_mode():
            values = torch.nn.functional.normalize(
                model(**inputs).last_hidden_state[:,0,:],dim=1).cpu().numpy()
        for (_,meta),feature in zip(batches,values):
            if feature.shape != (384,) or not np.isfinite(feature).all():
                raise ValueError('Invalid frozen feature')
            cache_id = hashlib.sha256((meta['source']+'|'+meta['variant']).encode()).hexdigest()[:24]
            atomic_npz(cache/(cache_id+'.npz'),feature=feature,metadata=np.array(json.dumps(meta)))
            cached[(meta['source'],meta['variant'])] = dict(meta,feature=feature)
        batches = []
        del inputs
        torch.cuda.empty_cache()

    while True:
        paths = list((args.run/'extraction'/'records').glob('*.json'))
        for path in paths:
            if path.name in seen:
                continue
            record = json.loads(path.read_text(encoding='utf-8'))
            seen[path.name] = record
            if int(record['view']) not in QUARTERS or not p.truth(record['qc_pass']):
                continue
            variants = ['recorded'] + (['blue_triplicate'] if record['camera']=='noir' else [])
            record_hash = p.digest(path)
            pending = []
            for variant in variants:
                old = cached.get((record['source'],variant))
                if old is not None:
                    if old['record_sha256']!=record_hash:
                        raise ValueError('Segmentation record changed beneath cached encoder')
                else:
                    pending.append(variant)
            if not pending:
                continue
            try:
                source = args.intake/record['source']
                maskpath = args.run/record['mask']
                if p.digest(source)!=record['source_sha256'] or p.digest(maskpath)!=record['mask_sha256']:
                    raise ValueError('Source or mask hash mismatch')
                image = load_processing_image(source)
                with Image.open(maskpath) as maskimage:
                    mask = np.array(maskimage)>0
                if mask.ndim != 2:
                    raise ValueError('Expected binary single-channel mask')
                for variant in pending:
                    crop,box = make_crop(image,mask,variant)
                    meta = {k:record[k] for k in (*p.KEYS,'session','camera','view','source',
                            'source_sha256','mask','mask_sha256','configuration_signature')}
                    meta.update(variant=variant,record_sha256=record_hash,
                                operator_signature=signature,crop_xyxy=box)
                    batches.append((crop,meta))
                    if len(batches)>=args.batch_size:
                        flush()
            except Exception as exc:
                if isinstance(exc,RuntimeError) or 'hash mismatch' in str(exc):
                    raise
                p.save_json(errors/path.name,dict(source=record['source'],
                    record_sha256=record_hash,error_type=type(exc).__name__,error=str(exc)))
        flush()
        arrays,counts = aggregate(references,cached.values())
        atomic_npz(dest/'session_embeddings.npz',keys=np.array([p.key(r) for r in references]),
            **arrays,**{k+'_count':v for k,v in counts.items()},signature=np.array(signature))
        p.save_csv(dest/'availability.csv',[dict(**{k:r[k] for k in p.KEYS},
            **{camera+'_accepted_quarter_views':int(counts[camera][i]) for camera in counts})
            for i,r in enumerate(references)])
        quarter_rows = []
        for record in seen.values():
            if int(record['view']) in QUARTERS:
                quarter_rows.append({k:record.get(k,'') for k in
                    (*p.KEYS,'session','camera','view','source','qc_pass','qc_issues')})
        p.save_csv(dest/'quarter_view_qc.csv',quarter_rows)
        config = json.loads((args.run/'extraction'/'configuration_shard_0.json').read_text())
        final = len(paths)>=config['tasks'] and (args.run/'extraction'/'session_traits.csv').exists()
        status = dict(status='completed' if final else 'streaming',records_seen=len(seen),
            expected_records=config['tasks'],encoded_views=len(cached),
            encoder_errors=len(list(errors.glob('*.json'))),
            elapsed_seconds=round(time.time()-started,1),
            max_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
            max_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
            sessions_with_any={name:int((counts[name]>0).sum()) for name in counts},
            sessions_with_four={name:int((counts[name]==4).sum()) for name in counts},
            operator_signature=signature)
        p.save_json(dest/'progress.json',status)
        print(json.dumps(status),flush=True)
        if final or not args.watch:
            break
        time.sleep(20)


if __name__=='__main__':
    main()
