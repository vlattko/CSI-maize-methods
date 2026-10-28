"""Bounded independent check of stricter affine frame support after mask erosion."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import validity_audit as a

def load_mask(binding,root):
    row=binding['record'];path=a.contained(root,row['mask'])
    if a.sha(path)!=row['mask_sha256'] or a.sha(path)!=binding['actual_mask_sha256']:raise ValueError('Frozen mask digest mismatch')
    with Image.open(path) as im:return np.asarray(im.convert('L').resize((512,385),Image.Resampling.NEAREST))>0

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','repo','intake','run','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--limit',type=int,default=12);args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a new field-audit path')
    c,s,_=ops=a.configure(args.repo);bindings=json.loads((args.audit/'source_bindings.json').read_text());verified={};rows=[]
    for name,binding in list(bindings.items())[:args.limit]:
        record=json.loads((args.run/'pairing/full/records'/name).read_text());nominal=next(e for e in record['evaluations'] if e['pair_kind']=='nominal' and e['method']=='ECC')
        rm=load_mask(binding['rgb'],args.run);nm=load_mask(binding['noir'],args.run);matrix=np.asarray(nominal['source_to_target'],np.float32)
        ones,old_field=c.warp_to_rgb(np.ones(nm.shape,np.float32),matrix,rm.shape);strict_field=ones>=a.STRICT_MINIMUM
        wn,_=c.warp_to_rgb(c.interior(nm).astype(np.uint8),matrix,rm.shape,True)
        old_noir=(wn>0)&old_field;new_noir=(wn>0)&strict_field
        rows.append(dict(record_file=name,field_pixels_lost=int((old_field&~strict_field).sum()),noir_interior_pixels_lost=int((old_noir&~new_noir).sum()),common_interior_pixels_lost=int((old_noir&~new_noir&c.interior(rm)).sum())))
    output=dict(status='completed',anchors=len(rows),selection=f'First {args.limit} accepted anchors in deterministic filename order; all accepted anchors when limit covers the input',operator_sha256={k:a.sha(ops[2]/k) for k in ('channel_fusion_core.py','spatial_fusion_core.py')},field_pixels_lost=sum(r['field_pixels_lost'] for r in rows),noir_interior_pixels_lost=sum(r['noir_interior_pixels_lost'] for r in rows),common_interior_pixels_lost=sum(r['common_interior_pixels_lost'] for r in rows),rows=rows)
    a.dump(args.output,output);print(json.dumps({k:v for k,v in output.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
