"""Read-only comparison of two interpolation-validity tolerances on frozen pairings."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,json,math,shutil,sys,time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
cv2.setNumThreads(2)

STRICT_MINIMUM=1-1e-7
STAT_FIELDS=('eligible_pixels','valid_pixels','qualified','valid_fraction','median','q10','q25','q75','q90','iqr')

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def dump(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def table(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,keys);writer.writeheader();writer.writerows(rows)

def contained(root,name):
    root=Path(root).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Input path escapes declared root')
    return path

def configure(repo):
    folder=Path(repo).resolve()/'src/csi_maize/_methods';sys.path.insert(0,str(folder))
    import channel_fusion_core as c
    import spatial_fusion_core as s
    for module in (c,s):
        if Path(module.__file__).resolve().parent!=folder:raise ValueError('Unexpected scientific import')
    return c,s,folder

def compare_pair(rgb,rm,noir,nm,matrix,operators,usable=True):
    c,s,_=operators
    moved,field=c.warp_to_rgb(noir.astype(np.float32),matrix,rm.shape)
    warped_mask,_=c.warp_to_rgb(c.interior(nm).astype(np.uint8),matrix,rm.shape,True)
    supports=dict(rgb=c.interior(rm),noir=(warped_mask>0)&field)
    supports['cross']=supports['rgb']&supports['noir'] if usable else np.zeros_like(rm)
    validity,_=c.warp_to_rgb(((noir>1)&(noir<254)).astype(np.float32),matrix,rm.shape)
    six=np.concatenate([rgb.astype(np.float32),moved],axis=-1)
    guards=[np.concatenate([(rgb>1)&(rgb<254),v],axis=-1) for v in (validity>.999,validity>=STRICT_MINIMUM)]
    maps=[s.spatial_indices(six,supports,v) for v in guards]
    statistics=[]
    for definition in s.index_definitions():
        old,new=[s.index_statistics(m[definition['id']],supports[definition['support']]) for m in maps]
        row=dict(index=definition['id'],family=definition['support'])
        for key in STAT_FIELDS:
            row['old_'+key]=old[key];row['strict_'+key]=new[key]
            row['delta_'+key]=(new[key]-old[key]) if isinstance(old[key],(int,float)) and not isinstance(old[key],bool) and new[key] is not None else None
        row['lost_valid_pixels']=old['valid_pixels']-new['valid_pixels'];row['qualification_changed']=old['qualified']!=new['qualified']
        statistics.append(row)
    channels=[]
    for i,name in enumerate(c.CHANNELS):
        for scope,support in [('camera_interior',supports['rgb' if i<3 else 'noir']),('common_interior',supports['cross'])]:
            old_valid=support&guards[0][...,i];new_valid=support&guards[1][...,i]
            if scope=='common_interior':
                signal=np.isfinite(six[...,i])&(six[...,i]>1)&(six[...,i]<254)
                old_valid&=signal;new_valid&=signal
            a=six[...,i][old_valid];b=six[...,i][new_valid]
            row=dict(channel=name,scope=scope,eligible_pixels=int(support.sum()),old_valid_pixels=len(a),strict_valid_pixels=len(b),lost_valid_pixels=len(a)-len(b),old_qualified=len(a)>=40,strict_qualified=len(b)>=40)
            for label,values in [('old',a),('strict',b)]:
                row[label+'_median']=float(np.median(values)) if len(values)>=40 else None
                row[label+'_iqr']=float(np.quantile(values,.75)-np.quantile(values,.25)) if len(values)>=40 else None
            channels.append(row)
    # Separate diagnostic: do not change inherited supports in the main comparison.
    ones,_=c.warp_to_rgb(np.ones(nm.shape,np.float32),matrix,rm.shape)
    border=dict(inherited_field_pixels=int(field.sum()),strict_field_pixels=int((ones>=STRICT_MINIMUM).sum()),
                differing_field_pixels=int(np.sum(field&(ones<STRICT_MINIMUM))),
                differing_field_in_common_pixels=int(np.sum(supports['cross']&(ones<STRICT_MINIMUM))))
    return statistics,channels,border

def load_binding(binding,intake,run,verified):
    row=binding['record'];source=contained(intake,row['source']);mask=contained(run,row['mask'])
    for path,wanted in [(source,row['source_sha256']),(mask,row['mask_sha256'])]:
        key=str(path)
        actual=verified.get(key)
        if actual is None:actual=verified[key]=sha(path)
        if actual!=wanted:raise ValueError('Source or mask digest mismatch: '+key)
    if verified[str(source)]!=binding['actual_source_sha256'] or verified[str(mask)]!=binding['actual_mask_sha256']:raise ValueError('Pair binding digest mismatch')
    with Image.open(source) as im:im.load();image=np.asarray(im.convert('RGB').resize((512,385),Image.Resampling.LANCZOS))
    with Image.open(mask) as im:
        if im.size!=(int(row['processing_width']),int(row['processing_height'])):raise ValueError('Declared mask canvas mismatch')
        foreground=np.asarray(im.convert('L').resize((512,385),Image.Resampling.NEAREST))>0
    return image,foreground

def same(a,b):
    if a is None or b is None:return a is b
    if isinstance(a,(bool,str)) or isinstance(b,(bool,str)):return a==b
    return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','intake','run','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--max-accepted',type=int)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty output root')
    out.mkdir(parents=True,exist_ok=True);start=time.time();root=args.run/'pairing/full'
    configuration=json.loads((root/'configuration.json').read_text());operators=configure(args.repo);folder=operators[2]
    hashes={name:sha(folder/name) for name in configuration['scientific_source_sha256']}
    if hashes!=configuration['scientific_source_sha256']:raise ValueError('Frozen scientific source hashes differ')
    (out/'scientific_operator_snapshot').mkdir()
    for name in hashes:shutil.copyfile(folder/name,out/'scientific_operator_snapshot'/name)
    for path in Path(__file__).parent.glob('*'):
        if path.is_file():shutil.copyfile(path,out/path.name)
    dump(out/'configuration.json',dict(status='running',grid=[512,385],old_guard='>0.999',strict_guard='>=1-1e-7',support_policy='unchanged frozen supports and gates',maximum_accepted=args.max_accepted,scientific_source_sha256=hashes,configuration_sha256=sha(root/'configuration.json'),audit_sha256=sha(__file__)))
    rows=[];channels=[];coverage=[];borders=[];failures=[];bindings={};verified={};processed=0
    for path in sorted((root/'records').glob('*.json')):
        record=json.loads(path.read_text());payload=record.pop('payload_sha256')
        actual=hashlib.sha256(json.dumps(record,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
        if actual!=payload:raise ValueError('Frozen numerical payload digest mismatch')
        identity={k:record[k] for k in ('session','treatment','measurement','genotype','replicate','view')};identity['record_file']=path.name
        nominal=[e for e in record['evaluations'] if e['pair_kind']=='nominal' and e['method']=='ECC']
        status='nominal_ECC_unavailable' if not nominal else 'frozen_support_gate_rejected' if not nominal[0]['cross_support_usable'] else 'eligible'
        if status=='eligible' and args.max_accepted is not None and processed>=args.max_accepted:status='outside_prespecified_pilot'
        if status=='eligible':
            try:
                a,am=load_binding(record['source_bindings']['rgb'],args.intake,args.run,verified)
                b,bm=load_binding(record['source_bindings']['nominal'],args.intake,args.run,verified)
                evaluations,cc,border=compare_pair(a,am,b,bm,np.asarray(nominal[0]['source_to_target'],np.float32),operators)
                stored={r['index']:r for r in nominal[0]['descriptors']}
                for row in evaluations:
                    for key in STAT_FIELDS:
                        if not same(row['old_'+key],stored[row['index']][key]):raise ValueError('Inherited descriptor replay differs: '+row['index']+'/'+key)
                stored_channels={r['channel']:r for r in nominal[0]['channels']}
                for row in cc:
                    if row['scope']=='camera_interior' and row['old_valid_pixels']!=stored_channels[row['channel']]['valid_pixels']:raise ValueError('Inherited channel count replay differs')
                rows.extend(dict(identity,**r) for r in evaluations);channels.extend(dict(identity,**r) for r in cc);borders.append(dict(identity,**border));processed+=1;status='audited_exact_inherited_replay'
                bindings[path.name]=dict(payload_sha256=payload,rgb=record['source_bindings']['rgb'],noir=record['source_bindings']['nominal'])
            except Exception as exc:
                failures.append(dict(identity,error=repr(exc)));status='failed'
        coverage.append(dict(identity,status=status))
        if processed and processed%50==0 and status=='audited_exact_inherited_replay':
            event=dict(event='progress',audited=processed,elapsed_seconds=time.time()-start)
            with (out/'experiment_log.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
    table(out/'formula_changes.csv',rows);table(out/'channel_changes.csv',channels);table(out/'coverage.csv',coverage);table(out/'field_border_diagnostic.csv',borders);table(out/'failures.csv',failures)
    by_formula=[]
    for index in sorted({r['index'] for r in rows}):
        rr=[r for r in rows if r['index']==index];d=[abs(r['delta_median']) for r in rr if r['delta_median'] is not None]
        by_formula.append(dict(index=index,rows=len(rr),changed_valid_pixel_rows=sum(r['lost_valid_pixels']>0 for r in rr),lost_valid_pixels=sum(r['lost_valid_pixels'] for r in rr),qualification_changes=sum(r['qualification_changed'] for r in rr),maximum_absolute_median_delta=max(d,default=None),changed_median_rows=sum(v>0 for v in d)))
    summary=dict(status='completed' if not failures else 'completed_with_failures',planned_anchors=len(coverage),audited_accepted_anchors=processed,coverage_status_counts={s:sum(r['status']==s for r in coverage) for s in sorted({r['status'] for r in coverage})},formula_rows=len(rows),channel_rows=len(channels),formula_changed_valid_pixel_rows=sum(r['lost_valid_pixels']>0 for r in rows),formula_lost_valid_pixels=sum(r['lost_valid_pixels'] for r in rows),formula_qualification_changes=sum(r['qualification_changed'] for r in rows),formula_summaries=by_formula,channel_lost_valid_pixels_by_scope={scope:sum(r['lost_valid_pixels'] for r in channels if r['scope']==scope) for scope in ('camera_interior','common_interior')},field_differing_pixels=sum(r['differing_field_pixels'] for r in borders),field_differing_common_pixels=sum(r['differing_field_in_common_pixels'] for r in borders),failures=len(failures),elapsed_seconds=time.time()-start,versions=dict(numpy=np.__version__,opencv=cv2.__version__,python=sys.version),limits=['Audit is teacher-support JPEG-DN sensitivity, not physiological or anatomical accuracy.','Formula pixel counts overlap and must not be treated as unique physical pixels.','Existing outputs are unchanged; strict comparison is a separate numerical operator.'])
    if any(sha(folder/name)!=value for name,value in hashes.items()):raise ValueError('Scientific source changed during audit')
    dump(out/'summary.json',summary);dump(out/'source_bindings.json',bindings);dump(out/'verified_input_hashes.json',verified)
    print(json.dumps(summary,indent=2),flush=True)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
