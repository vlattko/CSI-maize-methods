"""Strict same-camera descriptors from the frozen gate-rejected nominal ECC anchors."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import validity_audit as a

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','intake','run','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty audit root')
    out.mkdir(parents=True,exist_ok=True);root=args.run/'pairing/full';config=json.loads((root/'configuration.json').read_text());ops=a.configure(args.repo)
    hashes={name:a.sha(ops[2]/name) for name in config['scientific_source_sha256']}
    if hashes!=config['scientific_source_sha256']:raise ValueError('Frozen scientific source mismatch')
    rows=[];channels=[];coverage=[];bindings={};verified={};start=time.time()
    for path in sorted((root/'records').glob('*.json')):
        record=json.loads(path.read_text());payload=record.pop('payload_sha256')
        actual=hashlib.sha256(json.dumps(record,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
        if actual!=payload:raise ValueError('Frozen payload mismatch')
        identity={k:record[k] for k in ('session','treatment','measurement','genotype','replicate','view')};identity['record_file']=path.name
        nominal=[e for e in record['evaluations'] if e['pair_kind']=='nominal' and e['method']=='ECC']
        selected=bool(nominal and not nominal[0]['cross_support_usable'])
        coverage.append(dict(identity,status='selected_rejected_anchor' if selected else 'outside_rejected_anchor_supplement'))
        if not selected:continue
        rgb,rm=a.load_binding(record['source_bindings']['rgb'],args.intake,args.run,verified)
        noir,nm=a.load_binding(record['source_bindings']['nominal'],args.intake,args.run,verified)
        stats,cc,_=a.compare_pair(rgb,rm,noir,nm,np.asarray(nominal[0]['source_to_target'],np.float32),ops,usable=False)
        stored={r['index']:r for r in nominal[0]['descriptors']}
        for row in stats:
            for key in a.STAT_FIELDS:
                if not a.same(row['old_'+key],stored[row['index']][key]):raise ValueError('Rejected-anchor inherited replay differs')
        rows.extend(dict(identity,**r) for r in stats);channels.extend(dict(identity,**r) for r in cc)
        bindings[path.name]=dict(payload_sha256=payload,rgb=record['source_bindings']['rgb'],noir=record['source_bindings']['nominal'])
    a.table(out/'formula_changes.csv',rows);a.table(out/'channel_changes.csv',channels);a.table(out/'coverage.csv',coverage)
    summary=dict(status='completed',audited_rejected_nominal_ECC_anchors=len(bindings),formula_rows=len(rows),formula_changed_valid_pixel_rows=sum(r['lost_valid_pixels']>0 for r in rows),formula_lost_valid_pixels=sum(r['lost_valid_pixels'] for r in rows),formula_qualification_changes=sum(r['qualification_changed'] for r in rows),elapsed_seconds=time.time()-start,policy='Cross supports remain empty as archived; only same-camera NoIR validity can change. This supplements all accepted nominal anchors for complete available nominal-ECC strict features.',operator_sha256=hashes,source_sha256={p.name:a.sha(p) for p in (Path(__file__),Path(a.__file__))},input_configuration_sha256=a.sha(root/'configuration.json'))
    a.dump(out/'source_bindings.json',bindings);a.dump(out/'verified_input_hashes.json',verified);a.dump(out/'summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
