"""Target-free session aggregation of completed nominal paired-DN estimates.

Staged in pairing_support while the numerical producer is running, so adding
this analysis cannot change the producer's root-level source fingerprint.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time

IDENTITY=('treatment','measurement','genotype','replicate')
QUARTERS={0,10,20,30}
MINIMUM_VALID_QUARTERS=2
MINIMUM_VALID_PIXELS=40


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def truth(value):
    return str(value).lower() in ('true','1')


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)


def atomic_csv(path,rows):
    temporary=path.with_suffix('.csv.tmp')
    with temporary.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    os.replace(temporary,path)


def quantile(values,p):
    """Linear interpolation, equivalent to the public NumPy default quantile."""
    values=sorted(values)
    index=(len(values)-1)*p
    lo=math.floor(index);hi=math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def aggregate(sessions,descriptor_rows,registration_rows,definitions,method):
    canonical={s['session']:s for s in sessions}
    if len(canonical)!=len(sessions):raise ValueError('Duplicate canonical session')
    four_keys={tuple(s[k] for k in IDENTITY) for s in sessions}
    if len(four_keys)!=len(sessions):raise ValueError('Duplicate four-key canonical identity')
    ids=[d['id'] for d in definitions]
    if len(ids)!=27 or len(set(ids))!=27:raise ValueError('Expected 27 unique definitions')
    samples={};seen=set();available={}
    for row in descriptor_rows:
        if row['pair_kind']!='nominal' or row['method']!=method:continue
        key=(row['session'],row['index'],int(row['view']))
        if key in seen:raise ValueError('Duplicate descriptor-quarter key: '+str(key))
        seen.add(key)
        if key[0] not in canonical or key[1] not in ids or key[2] not in QUARTERS:
            raise ValueError('Unexpected nominal descriptor key: '+str(key))
        if any(str(row[k])!=str(canonical[key[0]][k]) for k in IDENTITY):
            raise ValueError('Descriptor identity disagrees with canonical four-key join')
        n=int(row['valid_pixels']);eligible=int(row['eligible_pixels'])
        if n>eligible or n<0:raise ValueError('Invalid pixel denominator')
        qualified=truth(row['qualified'])
        if qualified != (n>=MINIMUM_VALID_PIXELS):
            raise ValueError('Pixel qualification disagrees with original >=40 rule')
        if qualified:
            value=float(row['median'])
            if not math.isfinite(value):raise ValueError('Qualified median is nonfinite')
            samples.setdefault((key[0],key[1]),[]).append(value)
        elif row.get('median'):
            raise ValueError('Unqualified descriptor unexpectedly has a median')
    for row in registration_rows:
        if row['pair_kind']!='nominal' or row['method']!=method:continue
        if row['session'] not in canonical or int(row['view']) not in QUARTERS:
            raise ValueError('Unexpected registration key')
        values=available.setdefault(row['session'],{})
        if row['view'] in values:raise ValueError('Duplicate nominal registration quarter')
        values[row['view']]=truth(row['cross_support_usable'])
    output=[]
    for session in sessions:
        row={k:session[k] for k in (*IDENTITY,'session')}
        registered=available.get(session['session'],{})
        row.update(paired_registered_quarters=len(registered),
                   paired_cross_support_usable_quarters=sum(registered.values()))
        eligible_count=0
        for definition in ids:
            values=samples.get((session['session'],definition),[])
            n=len(values)
            if n>4:raise ValueError('More than four qualified quarter medians')
            eligible=n>=MINIMUM_VALID_QUARTERS
            eligible_count+=eligible
            prefix='paired_index__'+definition
            row[prefix+'__median']=statistics.median(values) if eligible else None
            row[prefix+'__iqr']=quantile(values,.75)-quantile(values,.25) if eligible else None
            row[prefix+'__n_views']=n
            row[prefix+'__eligible']=eligible
            row[prefix+'__missing_reason']=('' if eligible else 'no_qualified_quarters' if n==0
                                             else 'fewer_than_two_qualified_quarters')
        row.update(paired_index_n_eligible_definitions=eligible_count,
                   paired_index_any_eligible=eligible_count>0,
                   paired_index_all_27_eligible=eligible_count==27)
        output.append(row)
    return output


def schema(definitions):
    ids=[d['id'] for d in definitions]
    return dict(schema_version=1,expected_rows=360,definitions=ids,
                feature_columns=['paired_index__'+i+'__'+stat for i in ids for stat in ('median','iqr')],
                availability_columns=['paired_index__'+i+'__'+stat for i in ids
                                      for stat in ('n_views','eligible','missing_reason')],
                nonfeature_columns=[*IDENTITY,'session','paired_registered_quarters',
                    'paired_cross_support_usable_quarters','paired_index_n_eligible_definitions',
                    'paired_index_any_eligible','paired_index_all_27_eligible'],
                aggregation_policy=dict(pair_kind='nominal',primary_method='ECC',baseline_method='moment',
                    views=sorted(QUARTERS),minimum_valid_pixels_per_view=MINIMUM_VALID_PIXELS,
                    minimum_qualified_views_per_descriptor=MINIMUM_VALID_QUARTERS,
                    values='Unweighted median and linear IQR of per-quarter valid-pixel descriptor medians',
                    per_descriptor_denominator='qualified fixed quarters, out of four planned',
                    missing='Empty median/IQR when fewer than two qualified quarters; never zero-fill',
                    all_27_complete_required=False,targets_included=False,
                    registration_or_availability_columns_are_predictors=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('intake','run','repo'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--watch',action='store_true')
    args=p.parse_args()
    root=args.run.resolve()/'pairing/full'
    summary_path=root/'summary.json'
    while True:
        summary=json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
        if summary.get('status')=='completed':break
        if not args.watch:raise ValueError('Full pairing summary must be completed before publishing session features')
        time.sleep(30)
    if summary.get('mode')!='full' or summary.get('planned_anchor_pairs')!=1440:
        raise ValueError('Refusing a pilot/partial cohort as full session features')
    sessions=[r for r in read(args.intake.resolve()/'session_decode_qc.csv') if truth(r['known_reference_identity'])]
    if len(sessions)!=360:raise ValueError('Canonical intake must contain all 360 assay rows')
    sessions=sorted(sessions,key=lambda r:tuple(r[k] for k in IDENTITY))
    definitions=json.loads((root/'index_definitions.json').read_text(encoding='utf-8'))
    descriptors=read(root/'descriptors_27.csv')
    registration=read(root/'registration.csv')
    if len(descriptors)!=27*len(registration):raise ValueError('Completed descriptor table count mismatch')
    outputs={}
    for method,filename in [('ECC','session_descriptors.csv'),('moment','session_descriptors_moment.csv')]:
        rows=aggregate(sessions,descriptors,registration,definitions,method)
        atomic_csv(root/filename,rows)
        outputs[method]=dict(path=filename,sha256=sha(root/filename),rows=len(rows),
                            sessions_with_any_eligible_index=sum(r['paired_index_any_eligible'] for r in rows),
                            sessions_with_all_27_eligible=sum(r['paired_index_all_27_eligible'] for r in rows),
                            eligible_sessions_per_definition={i:sum(r['paired_index__'+i+'__eligible'] for r in rows)
                                                              for i in schema(definitions)['definitions']})
    manifest=dict(schema(definitions),status='completed',source_pairing_summary_status=summary['status'],
                  session_descriptors_sha256=outputs['ECC']['sha256'],outputs=outputs,
                  aggregate_source_sha256=sha(__file__),
                  inputs_sha256={filename:sha(root/filename) for filename in
                                 ('summary.json','configuration.json','descriptors_27.csv','registration.csv','index_definitions.json')},
                  canonical_intake_sha256=sha(args.intake.resolve()/'session_decode_qc.csv'))
    snapshot=root/'source_snapshot/aggregation'/manifest['aggregate_source_sha256']
    snapshot.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(__file__,snapshot/'pairing_aggregate.py')
    atomic_json(root/'session_descriptors_manifest.json',manifest)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
