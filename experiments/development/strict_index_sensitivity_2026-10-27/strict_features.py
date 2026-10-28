"""Target-free replacement of audited nominal ECC statistics."""
import csv
import math
from pathlib import Path

IDENTITY=('treatment','measurement','genotype','replicate')
FIELDS=('eligible_pixels','valid_pixels','qualified','valid_fraction','median','q10','q25','q75','q90','iqr')


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def truth(value):return str(value).lower() in ('true','1')


def same(a,b):
    if a in ('',None) or b in ('',None):return a in ('',None) and b in ('',None)
    if isinstance(a,bool) or isinstance(b,bool) or str(a) in ('True','False') or str(b) in ('True','False'):
        return truth(a)==truth(b)
    try:
        aa,bb=float(a),float(b)
        return math.isfinite(aa) and math.isfinite(bb) and aa==bb
    except (TypeError,ValueError):return str(a)==str(b)


def keyed(rows,kind):
    output={}
    for row in rows:
        key=(str(row['session']),int(row['view']),row[kind])
        if key in output:raise ValueError('Duplicate audited key: '+str(key))
        output[key]=row
    return output


def binding(old,audit):
    if any(str(old[k])!=str(audit[k]) for k in (*IDENTITY,'session')):
        raise ValueError('Audit identity mismatch')


def replace_formulas(original,audits):
    base=keyed(original,'index');lookup=keyed(audits,'index')
    if set(base)!=set(lookup):raise ValueError('Incomplete or foreign formula audit coverage')
    output=[]
    for row in original:
        if row['pair_kind']!='nominal' or row['method']!='ECC':raise ValueError('Only nominal ECC rows may be replaced')
        audit=lookup[(row['session'],int(row['view']),row['index'])];binding(row,audit)
        if row['family']!=audit['family']:raise ValueError('Formula family mismatch')
        for field in FIELDS:
            if not same(row[field],audit['old_'+field]):raise ValueError('Audit old statistic mismatch: '+field)
        result=dict(row,**{field:audit['strict_'+field] for field in FIELDS})
        count=int(result['valid_pixels']);eligible=int(result['eligible_pixels'])
        if not 0<=count<=eligible or truth(result['qualified'])!=(count>=40):raise ValueError('Strict pixel denominator mismatch')
        if truth(result['qualified'])!=(result['median'] not in ('',None)):raise ValueError('Strict median qualification mismatch')
        output.append(result)
    return output


def replace_dns(original,audits):
    lookup=keyed([r for r in audits if r['scope']=='common_interior'],'channel')
    base=keyed(original,'channel')
    if set(lookup)-set(base):raise ValueError('Foreign DN audit coverage')
    output=[]
    for row in original:
        key=(row['session'],int(row['view']),row['channel']);audit=lookup.get(key)
        if audit is None:
            if truth(row['qualified']) or int(row['valid_pixels'])!=0:raise ValueError('Unaudited available DN row')
            output.append(dict(row));continue
        binding(row,audit)
        if not same(row['eligible_pixels'],audit['eligible_pixels']):raise ValueError('DN support denominator mismatch')
        for field in ('valid_pixels','qualified','median','iqr'):
            if not same(row[field],audit['old_'+field]):raise ValueError('DN old statistic mismatch: '+field)
        result=dict(row,**{field:audit['strict_'+field] for field in ('valid_pixels','qualified','median','iqr')})
        count=int(result['valid_pixels']);eligible=int(result['eligible_pixels'])
        if not 0<=count<=eligible or truth(result['qualified'])!=(count>=40):raise ValueError('Strict DN count mismatch')
        result['valid_fraction']=count/eligible if eligible else None
        result['failure_reason']='' if truth(result['qualified']) else 'fewer_than_40_valid_common_pixels'
        output.append(result)
    return output


def compare_tables(expected,actual):
    def key(r):return tuple(str(r[k]) for k in (*IDENTITY,'session'))
    old={key(r):r for r in expected};new={key(r):r for r in actual}
    if len(old)!=len(expected) or len(new)!=len(actual) or set(old)!=set(new):raise ValueError('Session key mismatch')
    for key,row in old.items():
        if set(row)!=set(new[key]):raise ValueError('Session field mismatch')
        for field,value in row.items():
            if not same(value,new[key][field]):raise ValueError('Session replay mismatch: '+str((key,field,value,new[key][field])))


def feature_changes(old,new,definitions,prefix,primary_keys):
    def key(r):return tuple(str(r[k]) for k in IDENTITY)
    old={key(r):r for r in old};new={key(r):r for r in new};changes=[];availability=[]
    for definition in definitions:
        stem=prefix+'__'+definition+'__'
        record=dict(feature=definition,family=prefix)
        for label,keys in [('all360',set(old)),('primary329',set(primary_keys))]:
            losses=sum(truth(old[k][stem+'eligible']) and not truth(new[k][stem+'eligible']) for k in keys)
            gains=sum(not truth(old[k][stem+'eligible']) and truth(new[k][stem+'eligible']) for k in keys)
            record.update({label+'_old_eligible':sum(truth(old[k][stem+'eligible']) for k in keys),
                           label+'_strict_eligible':sum(truth(new[k][stem+'eligible']) for k in keys),
                           label+'_eligibility_losses':losses,label+'_eligibility_gains':gains})
        for stat in ('median','iqr','n_views','eligible'):
            name=stem+stat;deltas=[]
            for k in old:
                a,b=old[k][name],new[k][name]
                if not same(a,b):
                    delta=float(b)-float(a) if a not in ('',None) and b not in ('',None) and stat!='eligible' else None
                    changes.append(dict(**dict(zip(IDENTITY,k)),feature=name,old=a,strict=b,delta=delta,primary_paired=k in primary_keys))
                    if stat in ('median','iqr') and delta is not None:deltas.append(abs(delta))
            record['changed_'+stat+'_rows']=sum(c['feature']==name for c in changes)
            if stat in ('median','iqr'):record['maximum_absolute_'+stat+'_delta']=max(deltas,default=0.)
        availability.append(record)
    return changes,availability
