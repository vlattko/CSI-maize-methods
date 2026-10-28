"""Apply explicit experimenter-confirmed pot continuity without changing raw data."""
import argparse,csv,hashlib,json
from pathlib import Path


FIELDS=('treatment','measurement','genotype','replicate')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def make_registry(rows,confirmation):
    if confirmation.get('same_labels_refer_to_same_physical_pots') is not True:raise ValueError('Explicit pot-continuity confirmation required')
    ages=confirmation['declared_das_by_visit'];ordered=[ages[k] for k in sorted(ages,key=int)]
    if any(not isinstance(v,int) or v<0 for v in ordered) or ordered!=sorted(set(ordered)):raise ValueError('Declared ages must increase across visits')
    result=[];seen=set()
    for row in rows:
        key=tuple(str(row[k]) for k in FIELDS)
        if not all(key):raise ValueError('Unresolved labels cannot define a physical pot')
        if key in seen:raise ValueError('Duplicate physical identity/visit')
        seen.add(key);pot=(key[0],key[2],key[3]);uid='pot_'+hashlib.sha256(json.dumps(pot,separators=(',',':')).encode()).hexdigest()[:20]
        result.append(dict(zip(FIELDS,key))|dict(plant_uid=uid,declared_das=ages[key[1]],physical_identity_status='experimenter_confirmed_same_pot',
            confirmation_id=confirmation['confirmation_id'],organ_identity_status='geometry_hypotheses_not_anatomical_validation'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--confirmation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.reference.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    confirmation=json.loads(a.confirmation.read_text());records=make_registry(rows,confirmation);a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'plant_identity_registry.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    result=dict(status='verified_experimenter_confirmed_identity_overlay',confirmation=confirmation,records=records,
        plant_count=len({r['plant_uid'] for r in records}),reference_visits=len(records),reference_sha256=sha(a.reference),confirmation_sha256=sha(a.confirmation),builder_sha256=sha(Path(__file__)))
    (a.output/'identity_registry.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','plant_count','reference_visits']},indent=2))


if __name__=='__main__':main()
