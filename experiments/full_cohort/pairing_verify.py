"""Final source-cache, feature-schema and completed-denominator verification."""
import argparse
import csv
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
from pairing_core import sha,digest,atomic_json,true


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def check_features(path,manifest,definitions,prefix):
    rows=read(path)
    assert len(rows)==360
    assert len({tuple(r[k] for k in ('treatment','measurement','genotype','replicate')) for r in rows})==360
    for row in rows:
        for definition in definitions:
            base=prefix+definition
            n=int(row[base+'__n_views']);eligible=true(row[base+'__eligible'])
            assert 0<=n<=4 and eligible==(n>=2)
            for statistic in ('median','iqr'):
                value=row[base+'__'+statistic]
                assert bool(value)==eligible
                if eligible:assert math.isfinite(float(value))
            if eligible:assert float(row[base+'__iqr'])>=0
    assert all(name in rows[0] for name in manifest['feature_columns'])
    return len(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('intake','run','repo'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();root=args.run.resolve()/'pairing/full'
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    config=json.loads((root/'configuration.json').read_text(encoding='utf-8'))
    assert summary['status']=='completed' and summary['pending_decodable_quarter_sources']==0
    assert summary['planned_anchor_pairs']==summary['retained_anchor_pairs']==1440
    assert summary['expected_decodable_quarter_sources']==summary['extraction_records_present']==2716
    snapshot=root/'source_snapshot'/config['signature']
    original_sources={name:sha(snapshot/name) for name in config['producer_sha256']}
    assert original_sources==config['producer_sha256']
    original_keys=set();optimizer={}
    from collections import Counter
    optimizer=Counter()
    for path in (root/'records').glob('*.json'):
        record=json.loads(path.read_text(encoding='utf-8'))
        expected=record.pop('payload_sha256');assert digest(record)==expected
        assert record['signature']==digest(dict(configuration=config['signature'],bindings=record['source_bindings']))
        key=record['session'],record['view'];assert key not in original_keys;original_keys.add(key)
        for evaluation in record['evaluations']:
            if evaluation['method']=='ECC' and evaluation['pair_kind'] in ('nominal','wrong_view','wrong_plant'):
                optimizer[evaluation['pair_kind'],evaluation['ECC_optimizer']['status']]+=1
    assert len(original_keys)==1440 and len({k[0] for k in original_keys})==360
    index_manifest=json.loads((root/'session_descriptors_manifest.json').read_text(encoding='utf-8'))
    assert index_manifest['status']=='completed'
    assert sha(root/'session_descriptors.csv')==index_manifest['session_descriptors_sha256']
    assert sha(root/'session_descriptors_moment.csv')==index_manifest['outputs']['moment']['sha256']
    index_rows=check_features(root/'session_descriptors.csv',index_manifest,index_manifest['definitions'],'paired_index__')
    moment_rows=check_features(root/'session_descriptors_moment.csv',index_manifest,index_manifest['definitions'],'paired_index__')
    dn_manifest=json.loads((root/'session_camera_dn_manifest.json').read_text(encoding='utf-8'))
    assert dn_manifest['status']=='completed' and not dn_manifest['failures']
    assert sha(root/'session_camera_dn.csv')==dn_manifest['session_camera_dn_sha256']
    channels=('rgb_R','rgb_G','rgb_B','noir_R','noir_G','noir_B')
    dn_rows=check_features(root/'session_camera_dn.csv',dn_manifest,channels,'paired_dn__')
    atlas=json.loads((root/'atlas_manifest.json').read_text(encoding='utf-8'))
    assert atlas['status']=='completed' and atlas['rendered_strata']==6 and not atlas['unavailable_strata']
    assert sum(x['verified_definition_statistics'] for x in atlas['selection'])==162
    for entry in atlas['selection']:assert sha(root/entry['figure'])==entry['figure_sha256']
    source_base=Path(__file__).resolve().parent
    source_fields=[('pairing_aggregate.py',index_manifest['aggregate_source_sha256']),
                   ('pairing_dn_controls.py',dn_manifest['source_sha256']),
                   ('pairing_atlas.py',atlas['producer_source_sha256'])]
    input_audit=json.loads((root.parent/'quarter_input_audit.json').read_text(encoding='utf-8'))
    source_fields.append(('pairing_input_audit.py',input_audit['source_sha256']))
    source_fields.append((Path(__file__).name,sha(__file__)))
    report=json.loads((root/'report_verification.json').read_text(encoding='utf-8'))
    source_fields.append(('pairing_report.py',report['report_source_sha256']))
    cached={}
    for name,expected in source_fields:
        source=source_base/name
        if not source.exists():source=source_base/'pairing_support'/name
        assert sha(source)==expected
        target=root/'source_snapshot/postprocessing'/expected/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);assert sha(target)==expected
        cached[name]=dict(sha256=expected,path=target.relative_to(root).as_posix())
    result=dict(status='verified_completed_full_cohort',anchors=1440,acquisitions=360,
                original_numerical_signature=config['signature'],original_numerical_sources_preserved=original_sources,
                per_anchor_payloads_and_configuration_bindings_verified=1440,
                session_feature_rows=dict(ECC=index_rows,moment=moment_rows,matched_DN=dn_rows),
                verified_atlas_definition_statistics=162,postprocessing_source_caches=cached,
                ECC_optimizer_statuses=[dict(pair_kind=k[0],status=k[1],n=v) for k,v in sorted(optimizer.items())],
                environment_versions={name:importlib.metadata.version(name) for name in
                                      ('numpy','opencv-python','Pillow','scipy','matplotlib')},
                verifier_source_sha256=sha(__file__))
    atomic_json(root/'final_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
