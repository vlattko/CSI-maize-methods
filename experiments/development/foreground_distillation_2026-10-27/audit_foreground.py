"""Independent identity, split and full-reference denominator checks."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import foreground_core as core

KEYS=('treatment','measurement','genotype','replicate')


def check_source_bindings(original,prepared):
    lookup={r['source']:r for r in original}
    if len(lookup)!=len(original) or len({r['source'] for r in prepared})!=len(prepared):
        raise ValueError('Source paths are not unique')
    for row in prepared:
        source=lookup.get(row['source'])
        if source is None or any(str(row[k])!=str(source[k]) for k in (*KEYS,'session','camera','view','source_sha256','mask_sha256')):
            raise ValueError('Prepared image identity or provenance differs from the teacher inventory')
        if str(row['teacher_accepted']).lower()!=str(source['qc_pass']).lower():
            raise ValueError('Prepared teacher acceptance differs from original operational QC')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','intake','teacher-run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    original=core.read_csv(args.teacher_run/'extraction/view_traits.csv')
    references=core.read_csv(args.intake/'research_audit/reference_master.csv')
    reference_keys={tuple(r[k] for k in KEYS) for r in references}
    if len(references)!=360 or len(reference_keys)!=360:
        raise ValueError('All360unique reference identities are required')
    splits=core.genotype_splits(references)
    results={}
    for kind,relative in [('quarter','quarter_frames.csv'),('all_views','all_views/frame_inventory.csv')]:
        rows=core.read_csv(args.output/relative)
        check_source_bindings(original,rows)
        expected=[r for r in original if kind=='all_views' or int(r['view']) in core.QUARTERS]
        if {r['source'] for r in rows}!={r['source'] for r in expected}:
            raise ValueError('Prepared image inventory is incomplete or contains unsupported views')
        for row in rows:
            if tuple(row[k] for k in KEYS) not in reference_keys or row['split']!=splits[row['genotype']]:
                raise ValueError('Prepared row violates the reference key or genotype partition')
        hashes=defaultdict(list); candidates=defaultdict(set)
        for row in rows:
            hashes[row['source_sha256']].append(row)
            candidates[(row['treatment'],row['genotype'],row['replicate'])].add(row['split'])
        if any(len(value)>1 for value in candidates.values()):
            raise ValueError('A whole candidate trajectory crosses partitions')
        duplicates=[group for group in hashes.values() if len(group)>1]
        cross=[group for group in duplicates if len({r['split'] for r in group})>1]
        if cross:
            raise ValueError('Identical source-image bytes cross genotype partitions')
        by_split={}
        for split in ('train','val','test'):
            subset=[r for r in rows if r['split']==split]
            observed={tuple(r[k] for k in KEYS) for r in subset}
            expected_keys={tuple(r[k] for k in KEYS) for r in references if splits[r['genotype']]==split}
            by_split[split]=dict(reference_acquisitions=len(expected_keys),acquisitions_with_any_source=len(observed),
                acquisitions_without_any_source=len(expected_keys-observed),frames=len(subset),
                accepted_frames=sum(r['teacher_accepted'].lower()=='true' for r in subset),
                rejected_frames=sum(r['teacher_accepted'].lower()!='true' for r in subset))
        results[kind]=dict(frames=len(rows),source_hash_duplicate_groups=len(duplicates),cross_partition_duplicate_groups=0,
            candidate_partition_violations=0,genotype_splits=splits,denominators=by_split,
            frame_table_sha256=core.digest(args.output/relative))
    core.write_json(args.output/'foreground_data_audit.json',dict(status='passed',datasets=results,
        reference_records=360,original_teacher_table_sha256=core.digest(args.teacher_run/'extraction/view_traits.csv'),
        audit_source_sha256=core.digest(__file__)))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    main()
