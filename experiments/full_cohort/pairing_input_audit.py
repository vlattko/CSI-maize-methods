"""Unique-source denominators for the prespecified full-quarter pairing grid."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('intake','run','repo'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();intake=args.intake.resolve();output=args.run.resolve()/'pairing'
    output.mkdir(parents=True,exist_ok=True)
    canonical=intake/'session_decode_qc.csv';manifest=intake/'jpeg_decode_manifest.csv'
    sessions=[r for r in read(canonical) if r['known_reference_identity']=='True']
    if len(sessions)!=360 or len({tuple(s[k] for k in ('treatment','measurement','genotype','replicate')) for s in sessions})!=360:
        raise ValueError('Expected360 unique canonical assay rows')
    decoded={r['path']:r for r in read(manifest)}
    rows=[];by_anchor=defaultdict(list)
    for s in sessions:
        for view in (0,10,20,30):
            for camera in ('rgb','noir'):
                source=f"{s['session']}/{camera}/image_{camera}_{view:03d}.jpg"
                found=decoded.get(source)
                state=('missing_expected_source' if found is None else
                       'empty_placeholder' if int(found['bytes'])==0 else
                       'decodable' if found['decode_ok']=='True' else 'nonempty_decode_failure')
                row={k:s[k] for k in ('treatment','measurement','genotype','replicate','session')}
                row.update(view=view,camera=camera,source=source,input_state=state,
                           source_sha256=found['sha256'] if found else None,
                           source_bytes=int(found['bytes']) if found else None,
                           original_decode_error=found['decode_error'] if found else None)
                rows.append(row);by_anchor[s['session'],view].append(state)
    with (output/'quarter_source_availability.csv.tmp').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(output/'quarter_source_availability.csv.tmp',output/'quarter_source_availability.csv')
    session_pairs=Counter(session for (session,view),states in by_anchor.items() if states==['decodable','decodable'])
    paired=sum(v for v in session_pairs.values())
    summary=dict(status='completed_input_audit',canonical_sessions=360,planned_quarter_anchors=1440,
                 planned_unique_nominal_camera_slots=len(rows),unique_source_states=dict(Counter(r['input_state'] for r in rows)),
                 both_sources_decodable_quarter_anchors=paired,both_sources_decodable_fraction=paired/1440,
                 sessions_with_any_decodable_quarter_pair=sum(session_pairs.get(s['session'],0)>0 for s in sessions),
                 sessions_with_at_least_two_decodable_quarter_pairs=sum(session_pairs.get(s['session'],0)>=2 for s in sessions),
                 sessions_with_all_four_decodable_quarter_pairs=sum(session_pairs.get(s['session'],0)==4 for s in sessions),
                 limitations=['This is input availability before maskQC and registration; not final feature availability.',
                              'Control roles may reuse originals. Counts here concern unique expected nominal source slots.',
                              'Unidentified extras are outside supervised joins.'],
                 source_sha256=sha(Path(__file__)),canonical_sha256=sha(canonical),decode_manifest_sha256=sha(manifest),
                 source_table_sha256=sha(output/'quarter_source_availability.csv'))
    temp=output/'quarter_input_audit.json.tmp'
    temp.write_text(json.dumps(summary,indent=2),encoding='utf-8');os.replace(temp,output/'quarter_input_audit.json')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
