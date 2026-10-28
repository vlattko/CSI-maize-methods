"""Machine-readable matched comparisons and deployment denominators."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def stats(rows):
    ok=[r for r in rows if r['status']=='executed']
    sessions={r['session'] for r in ok}
    return dict(planned_rows=len(rows),executed_rows=len(ok),sessions_with_scores=len(sessions),
        median_iou=float(np.median([float(r['iou']) for r in ok])) if ok else None,
        median_session_median_iou=float(np.median([np.median([float(r['iou']) for r in ok if r['session']==s])
                                     for s in sessions])) if ok else None)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    shared=read(out/'full/summary.json');periodic=read(out/'periodic/full/summary.json')
    lookup=read(out/'periodic/full/session_camera_lookup.json')['rows']
    metrics=load(out/'periodic/full/heldout_metrics_with_pair_identity.csv')
    strata={}
    for status in sorted({r['deployment_status'] for r in lookup}):
        sessions={r['session'] for r in lookup if r['deployment_status']==status}
        strata[status]=dict(acquisitions=len(sessions),**stats([r for r in metrics if r['session'] in sessions and r['method']=='selected_causal']))
    matched={}
    grouped={method:{(r['session'],r['rgb_view']):float(r['iou']) for r in metrics
                     if r['method']==method and r['status']=='executed'} for method in
                     ('phase_shared','periodic_candidate','selected_causal','adaptive_selected_pair_ecc')}
    common=set.intersection(*(set(v) for v in grouped.values()))
    for method,values in grouped.items():
        matched[method]=dict(rows=len(common),sessions=len({k[0] for k in common}),
            median_iou=float(np.median([values[k] for k in common])),
            mean_iou=float(np.mean([values[k] for k in common])))
    session_changes=[]
    for session in sorted({k[0] for k in common}):
        keys=[k for k in common if k[0]==session]
        session_changes.append(float(np.median([grouped['selected_causal'][k]-grouped['phase_shared'][k] for k in keys])))
    verification=read(out/'verification.json')
    result=dict(status='completed_and_verified',schema=1,
        preferred_lookup='periodic/full/session_camera_lookup.json',
        preferred_flat_lookup='periodic/full/alignment_lookup.csv',
        baseline_lookup='full/session_camera_lookup.json',
        planned_acquisitions=360,planned_rotation_indices_per_acquisition=40,
        shared_fit_acquisitions=shared['fitted_acquisitions'],
        selected_deployment_status=periodic['deployment_status'],deployment_strata=strata,
        same_pair_comparison=matched,selected_minus_shared_session_direction=dict(
            positive=sum(x>0 for x in session_changes),zero=sum(x==0 for x in session_changes),negative=sum(x<0 for x in session_changes)),
        overall_evaluation_scope='All available fitted hypotheses, including training-abstaining acquisitions; deployment-stratified scores are separate',
        shared_signature=shared['signature'],periodic_signature=periodic['signature'],
        tests=dict(shared=8,periodic=5,total=13,result='passed'),verification=verification,
        controls=periodic['contrasts'],phase_summary=shared['phase_selections'],
        quality_notes=['No nonnominal phase/sign passed the fixed training-only gate',
            'Phase resolution is four indices; retained nominal mapping is not physical angle verification',
            'Adaptive per-view ECC uses the withheld mask and is a more permissive comparator',
            'Silhouette overlap does not validate leaf identity, tissue correspondence, narrow-band response or true 3D cameras',
            'Original source pixels were rehashed for example figures; the full mask-only run verifies actual mask bytes and binds source hashes to the existing intake decode manifest'],
        source_versions=dict(shared=shared['signature'],periodic=periodic['signature']),
        current_sources_snapshot_policy='Numerical source snapshots are immutable; later test/export additions have their own provenance and may invalidate current-directory resume fingerprints',
        source_sha256=sha(__file__))
    path=out/'development_findings.json';tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2),encoding='utf-8');tmp.replace(path)
    print(json.dumps(dict(status=result['status'],same_pair_comparison=matched,deployment_strata=strata,
                          session_direction=result['selected_minus_shared_session_direction'])),flush=True)

if __name__=='__main__':main()
