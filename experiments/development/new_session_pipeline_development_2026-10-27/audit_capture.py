"""Audit source bindings, snapshots, fixed slots and numerical output schema."""
import argparse,base64,gzip,json,re
from pathlib import Path
from capture_core import digest
from run_capture import dump


def audit(folder):
    folder=Path(folder);capture=json.loads((folder/'capture.json').read_text(encoding='utf-8'));findings=[];rows=[];root=Path(capture['provenance']['raw_root']);steps=capture['session']['steps']
    for camera in ['rgb','noir']:
        for v in range(steps):
            r=json.loads((folder/f'records/{camera}_{v:03d}.json').read_text(encoding='utf-8'));rows.append(r)
            if r['camera']!=camera or r['view']!=v:findings.append('Fixed slot identity mismatch')
            if r.get('source_sha256'):
                source=(root/r['source']).resolve()
                if not source.is_relative_to(root.resolve()) or digest(source)!=r['source_sha256']:findings.append('Source identity changed')
            if r.get('mask'):
                mask=(folder/r['mask']).resolve()
                if not mask.is_relative_to(folder.resolve()) or digest(mask)!=r['mask_sha256']:findings.append('Mask binding mismatch')
            if r.get('traits') and len(r['traits'])!=39:findings.append('Trait schema differs from39')
    if capture['source_changed_during_run']:findings.append('Source changed during run')
    for name,sha in capture['provenance']['pipeline_source_sha256'].items():
        if digest(folder/'execution_sources'/name)!=sha:findings.append('Pipeline execution snapshot mismatch '+name)
    if digest(folder/'execution_sources/public_methods/traits.py')!=capture['provenance']['traits_sha256']:findings.append('Public trait source snapshot mismatch')
    manifest=json.loads((folder/'output_manifest.json').read_text())
    for item in manifest['files']:
        if digest(folder/item['path'])!=item['sha256']:findings.append('Output manifest mismatch '+item['path'])
    if digest(capture['provider']['checkpoint'])!=capture['provider']['checkpoint_sha256']:findings.append('Model checkpoint mismatch')
    if capture['session']['plant_uid'] is not None:physical='Explicitly user supplied; not independently verified'
    else:physical='No physical UID inferred'
    html=(folder/'report.html').read_text(encoding='utf-8');encoded=re.search(r'<script id="payload" type="application/octet-stream">([^<]+)</script>',html).group(1);payload=json.loads(gzip.decompress(base64.b64decode(encoded)))
    if len(payload['frames'])!=2*steps or len({(r['camera'],r['view']) for r in payload['frames']})!=2*steps:findings.append('Report dropped/duplicated fixed slots')
    trackcounts={}
    for camera,t in capture['tracking'].items():
        trackcounts[camera]=len(t['tracks'])
        for track in t['tracks']:
            if any(a['view']%4==2 for a in track['assignments']):findings.append('Heldout angle in training assignments')
            if len(track['angle_evidence'])!=steps:findings.append('Tracking angular evidence count')
    result=dict(status='passed' if not findings else 'failed',findings=findings,sample_id=capture['session']['sample_id'],
        planned_frames=2*steps,source_bound_frames=sum(bool(r.get('source_sha256')) for r in rows),decoded_frames=capture['decoded_frames'],
        mask_bound_frames=sum(bool(r.get('mask')) for r in rows),trait_frames=sum(bool(r.get('traits')) for r in rows),
        usable_views={c:s['usable_views'] for c,s in capture['camera_summaries'].items()},terminal_hypotheses=trackcounts,
        duplicate_source_hash_warning_groups=len(capture['duplicate_source_hash_warnings']),physical_identity=physical,
        assays_predicted=capture['assays']['predicted'],report_embedded_frames=len(payload['assets']),
        report_sha256=digest(folder/'report.html'),capture_json_sha256=digest(folder/'capture.json'),audit_source_sha256=digest(__file__))
    dump(folder/'independent_audit.json',result);print(json.dumps(result,indent=2),flush=True)
    if findings:raise ValueError(findings)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True);a=p.parse_args();audit(a.capture)
