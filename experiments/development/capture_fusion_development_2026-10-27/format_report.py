"""Apply presentation-only corrections while preserving numerical output bytes."""
import argparse,hashlib,json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def format_report(path,update_manifest=False):
    path=Path(path);source=path.read_text(encoding='utf-8')
    if 'data-presentation="responsive-status"' in source:return
    original=path.parent/'presentation/original.html';original.parent.mkdir(exist_ok=True)
    if original.exists() and original.read_text(encoding='utf-8')!=source:raise ValueError('Existing archived presentation differs')
    original.write_text(source,encoding='utf-8')
    replacement="const p=r.pair;$('status').textContent='RGB index '+p.view+' → NoIR index '+p.noir_view+' · '+p.split+' angles · '+(p.deployment_abstain?'Camera abstained; fused maps unavailable.':typeof p.map_grid_iou_512!=='number'?'Source pair unavailable.':'Atlas-grid overlap '+(100*p.map_grid_iou_512).toFixed(1)+'%; '+(p.cross_formulas_usable?'cross-camera maps supported.':'cross-camera maps withheld: '+p.cross_formula_reasons.join(', ')));"
    needle="$('status').textContent=JSON.stringify(r.pair);"
    if source.count(needle)!=1:raise ValueError('Unexpected report presentation program')
    formatted=source.replace('<html lang="en">','<html lang="en" data-presentation="responsive-status">',1).replace('body{font:','body{overflow-wrap:anywhere;font:',1).replace(needle,replacement,1)
    formatted=formatted.replace('<h2>Camera and provenance</h2><pre>','<details><summary>Camera and provenance</summary><pre>',1).replace('</pre><p>Data and reference','</pre></details><p>Data and reference',1)
    path.write_text(formatted,encoding='utf-8')
    manifest=path.parent/'output_manifest.json';old_manifest_sha=sha(manifest) if manifest.exists() else None
    record=dict(status='presentation_only',original_html_sha256=sha(original),formatted_html_sha256=sha(path),formatter_sha256=sha(Path(__file__)),prior_output_manifest_sha256=old_manifest_sha,
        changes=['Responsive wrapping removes browser-confirmed horizontal overflow','Human-readable frame/support status replaces raw JSON line','Detailed provenance remains expandable'],numerical_files_changed=False)
    (path.parent/'presentation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    if update_manifest:
        data=json.loads(manifest.read_text());data['files']=[dict(path=p.relative_to(path.parent).as_posix(),sha256=sha(p)) for p in path.parent.rglob('*') if p.is_file() and p!=manifest]
        manifest.write_text(json.dumps(data,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--update-manifest',action='store_true');a=p.parse_args();format_report(a.report,a.update_manifest)
