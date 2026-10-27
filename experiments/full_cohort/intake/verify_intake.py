"""Independent consistency checks for full-archive intake and diagnostic replays."""
import argparse
import base64
from collections import defaultdict
import csv
import hashlib
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path
from PIL import Image


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class Report(HTMLParser):
    def __init__(self):super().__init__();self.images=[];self.ids=set();self.anchors=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d:
            assert d['id'] not in self.ids,'Duplicate anchor';self.ids.add(d['id'])
        if tag=='img':self.images.append(d)
        if tag=='a' and d.get('href','').startswith('#'):self.anchors.append(d['href'][1:])
        assert tag not in ('script','iframe','link'),'Unexpected external/script dependency'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--intake',type=Path,required=True)
    root=p.parse_args().intake
    summary=json.loads((root/'intake_summary.json').read_text(encoding='utf-8'))
    assert sha(root/'intake_execution_source.py')==summary['script_sha256']
    images=read(root/'jpeg_decode_manifest.csv')
    raw=Path(summary['raw_root'])
    assert len(images)==len({r['path'] for r in images})==summary['image_files']
    paths={p.relative_to(raw).as_posix() for p in raw.glob('IMAGES/*/capture_session*/*/*.jpg')}
    assert paths=={r['path'] for r in images}
    assert sum(int(r['bytes']) for r in images)==summary['image_bytes']
    assert sum(r['decode_ok']=='True' for r in images)==summary['decode_pass']
    by_session=defaultdict(lambda:defaultdict(set))
    source_hashes={r['path']:r['sha256'] for r in images}
    for r in images:
        path=Path(r['path'])
        if r['decode_ok']=='True':by_session[path.parent.parent.as_posix()][path.parent.name].add(int(path.stem.rsplit('_',1)[1]))
    sessions=read(root/'session_decode_qc.csv')
    known=[r for r in sessions if r['known_reference_identity']=='True']
    fields=('treatment','measurement','genotype','replicate')
    assert len({tuple(r[k] for k in fields) for r in known})==360
    assert len(sessions)==summary['sessions']
    complete=set()
    for r in sessions:
        d=by_session[r['session']]
        strict=d['rgb']==d['noir']==set(range(40))
        assert strict==(r['strict_complete_40_pairs']=='True')
        assert len(d['rgb']&d['noir'])==int(r['paired_decodable_views'])
        if strict and r['known_reference_identity']=='True':complete.add(r['session'])
    assert len(complete)==summary['complete_identified_40_pair_sessions']
    master=read(root/'research_audit/reference_master.csv')
    assert {tuple(r[k] for k in fields) for r in master}=={tuple(r[k] for k in fields) for r in known}
    for r in json.loads((root/'workbook_sources.json').read_text(encoding='utf-8')):
        assert sha(Path(r['source']))==sha(root/r['staged'])==r['sha256']
    replays={};configs={}
    for folder in ['segmentation_triage','segmentation_prompted']:
        rows=read(root/folder/'view_diagnostics.csv')
        config=json.loads((root/folder/'configuration.json').read_text(encoding='utf-8'))
        finish=json.loads((root/folder/'summary.json').read_text(encoding='utf-8'))
        assert finish['status']=='completed'
        assert finish['images']==len(rows)==config['planned_images']==720
        assert sum(r['qc_pass']=='True' for r in rows)==finish['qc_pass']
        assert sha(root/folder/'execution_source.py')==config['script_sha256']
        keys={(r['source'],r['camera'],r['view']) for r in rows}
        assert len(keys)==len(rows)
        assert len({(r['treatment'],r['measurement'],r['genotype']) for r in rows})==90
        for r in rows:
            assert r['session'] in complete
            assert source_hashes[r['source']]==r['source_sha256']
            if r.get('mask'):
                with Image.open(root/r['mask']) as mask:
                    assert mask.size==(int(r['processing_width']),int(r['processing_height']))
                    mask.verify()
        replays[folder]=keys;configs[folder]=config
    assert replays['segmentation_triage']==replays['segmentation_prompted']
    assert configs['segmentation_triage']['source_hashes']==configs['segmentation_prompted']['source_hashes']
    assert configs['segmentation_triage']['asset_hashes']==configs['segmentation_prompted']['asset_hashes']
    baseline=json.loads((root/'metadata_baselines/summary.json').read_text(encoding='utf-8'))
    assert baseline['status']=='completed' and baseline['images_used'] is False
    assert baseline['reference_sha256']==sha(root/'research_audit/reference_master.csv')
    assert baseline['script_sha256']==sha(root/'metadata_baselines/execution_source.py')
    predictions=read(root/'metadata_baselines/predictions.csv')
    assert len(predictions)==4320
    model_fields={'training_mean':(),'visit_only':('measurement',),
                  'treatment_only':('treatment',),'treatment_visit':('treatment','measurement')}
    observed={tuple(r[k] for k in fields):r for r in master}
    for r in predictions:
        key=(r['treatment'],r['visit'],r['genotype'],r['replicate'])
        origin=observed[key]
        assert float(r['observed'])==float(origin[r['target']])
        training=[a for a in master if a['genotype']!=r['genotype'] and
                  all(a[f]==origin[f] for f in model_fields[r['model']])]
        assert len(training)==int(r['training_group_support'])
        expected=math.fsum(float(a[r['target']]) for a in training)/len(training)
        assert math.isclose(expected,float(r['predicted']),rel_tol=0,abs_tol=1e-12)
    parser=Report();text=(root/'CSI_maize_full_archive_report.html').read_text(encoding='utf-8');parser.feed(text)
    assert parser.images and all(r['alt'] for r in parser.images)
    for r in parser.images:
        assert r['src'].startswith('data:image/png;base64,')
        with Image.open(io.BytesIO(base64.b64decode(r['src'].split(',',1)[1]))) as im:im.verify()
    assert all(x in parser.ids for x in parser.anchors)
    assert 'Times New Roman' in text and 'background:white' in text
    assert '\u2014' not in text
    report=dict(status='passed',image_manifest_rows=len(images),reference_keys=len(master),
                complete_identified_acquisitions=len(complete),compared_diagnostic_frames=720,
                independently_checked_metadata_predictions=len(predictions),
                embedded_report_images=len(parser.images),
                raw_sha_scope='All images hashed at intake; diagnostic source hashes cross-checked in each replay. No second full-archive rehash.',
                html_sha256=sha(root/'CSI_maize_full_archive_report.html'))
    (root/'validation_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
