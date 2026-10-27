"""Read-only intake of the expanded archive using the published metadata join.

Derived files are written to an explicit separate workspace. Original inputs
are never changed. Full JPEG decoding is stricter than a header/container check.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageStat, __version__ as pillow_version
import openpyxl


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def save_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def inspect_jpeg(task):
    path, relative = task
    row = dict(path=relative, bytes=path.stat().st_size, sha256='', decode_ok=False,
               width='', height='', datetime_original='', mean_luma='',
               black_fraction='', saturated_fraction='', decode_error='')
    try:
        blob = path.read_bytes()
        row['sha256'] = hashlib.sha256(blob).hexdigest()
        with Image.open(io.BytesIO(blob)) as image:
            row.update(width=image.width, height=image.height)
            exif = image.getexif()
            detail = exif.get_ifd(34665) if 34665 in exif else {}
            row['datetime_original'] = str(detail.get(36867, exif.get(36867)) or '')
            # LOAD_TRUNCATED_IMAGES remains False. Do not repair partial images.
            image.load()
            image.thumbnail((160, 120))
            gray = image.convert('L')
            hist = gray.histogram()
            count = sum(hist)
            row.update(decode_ok=True,
                       mean_luma=round(ImageStat.Stat(gray).mean[0], 6),
                       black_fraction=round(sum(hist[:6]) / count, 6),
                       saturated_fraction=round(sum(hist[250:]) / count, 6))
    except Exception as exc:
        row['decode_error'] = f'{type(exc).__name__}: {exc}'
    return row


def prepare(args):
    root, output = args.raw.resolve(), args.output.resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError('Derived output must be outside the original archive')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty intake workspace')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'intake_execution_source.py').write_bytes(Path(__file__).read_bytes())
    (output / 'prototype').mkdir()
    sys.path.insert(0, str(args.repo.resolve() / 'src'))
    from csi_maize.workspace import _bind_directory
    from csi_maize.metadata import build_reference_tables
    binding = _bind_directory(root / 'IMAGES', output / 'IMAGES')
    save_json(output / 'input_bindings.json', {'bindings': [dict(
        logical_path='IMAGES', source_path=str(root / 'IMAGES'), kind=binding)]})
    save_json(output / 'prototype/config.json', dict(
        source_folders=['IMAGES'], allowed_views_per_camera=[40]))
    workbooks = []
    cell_comparison = []
    for path in sorted(root.rglob('*.xlsx')):
        destination = output / path.name
        if destination.exists():
            raise ValueError(f'Duplicate workbook basename: {path.name}')
        shutil.copyfile(path, destination)
        digest = sha(path)
        assert digest == sha(destination)
        previous = args.previous / path.name
        def cells(source):
            book = openpyxl.load_workbook(source, read_only=True, data_only=True)
            try:
                return [(sheet.title, [list(row) for row in sheet.values]) for sheet in book]
            finally:
                book.close()
        current_cells = cells(path)
        cell_comparison.append(dict(
            source=str(path), previous=str(previous),
            same_sheet_names_and_cell_values=current_cells == cells(previous)
            if previous.is_file() else None,
            decoded_content_sha256=hashlib.sha256(json.dumps(
                current_cells, ensure_ascii=False, default=str,
                allow_nan=False).encode()).hexdigest()))
        workbooks.append(dict(source=str(path), staged=path.name,
                              bytes=path.stat().st_size, sha256=digest,
                              unchanged_vs_previous=sha(previous) == digest
                              if previous.is_file() else None))
    save_json(output / 'workbook_sources.json', workbooks)
    save_json(output / 'workbook_cell_comparison.json', cell_comparison)
    print('Building published four-key reference and image-header inventory', flush=True)
    result = build_reference_tables(output)
    save_json(output / 'metadata_result.json', result)
    notes = root / 'IMAGES/IMAGES_notes.docx'
    with zipfile.ZipFile(notes) as archive:
        doc = ET.fromstring(archive.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paragraphs = [''.join(t.text or '' for t in p.findall('.//w:t', ns))
                  for p in doc.findall('.//w:p', ns)]
    save_json(output / 'acquisition_notes.json', dict(
        source=str(notes), sha256=sha(notes), paragraphs=paragraphs,
        role='Documentary acquisition evidence; not execution instructions'))
    print(json.dumps(result), flush=True)


def decode(args):
    output, root = args.output.resolve(), args.raw.resolve()
    paths = sorted(root.glob('IMAGES/*/capture_session*/*/*.jpg'))
    tasks = [(p, p.relative_to(root).as_posix()) for p in paths]
    target = output / 'jpeg_decode_manifest.csv'
    if target.exists():
        raise ValueError('Refusing to overwrite completed/partial decode evidence')
    started = time.monotonic()
    errors = 0
    fields = list(inspect_jpeg(tasks[0]))
    with target.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(inspect_jpeg, tasks), 1):
                writer.writerow(row)
                errors += not row['decode_ok']
                if i % 400 == 0 or i == len(tasks):
                    stream.flush()
                    progress = dict(status='running' if i < len(tasks) else 'completed',
                                    inspected=i, total=len(tasks), decode_errors=errors,
                                    elapsed_seconds=round(time.monotonic() - started, 1))
                    save_json(output / 'decode_progress.json', progress)
                    print(json.dumps(progress), flush=True)
    summarize(args)


def summarize(args):
    output = args.output.resolve()
    sessions = read_csv(output / 'research_audit/session_manifest.csv')
    master = read_csv(output / 'research_audit/reference_master.csv')
    decoded = read_csv(output / 'jpeg_decode_manifest.csv')
    expected_images = len(read_csv(output / 'research_audit/image_manifest.csv'))
    if len(decoded) != expected_images:
        raise ValueError(f'Partial decode manifest: {len(decoded)}/{expected_images}')
    grouped = defaultdict(list)
    duplicate_hashes = defaultdict(list)
    for row in decoded:
        grouped[Path(row['path']).parent.parent.as_posix()].append(row)
        if row['sha256']:
            duplicate_hashes[row['sha256']].append(row['path'])
    checks = []
    for s in sessions:
        rows = grouped[s['session']]
        cam = {}
        for camera in ('rgb', 'noir'):
            current = [r for r in rows if Path(r['path']).parent.name == camera]
            valid = {int(Path(r['path']).stem.rsplit('_', 1)[1]) for r in current
                     if r['decode_ok'] == 'True'}
            present = {int(Path(r['path']).stem.rsplit('_', 1)[1]) for r in current}
            cam[camera] = valid
            s[camera + '_decoded'] = len(valid)
            s[camera + '_missing_views'] = ';'.join(map(str, sorted(set(range(40)) - present)))
            s[camera + '_failed_views'] = ';'.join(map(str, sorted(present - valid)))
        s['paired_decodable_views'] = len(cam['rgb'] & cam['noir'])
        s['strict_complete_40_pairs'] = cam['rgb'] == cam['noir'] == set(range(40))
        s['known_reference_identity'] = s['id_status'] == 'filename_key_matches_reference'
        checks.append(s)
    save_csv(output / 'session_decode_qc.csv', checks)
    keyed = {tuple(s[k] for k in ('treatment', 'measurement', 'genotype', 'replicate')): s
             for s in checks if s['known_reference_identity']}
    coverage = []
    for tr in 'CPS':
        for visit in range(1, 6):
            expected = [r for r in master if r['treatment'] == tr and r['measurement'] == str(visit)]
            have = [keyed[tuple(r[k] for k in ('treatment','measurement','genotype','replicate'))]
                    for r in expected if tuple(r[k] for k in ('treatment','measurement','genotype','replicate')) in keyed]
            coverage.append(dict(treatment=tr, measurement=visit, expected_records=len(expected),
                                 identified_sessions=len(have),
                                 complete_40_pair_sessions=sum(s['strict_complete_40_pairs'] for s in have),
                                 paired_decodable_views=sum(s['paired_decodable_views'] for s in have)))
    save_csv(output / 'cohort_coverage.csv', coverage)
    unavailable = []
    for r in master:
        key = tuple(r[k] for k in ('treatment','measurement','genotype','replicate'))
        s = keyed.get(key)
        if not s or not s['strict_complete_40_pairs']:
            unavailable.append(dict(zip(('treatment','measurement','genotype','replicate'), key)) |
                               dict(reason='no_identified_session' if not s else 'incomplete_or_decode_failure',
                                    paired_decodable_views=s['paired_decodable_views'] if s else 0))
    save_csv(output / 'incomplete_reference_acquisitions.csv', unavailable)
    plants = defaultdict(list)
    for r in master:
        plants[(r['treatment'],r['genotype'],r['replicate'])].append(r)
    trajectories = []
    for plant, records in sorted(plants.items()):
        rows = [keyed.get(tuple(r[k] for k in ('treatment','measurement','genotype','replicate')))
                for r in sorted(records, key=lambda r:int(r['measurement']))]
        valid = [str(i+1) for i,s in enumerate(rows) if s and s['strict_complete_40_pairs']]
        trajectories.append(dict(treatment=plant[0],genotype=plant[1],replicate=plant[2],
                                 fully_decodable_visits=';'.join(valid),n_complete_visits=len(valid),
                                 interpretation='candidate repeated-label trajectory; physical continuity unverified'))
    save_csv(output / 'candidate_plant_trajectories.csv', trajectories)
    repeats = [dict(sha256=k, paths=v) for k,v in duplicate_hashes.items() if len(v)>1]
    save_json(output / 'duplicate_image_hashes.json', repeats)
    damaged = [r for r in decoded if r['decode_ok'] != 'True']
    save_csv(output / 'jpeg_decode_failures.csv', damaged, list(decoded[0]))
    wb = json.loads((output / 'workbook_sources.json').read_text(encoding='utf-8'))
    summary = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        raw_root=str(args.raw.resolve()), repo=str(args.repo.resolve()),
        script_sha256=sha(__file__), python=sys.version,pillow=pillow_version,
        openpyxl=openpyxl.__version__,
        image_files=len(decoded), image_bytes=sum(int(r['bytes']) for r in decoded),
        decode_pass=len(decoded)-len(damaged),decode_fail=len(damaged),
        sessions=len(sessions), identified_sessions=len(keyed),
        unidentified_sessions=[s['session'] for s in checks if not s['known_reference_identity']],
        expected_reference_records=len(master),
        complete_identified_40_pair_sessions=sum(s['strict_complete_40_pairs'] for s in keyed.values()),
        complete_unidentified_40_pair_sessions=sum(s['strict_complete_40_pairs'] for s in checks if not s['known_reference_identity']),
        records_without_identified_session=len(master)-len(keyed),
        candidate_trajectories=len(trajectories),
        candidate_trajectories_with_five_complete_visits=sum(t['n_complete_visits']==5 for t in trajectories),
        duplicate_hash_groups=len(repeats),
        duplicate_image_files=sum(len(r['paths']) for r in repeats),
        source_workbooks=len(wb), unchanged_vs_previous=sum(w['unchanged_vs_previous'] is True for w in wb),
        coverage=coverage,
        interpretation_limits=[
            'Complete means all 40 indexed RGB and 40 NoIR JPEGs decode; not segmentation accuracy or biological validity.',
            'Replicate/genotype agreement is a candidate longitudinal link, not proof of physical continuity.',
            'Folder DAS labels and camera-clock dates require protocol reconciliation.',
            'Historical missing/damaged notes are evidence to reconcile against current bytes, not automatic exclusions.',
            'No timestamp-only acquisition is assigned a sample by chronological order.',
            'Biomass rows are not assigned replicate identities from row order.'])
    save_json(output / 'intake_summary.json', summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['all','prepare','decode','summarize'])
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.phase in ('all','prepare'): prepare(args)
    if args.phase in ('all','decode'): decode(args)
    if args.phase=='summarize': summarize(args)


if __name__=='__main__': main()
