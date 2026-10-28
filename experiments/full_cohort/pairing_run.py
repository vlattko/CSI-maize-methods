"""Paired JPEG-DN descriptors and silhouette registration for all fixed quarters.

No calibrated reflectance, narrow-band response or corresponding tissue is
established. Whole-mask common support is an engineering gate, never leaf truth.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import shutil
from collections import Counter

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from pairing_core import (QUARTERS, IDENTITY, read_csv, write_csv, sha, digest,
                          atomic_json, true, frame_lookup, wrong_plant,
                          moment_affine, cache_read)

HASH_CACHE = {}


def verified_hash(path):
    """Hash immutable inputs once per observed size/mtime, within this process."""
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        HASH_CACHE[key] = sha(path)
    return HASH_CACHE[key]


def configure(repo):
    global np, cv2, Image, affine_align, mask_iou, warp_to_rgb, interior
    global CHANNELS, POLICY, spatial_indices, index_statistics, index_definitions
    sys.path.insert(0, str(repo / 'src/csi_maize/_methods'))
    import numpy as np
    import cv2
    from PIL import Image
    from multimodal_leaf_core import affine_align, mask_iou
    from channel_fusion_core import warp_to_rgb, interior, CHANNELS, POLICY
    from spatial_fusion_core import spatial_indices, index_statistics, index_definitions
    cv2.setNumThreads(2)


def load(row, intake, mask_root, decoded):
    if row is None:
        raise ValueError('extraction_record_unavailable')
    source = intake / row['source']
    manifest = decoded.get(row['source'])
    if not manifest or not true(manifest['decode_ok']):
        raise ValueError('intake_decode_not_valid')
    actual = verified_hash(source)
    if actual != manifest['sha256'] or actual != row['source_sha256']:
        raise ValueError('source_hash_mismatch')
    if not true(row.get('qc_pass')):
        raise ValueError('mask_qc_rejected:' + str(row.get('qc_issues', 'unspecified')))
    mask_path = mask_root / row['mask']
    mask_digest = verified_hash(mask_path)
    if row.get('mask_sha256') and row['mask_sha256'] != mask_digest:
        raise ValueError('mask_hash_mismatch')
    with Image.open(source) as im:
        im.load()
        image = np.asarray(im.convert('RGB').resize((512, 385), Image.Resampling.LANCZOS))
    with Image.open(mask_path) as im:
        declared=(int(row['processing_width']),int(row['processing_height']))
        if im.size != declared:
            raise ValueError('mask_processing_dimensions_mismatch')
        mask = np.asarray(im.convert('L').resize((512, 385), Image.Resampling.NEAREST)) > 0
    if mask.sum() < 10:
        raise ValueError('empty_or_tiny_foreground')
    return image, mask, dict(source=row['source'], source_sha256=actual,
                            mask=row['mask'], mask_sha256=mask_digest)


def describe(a, am, b, bm, matrix, usable):
    wb, field = warp_to_rgb(b.astype(np.float32), matrix, am.shape)
    wn, _ = warp_to_rgb(interior(bm).astype(np.uint8), matrix, am.shape, True)
    ai = interior(am)
    ni = (wn > 0) & field
    common = ai & ni
    # Require *all* nonzero interpolation contributors to be valid before the
    # interpolation. A bright averaged value cannot repair clipped/dark inputs.
    native_valid = ((b > 1) & (b < 254)).astype(np.float32)
    warped_valid, _ = warp_to_rgb(native_valid, matrix, am.shape)
    six = np.concatenate([a.astype(np.float32), wb], axis=-1)
    channel_valid = np.concatenate([(a > 1) & (a < 254), warped_valid >= 1 - 1e-7], axis=-1)
    supports = dict(rgb=ai, noir=ni, cross=common if usable else np.zeros_like(common))
    maps = spatial_indices(six, supports, channel_valid)
    descriptors = []
    for definition in index_definitions():
        family = definition['support']
        descriptors.append(dict(index=definition['id'], family=family,
                                **index_statistics(maps[definition['id']], supports[family])))
    channels = []
    for i, name in enumerate(CHANNELS):
        eligible = supports['rgb' if i < 3 else 'noir']
        values = six[..., i][eligible]
        n = int(eligible.sum())
        # Unregistered processing-grid saturation is separate in camera_qc.
        valid_n = int((eligible & channel_valid[..., i]).sum())
        channels.append(dict(channel=name, eligible_pixels=n, valid_pixels=valid_n,
                             valid_fraction=valid_n/n if n else None,
                             median=float(np.median(values)) if n else None,
                             dark_fraction=float(np.mean(values <= 1)) if n else None,
                             clipped_fraction=float(np.mean(values >= 254)) if n else None))
    return descriptors, channels, maps, supports, wb, common


def camera_qc(image, mask):
    values = image[interior(mask)]
    return dict(pixels=len(values),
                dark_fraction=np.mean(values <= 1, axis=0).tolist() if len(values) else None,
                clipped_fraction=np.mean(values >= 254, axis=0).tolist() if len(values) else None)


def evaluate(a, am, b, bm, kind, method, fixed=None, figure=None):
    initial = moment_affine(am, bm, np)
    optimizer = dict(status='not_attempted')
    if fixed is not None:
        matrix, status = fixed, 'nominal_transform_without_refitting'
    elif method == 'moment':
        matrix, status = initial, 'moment_initialisation'
    else:
        # Observe the exact original solver invocation without changing inputs,
        # output, iteration policy, exception propagation or fallback selection.
        # The public operator catches cv2.error; this audit retains that reason.
        original_solver = cv2.findTransformECC
        def audited_solver(*args, **kwargs):
            try:
                result = original_solver(*args, **kwargs)
                optimizer.update(status='converged',correlation=float(result[0]))
                return result
            except cv2.error as exc:
                optimizer.update(status='opencv_error',reason=str(exc))
                raise
        cv2.findTransformECC = audited_solver
        try:
            matrix, _, status = affine_align(am, bm)
        finally:
            cv2.findTransformECC = original_solver
    if not np.isfinite(matrix).all():
        raise ValueError('nonfinite_affine_matrix')
    warped_mask, _ = warp_to_rgb(bm.astype(np.uint8), matrix, am.shape, True)
    iou = mask_iou(am, warped_mask > 0)
    ni, field = warp_to_rgb(interior(bm).astype(np.uint8), matrix, am.shape, True)
    common = interior(am) & (ni > 0) & field
    fraction = float(common.sum() / max(1, min(interior(am).sum(), (ni > 0).sum())))
    reasons = []
    if iou < POLICY['whole_iou_min']:
        reasons.append('whole_mask_IoU_below_gate')
    if common.sum() < POLICY['min_common_interior_pixels']:
        reasons.append('too_few_common_interior_pixels')
    if fraction < POLICY['min_interior_overlap_fraction']:
        reasons.append('low_common_interior_fraction')
    record = dict(pair_kind=kind, method=method, registration_status=status,
                  ECC_optimizer=optimizer,
                  source_to_target=matrix.tolist(), whole_iou_rgb_grid=iou,
                  common_interior_pixels=int(common.sum()), common_interior_fraction=fraction,
                  cross_support_usable=not reasons, failure_reasons=reasons,
                  determinant=float(np.linalg.det(matrix[:, :2])),
                  independently_verified_correspondence=False)
    descriptors, channels, maps, supports, wb, common = describe(a, am, b, bm, matrix, not reasons)
    record.update(descriptors=descriptors, channels=channels)
    if figure:
        panels = [a.copy(), np.clip(wb, 0, 255).astype(np.uint8)]
        panels[0][~am] = 0
        panels[1][warped_mask == 0] = 0
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for key in ('ND_rgb_R__noir_B', 'CAMSUM6', 'GLICONTRAST6', 'ExG'):
            values = maps[key]
            definition = next(x for x in index_definitions() if x['id'] == key)
            color = (plt.colormaps['viridis'](np.clip((values-definition['vmin']) /
                     (definition['vmax']-definition['vmin']), 0, 1))[..., :3] * 255).astype(np.uint8)
            color[~np.isfinite(values)] = 0
            panels.append(color)
        Image.fromarray(np.concatenate(panels, axis=1)).save(figure)
    return record, matrix


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('intake', 'run', 'repo'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--mode', choices=('pilot', 'full'), default='full')
    p.add_argument('--max-pairs', type=int)
    p.add_argument('--watch', action='store_true', help='Resume snapshots until all decodable quarter records exist')
    p.add_argument('--poll-seconds', type=float, default=30)
    return p.parse_args()


def run_pass(args):
    intake, run, repo = [getattr(args, k).resolve() for k in ('intake', 'run', 'repo')]
    configure(repo)
    output = run/'pairing'/args.mode
    output.mkdir(parents=True, exist_ok=True)
    (output/'records').mkdir(exist_ok=True)
    (output/'figures').mkdir(exist_ok=True)
    source = (intake/'segmentation_triage/view_diagnostics.csv' if args.mode == 'pilot'
              else run/'extraction/view_traits.csv')
    mask_root = intake if args.mode == 'pilot' else run
    record_directory = run/'extraction/records'
    if args.mode=='full' and record_directory.is_dir():
        rows = [json.loads(p.read_text(encoding='utf-8')) for p in record_directory.glob('*.json')]
        source = record_directory
    else:
        rows = read_csv(source)
    frames = frame_lookup(rows)
    sessions = [r for r in read_csv(intake/'session_decode_qc.csv') if true(r['known_reference_identity'])]
    if len(sessions) != 360:
        raise ValueError('Expected all 360 four-key assay acquisitions in intake')
    if args.mode == 'pilot':
        sessions = [s for s in sessions if s['session'] in {r['session'] for r in rows}]
    session_lookup = {s['session']:s for s in sessions}
    for row in rows:
        if row['session'] not in session_lookup:
            continue
        expected_source = f"{row['session']}/{row['camera']}/image_{row['camera']}_{int(row['view']):03d}.jpg"
        if row['source'] != expected_source:
            raise ValueError('Wrong source/frame join: '+row['source'])
        if any(str(row[k])!=str(session_lookup[row['session']][k]) for k in IDENTITY):
            raise ValueError('Wrong four-key identity join: '+row['source'])
    decoded = {r['path']: r for r in read_csv(intake/'jpeg_decode_manifest.csv')}
    expected_sources = {f"{s['session']}/{camera}/image_{camera}_{view:03d}.jpg"
                        for s in sessions for camera in ('rgb','noir') for view in QUARTERS}
    expected_decodable = {s for s in expected_sources if s in decoded and true(decoded[s]['decode_ok'])}
    available_sources = {r['source'] for r in rows}
    pending_sources = sorted(expected_decodable - available_sources)
    methods = repo/'src/csi_maize/_methods'
    scientific = {str(p.name): sha(p) for p in methods.glob('*.py')}
    upstream_paths = (list((intake/'segmentation_triage').glob('configuration.json'))
                      if args.mode=='pilot' else list((run/'extraction').glob('configuration*.json')))
    upstream = {p.name:dict(sha256=sha(p),contents=json.loads(p.read_text(encoding='utf-8-sig')))
                for p in upstream_paths}
    if args.mode=='full':
        declared_signatures={v['contents']['signature'] for v in upstream.values()}
        if not declared_signatures:
            raise ValueError('No full-extraction model/operator configurations available')
        if any(r.get('configuration_signature') not in declared_signatures for r in rows):
            raise ValueError('Extraction record is not bound to the recorded model/operator configuration')
    config = dict(mode=args.mode, processing_grid=[512,385], fixed_views=QUARTERS,
                  channels=CHANNELS, policy=POLICY, scientific_source_sha256=scientific,
                  signal_validity_scope='512x385 JPEG processing grid before affine interpolation; not native-sensor clipping',
                  producer_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('pairing*.py')},
                  decode_manifest_sha256=sha(intake/'jpeg_decode_manifest.csv'),
                  intake_sessions_sha256=sha(intake/'session_decode_qc.csv'),
                  extraction_source=str(source),
                  segmentation_provenance=upstream,
                  provenance='90-acquisition stratified box-mask diagnostic pilot' if args.mode=='pilot'
                             else 'full 360-assay cohort; incomplete sessions retained',
                  support_scope='whole-plant mask interiors, without anatomical correspondence',
                  control_selection='wrong view +20 modulo 40; wrong candidate identity matched labels',
                  original_affine_ECC_iterations=80, cpu_threads=2,
                  versions=dict(numpy=np.__version__,opencv=cv2.__version__,python=sys.version),
                  limitations=['Processed JPEG DN only; no reflectance or calibrated NIR band.',
                               'Silhouette overlap does not validate corresponding tissue.',
                               'Quarter views are technical observations within an acquisition.',
                               'QC and overlap gates are engineering choices, not anatomical accuracy.'])
    config_signature = digest(config)
    snapshot = output/'source_snapshot'/config_signature
    snapshot.mkdir(parents=True, exist_ok=True)
    for path in Path(__file__).parent.glob('pairing*.py'):
        target = snapshot/path.name
        if not target.exists():
            shutil.copyfile(path,target)
    table_hash = sha(source) if source.is_file() else digest(sorted(
        (r['source'],digest(r)) for r in rows))
    atomic_json(output/'configuration.json', dict(config, signature=config_signature,
                                                extraction_table_sha256=table_hash))
    atomic_json(output/'index_definitions.json', index_definitions())
    allresults = []
    started = time.monotonic()
    planned = len(sessions)*len(QUARTERS)
    for session in sorted(sessions, key=lambda s:s['session']):
        other, matching = wrong_plant(sessions, session)
        for view in QUARTERS:
            if args.max_pairs and len(allresults) >= args.max_pairs:
                break
            refs = dict(rgb=frames.get((session['session'],'rgb',view)),
                        nominal=frames.get((session['session'],'noir',view)),
                        wrong_view=frames.get((session['session'],'noir',(view+20)%40)),
                        wrong_plant=frames.get((other['session'],'noir',view)) if other else None)
            bindings = {}
            for label, row in refs.items():
                bindings[label] = dict(record=row,
                                       actual_mask_sha256=verified_hash(mask_root/row['mask'])
                                       if row and row.get('mask') and (mask_root/row['mask']).exists() else None,
                                       actual_source_sha256=verified_hash(intake/row['source'])
                                       if row and (intake/row['source']).exists() else None)
            signature = digest(dict(configuration=config_signature, bindings=bindings))
            key = Path(session['session']).name+f'__{view:03d}'
            cache = output/'records'/(key+'.json')
            result = cache_read(cache, signature)
            if result is None:
                result = dict(signature=signature, session=session['session'], view=view,
                              **{k:session[k] for k in IDENTITY},
                              source_bindings=bindings, wrong_plant_session=other['session'] if other else None,
                              wrong_plant_matching=matching, evaluations=[], failures=[])
                loaded = {}
                for label, row in refs.items():
                    try:
                        if row is None:
                            expected_session = other['session'] if label=='wrong_plant' and other else session['session']
                            expected_camera = 'rgb' if label=='rgb' else 'noir'
                            expected_view = (view+20)%40 if label=='wrong_view' else view
                            expected_source = f'{expected_session}/{expected_camera}/image_{expected_camera}_{expected_view:03d}.jpg'
                            info = decoded.get(expected_source)
                            if not info:
                                raise ValueError('source_missing_from_intake')
                            if not true(info['decode_ok']):
                                raise ValueError('intake_decode_failed:'+info.get('decode_error',''))
                        loaded[label] = load(row, intake, mask_root, decoded)
                    except Exception as exc:
                        result['failures'].append(dict(pair_kind=label, reason=type(exc).__name__+':'+str(exc)))
                if 'rgb' in loaded:
                    a, am, _ = loaded['rgb']
                    transforms = {}
                    for kind in ('nominal','wrong_view','wrong_plant'):
                        if kind not in loaded:
                            continue
                        b, bm, _ = loaded[kind]
                        for method in ('moment','ECC'):
                            try:
                                figure = output/'figures'/(key+'.png') if (kind=='nominal' and method=='ECC' and view==0) else None
                                evaluation, matrix = evaluate(a, am, b, bm, kind, method, figure=figure)
                                evaluation['rgb_camera_qc'] = camera_qc(a, am)
                                evaluation['noir_camera_qc'] = camera_qc(b, bm)
                                result['evaluations'].append(evaluation)
                                if kind=='nominal':
                                    transforms[method] = matrix
                                elif method in transforms:
                                    fixed, _ = evaluate(a, am, b, bm, kind+'_fixed', method, fixed=transforms[method])
                                    result['evaluations'].append(fixed)
                            except Exception as exc:
                                result['failures'].append(dict(pair_kind=kind,method=method,
                                                              reason=type(exc).__name__+':'+str(exc)))
                result['payload_sha256'] = digest(result)
                atomic_json(cache, result)
            allresults.append(result)
            if len(allresults)%10==0:
                atomic_json(output/'progress.json',dict(status='running',processed=len(allresults),planned=planned,
                                                        elapsed_seconds=time.monotonic()-started))
                print(json.dumps(dict(processed=len(allresults), planned=planned,
                                      elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
    registration, descriptors, channels, availability = [], [], [], []
    for result in allresults:
        base = {k:result[k] for k in ('session','view',*IDENTITY)}
        availability.append(dict(base, nominal_ready=all(x['pair_kind'] not in ('rgb','nominal')
                                                        for x in result['failures']),
                                 failures=result['failures'], wrong_plant_session=result['wrong_plant_session'],
                                 wrong_plant_matching=result['wrong_plant_matching']))
        for evaluation in result['evaluations']:
            rb = dict(base, **{k:v for k,v in evaluation.items() if k not in ('descriptors','channels')})
            registration.append(rb)
            eb = dict(base,pair_kind=evaluation['pair_kind'],method=evaluation['method'])
            descriptors.extend(dict(eb,**v) for v in evaluation['descriptors'])
            channels.extend(dict(eb,**v) for v in evaluation['channels'])
    write_csv(output/'registration.csv', registration)
    write_csv(output/'descriptors_27.csv', descriptors)
    write_csv(output/'channel_qc.csv', channels)
    write_csv(output/'availability.csv', availability)
    groups = []
    for kind in ('nominal','wrong_view','wrong_plant','wrong_view_fixed','wrong_plant_fixed'):
        for method in ('moment','ECC'):
            selected = [r for r in registration if r['pair_kind']==kind and r['method']==method]
            groups.append(dict(pair_kind=kind,method=method,planned_anchor_pairs=planned,
                               executed=len(selected),fraction_of_planned=len(selected)/planned,
                               cross_support_usable=sum(r['cross_support_usable'] for r in selected),
                               whole_iou_median=float(np.median([r['whole_iou_rgb_grid'] for r in selected])) if selected else None,
                               determinant_nonpositive=sum(r['determinant']<=0 for r in selected),
                               determinant_min=min((r['determinant'] for r in selected),default=None),
                               determinant_max=max((r['determinant'] for r in selected),default=None),
                               status_counts=dict(Counter(r['registration_status'] for r in selected))))
    paired_comparisons = []
    for method in ('moment','ECC'):
        for kind in ('wrong_view','wrong_plant','wrong_view_fixed','wrong_plant_fixed'):
            pairs = []
            for result in allresults:
                ev = {r['pair_kind']:r for r in result['evaluations'] if r['method']==method}
                if 'nominal' in ev and kind in ev:
                    pairs.append((ev['nominal'],ev[kind]))
            delta = [a['whole_iou_rgb_grid']-b['whole_iou_rgb_grid'] for a,b in pairs]
            paired_comparisons.append(dict(method=method,control=kind,paired_anchors=len(pairs),
                                           nominal_minus_control_median=float(np.median(delta)) if delta else None,
                                           nominal_greater_count=sum(d>0 for d in delta),
                                           interpretation='within-acquisition technical comparison; no independent biological p value'))
    summary = dict(status=('waiting_for_extraction' if pending_sources else 'completed')
                   if len(allresults)==planned else 'partial_explicit_limit',
                   mode=args.mode,sessions=len(sessions),planned_anchor_pairs=planned,
                   expected_nominal_source_slots=len(expected_sources),
                   expected_decodable_quarter_sources=len(expected_decodable),
                   extraction_records_present=len(expected_decodable & available_sources),
                   pending_decodable_quarter_sources=len(pending_sources),
                   retained_anchor_pairs=len(allresults),evaluated_rows=len(registration),
                   descriptor_rows=len(descriptors),definitions=27,
                   failures=dict(Counter(f['reason'] for r in allresults for f in r['failures'])),
                   experiments=groups,paired_comparisons=paired_comparisons,
                   elapsed_seconds=time.monotonic()-started,limitations=config['limitations'])
    atomic_json(output/'summary.json',summary)
    atomic_json(output/'pending_sources.json',pending_sources)
    atomic_json(output/'progress.json',dict(status=summary['status'],processed=len(allresults),planned=planned))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    args = parse()
    while True:
        summary = run_pass(args)
        if not args.watch or summary['status'] != 'waiting_for_extraction':
            break
        time.sleep(max(5, args.poll_seconds))


if __name__=='__main__':
    main()
