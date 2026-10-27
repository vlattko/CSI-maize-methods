"""CPU-only, resumable prediction controls for the 360-row maize assay cohort.

Native spectral values are instrument targets, never calibrated image spectra.
All transformations and ridge-alpha selection are fitted inside training folds.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

KEYS = ('treatment', 'measurement', 'genotype', 'replicate')
TARGETS = ('ojip__Fv/Fm', 'ojip__Pi_Abs', 'nir__NDVI')
ALPHAS = (0.1, 1., 10., 100., 1000.)
SEEDS = (1979, 2026, 104729)


def key(row):
    return tuple(str(row[k]) for k in KEYS)


def candidate(row):
    return tuple(str(row[k]) for k in ('treatment', 'genotype', 'replicate'))


def number(value):
    return float(value) if value not in ('', None) else np.nan


def truth(value):
    return str(value).strip().lower() in ('true', '1', 'yes')


def read_keyed(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError(f'Duplicate four-key records: {path}')
    return rows


def save_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_extraction(table):
    summary = json.loads((table.parent/'summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'completed' or summary['reference_samples'] != 360:
        raise ValueError('Full extraction must complete before image prediction')
    if summary['minimum_accepted_views'] != 20 or summary['required_quadrants'] != 4:
        raise ValueError('Unrecognized primary coverage operator')
    if summary['session_table_sha256'] != digest(table):
        raise ValueError('Final extraction table hash mismatch')
    return summary


def cohort_indices(rows, cohort):
    cameras = {'paired_primary': ('rgb','noir'), 'paired_strict30': ('rgb','noir'),
               'paired_strict40': ('rgb','noir'),
               'rgb_primary': ('rgb',), 'noir_primary': ('noir',), 'all360': ()}[cohort]
    threshold = 40 if cohort.endswith('40') else 30 if cohort.endswith('30') else 20
    return np.array([i for i,r in enumerate(rows)
                     if all(truth(r.get(c+'_eligible', False)) for c in cameras)
                     and all(number(r.get(c+'_usable_views')) >= threshold and
                             number(r.get(c+'_quadrants')) == 4 for c in cameras)], int)


def partitions(rows, mode):
    field = {'genotype': 'genotype', 'treatment': 'treatment'}[mode]
    for group in sorted({r[field] for r in rows}):
        train = np.array([i for i,r in enumerate(rows) if r[field] != group], int)
        test = np.array([i for i,r in enumerate(rows) if r[field] == group], int)
        if not len(train) or not len(test):
            raise ValueError('Empty outer partition')
        if {candidate(rows[i]) for i in train} & {candidate(rows[i]) for i in test}:
            raise ValueError('Candidate plant leakage')
        yield train, test, str(group)


def metadata_predict(rows, y, train, test, fields):
    buckets = defaultdict(list)
    for i in train:
        buckets[tuple(rows[i][f] for f in fields)].append(i)
    pred = []
    for i in test:
        bucket = buckets.get(tuple(rows[i][f] for f in fields))
        if not bucket:
            raise ValueError('No training support for test metadata; no fallback invented')
        pred.append(np.mean(y[bucket], axis=0))
    return np.array(pred)


def permute_trajectories(rows, y, train, seed):
    """Shuffle entire training candidate trajectories within treatment/pattern.

    Identical observed-visit patterns are required so missingness is retained.
    Test targets never enter the permutation or any training operation.
    """
    rng = np.random.default_rng(seed)
    plants = defaultdict(dict)
    for i in train:
        plants[candidate(rows[i])][str(rows[i]['measurement'])] = i
    strata = defaultdict(list)
    for plant, visits in plants.items():
        strata[(plant[0], tuple(sorted(visits)))].append(plant)
    result = y.copy()
    mapping = {}
    for members in strata.values():
        permuted = [members[j] for j in rng.permutation(len(members))]
        for recipient, donor in zip(members, permuted):
            for visit,i in plants[recipient].items():
                j = plants[donor][visit]
                mapping[i] = j
                result[i] = y[j]
    return result, mapping


def fit_outer(rows, x, y, train, test, seed=None):
    groups = np.array([r['genotype'] for r in rows])
    inner_groups = groups[train]
    unique = sorted(set(inner_groups))
    if len(unique) < 2:
        raise ValueError('At least two training genotypes required for nested tuning')
    yy = y
    mapping = {}
    if seed is not None:
        yy, mapping = permute_trajectories(rows, y, train, seed)
    estimator = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                              StandardScaler(), Ridge(solver='cholesky'))
    search = GridSearchCV(estimator, {'ridge__alpha': ALPHAS},
                          cv=LeaveOneGroupOut(), scoring='neg_mean_squared_error',
                          n_jobs=1, error_score='raise')
    with threadpool_limits(limits=2):
        if seed is None:
            search.fit(x[train], yy[train], groups=inner_groups)
            predicted = search.predict(x[test])
            alpha = float(search.best_params_['ridge__alpha'])
            best_mse = float(-search.best_score_)
        else:
            # Re-shuffle only the current inner training subset. Its validation
            # labels never enter the shuffle or the estimator fit.
            scores = []
            for alpha in ALPHAS:
                losses = []
                for inner_fold,(a,b) in enumerate(LeaveOneGroupOut().split(
                        x[train],groups=inner_groups)):
                    itrain, itest = train[a],train[b]
                    null_y,_ = permute_trajectories(rows,y,itrain,seed+inner_fold*1009)
                    model = clone(estimator).set_params(ridge__alpha=alpha)
                    model.fit(x[itrain],null_y[itrain])
                    pp = np.asarray(model.predict(x[itest])).reshape(len(itest),y.shape[1])
                    losses.append(float(np.mean((pp-y[itest])**2)))
                scores.append(float(np.mean(losses)))
            best = int(np.argmin(scores))
            alpha, best_mse = ALPHAS[best],scores[best]
            model = clone(estimator).set_params(ridge__alpha=alpha)
            model.fit(x[train],yy[train])
            predicted = model.predict(x[test])
    predicted = np.asarray(predicted).reshape(len(test), y.shape[1])
    return predicted, dict(alpha=float(alpha),
                           inner_folds=len(unique), inner_group='genotype',
                           inner_train_genotypes=';'.join(unique),
                           inner_best_mse=best_mse,
                           null_seed=seed,
                           permutation=json.dumps({str(i):int(j) for i,j in mapping.items()}))


def read_spectra(intake, band):
    source = intake/'research_audit'/'tables'/f'{band}_P6_transmitance__List1.csv'
    with source.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        waves = np.array([float(v) for v in header[4:]])
        values = {}
        for record in reader:
            kk = tuple(record[:4])
            if kk in values:
                raise ValueError('Duplicate spectral four-key record')
            values[kk] = np.array([number(v) for v in record[4:]])
    flags = defaultdict(set)
    if band == 'UVIS':
        source_flags = intake/'research_audit'/'uvis_consistency_issues.json'
        for issue in json.loads(source_flags.read_text(encoding='utf-8')):
            flags[tuple(str(v) for v in issue['key'])].add(issue['reason'])
    for kk, yy in values.items():
        if np.any(yy <= 0):
            flags[kk].add('nonpositive_transmittance')
        if not np.isfinite(yy).all():
            flags[kk].add('nonfinite_source_value')
    return waves, values, {kk: sorted(v) for kk,v in flags.items()}


def metadata_features(rows, transfer=False):
    # Protocol-level labels are declared here, not estimated from target data.
    visits = np.array([[str(r['measurement']) == str(v) for v in range(1,6)]
                       for r in rows], float)
    if transfer:
        return visits, [f'visit_{v}' for v in range(1,6)]
    interactions = np.array([[r['treatment'] == t and str(r['measurement']) == str(v)
                              for t in ('C','P','S') for v in range(1,6)]
                             for r in rows], float)
    return interactions, [f'treatment_{t}_visit_{v}' for t in ('C','P','S')
                          for v in range(1,6)]


def feature_sets(rows, image, mode, cohort):
    meta, meta_names = metadata_features(rows, transfer=mode == 'treatment')
    sets = {'metadata_ridge': (meta, meta_names)}
    if not image:
        return sets
    names = sorted(n for n in rows[0] if n.startswith(('rgb__','noir__'))
                   and n.endswith(('__median','__max','__iqr')))
    if not names:
        raise ValueError('No declared session image traits found')
    cameras = ('rgb','noir') if cohort.startswith('paired') else (cohort.split('_')[0],)
    for camera in cameras:
        selected = [n for n in names if n.startswith(camera+'__')]
        matrix = np.array([[number(r[n]) for n in selected] for r in rows])
        sets[camera] = matrix, selected
    if len(cameras) == 2:
        sets['combined'] = (np.hstack([sets[c][0] for c in cameras]),
                            sum([sets[c][1] for c in cameras], []))
    image_name = 'combined' if len(cameras) == 2 else cameras[0]
    matrix, image_names = sets[image_name]
    sets['metadata_plus_image'] = np.hstack([meta,matrix]), meta_names+image_names
    dino_cameras = []
    encoder_cameras = ((*cameras, *(['noir_blue'] if 'noir' in cameras else []))
                       if cohort == 'paired_primary' else ())
    for camera in encoder_cameras:
        selected = sorted(n for n in rows[0] if n.startswith('dino_'+camera+'__'))
        if selected:
            name = 'dino_'+camera
            sets[name] = np.array([[number(r[n]) for n in selected] for r in rows]),selected
            if camera != 'noir_blue':
                dino_cameras.append(camera)
    if dino_cameras:
        dn = 'dino_combined' if len(dino_cameras)==2 else 'dino_'+dino_cameras[0]
        if len(dino_cameras)==2:
            sets[dn] = (np.hstack([sets['dino_'+c][0] for c in dino_cameras]),
                        sum([sets['dino_'+c][1] for c in dino_cameras],[]))
        dm, names_dino = sets[dn]
        sets['metadata_plus_dino'] = np.hstack([meta,dm]),meta_names+names_dino
        if 'dino_rgb' in sets and 'dino_noir_blue' in sets:
            dm = np.hstack([sets['dino_rgb'][0],sets['dino_noir_blue'][0]])
            names_dino = sets['dino_rgb'][1]+sets['dino_noir_blue'][1]
            sets['dino_rgb_plus_noir_blue'] = dm,names_dino
    return sets


def metrics(y, prediction):
    errors = prediction-y
    denom = np.sum((y-y.mean(axis=0))**2, axis=0)
    r2 = np.divide(np.sum(errors**2, axis=0), denom,
                   out=np.full(denom.shape, np.nan), where=denom > 0)
    return dict(rmse=float(np.sqrt(np.mean(errors**2))),
                mae=float(np.mean(np.abs(errors))),
                mean_target_r2=float(np.nanmean(1-r2)) if np.any(denom > 0) else None)


def cluster_comparison(rows, y, prediction, baseline, seed=20261027):
    groups = sorted({r['genotype'] for r in rows})
    errors = np.mean((prediction-y)**2, axis=1)
    reference = np.mean((baseline-y)**2, axis=1)
    index = [np.array([i for i,r in enumerate(rows) if r['genotype'] == g]) for g in groups]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        idx = np.concatenate([index[j] for j in rng.integers(0,len(groups),len(groups))])
        values.append(float(np.sqrt(reference[idx].mean())-np.sqrt(errors[idx].mean())))
    lo, hi = np.quantile(values,[.025,.975])
    return dict(genotype_clusters=len(groups), rmse_improvement_vs_metadata=float(
        np.sqrt(reference.mean())-np.sqrt(errors.mean())),
        exploratory_cluster_bootstrap_low=float(lo), exploratory_cluster_bootstrap_high=float(hi))


def run_experiment(destination, rows, y, targets, *, mode, cohort, domain,
                   image, flags, signature, nulls, designs=None, null_models_override=None):
    destination.mkdir(parents=True, exist_ok=True)
    state = destination/'complete.json'
    if state.exists():
        old = json.loads(state.read_text())
        if old['signature'] != signature:
            raise ValueError(f'Completed experiment provenance mismatch: {destination}')
        print('RESUME', destination.name, flush=True)
        return old
    if len({r['genotype'] for r in rows}) != 6:
        raise ValueError('Every experiment requires all six genotype clusters')
    sets = feature_sets(rows, image, mode, cohort) if designs is None else designs
    save_json(destination/'features.json', {k:v[1] for k,v in sets.items()})
    save_csv(destination/'cohort.csv', [{**{k:r[k] for k in KEYS},
        'session': r.get('session',r.get('image_session','')),
        'source_flags': ';'.join(flags.get(key(r), []))} for r in rows])
    splits = list(partitions(rows, mode))
    means = {'training_mean': (), 'visit_mean': ('measurement',)}
    if mode == 'genotype':
        means.update(treatment_mean=('treatment',),
                     treatment_visit_mean=('treatment','measurement'))
    predictions = {}
    fold_records = []
    for model, fields in means.items():
        pp = np.full_like(y,np.nan)
        for train,test,label in splits:
            pp[test] = metadata_predict(rows,y,train,test,fields)
            fold_records.append(dict(model=model, outer_group=label,
                train_n=len(train),test_n=len(test), alpha=None,inner_folds=0,
                inner_group='',inner_train_genotypes='',inner_best_mse=None,
                null_seed=None,permutation='{}',
                train_keys=json.dumps([key(rows[i]) for i in train]),
                test_keys=json.dumps([key(rows[i]) for i in test])))
        predictions[model] = pp
    jobs = [(model,x,None) for model,(x,names) in sets.items()]
    if nulls:
        null_models = (['metadata_plus_image' if image else 'metadata_ridge']
                       if null_models_override is None else null_models_override)
        if null_models_override is None and 'metadata_plus_dino' in sets:
            null_models.append('metadata_plus_dino')
        jobs.extend((f'{null_model}_trajectory_null_{seed}',sets[null_model][0],seed)
                    for null_model in null_models for seed in SEEDS)
    for model,x,seed in jobs:
        checkpoint = destination/(model+'.npz')
        record_file = destination/(model+'_folds.json')
        if checkpoint.exists() and record_file.exists():
            with np.load(checkpoint) as loaded:
                if str(loaded['signature']) != signature:
                    raise ValueError('Checkpoint provenance mismatch')
                pp = loaded['prediction']
            records = json.loads(record_file.read_text())
        else:
            pp = np.full_like(y,np.nan)
            records = []
            for fold,(train,test,label) in enumerate(splits):
                actual_seed = None if seed is None else seed+fold*100003
                pp[test], detail = fit_outer(rows,x,y,train,test,seed=actual_seed)
                records.append(dict(model=model,outer_group=label,
                    train_n=len(train),test_n=len(test),**detail,
                    train_keys=json.dumps([key(rows[i]) for i in train]),
                    test_keys=json.dumps([key(rows[i]) for i in test])))
            np.savez_compressed(checkpoint, prediction=pp, signature=np.array(signature))
            save_json(record_file, records)
        predictions[model] = pp
        fold_records.extend(records)
        print(domain,cohort,mode,model,metrics(y,pp),flush=True)
    baseline = predictions['treatment_visit_mean' if mode == 'genotype' else 'visit_mean']
    summaries = []
    per_target = []
    per_genotype = []
    for model,pp in predictions.items():
        if not np.isfinite(pp).all():
            raise ValueError('Missing/nonfinite OOF prediction')
        summary = dict(domain=domain,cohort=cohort,split=mode,model=model,n=len(rows),
            target_channels=y.shape[1],**metrics(y,pp),
            **cluster_comparison(rows,y,pp,baseline))
        summaries.append(summary)
        for j,target in enumerate(targets):
            per_target.append(dict(model=model,target=target,n=len(rows),
                                   **metrics(y[:,j:j+1],pp[:,j:j+1])))
        for genotype in sorted({r['genotype'] for r in rows}):
            ii = np.array([i for i,r in enumerate(rows) if r['genotype'] == genotype])
            per_genotype.append(dict(model=model,genotype=genotype,n=len(ii),
                **metrics(y[ii],pp[ii])))
    save_csv(destination/'metrics.csv',summaries)
    save_csv(destination/'target_metrics.csv',per_target)
    save_csv(destination/'genotype_metrics.csv',per_genotype)
    save_csv(destination/'folds.csv',fold_records)
    # NPZ is the complete OOF record for spectra without 360*256*models CSV rows.
    np.savez_compressed(destination/'oof.npz', observed=y, targets=np.array(targets),
        keys=np.array([key(r) for r in rows]),
        session=np.array([r.get('session',r.get('image_session','')) for r in rows]),
        source_flags=np.array([';'.join(flags.get(key(r),[])) for r in rows]),
        outer_fold=np.array([r['genotype' if mode=='genotype' else 'treatment'] for r in rows]),
        **predictions)
    if domain.startswith('assay'):
        save_csv(destination/'oof.csv', [dict(**{k:r[k] for k in KEYS},model=model,
            target=target, observed=float(y[i,j]),predicted=float(pp[i,j]),
            outer_fold=r['genotype' if mode=='genotype' else 'treatment'])
            for model,pp in predictions.items() for i,r in enumerate(rows)
            for j,target in enumerate(targets)])
    result = dict(status='completed',signature=signature,domain=domain,cohort=cohort,
        split=mode,n=len(rows),target_channels=len(targets),models=list(predictions),
        summary=summaries)
    save_json(state,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('intake','run','repo'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=('metadata','images'),default='images')
    parser.add_argument('--no-nulls',action='store_true')
    parser.add_argument('--embeddings',type=Path,
                        help='Completed hash-bound360-row frozen encoder NPZ')
    args = parser.parse_args()
    if not (args.repo/'src'/'csi_maize').exists():
        raise ValueError('--repo must identify the public methods checkout')
    dest = args.run/'prediction'
    dest.mkdir(parents=True,exist_ok=True)
    source = args.intake/'research_audit'/'reference_master.csv'
    references = read_keyed(source)
    if len(references) != 360 or len({r['genotype'] for r in references}) != 6:
        raise ValueError('Expected 360 reference rows and six genotypes')
    image = args.stage != 'metadata'
    sessions_path = args.run/'extraction'/'session_traits.csv'
    joined = references
    hashes = dict(script=digest(__file__),reference=digest(source))
    if image:
        validate_extraction(sessions_path)
        sessions = read_keyed(sessions_path)
        lookup = {key(r):r for r in sessions}
        if set(lookup) != {key(r) for r in references} or len(sessions)!=360:
            raise ValueError('Image table must retain exactly the 360 reference keys')
        joined = [{**r,**lookup[key(r)]} for r in references]
        hashes['sessions'] = digest(sessions_path)
        if args.embeddings is not None:
            encoder_status = args.embeddings.parent/'progress.json'
            status = json.loads(encoder_status.read_text())
            if status['status'] != 'completed':
                raise ValueError('Frozen encoder stream must finish before image comparisons')
            with np.load(args.embeddings) as data:
                if str(data['signature']) != status['operator_signature']:
                    raise ValueError('Frozen encoder operator binding mismatch')
                embedding_keys = [tuple(v) for v in data['keys']]
                if len(set(embedding_keys))!=360 or set(embedding_keys)!=set(lookup):
                    raise ValueError('Frozen embeddings must retain exactly360 unique keys')
                order = {k:i for i,k in enumerate(embedding_keys)}
                for row in joined:
                    i = order[key(row)]
                    for camera in ('rgb','noir','noir_blue'):
                        for j,value in enumerate(data[camera][i]):
                            row[f'dino_{camera}__{j:03}'] = float(value)
            hashes['embeddings'] = digest(args.embeddings)
            hashes['encoder_operator'] = digest(args.embeddings.parent/'operator.json')
    spectra = {}
    for band in ('UVIS','NIR'):
        waves,values,flags = read_spectra(args.intake,band)
        spectra[band] = waves,values,flags
        hashes[band] = digest(args.intake/'research_audit'/'tables'/
                              f'{band}_P6_transmitance__List1.csv')
    flags_path = args.intake/'research_audit'/'uvis_consistency_issues.json'
    hashes['source_flags'] = digest(flags_path)
    config = dict(hashes=hashes,alphas=ALPHAS,null_seeds=SEEDS,
        nulls=not args.no_nulls,stage=args.stage,intake=str(args.intake),
        run=str(args.run),repo=str(args.repo),python=sys.version,
        limitations=['One experiment; six genotype clusters; no external validation.',
        'Metadata-only performance does not establish image physiology prediction.',
        'Source native transmittance grids are not calibrated spatial HSI.',
        'Cluster bootstrap intervals are exploratory with only six clusters.',
        'Repeated-label plant continuity is assumed only for partition protection.',
        'Three seeded trajectory shuffles diagnose label correspondence; no permutation p-value.',
        'Treatment transfer retains genotypes across treatments and tests unseen treatment only.'])
    save_json(dest/f'provenance_{args.stage}.json',config)
    (dest/f'execution_source_{args.stage}.py').write_bytes(Path(__file__).read_bytes())
    cohorts = ['all360'] if not image else ['paired_primary','paired_strict30','paired_strict40',
                                           'rgb_primary','noir_primary']
    results = []
    selection = []
    for cohort in cohorts:
        eligible = cohort_indices(joined,cohort)
        for i,r in enumerate(joined):
            selection.append(dict(**{k:r[k] for k in KEYS},cohort=cohort,
                eligible=bool(i in eligible),
                rgb_eligible=r.get('rgb_eligible',''),noir_eligible=r.get('noir_eligible',''),
                rgb_usable_views=r.get('rgb_usable_views',''),
                noir_usable_views=r.get('noir_usable_views','')))
        if not len(eligible):
            continue
        records = [joined[i] for i in eligible]
        domains = [('assay_'+label,(target,),np.array([[number(r[target])]
                     for r in records]),{}) for target,label in zip(
                     TARGETS,('FvFm','PiAbs','NIR_NDVI'))]
        for band,(waves,values,flags) in spectra.items():
            if any(key(r) not in values for r in records):
                raise ValueError('Missing spectral source key')
            domains.append((band,tuple(str(v) for v in waves),
                            np.array([values[key(r)] for r in records]),flags))
        for domain,targets,y,flags in domains:
            valid = np.isfinite(y).all(axis=1)
            for sensitivity in ('source_retained','source_flag_sensitivity') if flags else ('source_retained',):
                keep = valid.copy()
                if sensitivity == 'source_flag_sensitivity':
                    keep &= np.array([key(r) not in flags for r in records])
                rr = [r for i,r in enumerate(records) if keep[i]]
                yy = y[keep]
                for mode in ('genotype','treatment'):
                    # Modality-specific and 40-view cohorts are sensitivity analyses.
                    nulls = not args.no_nulls and cohort in ('all360','paired_primary')
                    identity = f'{cohort}__{domain}__{sensitivity}__{mode}'
                    signature = hashlib.sha256(json.dumps(dict(config=config,
                        identity=identity,keys=[key(r) for r in rr]),sort_keys=True).encode()).hexdigest()
                    results.append(run_experiment(dest/identity,rr,yy,targets,mode=mode,
                        cohort=cohort,domain=domain,image=image,flags=flags,
                        signature=signature,nulls=nulls))
                    save_json(dest/f'progress_{args.stage}.json',dict(
                        completed_experiments=len(results),last_experiment=identity,
                        updated_unix=time.time()))
    save_csv(dest/f'selection_{args.stage}.csv',selection)
    save_csv(dest/f'metrics_{args.stage}.csv',[s for r in results for s in r['summary']])
    save_json(dest/f'summary_{args.stage}.json',dict(status='completed',
        experiments=len(results),reference_rows=360,results=results,config=config))
    print('COMPLETED',len(results),'experiments',flush=True)


if __name__ == '__main__':
    main()
