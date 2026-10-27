"""Recompute output invariants without refitting any full-cohort estimator."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path,PurePosixPath
import numpy as np
import prediction as p


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def audit_preflight(intake,run,repo):
    """Verify completed atomic frame records before final tables are written.

    The prior full-decode manifest binds original image bytes. This bounded
    preflight rehashes current masks and frozen operators/source snapshots;
    it does not reread tens of gigabytes of unchanged original JPEGs.
    """
    identified = {}
    for row in read_rows(intake/'session_decode_qc.csv'):
        if not p.truth(row['known_reference_identity']):
            continue
        require(row['session'] not in identified,'Duplicate identified session alias')
        identified[row['session']] = row
    tasks = []
    for row in read_rows(intake/'jpeg_decode_manifest.csv'):
        source = PurePosixPath(row['path'])
        session = source.parent.parent.as_posix()
        if not p.truth(row['decode_ok']) or session not in identified:
            continue
        require(not source.is_absolute() and '..' not in source.parts and
                source.parts[0]=='IMAGES','Invalid intake source path')
        camera = source.parent.name
        view = int(source.stem.rsplit('_',1)[1])
        require(camera in ('rgb','noir') and 0<=view<40,'Invalid intake camera/view')
        tasks.append(dict(source=row['path'],source_sha256=row['sha256'],
                          session=session,camera=camera,view=view,
                          **{k:identified[session][k] for k in p.KEYS}))
    tasks.sort(key=lambda r:r['source'])
    require(len({r['source'] for r in tasks})==len(tasks),'Duplicate intake source task')
    task_hash = hashlib.sha256(json.dumps(
        [(r['source'],r['source_sha256']) for r in tasks]).encode()).hexdigest()
    folder = run/'extraction'
    configs = {}
    for path in folder.glob('configuration_shard_*.json'):
        config = json.loads(path.read_text(encoding='utf-8'))
        shard = int(path.stem.rsplit('_',1)[1])
        require(config['tasks']==len(tasks) and config['task_digest']==task_hash,
                'Configuration task inventory mismatch')
        unsigned = {k:v for k,v in config.items() if k!='signature'}
        require(config['signature']==hashlib.sha256(json.dumps(
            unsigned,sort_keys=True).encode()).hexdigest(),'Configuration self-hash mismatch')
        for name,expected in config['operator_hashes'].items():
            require(p.digest(repo/'src'/'csi_maize'/'_methods'/name)==expected,
                    'Frozen scientific operator changed')
        source_hash = config['extraction_source_sha256']
        require(p.digest(folder/'sources'/f'extraction_{source_hash}.py')==source_hash,
                'Frozen extraction source changed')
        configs[shard] = config
    require(bool(configs),'No extraction shard configuration')
    shards = next(iter(configs.values()))['shards']
    require(set(configs)==set(range(shards)),'Incomplete shard configuration set')
    expected_files = set()
    mask_owners = {}
    view_owners = set()
    accepted = 0
    masks = 0
    for index,task in enumerate(tasks):
        name = hashlib.sha256(task['source'].encode()).hexdigest()[:24]+'.json'
        expected_files.add(name)
        path = folder/'records'/name
        require(path.exists(),'Incomplete extraction: missing atomic frame record')
        record = json.loads(path.read_text(encoding='utf-8'))
        require(all(str(record.get(k))==str(task[k]) for k in
                    (*p.KEYS,'session','camera','view','source','source_sha256')),
                'Frame identity/source mismatch against intake task')
        shard = index%shards
        require(int(record['shard'])==shard and
                record['configuration_signature']==configs[shard]['signature'],
                'Frame configuration/shard mismatch')
        unique = (p.key(record),record['camera'],int(record['view']))
        require(unique not in view_owners,'Duplicate four-key camera/view alias')
        view_owners.add(unique)
        if p.truth(record['qc_pass']):
            accepted += 1
            require(bool(record.get('mask')),'Accepted frame lacks mask')
        if record.get('mask'):
            mask = (run/record['mask']).resolve()
            require(mask.is_relative_to((run/'extraction'/'masks').resolve()),
                    'Mask path escapes extraction output')
            require(mask not in mask_owners,'Shared mask filename alias across source frames')
            mask_owners[mask] = task['source']
            require(mask.exists() and p.digest(mask)==record['mask_sha256'],
                    'Actual mask hash mismatch')
            masks += 1
    require({v.name for v in (folder/'records').glob('*.json')}==expected_files,
            'Unexpected atomic records outside intake task inventory')
    return dict(status='verified',frames=len(tasks),accepted_frames=accepted,
                checked_mask_hashes=masks,shards=shards,mask_aliases=0,
                source_byte_policy='Originals bound to full-decode intake hashes; masks rehashed now')


def audit_aggregation(sessions,frames,min_views=20):
    """Recompute coverage and every declared session image statistic from views."""
    lookup = {p.key(r):r for r in sessions}
    require(len(lookup)==len(sessions),'Duplicate session identity')
    groups = defaultdict(list)
    unique = set()
    for frame in frames:
        kk = p.key(frame)
        camera,view = frame['camera'],int(frame['view'])
        require(kk in lookup,'Frame lacks session identity')
        require(camera in ('rgb','noir') and 0<=view<40,'Invalid aggregate camera/view')
        require((kk,camera,view) not in unique,'Duplicate aggregate camera/view')
        unique.add((kk,camera,view))
        groups[(kk,camera)].append(frame)
    trait_columns = [n for n in sessions[0] if n.startswith(('rgb__','noir__'))]
    checked = 0
    missing_values = 0
    for row in sessions:
        eligibility = {}
        for camera in ('rgb','noir'):
            records = groups[(p.key(row),camera)]
            good = [r for r in records if p.truth(r['qc_pass'])]
            quadrants = len({int(r['view'])//10 for r in good})
            require(int(row[camera+'_decodable_views'])==len(records) and
                    int(row[camera+'_usable_views'])==len(good) and
                    int(row[camera+'_quadrants'])==quadrants,'Session coverage mismatch')
            eligible = len(good)>=min_views and quadrants==4
            eligibility[camera] = eligible
            require(p.truth(row[camera+'_eligible'])==eligible,'Session eligibility mismatch')
            for column in trait_columns:
                if not column.startswith(camera+'__'):
                    continue
                _,trait,statistic = column.split('__')
                values = np.array([p.number(r.get(trait)) for r in good])
                missing_values += int((~np.isfinite(values)).sum())
                values = values[np.isfinite(values)]
                actual = p.number(row[column])
                if not len(values):
                    require(np.isnan(actual),'Empty image aggregate became a numeric value')
                else:
                    expected = {'median':lambda:np.median(values),
                                'max':lambda:np.max(values),
                                'iqr':lambda:np.quantile(values,.75)-np.quantile(values,.25)}[statistic]()
                    require(np.isclose(actual,expected,rtol=1e-12,atol=1e-12),
                            'Image aggregate statistic mismatch')
                checked += 1
        require(p.truth(row['paired_eligible'])==all(eligibility.values()),
                'Paired eligibility mismatch')
    return dict(status='verified',sessions=len(sessions),frames=len(frames),
                checked_trait_statistics=checked,accepted_frame_missing_trait_cells_times3=missing_values)


def selection_audit(references,sessions):
    """Describe target distributions for retained/excluded rows without tests."""
    lookup = {p.key(r):r for r in sessions}
    require(set(lookup)=={p.key(r) for r in references},'Selection audit key mismatch')
    joined = [{**r,**lookup[p.key(r)]} for r in references]
    groupings = [('overall',())]+[(f,(f,)) for f in p.KEYS]+[
                 ('treatment_visit',('treatment','measurement'))]
    output = []
    for cohort in ('paired_primary','paired_strict30','paired_strict40','rgb_primary','noir_primary'):
        eligible = set(p.cohort_indices(joined,cohort))
        for label,fields in groupings:
            levels = sorted({tuple(r[f] for f in fields) for r in joined})
            for level in levels:
                for state in (True,False):
                    records = [r for i,r in enumerate(joined) if (i in eligible)==state
                               and tuple(r[f] for f in fields)==level]
                    for target in p.TARGETS:
                        values = np.array([p.number(r[target]) for r in records])
                        values = values[np.isfinite(values)]
                        output.append(dict(cohort=cohort,grouping=label,group=';'.join(level),
                            eligible=state,target=target,sessions=len(records),
                            observed_targets=len(values),
                            mean=float(values.mean()) if len(values) else None,
                            median=float(np.median(values)) if len(values) else None,
                            sd=float(values.std(ddof=1)) if len(values)>1 else None,
                            minimum=float(values.min()) if len(values) else None,
                            maximum=float(values.max()) if len(values) else None))
    return output


def incremental_contrast(folder,baseline,augmented):
    with np.load(Path(folder)/'oof.npz') as data:
        rows=[dict(zip(p.KEYS,k)) for k in data['keys']]
        stats=p.cluster_comparison(rows,data['observed'],data[augmented],data[baseline])
        return dict(experiment=Path(folder).name,baseline=baseline,augmented=augmented,
            rows=len(rows),target_channels=data['observed'].shape[1],
            baseline_rmse=p.metrics(data['observed'],data[baseline])['rmse'],
            augmented_rmse=p.metrics(data['observed'],data[augmented])['rmse'],
            rmse_improvement=stats.pop('rmse_improvement_vs_metadata'),**stats)


def audit_experiment(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'complete.json').read_text())
    with np.load(folder/'oof.npz') as archive:
        data = {k:archive[k] for k in archive.files}
    keys = [tuple(v) for v in data['keys']]
    require(len(set(keys)) == len(keys), 'Duplicate OOF keys')
    rows = [dict(zip(p.KEYS,k)) for k in keys]
    observed = data['observed']
    lookup = {k:i for i,k in enumerate(keys)}
    require(np.isfinite(observed).all(), 'Nonfinite source observation')
    require(observed.shape == (manifest['n'],manifest['target_channels']), 'Wrong OOF shape')
    split_field = 'genotype' if manifest['split']=='genotype' else 'treatment'
    require(data['outer_fold'].tolist() == [r[split_field] for r in rows],
            'OOF fold labels disagree with keys')
    with (folder/'folds.csv').open(encoding='utf-8-sig',newline='') as f:
        folds = list(csv.DictReader(f))
    model_tests = {m:[] for m in manifest['models']}
    mean_fields = {'training_mean':(), 'visit_mean':('measurement',),
                   'treatment_mean':('treatment',),
                   'treatment_visit_mean':('treatment','measurement')}
    for fold in folds:
        train_keys = [tuple(k) for k in json.loads(fold['train_keys'])]
        test_keys = [tuple(k) for k in json.loads(fold['test_keys'])]
        require(not set(train_keys)&set(test_keys), 'Training/test key overlap')
        require(set(train_keys)|set(test_keys)==set(keys), 'Fold misses cohort keys')
        train = [lookup[k] for k in train_keys]
        test = [lookup[k] for k in test_keys]
        require(not {p.candidate(rows[i]) for i in train}&
                {p.candidate(rows[i]) for i in test}, 'Candidate plant overlap')
        require(not {rows[i][split_field] for i in train}&
                {rows[i][split_field] for i in test}, 'Outer group overlap')
        require(all(rows[i][split_field] == fold['outer_group'] for i in test),
                'Wrong held-out group')
        require(int(fold['train_n'])==len(train) and int(fold['test_n'])==len(test),
                'Fold denominator mismatch')
        model = fold['model']
        model_tests[model].extend(test_keys)
        if model in mean_fields:
            expected = p.metadata_predict(rows,observed,train,test,mean_fields[model])
            require(np.allclose(expected,data[model][test],rtol=0,atol=1e-12),
                    'Training mean predictions disagree')
        else:
            require(float(fold['alpha']) in p.ALPHAS,'Undeclared alpha')
            require(fold['inner_group']=='genotype', 'Wrong inner group')
            require(set(fold['inner_train_genotypes'].split(';')) ==
                    {rows[i]['genotype'] for i in train}, 'Inner group support mismatch')
            require(int(fold['inner_folds'])==len({rows[i]['genotype'] for i in train}),
                    'Wrong nested-fold denominator')
            mapping = {int(i):int(j) for i,j in json.loads(fold['permutation']).items()}
            if mapping:
                require(set(mapping)==set(train) and set(mapping.values())==set(train),
                        'Permutation touches test rows or is not bijective')
                for i,j in mapping.items():
                    require(rows[i]['treatment']==rows[j]['treatment'] and
                            rows[i]['measurement']==rows[j]['measurement'],
                            'Permutation breaks treatment/visit strata')
                donors = {}
                for i,j in mapping.items():
                    recipient = p.candidate(rows[i])
                    donor = p.candidate(rows[j])
                    require(recipient not in donors or donors[recipient]==donor,
                            'Permutation breaks candidate trajectory')
                    donors[recipient] = donor
    for model in manifest['models']:
        require(len(model_tests[model])==len(keys) and set(model_tests[model])==set(keys),
                'OOF coverage is not exactly once per row/model')
        require(data[model].shape==observed.shape and np.isfinite(data[model]).all(),
                'Missing model output')
        actual = p.metrics(observed,data[model])
        claimed = next(s for s in manifest['summary'] if s['model']==model)
        for name,value in actual.items():
            require((value is None and claimed[name] is None) or
                    (value is not None and np.isclose(value,claimed[name],rtol=1e-10,atol=1e-12)),
                    'Saved metric disagrees with OOF array')
    return dict(experiment=folder.name,rows=len(keys),models=len(manifest['models']),
                targets=observed.shape[1],fold_records=len(folds),status='verified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('intake','run','repo'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=('outputs','extraction-preflight','extraction-tables'),
                        default='outputs')
    args = parser.parse_args()
    root = args.run/'prediction'
    root.mkdir(parents=True,exist_ok=True)
    if args.stage=='extraction-preflight':
        result = audit_preflight(args.intake,args.run,args.repo)
        p.save_json(root/'extraction_preflight_audit.json',result)
        print(json.dumps(result,indent=2))
        return
    extraction_results = []
    if (args.run/'extraction'/'summary.json').exists():
        sessions = p.read_keyed(args.run/'extraction'/'session_traits.csv')
        require(len(sessions)==360,'Extraction table must retain all360 reference identities')
        p.validate_extraction(args.run/'extraction'/'session_traits.csv')
        frames = read_rows(args.run/'extraction'/'view_traits.csv')
        summary = json.loads((args.run/'extraction'/'summary.json').read_text())
        require(p.digest(args.run/'extraction'/'view_traits.csv')==summary['frame_table_sha256'],
                'Final frame table hash mismatch')
        for threshold in (20,30,40):
            rows = sessions if threshold==20 else p.read_keyed(
                args.run/'extraction'/f'session_traits_min{threshold}.csv')
            extraction_results.append(dict(threshold=threshold,
                **audit_aggregation(rows,frames,threshold)))
        p.save_json(root/'extraction_tables_audit.json',dict(status='verified',
                                                           results=extraction_results))
        reference_rows = p.read_keyed(args.intake/'research_audit'/'reference_master.csv')
        p.save_csv(root/'eligibility_assay_distributions.csv',selection_audit(reference_rows,sessions))
    if args.stage=='extraction-tables':
        require(bool(extraction_results),'Final extraction tables not yet published')
        print(json.dumps(dict(status='verified',results=extraction_results),indent=2))
        return
    reference = {p.key(r):r for r in p.read_keyed(
        args.intake/'research_audit'/'reference_master.csv')}
    spectra = {band:p.read_spectra(args.intake,band) for band in ('UVIS','NIR')}
    results = []
    contrasts = []
    planned_contrasts = [
        ('metadata_ridge','metadata_plus_image'),
        ('metadata_ridge','metadata_plus_dino'),
        ('rgb','combined'),('noir','combined'),
        ('registered_camera_dns','camera_dns_plus_cross_indices'),
        ('metadata_plus_dns','metadata_plus_dns_plus_cross_indices'),
        ('sam_traits','sam_traits_plus_indices'),
        ('metadata_ridge','metadata_plus_indices'),
    ]
    for complete in sorted(root.glob('*/complete.json')):
        folder = complete.parent
        result = audit_experiment(folder)
        manifest = json.loads(complete.read_text())
        for baseline,augmented in planned_contrasts:
            if baseline in manifest['models'] and augmented in manifest['models']:
                contrasts.append(incremental_contrast(folder,baseline,augmented))
        with np.load(folder/'oof.npz') as data:
            keys = [tuple(v) for v in data['keys']]
            if manifest['domain'].startswith('assay'):
                target = str(data['targets'][0])
                expected = np.array([[p.number(reference[k][target])] for k in keys])
            else:
                waves, values, flags = spectra[manifest['domain']]
                require(np.array_equal(np.array(data['targets'],float),waves),
                        'Native wavelength order changed')
                expected = np.array([values[k] for k in keys])
                require(data['source_flags'].tolist()==[';'.join(flags.get(k,[])) for k in keys],
                        'Missing source flags')
            require(np.array_equal(data['observed'],expected), 'Source observation mismatch')
        results.append(result)
    p.save_json(root/'audit.json',dict(status='verified',experiments=len(results),
                                     results=results))
    p.save_csv(root/'incremental_contrasts.csv',contrasts)
    print(json.dumps(dict(status='verified',experiments=len(results),
                         row_model_records=sum(r['rows']*r['models'] for r in results)),indent=2))


if __name__=='__main__':
    main()
