"""Independent verification of the frozen new run; does not import its code.

All feature rows, cohorts, folds and metrics are checked. A deterministic
structural sample of fits is replayed with scikit-learn, including inner tuning.
"""
from pathlib import Path, PureWindowsPath
import argparse
import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = RUN = SOURCE = None
AGES = [13, 19, 20, 22, 25]
KEY = ['treatment', 'genotype', 'replicate']
SHAPE = ['area_fraction__median', 'height_fraction__max', 'width_fraction__max', 'solidity__median']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return pd.read_csv(path, dtype={'replicate': str, 'measurement': str})


def independently_selected(frame, cohort):
    if cohort == 'all_pots':
        return frame
    if cohort == 'available_pair':
        return frame.loc[frame.current_pair_eligible]
    if cohort == 'common_five_visits':
        return frame.loc[frame.all_five_pair_eligible]
    if cohort == 'single_view_matched':
        return frame.loc[frame.current_pair_eligible & frame.rgb_view0_available]
    raise AssertionError(cohort)


def independent_splits(frame, regime):
    if regime == 'genotype':
        for g in sorted(frame.genotype.unique()):
            yield frame.genotype.ne(g).to_numpy(), frame.genotype.eq(g).to_numpy()
    elif regime == 'treatment':
        for t in sorted(frame.treatment.unique()):
            yield frame.treatment.ne(t).to_numpy(), frame.treatment.eq(t).to_numpy()
    elif regime == 'joint':
        for g in sorted(frame.genotype.unique()):
            for t in sorted(frame.treatment.unique()):
                yield (frame.genotype.ne(g) & frame.treatment.ne(t)).to_numpy(), (frame.genotype.eq(g) & frame.treatment.eq(t)).to_numpy()
    else:
        raise AssertionError(regime)


def sklearn_design(train, test, numeric, categorical):
    parts, held = [], []
    state = {}
    if numeric:
        imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
        scaler = StandardScaler()
        a = imputer.fit_transform(train[numeric])
        b = imputer.transform(test[numeric])
        parts.append(scaler.fit_transform(a))
        held.append(scaler.transform(b))
        state.update(medians=imputer.statistics_, indicators=imputer.indicator_.features_,
                     means=scaler.mean_, scales=scaler.scale_)
    if categorical:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        parts.append(encoder.fit_transform(train[categorical].astype(str)))
        held.append(encoder.transform(test[categorical].astype(str)))
        state['categories'] = {c: list(x) for c, x in zip(categorical, encoder.categories_)}
    return (np.hstack(parts) if parts else np.empty((len(train), 0)),
            np.hstack(held) if held else np.empty((len(test), 0)), state)


def fit_predict(a, b, y, alpha):
    if a.shape[1] == 0:
        return np.tile(np.mean(y, axis=0), (len(b), 1))
    model = LinearRegression() if alpha == 0 else Ridge(alpha=alpha)
    return np.asarray(model.fit(a, y).predict(b)).reshape(len(b), -1)


def resolve_inputs(protocol_inputs, input_root=None):
    """Preserve protocol keys, optionally resolving a unique portable basename."""
    if input_root is None:
        resolved = {key: Path(key) for key in protocol_inputs}
        missing = [str(path) for path in resolved.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError('Original input path(s) unavailable: ' + ', '.join(missing))
        return resolved
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError('Input root is not a directory: ' + str(root))
    inventory = {}
    for path in root.rglob('*'):
        if path.is_file():
            inventory.setdefault(path.name.casefold(), []).append(path)
    resolved = {}
    for key in protocol_inputs:
        basename = PureWindowsPath(str(key)).name
        matches = inventory.get(basename.casefold(), [])
        if not matches:
            raise FileNotFoundError('No rebased input for ' + str(key) + ' under ' + str(root))
        if len(matches) != 1:
            raise ValueError('Ambiguous rebased input ' + basename + ': ' + ', '.join(map(str, matches)))
        resolved[key] = matches[0]
    return resolved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed core run folder')
    parser.add_argument('--source', type=Path, required=True, help='Folder containing the exact execution source files')
    parser.add_argument('--output', type=Path, required=True, help='New folder for independent audit outputs')
    parser.add_argument('--input-root', type=Path, help='Rebase protocol inputs by unique basename under this folder')
    return parser.parse_args(argv)


def main():
    global ROOT, RUN, SOURCE
    args = parse_args()
    ROOT, RUN, SOURCE = args.output, args.run, args.source
    ROOT.mkdir(parents=True, exist_ok=False)
    started = time.time()
    protocol = json.loads((RUN / 'protocol.json').read_text())
    completion = json.loads((RUN / 'completion.json').read_text())
    targets, models, alphas = protocol['targets'], protocol['models'], protocol['alpha_grid']
    resolved_inputs = resolve_inputs(protocol['inputs'], args.input_root)
    checks = {p: sha(resolved_inputs[p]) == h for p, h in protocol['inputs'].items()}
    checks.update({str(RUN / p): sha(RUN / p) == h for p, h in completion['outputs'].items()})
    assert all(checks.values())
    source_root = SOURCE
    source_checks = {str(source_root / p): sha(source_root / p) == h for p, h in protocol['source_hashes'].items()}
    assert all(source_checks.values())
    paths = list(resolved_inputs.values())
    mass = read(next(p for p in paths if Path(p).name == 'confirmed_mass.csv'))
    traits = read(next(p for p in paths if Path(p).name == 'session_traits.csv'))
    views = read(next(p for p in paths if Path(p).name == 'view_traits.csv'))
    registry = json.loads(Path(next(p for p in paths if Path(p).name == 'identity_registry.json')).read_text())
    ids = pd.DataFrame(registry['records'])
    assert ids.groupby(KEY).plant_uid.nunique().eq(1).all()
    id_by_pot = ids[KEY + ['plant_uid']].drop_duplicates()
    mass = mass.merge(id_by_pot, on=KEY, validate='one_to_one')
    assert len(mass) == mass.plant_uid.nunique() == 72
    assert not traits.duplicated(KEY + ['measurement']).any() and len(traits) == 360
    assert not views.duplicated(KEY + ['measurement', 'camera', 'view']).any()
    t = traits.set_index(KEY + ['measurement'])
    v = views[views.camera.eq('rgb') & views.view.eq(0)].set_index(KEY + ['measurement'])
    actual = read(RUN / 'pot_cutoff_features.csv').set_index(['plant_uid', 'cutoff_DAS'])
    assert len(actual) == 360 and not actual.index.duplicated().any()
    expected_rows = []
    feature_checks = 0
    maximum_feature_error = 0.
    for _, m in mass.iterrows():
        pot = tuple(m[k] for k in KEY)
        visits = [t.loc[pot + (str(i),)] for i in range(1, 6)]
        for j, age in enumerate(AGES):
            current = visits[j]
            r = {k: m[k] for k in KEY + ['plant_uid'] + targets}
            r.update(cutoff_DAS=age, lead_to_scheduled_harvest_days=26-age,
                     current_pair_eligible=bool(current.rgb_eligible and current.noir_eligible),
                     all_five_pair_eligible=all(z.rgb_eligible and z.noir_eligible for z in visits))
            for camera in ['rgb', 'noir']:
                ok = bool(current[camera + '_eligible'])
                for feature in SHAPE:
                    c = camera + '__' + feature
                    r[c] = current[c] if ok else np.nan
                # Prefix only: no values or availability from later visits enter.
                hist = np.array([z[camera + '__area_fraction__median'] if z[camera + '_eligible'] else np.nan for z in visits[:j+1]])
                observed = np.isfinite(hist)
                xx, yy = np.array(AGES[:j+1], float)[observed], hist[observed]
                r[camera + '__ratio_to_first'] = hist[-1] / hist[0] if np.isfinite(hist[0]) and hist[0] > 0 and np.isfinite(hist[-1]) else np.nan
                r[camera + '__slope_per_day'] = float(np.polyfit(xx, yy, 1)[0]) if len(xx) >= 2 else np.nan
                r[camera + '__observed_past_visits'] = int(observed.sum())
                r[camera + '__usable_fraction'] = current[camera + '_usable_views'] / 40
                r[camera + '__eligible_flag'] = float(ok)
            k = pot + (str(j+1),)
            vr = v.loc[k] if k in v.index else None
            r['rgb_view0_available'] = vr is not None and bool(vr.qc_pass)
            r['rgb_view0_area'] = float(vr.area_fraction) if r['rgb_view0_available'] else np.nan
            r['log_rgb_area'] = np.log(r['rgb__area_fraction__median']) if np.isfinite(r['rgb__area_fraction__median']) and r['rgb__area_fraction__median'] > 0 else np.nan
            a = actual.loc[(m.plant_uid, age)]
            for c, value in r.items():
                if c in ['plant_uid', 'cutoff_DAS']:
                    continue
                feature_checks += 1
                if isinstance(value, (str, bool, np.bool_)):
                    assert a[c] == value, (m.plant_uid, age, c, a[c], value)
                elif np.isnan(value):
                    assert pd.isna(a[c]), (m.plant_uid, age, c)
                else:
                    err = abs(float(a[c]) - float(value))
                    maximum_feature_error = max(maximum_feature_error, err)
                    assert err <= 1e-10, (m.plant_uid, age, c, err)
            expected_rows.append(r)
    expected = pd.DataFrame(expected_rows)
    for spec in models.values():
        assert not set(spec['numeric']) & {'all_five_pair_eligible', 'current_pair_eligible', 'rgb_view0_available'}
        assert set(spec['numeric']) <= set(expected.columns)

    folds = json.loads((RUN / 'folds.json').read_text())
    preds = read(RUN / 'predictions.csv')
    met = read(RUN / 'metrics.csv')
    coverage = read(RUN / 'coverage.csv')
    assert len(folds) == 8100 and len(met) == 2700 and len(preds) == 169560
    assert not preds.duplicated(['experiment_id', 'target', 'plant_uid']).any()
    prediction_lookup = preds.set_index(['experiment_id', 'target', 'plant_uid']).predicted
    source_targets = mass.set_index('plant_uid')[targets]
    assert np.allclose(preds.observed, [source_targets.loc[u, y] for u, y in zip(preds.plant_uid, preds.target)], atol=1e-14)
    selected_by_experiment = {}
    membership_checks = 0
    for experiment, fs in pd.DataFrame(folds).groupby('experiment_id', sort=False):
        first = fs.iloc[0]
        frame = independently_selected(expected[expected.cutoff_DAS.eq(first.cutoff_DAS)], first.cohort).reset_index(drop=True)
        selected_by_experiment[experiment] = frame
        expected_splits = list(independent_splits(frame, first.regime))
        assert len(fs) == len(expected_splits)
        all_tests = []
        for (_, f), (a, b) in zip(fs.iterrows(), expected_splits):
            train_uids, test_uids = frame.loc[a, 'plant_uid'].tolist(), frame.loc[b, 'plant_uid'].tolist()
            unused_uids = frame.loc[~(a | b), 'plant_uid'].tolist()
            assert train_uids == f.train_uids and test_uids == f.test_uids and unused_uids == f.unused_uids
            assert not set(train_uids) & set(test_uids)
            all_tests += test_uids
            membership_checks += 1
        assert len(all_tests) == len(set(all_tests)) == len(frame)
        assert set(all_tests) == set(frame.plant_uid)
        for treatment in ['C', 'P', 'S']:
            n = coverage.query('cutoff_DAS==@first.cutoff_DAS and cohort==@first.cohort and treatment==@treatment').n
            assert len(n) == 1 and n.iloc[0] == frame.treatment.eq(treatment).sum()
    maximum_metric_error = 0.
    summary_lookup = met.set_index(['experiment_id', 'target'])
    for (experiment, target), group in preds.groupby(['experiment_id', 'target'], sort=False):
        y, p = group.observed.to_numpy(), group.predicted.to_numpy()
        err = p-y
        scores = dict(r2=1-sum(err**2)/sum((y-y.mean())**2), mae=np.mean(abs(err)), rmse=np.sqrt(np.mean(err**2)), bias=np.mean(err))
        stored = summary_lookup.loc[(experiment, target)]
        assert stored.n == len(group)
        for c, value in scores.items():
            maximum_metric_error = max(maximum_metric_error, abs(value-stored[c]))
            assert np.isclose(value, stored[c], rtol=1e-10, atol=1e-10)

    # Select the first fold from all families/cohorts/regimes at 25 DAS,
    # plus all families at four additional structural settings/cutoffs.
    chosen = []
    seen = set()
    extra = {(13, 'all_pots', 'genotype'), (19, 'all_pots', 'joint'),
             (20, 'common_five_visits', 'treatment'), (22, 'single_view_matched', 'genotype')}
    for f in folds:
        e = f['experiment_id']
        setting = (f['cutoff_DAS'], f['cohort'], f['regime'])
        if e not in seen and (f['cutoff_DAS'] == 25 or setting in extra):
            chosen.append(f)
            seen.add(e)
    maximum_prediction_error = maximum_inner_error = maximum_state_error = maximum_smearing_error = 0.
    replay_rows = []
    alpha_ties = []
    for pos, fold in enumerate(chosen):
        frame = selected_by_experiment[fold['experiment_id']]
        train = frame.set_index('plant_uid').loc[fold['train_uids']].reset_index()
        test = frame.set_index('plant_uid').loc[fold['test_uids']].reset_index()
        spec = models[fold['model']]
        history_fields = {c+'__'+s for c in ['rgb','noir'] for s in ['ratio_to_first','slope_per_day']}
        numeric = [c for c in spec['numeric'] if not (fold['cutoff_DAS'] == 13 and c in history_fields)]
        categorical = spec['categorical']
        assert fold['effective_numeric'] == numeric
        a, b, state = sklearn_design(train, test, numeric, categorical)
        stored_state = fold['preprocessing']
        if numeric:
            for c, source in [('medians','medians'), ('means','means'), ('scales','scales')]:
                err = np.max(abs(np.array(stored_state[c])-state[source]), initial=0)
                maximum_state_error = max(maximum_state_error, float(err))
                assert err < 1e-10
            assert stored_state['indicator_columns'] == state['indicators'].tolist()
        assert stored_state['categories'] == state.get('categories', {})
        y = train[targets].to_numpy()
        if spec.get('shuffled'):
            donor = train.iloc[np.random.default_rng(fold['seed']).permutation(len(train))].plant_uid.tolist()
            assert donor == fold['null_donor_uids']
            y = train.set_index('plant_uid').loc[donor, targets].to_numpy()
        else:
            assert fold['null_donor_uids'] is None
        if spec['kind'] == 'ridge':
            total = np.zeros((len(alphas), len(targets)))
            count = 0
            for itrain, itest in independent_splits(train, fold['regime']):
                x, z, _ = sklearn_design(train.loc[itrain], train.loc[itest], numeric, categorical)
                for j, alpha in enumerate(alphas):
                    total[j] += np.sum((fit_predict(x, z, y[itrain], alpha)-y[itest])**2, axis=0)
                count += np.sum(itest)
            losses = total/count
            err = np.max(abs(losses-np.array(fold['inner_mse'])))
            maximum_inner_error = max(maximum_inner_error, float(err))
            assert err < 1e-9
            alpha = [alphas[i] for i in np.argmin(losses, axis=0)]
            if alpha != fold['alpha']:
                for j, (want, stored_alpha) in enumerate(zip(alpha, fold['alpha'])):
                    if want != stored_alpha:
                        excess = float(losses[alphas.index(stored_alpha), j]-losses[:,j].min())
                        assert excess < 1e-10, (fold['experiment_id'], targets[j], alpha, fold['alpha'], losses.tolist(), excess)
                        alpha_ties.append(dict(experiment_id=fold['experiment_id'], target=targets[j],
                                               sklearn_argmin_alpha=want, stored_alpha=stored_alpha, sklearn_loss_excess=excess))
            p = np.column_stack([fit_predict(a, b, y[:,j:j+1], x)[:,0] for j, x in enumerate(fold['alpha'])])
        elif spec['kind'] == 'allometry':
            log_y = np.log(y)
            train_fitted = fit_predict(a, a, log_y, 0.)
            smear = np.mean(np.exp(log_y-train_fitted), axis=0)
            maximum_smearing_error = max(maximum_smearing_error, float(np.max(abs(smear-fold['smearing_factor']))))
            assert np.allclose(smear, fold['smearing_factor'], rtol=1e-10, atol=1e-10)
            p = np.exp(fit_predict(a, b, log_y, 0.))*smear
        else:
            p = fit_predict(a, b, y, 0.)
        stored = np.array([[prediction_lookup.loc[(fold['experiment_id'], target, uid)] for target in targets] for uid in test.plant_uid])
        error = float(np.max(abs(p-stored)))
        maximum_prediction_error = max(maximum_prediction_error, error)
        assert error < 1e-8, (fold['experiment_id'], error)
        replay_rows.append(dict(experiment_id=fold['experiment_id'], heldout_group=fold['heldout_group'],
                                model=fold['model'], regime=fold['regime'], cohort=fold['cohort'],
                                cutoff_DAS=fold['cutoff_DAS'], n_train=len(train), n_test=len(test),
                                maximum_prediction_error=error))
        if pos % 50 == 0:
            print(json.dumps({'replayed':pos+1,'of':len(chosen),'elapsed_seconds':round(time.time()-started,1)}), flush=True)
    pd.DataFrame(replay_rows).to_csv(ROOT / 'new_run_sklearn_replayed_folds.csv', index=False)

    # Detect a temporal ablation's no-history limit without making it an error.
    first = preds[preds.cutoff_DAS.eq(13)]
    no_history = []
    for base, hist in [('rgb_shape', 'rgb_history'), ('rgb_noir_shape', 'rgb_noir_history')]:
        a = first[first.model.eq(base)].set_index(['cohort','regime','target','plant_uid'])
        b = first[first.model.eq(hist)].set_index(['cohort','regime','target','plant_uid']).loc[a.index]
        delta = (a.predicted-b.predicted).abs().groupby(['cohort','regime','target']).max()
        no_history += [dict(base=base, history=hist, cohort=k[0], regime=k[1], target=k[2],
                            maximum_13DAS_prediction_difference=float(x)) for k,x in delta.items()]
    pd.DataFrame(no_history).to_csv(ROOT / 'new_run_13DAS_history_ablation.csv', index=False)
    negative = preds.assign(negative=preds.predicted.lt(0), above_one=preds.predicted.gt(1)).groupby(['cutoff_DAS','cohort','regime','model','target'])[['negative','above_one']].sum().reset_index()
    negative.to_csv(ROOT / 'new_run_prediction_bounds.csv', index=False)
    result = dict(status='passed', feature_rows_checked=360, feature_fields_checked=feature_checks,
                  feature_prefix_causality='All model feature values independently reconstructed from raw visit prefixes; future-dependent all-five eligibility is used only in the labeled intersection cohort.',
                  all_fold_memberships_checked=membership_checks, all_experiments_checked=len(selected_by_experiment),
                  prediction_rows_identity_checked=len(preds), summary_metric_rows_checked=len(met),
                  sklearn_replayed_outer_fits=len(chosen), sklearn_replayed_target_predictions=sum(r['n_test']*len(targets) for r in replay_rows),
                  maximum_feature_error=maximum_feature_error, maximum_preprocessing_state_error=maximum_state_error,
                  maximum_prediction_error=maximum_prediction_error, maximum_inner_mse_error=maximum_inner_error,
                  maximum_metric_error=maximum_metric_error, maximum_smearing_error=maximum_smearing_error,
                  checks=checks, execution_source_hash_matches_protocol=source_checks,
                  input_resolution={str(k):str(v) for k,v in resolved_inputs.items()},
                  numerical_alpha_ties=alpha_ties,
                  source_code_imported=False, verifier_sha256=sha(__file__), elapsed_seconds=time.time()-started,
                  scope_limit='Only the listed structural sample of numerical fits was replayed; all identities, features, folds, cohorts and aggregate metric rows were checked. Numerical replay is not external scientific validation.')
    (ROOT / 'new_run_sklearn_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*valid feature names.*')
        main()
