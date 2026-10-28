"""Independent extension audit; imports only the reviewer's sklearn helpers.

No source/output mutations. Audit artifacts are saved only in the explicit --output folder.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from verify_new_run_sklearn import sklearn_design, fit_predict, independent_splits, sha, resolve_inputs

ROOT = RUN = FINAL = SOURCE = TABLES = None
KEY = ['treatment', 'genotype', 'replicate', 'measurement']
TARGETS = ['fresh_mass_g', 'dry_mass_g', 'dry_matter_fraction']
AGES = [13, 19, 20, 22, 25]
PHASES = [0, 10, 20, 30]


def read(path):
    return pd.read_csv(path, dtype={'replicate':str, 'measurement':str})


def cohort(frame, name):
    if name == 'physiology_all_pots':
        return frame
    if name == 'physiology_available_pair':
        return frame.loc[frame.current_pair_eligible]
    phase = int(name.rsplit('_', 1)[-1])
    assert phase in PHASES
    if name.startswith('phase_available_'):
        keep = frame.current_pair_eligible & frame[f'rgb_phase{phase}_valid']
    elif name.startswith('all_four_phase_matched_'):
        keep = frame.current_pair_eligible & frame[[f'rgb_phase{p}_valid' for p in PHASES]].all(axis=1)
    else:
        raise AssertionError(name)
    return frame.loc[keep]


def score(y, p):
    err = p-y
    return dict(r2=1-np.sum(err**2)/np.sum((y-y.mean())**2), mae=np.mean(abs(err)),
                rmse=np.sqrt(np.mean(err**2)), bias=np.mean(err))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed extension run folder')
    parser.add_argument('--source', type=Path, required=True, help='Folder containing the exact execution source files')
    parser.add_argument('--output', type=Path, required=True, help='New folder for independent audit outputs')
    parser.add_argument('--final', type=Path, required=True, help='Completed core final run folder for shared-result checks')
    parser.add_argument('--tables', type=Path, required=True, help='Folder containing original computed-workbook CSV exports')
    parser.add_argument('--input-root', type=Path, help='Rebase protocol inputs by unique basename under this folder')
    return parser.parse_args(argv)


def main():
    global ROOT, RUN, FINAL, SOURCE, TABLES
    args = parse_args()
    ROOT, RUN, FINAL, SOURCE, TABLES = args.output, args.run, args.final, args.source, args.tables
    ROOT.mkdir(parents=True, exist_ok=False)
    started = time.time()
    protocol = json.loads((RUN/'protocol.json').read_text())
    completion = json.loads((RUN/'completion.json').read_text())
    resolved_inputs = resolve_inputs(protocol['inputs'], args.input_root)
    checks = {p:sha(resolved_inputs[p])==h for p,h in protocol['inputs'].items()}
    checks.update({str(RUN/p):sha(RUN/p)==h for p,h in completion['outputs'].items()})
    source_root = SOURCE
    source_checks = {str(source_root/p):sha(source_root/p)==h for p,h in protocol['source_hashes'].items()}
    assert all(checks.values()) and all(source_checks.values())
    inputs = list(resolved_inputs.values())
    base = read(next(p for p in inputs if Path(p).name=='pot_cutoff_features.csv'))
    traits = read(next(p for p in inputs if Path(p).name=='session_traits.csv'))
    reference = read(next(p for p in inputs if Path(p).name=='reference_master.csv'))
    views = read(next(p for p in inputs if Path(p).name=='view_traits.csv'))
    extension = read(RUN/'features.csv')
    assert len(base)==len(extension)==len(reference)==len(traits)==360
    assert not reference.duplicated(KEY).any() and not traits.duplicated(KEY).any()
    assert not views.duplicated(KEY+['camera','view']).any()
    order = ['plant_uid','cutoff_DAS']
    base = base.set_index(order).sort_index()
    ext = extension.set_index(order).sort_index()
    assert base.index.equals(ext.index)
    max_old_error = max_new_error = 0.
    checked_old_fields = checked_new_fields = 0
    for c in base:
        a,b = base[c],ext[c]
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
            assert a.isna().equals(b.isna())
            err = np.nanmax(abs(a.to_numpy()-b.to_numpy()), initial=0.)
            max_old_error=max(max_old_error,float(err));assert err<1e-12
        else:
            assert a.fillna('').equals(b.fillna(''))
        checked_old_fields+=len(base)
    raw_t = traits.set_index(KEY)
    raw_r = reference.set_index(KEY)
    view_lookup = views[views.camera.eq('rgb') & views.view.isin(PHASES)].set_index(KEY+['view'])
    good_views = views.loc[views.qc_pass]
    grouped_views = {k:g for k,g in good_views.groupby(KEY+['camera'],sort=False)}
    assay_fields = protocol['assays']; appearance_fields = protocol['appearance']
    assert not any('lab' in c.lower() for s in protocol['models'].values() for c in s['numeric'])
    expected_rows=[]
    for (uid,age), row in base.iterrows():
        m=str(AGES.index(age)+1)
        k=tuple(row[f] for f in KEY[:3])+(m,)
        expected = row.to_dict()|dict(plant_uid=uid, cutoff_DAS=age)
        for c in assay_fields:
            expected[c]=float(raw_r.loc[k,c])
        for c in appearance_fields:
            camera,field=c.split('__',1)
            view_field=field.removesuffix('__median')
            if bool(raw_t.loc[k,camera+'_eligible']):
                vals=grouped_views[k+(camera,)][view_field].to_numpy(float)
                expected[c]=float(np.median(vals[np.isfinite(vals)]))
                assert np.isclose(expected[c],raw_t.loc[k,c],rtol=1e-12,atol=1e-12)
            else:
                expected[c]=np.nan
        for phase in PHASES:
            vk=k+(phase,)
            vr=view_lookup.loc[vk] if vk in view_lookup.index else None
            valid=vr is not None and bool(vr.qc_pass)
            expected[f'rgb_phase{phase}_valid']=valid
            expected[f'rgb_phase{phase}_area']=float(vr.area_fraction) if valid else np.nan
        for c in assay_fields+appearance_fields+[f'rgb_phase{p}_{suffix}' for p in PHASES for suffix in ['area','valid']]:
            a,b=ext.loc[(uid,age),c],expected[c]
            checked_new_fields+=1
            if isinstance(b,(bool,np.bool_)):
                assert bool(a)==b
            elif np.isnan(b):
                assert pd.isna(a)
            else:
                err=abs(float(a)-b);max_new_error=max(max_new_error,err);assert err<1e-12,(uid,age,c,err)
        expected_rows.append(expected)
    expected=pd.DataFrame(expected_rows).set_index(order).loc[pd.MultiIndex.from_frame(extension[order])].reset_index()

    # Cross-check all selected assays through the original computed-workbook
    # exports, rather than trusting only the merged reference table.
    rename={'Treatment':'treatment','Measurement':'measurement','GEN':'genotype','REP':'replicate'}
    raw_ojip=pd.read_csv(TABLES/'OJIP_P6_computed_data__SVE.csv',dtype=str).rename(columns=rename).set_index(KEY)
    raw_nir=pd.read_csv(TABLES/'NIR_P6_computed_data__List1.csv',dtype=str).rename(columns=rename).set_index(KEY)
    assert len(raw_ojip)==len(raw_nir)==360 and not raw_ojip.index.duplicated().any() and not raw_nir.index.duplicated().any()
    raw_assay_checks=0
    for k,row in raw_r.iterrows():
        for c in assay_fields:
            table,field=(raw_ojip,c.split('__',1)[1]) if c.startswith('ojip__') else (raw_nir,'NDVI')
            assert np.isclose(float(table.loc[k,field]),row[c],rtol=1e-12,atol=1e-12)
            raw_assay_checks+=1

    metrics=read(RUN/'metrics.csv');preds=read(RUN/'predictions.csv');folds=json.loads((RUN/'folds.json').read_text());cov=read(RUN/'coverage.csv')
    assert len(metrics)==1800 and len(preds)==120258 and len(folds)==5400
    assert not preds.duplicated(['experiment_id','target','plant_uid']).any()
    targets=expected.drop_duplicates('plant_uid').set_index('plant_uid')[TARGETS]
    assert np.allclose(preds.observed,[targets.loc[u,t] for u,t in zip(preds.plant_uid,preds.target)],atol=1e-13)
    prediction_lookup=preds.set_index(['experiment_id','target','plant_uid']).predicted
    metric_lookup=metrics.set_index(['experiment_id','target'])
    max_metric_error=0.
    for (experiment,target), group in preds.groupby(['experiment_id','target'],sort=False):
        stored=metric_lookup.loc[(experiment,target)]
        assert stored.n==len(group)
        for c,x in score(group.observed.to_numpy(),group.predicted.to_numpy()).items():
            err=abs(stored[c]-x);max_metric_error=max(max_metric_error,err);assert err<1e-10
    frames={};first_folds=[];membership_checks=0;coverage_rows=[];joint_unused_counts=[]
    for experiment, fs in pd.DataFrame(folds).groupby('experiment_id',sort=False):
        first=fs.iloc[0]
        frame=cohort(expected.loc[expected.cutoff_DAS.eq(first.cutoff_DAS)],first.cohort).reset_index(drop=True)
        frames[experiment]=frame
        splits=list(independent_splits(frame,first.regime));assert len(fs)==len(splits)
        tested=[]
        for (_,fold),(train,test) in zip(fs.iterrows(),splits):
            assert frame.loc[train,'plant_uid'].tolist()==fold.train_uids
            assert frame.loc[test,'plant_uid'].tolist()==fold.test_uids
            tested+=fold.test_uids;membership_checks+=1
            if first.regime=='joint':
                joint_unused_counts.append(dict(experiment_id=experiment,heldout_group=fold.heldout_group,
                                                n_train=int(train.sum()),n_test=int(test.sum()),n_unused=int((~(train|test)).sum())))
        assert len(tested)==len(set(tested))==len(frame) and set(tested)==set(frame.plant_uid)
        for treatment in ['C','P','S']:
            n=int(frame.treatment.eq(treatment).sum())
            coverage_rows.append(dict(experiment_id=experiment,cutoff_DAS=first.cutoff_DAS,cohort=first.cohort,treatment=treatment,n=n))
            if not first.cohort.startswith('physiology_'):
                stored=cov[(cov.cutoff_DAS==first.cutoff_DAS)&(cov.cohort==first.cohort)&(cov.treatment==treatment)]
                assert len(stored)==1 and stored.n.iloc[0]==n
        first_folds.append(fs.iloc[0].to_dict())
    assert len(first_folds)==600
    pd.DataFrame(coverage_rows).to_csv(ROOT/'extensions_audited_coverage.csv',index=False)
    pd.DataFrame(joint_unused_counts).to_csv(ROOT/'extensions_joint_partition_counts.csv',index=False)

    max_prediction_error=max_inner_error=max_state_error=0.;alpha_ties=[];replays=[]
    history_fields={c+'__'+s for c in ['rgb','noir'] for s in ['ratio_to_first','slope_per_day']}
    for pos,fold in enumerate(first_folds):
        frame=frames[fold['experiment_id']].set_index('plant_uid')
        train=frame.loc[fold['train_uids']].reset_index();test=frame.loc[fold['test_uids']].reset_index()
        spec=protocol['models'][fold['model']]
        numeric=[c for c in spec['numeric'] if not(fold['cutoff_DAS']==13 and c in history_fields)]
        assert numeric==fold['effective_numeric']
        a,b,state=sklearn_design(train,test,numeric,spec['categorical'])
        if numeric:
            for c in ['medians','means','scales']:
                err=np.max(abs(np.array(fold['preprocessing'][c])-state[c]),initial=0.)
                max_state_error=max(max_state_error,float(err));assert err<1e-10
            assert state['indicators'].tolist()==fold['preprocessing']['indicator_columns']
        assert state.get('categories',{})==fold['preprocessing']['categories']
        y=train[TARGETS].to_numpy(float)
        assert fold['null_donor_uids'] is None and fold['smearing_factor'] is None
        if spec['kind']=='ridge':
            total=np.zeros((len(protocol['alpha_grid']),3));count=0
            for itrain,itest in independent_splits(train,fold['regime']):
                x,z,_=sklearn_design(train.loc[itrain],train.loc[itest],numeric,spec['categorical'])
                for j,alpha in enumerate(protocol['alpha_grid']):
                    total[j]+=np.sum((fit_predict(x,z,y[itrain],alpha)-y[itest])**2,axis=0)
                count+=itest.sum()
            losses=total/count;err=np.max(abs(losses-np.array(fold['inner_mse'])))
            max_inner_error=max(max_inner_error,float(err));assert err<1e-9
            for j,index in enumerate(np.argmin(losses,axis=0)):
                selected=protocol['alpha_grid'][index]
                if selected!=fold['alpha'][j]:
                    excess=float(losses[protocol['alpha_grid'].index(fold['alpha'][j]),j]-losses[:,j].min())
                    assert excess<1e-10,(fold['experiment_id'],excess)
                    alpha_ties.append(dict(experiment_id=fold['experiment_id'],target=TARGETS[j],sklearn_alpha=selected,
                                           stored_alpha=fold['alpha'][j],loss_excess=excess))
            p=np.column_stack([fit_predict(a,b,y[:,j:j+1],alpha)[:,0] for j,alpha in enumerate(fold['alpha'])])
        else:
            assert spec['kind']=='ols'
            p=fit_predict(a,b,y,0.)
        stored=np.array([[prediction_lookup.loc[(fold['experiment_id'],target,uid)] for target in TARGETS] for uid in test.plant_uid])
        err=float(np.max(abs(p-stored)));max_prediction_error=max(max_prediction_error,err);assert err<1e-8
        replays.append(dict(experiment_id=fold['experiment_id'],heldout_group=fold['heldout_group'],n_train=len(train),n_test=len(test),maximum_prediction_error=err))
        if pos%100==0:print(json.dumps(dict(replayed=pos+1,total=len(first_folds),elapsed_seconds=round(time.time()-started,1))),flush=True)
    pd.DataFrame(replays).to_csv(ROOT/'extensions_sklearn_replayed_folds.csv',index=False)

    # Unmodified shared model families must equal their frozen-final ancestors.
    final_metrics=read(FINAL/'metrics.csv').set_index(['cutoff_DAS','cohort','regime','model','target'])
    max_shared_difference=0.;shared_rows=0
    shared={'mean','metadata','rgb_area_ols','rgb_history','rgb_noir_history'}
    for _,row in metrics.iterrows():
        if row.cohort.startswith('physiology_') and row.model in shared:
            origin=final_metrics.loc[(row.cutoff_DAS,row.cohort.removeprefix('physiology_'),row.regime,row.model,row.target)]
            assert origin.n==row.n
            for c in ['r2','mae','rmse','bias']:
                max_shared_difference=max(max_shared_difference,abs(origin[c]-row[c]));assert abs(origin[c]-row[c])<1e-12
            shared_rows+=1
    info=dict(status='passed',feature_rows_checked=360,original_feature_fields_checked=checked_old_fields,
              extension_feature_fields_checked=checked_new_fields,assay_workbook_export_values_checked=raw_assay_checks,
              appearance_reconstruction='All ten current-camera appearance summaries reconstructed from accepted raw view-trait rows, then eligibility-masked.',
              cutoff_causality='Extension assays/appearance/phases use only the current scheduled visit; inherited prefix features match the independently audited final run.',
              all_fold_memberships_checked=membership_checks,model_family_experiments_checked=len(frames),
              prediction_rows_identity_checked=len(preds),aggregate_metric_rows_checked=len(metrics),
              sklearn_replayed_outer_fits=len(replays),sklearn_replayed_target_predictions=sum(r['n_test']*3 for r in replays),
              maximum_original_feature_error=max_old_error,maximum_extension_feature_error=max_new_error,
              maximum_preprocessing_state_error=max_state_error,maximum_prediction_error=max_prediction_error,
              maximum_inner_mse_error=max_inner_error,maximum_metric_error=max_metric_error,
              frozen_final_shared_metric_rows_checked=shared_rows,maximum_shared_metric_difference=max_shared_difference,
              numerical_alpha_ties=alpha_ties,input_output_hash_checks=checks,execution_source_hash_checks=source_checks,
              input_resolution={str(k):str(v) for k,v in resolved_inputs.items()},
              raw_assay_export_hashes={p.name:sha(p) for p in [TABLES/'OJIP_P6_computed_data__SVE.csv',TABLES/'NIR_P6_computed_data__List1.csv']},
              implementation_source_imported=False,verifier_sha256=sha(__file__),elapsed_seconds=time.time()-started,
              scope_limit='First outer fold replayed for each of all600 model-family settings; all5400 fold memberships, all input joins and all aggregate metrics checked. No external validation is implied.')
    (ROOT/'extensions_sklearn_verification.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
