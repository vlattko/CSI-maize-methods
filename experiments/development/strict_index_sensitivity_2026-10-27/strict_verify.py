"""Independent metric, partition, null-map and selected-fit replay checks."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,importlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import strict_features as sf


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def candidate(key):return key[0],key[2],key[3]
def metrics(y,pred):
    residual=np.square(pred-y);variance=np.square(y-y.mean(axis=0)).sum(axis=0)
    valid=variance>0
    return dict(rmse=float(np.sqrt(residual.mean())),mae=float(np.abs(pred-y).mean()),
        mean_target_r2=float(np.mean(1-residual.sum(axis=0)[valid]/variance[valid])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('intake','run','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();out=args.output;manifest=load(out/'feature_manifest.json');summary=load(out/'summary.json')
    if summary['status']!='completed' or summary['experiments']!=12:raise ValueError('All12 strict experiments required')
    for name,digest in manifest['operator_sha256'].items():
        if sha(out/'operators'/name)!=digest:raise ValueError('Frozen source changed')
    for name,digest in manifest['source_sha256'].items():
        if sha(out/'execution_sources'/name)!=digest:raise ValueError('Executed source snapshot changed')
    sys.path.insert(0,str(out/'operators'));p=importlib.import_module('prediction');pi=importlib.import_module('prediction_indices')
    reference=p.read_keyed(args.intake/'research_audit/reference_master.csv');traits=p.read_keyed(args.run/'extraction/session_traits.csv')
    indices=p.read_keyed(out/'features/session_descriptors_strict.csv');dns=p.read_keyed(out/'features/session_camera_dn_strict.csv')
    maps=[{p.key(r):r for r in rows} for rows in (traits,indices,dns)]
    joined=[{**r,**maps[0][p.key(r)],**maps[1][p.key(r)],**maps[2][p.key(r)]} for r in reference];lookup={p.key(r):r for r in joined}
    if len(joined)!=360 or len(p.cohort_indices(joined,'paired_primary'))!=329:raise ValueError('Wrong cohort denominator')
    metric_rows=[];case_rows=[];fold_count=0;replays=0;nulls=0;max_error=0.;max_metric_error=0.;alpha_changes=0;old_immutable_checks=0
    for case in sorted((out/'prediction').glob('indices__*')):
        integrity=load(case/'sensitivity_integrity.json');old=args.run/'prediction'/case.name
        for name,digest in integrity['files_sha256'].items():
            if sha(case/name)!=digest:raise ValueError('Strict output digest changed')
        for name,digest in [('oof.npz',integrity['archived_oof_sha256']),('complete.json',integrity['archived_complete_sha256'])]:
            if sha(old/name)!=digest:raise ValueError('Archived output changed')
            old_immutable_checks+=1
        new_result=load(case/'complete.json');old_result=load(old/'complete.json')
        features=load(case/'features.json')
        if features!=load(old/'features.json'):raise ValueError('Predictor whitelist changed')
        with np.load(case/'oof.npz',allow_pickle=False) as data,np.load(old/'oof.npz',allow_pickle=False) as legacy:
            keys=[tuple(k) for k in data['keys']];rows=[lookup[k] for k in keys];y=data['observed'];mode=new_result['split']
            if not np.array_equal(data['keys'],legacy['keys']) or not np.array_equal(y,legacy['observed']) or not np.array_equal(data['outer_fold'],legacy['outer_fold']):raise ValueError('OOF cohort/target/fold changed')
            if new_result['models']!=old_result['models']:raise ValueError('Model list changed')
            designs=pi.designs(rows,mode);stats={r['model']:r for r in new_result['summary']};oldstats={r['model']:r for r in old_result['summary']}
            fold_new=sf.read(case/'folds.csv');fold_old=sf.read(old/'folds.csv');fold_map={(r['model'],r['outer_group']):r for r in fold_old}
            if len(fold_map)!=len(fold_new):raise ValueError('Fold record count changed')
            case_rows.append(dict(experiment=case.name,n=len(keys),old_n=len(legacy['keys']),targets=y.shape[1],outer_folds=len(set(data['outer_fold'])),
                                  expected_inner_folds=5 if mode=='genotype' else 6,cohort_keys_identical=True))
            for model in new_result['models']:
                for source,computed,stored in [('strict',metrics(y,data[model]),stats[model]),('legacy',metrics(y,legacy[model]),oldstats[model])]:
                    for field,value in computed.items():
                        error=abs(value-stored[field]);max_metric_error=max(max_metric_error,error)
                        if error>1e-12:raise ValueError('Metric replay differs')
                model_delta=float(np.max(np.abs(data[model]-legacy[model])))
                if model in ('training_mean','visit_mean','treatment_mean','treatment_visit_mean','metadata_ridge','sam_traits') and model_delta!=0:raise ValueError('Unchanged baseline differs')
                metric_rows.append(dict(experiment=case.name,model=model,n=len(keys),target_channels=y.shape[1],max_abs_oof_prediction_delta=model_delta,
                    old_rmse=oldstats[model]['rmse'],strict_rmse=stats[model]['rmse'],delta_rmse=stats[model]['rmse']-oldstats[model]['rmse'],
                    old_r2=oldstats[model]['mean_target_r2'],strict_r2=stats[model]['mean_target_r2'],delta_r2=stats[model]['mean_target_r2']-oldstats[model]['mean_target_r2']))
            key_to_index={key:i for i,key in enumerate(keys)}
            for fold in fold_new:
                previous=fold_map[(fold['model'],fold['outer_group'])]
                train_keys=[tuple(k) for k in json.loads(fold['train_keys'])];test_keys=[tuple(k) for k in json.loads(fold['test_keys'])]
                train=np.array([key_to_index[k] for k in train_keys]);test=np.array([key_to_index[k] for k in test_keys])
                if set(train)&set(test) or set(train)|set(test)!=set(range(len(keys))) or {candidate(k) for k in train_keys}&{candidate(k) for k in test_keys}:raise ValueError('Outer candidate leakage')
                for field in ('train_keys','test_keys','inner_folds','inner_group','inner_train_genotypes','null_seed','permutation'):
                    if fold[field]!=previous[field]:raise ValueError('Partition/null protocol changed: '+field)
                fold_count+=1
                if fold['alpha']=='':continue
                alpha=float(fold['alpha'])
                if alpha not in p.ALPHAS:raise ValueError('Tuning grid changed')
                alpha_changes+=not sf.same(fold['alpha'],previous['alpha'])
                null_map={int(k):int(v) for k,v in json.loads(fold['permutation']).items()}
                target=y.copy();base_model=fold['model'].split('_trajectory_null_')[0]
                if null_map:
                    if set(null_map)!=set(train) or sorted(null_map.values())!=sorted(train):raise ValueError('Null uses test rows or is not a training bijection')
                    _,expected=p.permute_trajectories(rows,y,train,int(fold['null_seed']))
                    if null_map!=expected:raise ValueError('Seeded training null mapping differs')
                    for i,j in null_map.items():
                        if keys[i][0]!=keys[j][0] or keys[i][1]!=keys[j][1]:raise ValueError('Null treatment/visit changed')
                        target[i]=y[j]
                    nulls+=1
                x,names=designs[base_model]
                if names!=features[base_model]:raise ValueError('Design feature order differs')
                estimator=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=alpha,solver='cholesky'))
                with threadpool_limits(limits=2):
                    estimator.fit(x[train],target[train]);replayed=np.asarray(estimator.predict(x[test])).reshape(len(test),y.shape[1])
                error=float(np.max(np.abs(replayed-data[fold['model']][test])));max_error=max(max_error,error)
                if error>1e-10:raise ValueError('Selected outer model replay differs')
                replays+=1
    p.save_csv(out/'verified_metric_changes.csv',metric_rows);p.save_csv(out/'case_denominators.csv',case_rows)
    old_indices=sf.read(args.run/'pairing/full/session_descriptors.csv');new_indices=sf.read(out/'features/session_descriptors_strict.csv')
    result=dict(status='passed',cases=len(case_rows),metric_model_rows=len(metric_rows),outer_fold_records=fold_count,
        independent_selected_outer_fit_replays=replays,training_only_null_maps_verified=nulls,changed_selected_alphas=alpha_changes,
        maximum_absolute_selected_prediction_replay_error=max_error,maximum_absolute_metric_replay_error=max_metric_error,
        archived_output_hashes_reverified=old_immutable_checks,all360_retained=True,primary_paired_n=329,
        all27_eligible_old=sum(sf.truth(r['paired_index_all_27_eligible']) for r in old_indices),all27_eligible_strict=sum(sf.truth(r['paired_index_all_27_eligible']) for r in new_indices),
        limits=['Fixed-grid inner tuning reused the original source/protocol; full inner grid was not independently rerun by this verifier.',
                'Null mappings and selected outer fits independently replayed; one experiment remains retrospective.'])
    p.save_json(out/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
