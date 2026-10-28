"""Recompute matched-view, lead-time and transfer tests without changing source data."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(name,'2')

import numpy as np
import pandas as pd
from performance_core import partitions, trajectory, train_design, ridge_predict, cutoff_columns

AGES=[13,19,20,22,25]
TARGETS=['fresh_mass_g','dry_mass_g','dry_matter_fraction']
SHAPE=['area_fraction__median','height_fraction__max','width_fraction__max','solidity__median']
RGB=['rgb__'+x for x in SHAPE];NOIR=['noir__'+x for x in SHAPE]
HISTORY=[c+'__'+s for c in ['rgb','noir'] for s in ['ratio_to_first','slope_per_day']]
QUALITY=[c+'__'+s for c in ['rgb','noir'] for s in ['usable_fraction','eligible_flag']]
MODELS={
    'mean':dict(numeric=[],categorical=[],kind='ols'),
    'metadata':dict(numeric=[],categorical=['genotype','treatment'],kind='ridge'),
    'quality':dict(numeric=QUALITY,categorical=[],kind='ridge'),
    'rgb_area_ols':dict(numeric=[RGB[0]],categorical=[],kind='ols'),
    'rgb_area_ridge':dict(numeric=[RGB[0]],categorical=[],kind='ridge'),
    'rgb_area_allometry':dict(numeric=['log_rgb_area'],categorical=[],kind='allometry'),
    'rgb_shape':dict(numeric=RGB,categorical=[],kind='ridge'),
    'noir_shape':dict(numeric=NOIR,categorical=[],kind='ridge'),
    'rgb_noir_shape':dict(numeric=RGB+NOIR,categorical=[],kind='ridge'),
    'rgb_history':dict(numeric=RGB+HISTORY[:2],categorical=[],kind='ridge'),
    'rgb_noir_history':dict(numeric=RGB+NOIR+HISTORY,categorical=[],kind='ridge'),
    'metadata_rgb_area':dict(numeric=[RGB[0]],categorical=['genotype','treatment'],kind='ridge'),
    'metadata_history':dict(numeric=RGB+NOIR+HISTORY,categorical=['genotype','treatment'],kind='ridge'),
    'single_view0_area_ols':dict(numeric=['rgb_view0_area'],categorical=[],kind='ols'),
    'shuffled_rgb_area':dict(numeric=[RGB[0]],categorical=[],kind='ridge',shuffled=True),
}
ALPHAS=[.1,1.,10.,100.,1000.]


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def number(value):
    return float(value) if value is not None and value!='' else float('nan')


def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:
            writer.writerow({k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()})


def assemble(mass,registry,traits,views):
    identity={(r['treatment'],r['genotype'],r['replicate']):r['plant_uid'] for r in registry['records']}
    if len(identity)!=72:raise ValueError('Expected 72 confirmed pot IDs')
    trait_lookup={tuple(str(r[k]) for k in ['treatment','genotype','replicate','measurement']):r for r in traits}
    if len(trait_lookup)!=360 or len(traits)!=360:raise ValueError('Expected 360 distinct trait records')
    view_lookup={}
    for r in views:
        if r['view']!='0' or r['camera']!='rgb':continue
        k=tuple(str(r[f]) for f in ['treatment','genotype','replicate','measurement'])
        if k in view_lookup:raise ValueError('Duplicate fixed view')
        view_lookup[k]=r
    output=[]
    seen=set()
    for m in mass:
        pot=tuple(str(m[k]) for k in ['treatment','genotype','replicate'])
        if pot in seen:raise ValueError('Duplicate mass pot')
        seen.add(pot);uid=identity[pot]
        observations=[trait_lookup[(*pot,str(i))] for i in range(1,6)]
        available={c:np.array([r[c+'_eligible']=='True' for r in observations]) for c in ['rgb','noir']}
        both=available['rgb']&available['noir']
        for index,(age,observation) in enumerate(zip(AGES,observations)):
            row=dict(plant_uid=uid,treatment=pot[0],genotype=pot[1],replicate=pot[2],cutoff_DAS=age,lead_to_scheduled_harvest_days=26-age,
                     current_pair_eligible=bool(both[index]),all_five_pair_eligible=bool(both.all()))
            row.update({target:float(m[target]) for target in TARGETS})
            for camera in ['rgb','noir']:
                for feature in SHAPE:
                    col=camera+'__'+feature
                    row[col]=number(observation[col]) if available[camera][index] else float('nan')
                history=[number(r[camera+'__area_fraction__median']) if ok else float('nan') for r,ok in zip(observations,available[camera])]
                row.update({camera+'__'+k:v for k,v in trajectory(AGES,history,age).items()})
                row[camera+'__usable_fraction']=float(observation[camera+'_usable_views'])/40
                row[camera+'__eligible_flag']=float(available[camera][index])
            view=view_lookup.get((*pot,str(index+1)))
            view_valid=view is not None and view['qc_pass']=='True'
            row['rgb_view0_available']=bool(view_valid)
            row['rgb_view0_area']=float(view['area_fraction']) if view_valid else float('nan')
            area=row[RGB[0]];row['log_rgb_area']=float(np.log(area)) if np.isfinite(area) and area>0 else float('nan')
            output.append(row)
    if len(output)!=360 or len(seen)!=72:raise ValueError('Incomplete mass/visit assembly')
    return output


def score(y,pred):
    residual=y-pred;ss=np.sum((y-y.mean(axis=0))**2,axis=0)
    r2=np.divide(np.sum(residual**2,axis=0),ss,out=np.full(ss.shape,np.nan),where=ss>0)
    return dict(r2=1-r2,mae=np.mean(abs(residual),axis=0),rmse=np.sqrt(np.mean(residual**2,axis=0)),bias=np.mean(pred-y,axis=0))


def nested_fold(rows,train,test,regime,model,seed):
    spec=MODELS[model];numeric=cutoff_columns(spec['numeric'],rows[0]['cutoff_DAS'],AGES[0],HISTORY);categorical=spec['categorical']
    local=[rows[i] for i in train]
    y=np.array([[float(r[t]) for t in TARGETS] for r in local])
    donor=None
    if spec.get('shuffled'):
        order=np.random.default_rng(seed).permutation(len(y));y=y[order];donor=[local[i]['plant_uid'] for i in order]
    x,z,state=train_design(rows,train,test,numeric,categorical)
    losses=[]
    if spec['kind']=='ridge':
        inner=partitions([r['genotype'] for r in local],[r['treatment'] for r in local],regime)
        total=np.zeros((len(ALPHAS),len(TARGETS)));count=0
        for inner_train,inner_test,_ in inner:
            a,b,_=train_design(local,inner_train,inner_test,numeric,categorical)
            for j,alpha in enumerate(ALPHAS):total[j]+=np.sum((ridge_predict(a,b,y[inner_train],alpha)-y[inner_test])**2,axis=0)
            count+=len(inner_test)
        losses=(total/count).tolist();chosen=np.argmin(total,axis=0)
        alpha=[ALPHAS[i] for i in chosen]
        pred=np.column_stack([ridge_predict(x,z,y[:,j:j+1],a)[:,0] for j,a in enumerate(alpha)])
        smearing=None
    elif spec['kind']=='allometry':
        log_y=np.log(y)
        fitted=ridge_predict(x,x,log_y,0.)
        smearing=np.mean(np.exp(log_y-fitted),axis=0)
        pred=np.exp(ridge_predict(x,z,log_y,0.))*smearing
        smearing=smearing.tolist();alpha=[0.]*len(TARGETS)
    else:
        pred=ridge_predict(x,z,y,0.);alpha=[0.]*len(TARGETS);smearing=None
    return pred,dict(alpha=alpha,inner_mse=losses,preprocessing=state,effective_numeric=numeric,null_donor_uids=donor,smearing_factor=smearing)


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['mass','registry','traits','views','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cohort',action='append',choices=['available_pair','all_pots','common_five_visits','single_view_matched'])
    parser.add_argument('--regime',action='append',choices=['genotype','treatment','joint'])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'completion.json').exists():raise ValueError('Preserve completed analysis; use a new directory')
    inputs={str(p):sha(p) for p in [args.mass,args.registry,args.traits,args.views]}
    cohorts=args.cohort or ['available_pair','all_pots','common_five_visits','single_view_matched']
    regimes=args.regime or ['genotype','treatment','joint']
    protocol=dict(cutoffs_DAS=AGES,harvest_scheduled_DAS=26,targets=TARGETS,models=MODELS,alpha_grid=ALPHAS,cohorts=cohorts,regimes=regimes,
                  seed=20261028,version=2,first_visit_history_fields_omitted=True,inputs=inputs,source_hashes={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('performance_core.py')]},
                  design='Retrospective model development; no independent prospective test cohort. Raw measurements unchanged.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    start=time.time();rows=assemble(read(args.mass),json.loads(args.registry.read_text()),read(args.traits),read(args.views))
    write(args.output/'pot_cutoff_features.csv',rows)
    summaries=[];coverage=[];fold_metrics=[];fold_records=[];predictions=[];experiment_count=0
    for age in AGES:
        age_rows=[r for r in rows if r['cutoff_DAS']==age]
        for cohort in cohorts:
            selected=[r for r in age_rows if cohort=='all_pots' or (cohort=='available_pair' and r['current_pair_eligible']) or (cohort=='common_five_visits' and r['all_five_pair_eligible']) or (cohort=='single_view_matched' and r['current_pair_eligible'] and r['rgb_view0_available'])]
            for treatment in ['C','P','S']:
                coverage.append(dict(cutoff_DAS=age,cohort=cohort,treatment=treatment,n=sum(r['treatment']==treatment for r in selected),total_harvest_pots=24))
            y=np.array([[r[t] for t in TARGETS] for r in selected])
            for regime in regimes:
                outer=partitions([r['genotype'] for r in selected],[r['treatment'] for r in selected],regime)
                for model in MODELS:
                    experiment_id=f'{age}_{cohort}_{regime}_{model}'
                    all_pred=np.full(y.shape,np.nan)
                    for train,test,label in outer:
                        seed=int.from_bytes(hashlib.sha256((experiment_id+'|'+label).encode()).digest()[:4],'little')
                        pred,details=nested_fold(selected,train,test,regime,model,seed)
                        all_pred[test]=pred
                        train_uids=[selected[i]['plant_uid'] for i in train];test_uids=[selected[i]['plant_uid'] for i in test]
                        fold_records.append(dict(experiment_id=experiment_id,model=model,regime=regime,cohort=cohort,cutoff_DAS=age,heldout_group=label,
                                                 train_uids=train_uids,test_uids=test_uids,unused_uids=[r['plant_uid'] for i,r in enumerate(selected) if i not in set(train)|set(test)],
                                                 seed=seed,**details))
                        scores=score(y[test],pred)
                        for j,target in enumerate(TARGETS):
                            fold_metrics.append(dict(experiment_id=experiment_id,cutoff_DAS=age,cohort=cohort,regime=regime,model=model,target=target,heldout_group=label,n=len(test),
                                                     **{k:float(v[j]) for k,v in scores.items()}))
                    if not np.isfinite(all_pred).all():raise ValueError('Every selected pot must receive exactly one finite prediction')
                    scores=score(y,all_pred)
                    for j,target in enumerate(TARGETS):
                        summaries.append(dict(experiment_id=experiment_id,cutoff_DAS=age,lead_days=26-age,cohort=cohort,regime=regime,model=model,target=target,n=len(selected),
                                              **{k:float(v[j]) for k,v in scores.items()}))
                        for i,r in enumerate(selected):
                            predictions.append(dict(experiment_id=experiment_id,cutoff_DAS=age,cohort=cohort,regime=regime,model=model,target=target,plant_uid=r['plant_uid'],
                                                    treatment=r['treatment'],genotype=r['genotype'],replicate=r['replicate'],observed=y[i,j],predicted=all_pred[i,j]))
                    experiment_count+=1
                print(json.dumps(dict(age=age,cohort=cohort,regime=regime,n=len(selected),completed_family_experiments=experiment_count,elapsed_seconds=round(time.time()-start,1))),flush=True)
    write(args.output/'coverage.csv',coverage);write(args.output/'metrics.csv',summaries);write(args.output/'fold_metrics.csv',fold_metrics);write(args.output/'predictions.csv',predictions)
    (args.output/'folds.json').write_text(json.dumps(fold_records,indent=2,allow_nan=False),encoding='utf-8')
    result=dict(status='completed',model_family_experiments=experiment_count,target_experiments=len(summaries),outer_fits=len(fold_records),prediction_rows=len(predictions),
                elapsed_seconds=time.time()-start,python=sys.version,packages=dict(numpy=np.__version__,pandas=pd.__version__),
                inputs=inputs,outputs={p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.name!='completion.json'})
    (args.output/'completion.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
