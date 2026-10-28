"""Matched fixed-phase and physiology/appearance sensitivity, without view selection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import run_performance as r
from performance_core import partitions


ASSAYS=['ojip__Fv/Fm','ojip__Pi_Abs','nir__NDVI']
APPEARANCE=[c+'__'+f for c in ['rgb','noir'] for f in ['g_chromaticity_median__median','normalized_excess_green__median','yellow_hue_fraction__median','hue_median__median','saturation_median__median']]


def evaluate(rows, cohort, age, regime, names, summaries, predictions, records, fold_metrics):
    y=np.array([[float(row[t]) for t in r.TARGETS] for row in rows])
    splits=partitions([row['genotype'] for row in rows],[row['treatment'] for row in rows],regime)
    for model in names:
        identity=f'{age}_{cohort}_{regime}_{model}'
        complete=np.full(y.shape,np.nan)
        for train,test,label in splits:
            pred,details=r.nested_fold(rows,train,test,regime,model,20261028)
            complete[test]=pred
            records.append(dict(experiment_id=identity,cutoff_DAS=age,cohort=cohort,regime=regime,model=model,heldout_group=label,
                                train_uids=[rows[i]['plant_uid'] for i in train],test_uids=[rows[i]['plant_uid'] for i in test],**details))
            scores=r.score(y[test],pred)
            for j,target in enumerate(r.TARGETS):
                fold_metrics.append(dict(experiment_id=identity,cutoff_DAS=age,cohort=cohort,regime=regime,model=model,heldout_group=label,target=target,n=len(test),**{k:float(v[j]) for k,v in scores.items()}))
        if not np.isfinite(complete).all():raise ValueError('Missing held-out prediction')
        scores=r.score(y,complete)
        for j,target in enumerate(r.TARGETS):
            summaries.append(dict(experiment_id=identity,cutoff_DAS=age,cohort=cohort,regime=regime,model=model,target=target,n=len(rows),**{k:float(v[j]) for k,v in scores.items()}))
            for i,row in enumerate(rows):
                predictions.append(dict(experiment_id=identity,cutoff_DAS=age,cohort=cohort,regime=regime,model=model,target=target,plant_uid=row['plant_uid'],
                                        treatment=row['treatment'],genotype=row['genotype'],replicate=row['replicate'],observed=y[i,j],predicted=complete[i,j]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['features','traits','reference','views','output']:parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'completion.json').exists():raise ValueError('Preserve completed outputs')
    start=time.time();rows=r.read(a.features)
    numeric=list(set(sum([m['numeric'] for m in r.MODELS.values()],[])))
    for row in rows:
        row['cutoff_DAS']=int(row['cutoff_DAS'])
        for key in numeric+r.TARGETS:row[key]=r.number(row[key])
        for key in ['current_pair_eligible','all_five_pair_eligible','rgb_view0_available']:row[key]=row[key]=='True'
    fields=['treatment','genotype','replicate','measurement']
    reference={tuple(row[k] for k in fields):row for row in r.read(a.reference)}
    traits={tuple(row[k] for k in fields):row for row in r.read(a.traits)}
    views={tuple(row[k] for k in fields)+(int(row['view']),):row for row in r.read(a.views) if row['camera']=='rgb' and int(row['view']) in [0,10,20,30]}
    if len(reference)!=360 or len(traits)!=360:raise ValueError('Expected matched source grid')
    for row in rows:
        visit=str(r.AGES.index(row['cutoff_DAS'])+1)
        key=tuple(row[k] for k in fields[:3])+(visit,)
        for assay in ASSAYS:row[assay]=r.number(reference[key][assay])
        for feature in APPEARANCE:
            camera=feature.split('__')[0]
            row[feature]=r.number(traits[key][feature]) if traits[key][camera+'_eligible']=='True' else float('nan')
        for phase in [0,10,20,30]:
            source=views.get(key+(phase,));valid=source is not None and source['qc_pass']=='True'
            row[f'rgb_phase{phase}_area']=float(source['area_fraction']) if valid else float('nan')
            row[f'rgb_phase{phase}_valid']=bool(valid)
    for phase in [0,10,20,30]:r.MODELS[f'phase{phase}_area_ols']=dict(numeric=[f'rgb_phase{phase}_area'],categorical=[],kind='ols')
    additions={
        'assay_state':ASSAYS,
        'area_assays':[r.RGB[0]]+ASSAYS,
        'rgb_history_assays':r.RGB+r.HISTORY[:2]+ASSAYS,
        'joint_history_assays':r.RGB+r.NOIR+r.HISTORY+ASSAYS,
        'rgb_history_appearance':r.RGB+r.HISTORY[:2]+APPEARANCE[:5],
        'joint_history_appearance':r.RGB+r.NOIR+r.HISTORY+APPEARANCE,
    }
    for name,columns in additions.items():r.MODELS[name]=dict(numeric=columns,categorical=[],kind='ridge')
    r.MODELS['metadata_assays']=dict(numeric=ASSAYS,categorical=['genotype','treatment'],kind='ridge')
    protocol=dict(models=r.MODELS,assays=ASSAYS,appearance=APPEARANCE,alpha_grid=r.ALPHAS,cutoffs_DAS=r.AGES,
                  phases=[0,10,20,30],phase_choice='Fixed nominal acquisition phases, not anatomical azimuth; no best-view selection',
                  first_visit_history_fields_omitted=True,inputs={str(p):r.sha(p) for p in [a.features,a.traits,a.reference,a.views]},
                  source_hashes={p.name:r.sha(p) for p in [Path(__file__),Path(__file__).with_name('run_performance.py'),Path(__file__).with_name('performance_core.py')]})
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    r.write(a.output/'features.csv',rows)
    summaries=[];predictions=[];records=[];fold_metrics=[];coverage=[]
    for age in r.AGES:
        current=[row for row in rows if row['cutoff_DAS']==age]
        for mode in ['all_pots','available_pair']:
            selected=[row for row in current if mode=='all_pots' or row['current_pair_eligible']]
            names=['mean','metadata','rgb_area_ols','rgb_history','rgb_noir_history']+list(additions)+['metadata_assays']
            for regime in ['genotype','treatment','joint']:
                evaluate(selected,'physiology_'+mode,age,regime,names,summaries,predictions,records,fold_metrics)
        paired=[row for row in current if row['current_pair_eligible']]
        for mode in ['phase_available','all_four_phase_matched']:
            for phase in [0,10,20,30]:
                selected=[row for row in paired if row[f'rgb_phase{phase}_valid'] and (mode=='phase_available' or all(row[f'rgb_phase{p}_valid'] for p in [0,10,20,30]))]
                cohort=mode+f'_{phase}'
                for t in ['C','P','S']:coverage.append(dict(cutoff_DAS=age,cohort=cohort,treatment=t,n=sum(row['treatment']==t for row in selected)))
                for regime in ['genotype','treatment','joint']:
                    evaluate(selected,cohort,age,regime,[f'phase{phase}_area_ols','rgb_area_ols'],summaries,predictions,records,fold_metrics)
        print(json.dumps(dict(age=age,target_experiments=len(summaries),elapsed_seconds=time.time()-start)),flush=True)
    for name,values in [('metrics',summaries),('predictions',predictions),('fold_metrics',fold_metrics),('coverage',coverage)]:r.write(a.output/(name+'.csv'),values)
    (a.output/'folds.json').write_text(json.dumps(records,indent=2,allow_nan=False),encoding='utf-8')
    result=dict(status='completed',target_experiments=len(summaries),outer_fits=len(records),prediction_rows=len(predictions),elapsed_seconds=time.time()-start,
                outputs={p.name:r.sha(p) for p in a.output.iterdir() if p.is_file() and p.name!='completion.json'})
    (a.output/'completion.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
