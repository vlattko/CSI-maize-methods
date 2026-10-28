"""Pot-level image-to-harvest benchmarks with nested group-disjoint validation."""
import argparse
import base64
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from registry import sha


TARGETS = ['fresh_mass_g', 'dry_mass_g', 'dry_matter_fraction']
SHAPE = ['area_fraction__median', 'height_fraction__max', 'width_fraction__max', 'solidity__median']
RGB = ['rgb__'+f for f in SHAPE]
NOIR = ['noir__'+f for f in SHAPE]
DYNAMIC = [f'{c}__{kind}' for c in ['rgb','noir'] for kind in ['area_ratio_25_13','area_rate_19_25']]
MODELS = {'mean': [], 'metadata': ['treatment','genotype'], 'RGB': RGB, 'NoIR': NOIR,
          'RGB_NoIR': RGB+NOIR, 'metadata_RGB_NoIR': ['treatment','genotype']+RGB+NOIR,
          'RGB_NoIR_dynamics': RGB+NOIR+DYNAMIC,
          'shuffled_RGB_NoIR': RGB+NOIR}
ALPHAS = [.1, 1., 10., 100., 1000.]


def fit_pipeline(columns, alpha):
    numeric=[c for c in columns if c not in ['treatment','genotype']]
    categorical=[c for c in columns if c in ['treatment','genotype']]
    transformers=[]
    if numeric:
        transformers.append(('numeric', Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                                                   ('scale',StandardScaler())]),numeric))
    if categorical:
        transformers.append(('categories',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical))
    return Pipeline([('features',ColumnTransformer(transformers)),('ridge',Ridge(alpha=alpha))])


def predict_fold(frame, target, group_field, train, test, columns, shuffled=False):
    """All preprocessing, tuning and optional null permutation use training pots only."""
    y=frame[target].to_numpy(float)
    train_y=y[train].copy()
    if shuffled:
        # A fixed diagnostic, not a permutation significance test.
        train_y=np.random.default_rng(20261028+sum(train)).permutation(train_y)
    if not columns:
        return np.repeat(train_y.mean(),len(test)),None,[]
    groups=frame.iloc[train][group_field].to_numpy()
    splits=list(LeaveOneGroupOut().split(train,train_y,groups))
    scores=[]
    for alpha in ALPHAS:
        losses=[]
        for inner_train,inner_test in splits:
            model=fit_pipeline(columns,alpha)
            model.fit(frame.iloc[train[inner_train]],train_y[inner_train])
            pred=model.predict(frame.iloc[train[inner_test]])
            losses.extend((pred-train_y[inner_test])**2)
        scores.append(float(np.mean(losses)))
    selected=ALPHAS[int(np.argmin(scores))]
    model=fit_pipeline(columns,selected)
    model.fit(frame.iloc[train],train_y)
    return model.predict(frame.iloc[test]),selected,scores


def metrics(y,p):
    return dict(r2=float(r2_score(y,p)),mae=float(mean_absolute_error(y,p)),
                rmse=float(np.sqrt(mean_squared_error(y,p))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['mass','registry','traits','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    registry=json.loads(args.registry.read_text())
    ids=pd.DataFrame(registry['records']);ids=ids[ids.measurement=='5']
    masses=pd.read_csv(args.mass,dtype={'replicate':str})
    traits=pd.read_csv(args.traits,dtype={'replicate':str,'measurement':str})
    join=['treatment','genotype','replicate']
    joined=masses.merge(ids[join+['plant_uid','declared_das']],on=join,validate='one_to_one')
    if len(joined)!=72 or len(masses)!=72:raise ValueError('All 72 mass pots must match the confirmed registry exactly')
    final=traits[traits.measurement=='5']
    joined=joined.merge(final,on=join,validate='one_to_one')
    joined['endpoint_image_DAS']=joined['declared_das']
    joined['fresh_harvest_scheduled_DAS']=26
    joined['image_to_scheduled_fresh_harvest_days']=1
    for camera in ['rgb','noir']:
        area=camera+'__area_fraction__median'
        for visit,age in [('1',13),('2',19)]:
            previous=traits[traits.measurement==visit].set_index(join)
            values=[]
            for _,r in joined.iterrows():
                old=previous.loc[tuple(r[k] for k in join)]
                values.append(old[area] if old[camera+'_eligible'] else np.nan)
            joined[camera+f'__area_{age}']=values
        joined[camera+'__area_ratio_25_13']=joined[area]/joined[camera+'__area_13'].replace(0,np.nan)
        joined[camera+'__area_rate_19_25']=(joined[area]-joined[camera+'__area_19'])/6
    joined['benchmark_eligible']=joined.rgb_eligible & joined.noir_eligible
    joined.to_csv(args.output/'all_pot_mass_image_join.csv',index=False)
    frame=joined[joined.benchmark_eligible].reset_index(drop=True)
    if frame[RGB+NOIR].isna().any().any():raise ValueError('Eligible endpoint image traits must be observed')
    if len(frame)<36:raise ValueError('Too few paired-camera endpoint pots')
    predictions=[];folds=[];summary=[]
    for regime,field in [('unseen_genotype','genotype'),('unseen_treatment','treatment')]:
        groups=frame[field].to_numpy()
        splits=list(LeaveOneGroupOut().split(frame,groups=groups))
        for target in TARGETS:
            for model_name,columns in MODELS.items():
                all_pred=np.full(len(frame),np.nan)
                for train,test in splits:
                    if set(groups[train]) & set(groups[test]):raise ValueError('Outer group leakage')
                    pred,alpha,scores=predict_fold(frame,target,field,train,test,columns,model_name.startswith('shuffled'))
                    all_pred[test]=pred
                    label=str(groups[test[0]])
                    folds.append(dict(regime=regime,target=target,model=model_name,heldout_group=label,
                                      train_uids=frame.iloc[train].plant_uid.tolist(),test_uids=frame.iloc[test].plant_uid.tolist(),
                                      alpha=alpha,inner_mse=scores))
                    for pos,value in zip(test,pred):
                        row=frame.iloc[pos]
                        predictions.append(dict(regime=regime,target=target,model=model_name,heldout_group=label,
                                                plant_uid=row.plant_uid,treatment=row.treatment,genotype=row.genotype,replicate=row.replicate,
                                                observed=float(row[target]),predicted=float(value)))
                summary.append(dict(regime=regime,target=target,model=model_name,n=len(frame),**metrics(frame[target],all_pred)))
                print(f'{regime} {target} {model_name}: R2 {summary[-1]["r2"]:.4f}',flush=True)
    pd.DataFrame(predictions).to_csv(args.output/'heldout_predictions.csv',index=False)
    result=pd.DataFrame(summary);result.to_csv(args.output/'benchmark_metrics.csv',index=False)
    (args.output/'folds.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    descriptive=[];correlations=[]
    for treatment,group in joined.groupby('treatment'):
        for target in TARGETS:
            descriptive.append(dict(treatment=treatment,target=target,n=len(group),mean=group[target].mean(),
                                    sd=group[target].std(),minimum=group[target].min(),maximum=group[target].max()))
    pd.DataFrame(descriptive).to_csv(args.output/'harvest_summary.csv',index=False)
    for target in TARGETS:
        for feature in RGB+NOIR:
            for conditioning in ['pooled','within_genotype_treatment']:
                part=frame[[target,feature,'genotype','treatment']].copy()
                if conditioning!='pooled':
                    for col in [target,feature]:part[col]-=part.groupby(['genotype','treatment'])[col].transform('mean')
                correlations.append(dict(target=target,feature=feature,conditioning=conditioning,n=len(part),
                                         pearson_r=part[target].corr(part[feature]),
                                         spearman_r=part[target].corr(part[feature],method='spearman')))
    pd.DataFrame(correlations).to_csv(args.output/'descriptive_correlations.csv',index=False)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors={'C':'#27744f','P':'#b37a22','S':'#724d91'}
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    labels=['Whole aboveground fresh mass (g)','Whole aboveground dry mass (g)','Dry matter / fresh mass']
    rng=np.random.default_rng(20261028)
    for j,(target,label) in enumerate(zip(TARGETS,labels)):
        ax=axes[0,j]
        for i,t in enumerate(['C','S','P']):
            group=joined[joined.treatment==t];values=group[target].to_numpy()
            ax.scatter(i+rng.uniform(-.15,.15,len(values)),values,c=colors[t],s=22,alpha=.75)
            ax.plot([i-.2,i+.2],[values.mean()]*2,color='black',lw=2)
        ax.set_xticks(range(3),['Control','Stress','Priming']);ax.set_ylabel(label)
        ax.set_title('Harvest: 24 pots per treatment')
        ax=axes[1,j]
        for t in ['C','S','P']:
            group=frame[frame.treatment==t]
            ax.scatter(group['rgb__area_fraction__median'],group[target],c=colors[t],s=24,label=t)
        ax.set_xlabel('25 DAS median RGB projected area / canvas');ax.set_ylabel(label)
        ax.set_title(f'Eligible paired-camera pots: n = {len(frame)}')
    axes[1,0].legend();fig.tight_layout();fig.savefig(args.output/'harvest_and_image_support.png',dpi=190);fig.savefig(args.output/'harvest_and_image_support.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    pred_frame=pd.DataFrame(predictions)
    for i,regime in enumerate(['unseen_genotype','unseen_treatment']):
        for j,(target,label) in enumerate(zip(TARGETS,labels)):
            ax=axes[i,j]
            for model_name,color in [('mean','#999999'),('metadata','#b37a22'),('RGB_NoIR','#27744f')]:
                group=pred_frame[(pred_frame.regime==regime)&(pred_frame.target==target)&(pred_frame.model==model_name)]
                score=result[(result.regime==regime)&(result.target==target)&(result.model==model_name)].iloc[0]
                ax.scatter(group.observed,group.predicted,s=16,alpha=.65,color=color,label=f'{model_name}: R²={score.r2:.2f}')
            lim=[min(ax.get_xlim()[0],ax.get_ylim()[0]),max(ax.get_xlim()[1],ax.get_ylim()[1])]
            ax.plot(lim,lim,'k:',lw=1);ax.set_title(regime.replace('_',' '));ax.set_xlabel('Observed: '+label);ax.set_ylabel('Held-out prediction');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(args.output/'mass_grouped_predictions.png',dpi=190);fig.savefig(args.output/'mass_grouped_predictions.pdf');plt.close(fig)
    figures=''.join('<img alt="'+caption+'" src="data:image/png;base64,'+base64.b64encode((args.output/name).read_bytes()).decode()+'"><p>'+caption+'</p>' for name,caption in [
        ('harvest_and_image_support.png','Every harvested pot is retained in the top row, including the 0.31 g Priming plant. The lower row uses the same eligible paired-camera subset as the benchmarks. Black bars are treatment means. These comparisons do not separate treatment from calendar run.'),
        ('mass_grouped_predictions.png','One out-of-fold prediction per eligible pot and endpoint. Alpha selection, scaling, imputation and categorical encoding occur within outer training data. A metadata baseline is essential because treatment and genotype can explain pooled associations.')])
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Harvest mass validation</title><style>body{font:19px/1.5 "Times New Roman",serif;max-width:1200px;margin:35px auto;padding:0 24px;background:white;color:#171717}img{width:100%}table{border-collapse:collapse;font-size:16px}th,td{padding:6px;border-bottom:1px solid #ddd;text-align:left} .table{overflow-x:auto}h1,h2{font-weight:normal}</style><h1>Image architecture meets destructive harvest measurements</h1>'''
    page+=f'<p>We joined all 72 confirmed physical pots to whole aboveground fresh and dry mass in grams. The last imaging visit is 25 DAS; the timetable places fresh harvest at 26 DAS and dry-matter assessment at 27 DAS. The shared benchmark subset contains {len(frame)} pots with qualified RGB and NoIR support at 25 DAS. All 72 remain in the join table.</p>'
    page+='<p>This is an exploratory plant-level test of whether projected architecture carries information about subsequent harvest mass. It is neither a calibrated volumetric reconstruction nor a test of individual leaf mass. Images supply four fixed shape summaries per camera: median projected area, maximum visible height, maximum width and median solidity. The temporal model adds the 25/13 DAS area ratio and the 19–25 DAS area change per day from each camera. Missing earlier images remain missing and are imputed only inside training folds.</p>'
    page+=figures+'<h2>Group-disjoint benchmarks</h2><p>We leave one genotype out across six outer folds, or one treatment out across three folds. Inner folds leave out groups of the same type. Ridge alpha is selected from 0.1, 1, 10, 100 and 1000 using mean inner squared error. We compare training-mean and metadata baselines, each camera, both cameras, metadata plus both cameras, temporal features and one fixed training-only shuffled-label control. The control is diagnostic, not a permutation significance test. Every row below uses the same endpoint subset. R² is pooled across outer predictions, not averaged across folds.</p><div class="table">'+result.round(5).to_html(index=False,border=0)+'</div>'
    page+='<h2>Interpretation limits</h2><p>The timetable places treatments in different calendar blocks. Apparent treatment effects may include run effects; these data cannot identify a causal priming benefit independently of that design. The first imaging visit follows priming for P. We retain the small 0.31 g observation and report it for inspection rather than deleting it. Dry-matter fraction is an endpoint ratio, not relative water content, which would require turgid mass. Projected area can change with pose, occlusion, cropping and camera position. A one-day image-to-harvest interval also permits growth or tissue removal between measurements. The destructive masses validate whole-plant outcomes; they do not validate anatomical leaf correspondences.</p><p>CSV outputs include every source-cell address, all pot joins, individual held-out predictions, fold memberships, eligibility flags and descriptive pooled/within-genotype-and-treatment correlations. Data will be made available through Zenodo after publication.</p></html>'
    (args.output/'mass_validation.html').write_text(page,encoding='utf-8')
    manifest=dict(status='completed',mass_pots=len(joined),benchmark_pots=len(frame),experiments=len(summary),predictions=len(predictions),
                  targets=TARGETS,features=MODELS,alpha_grid=ALPHAS,
                  inputs={str(p):sha(p) for p in [args.mass,args.registry,args.traits]},source_sha256=sha(Path(__file__)),
                  outputs={p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.name!='mass_validation_manifest.json'})
    (args.output/'mass_validation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['status','mass_pots','benchmark_pots','experiments','predictions']}))


if __name__=='__main__':main()
