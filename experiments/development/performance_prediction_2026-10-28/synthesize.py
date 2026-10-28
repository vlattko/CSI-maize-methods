"""Build publication figures, paired conditional uncertainty and readable evidence tables."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


COLORS={'C':'#27744f','P':'#b37a22','S':'#724d91'}
MODELCOLORS={'rgb_area_ols':'#686868','rgb_history':'#27744f','rgb_noir_history':'#724d91'}
LABELS={'rgb_area_ols':'One median RGB area','rgb_history':'RGB state and history','rgb_noir_history':'RGB + NoIR state and history'}
TARGETLABELS={'fresh_mass_g':'Fresh mass (g)','dry_mass_g':'Dry mass (g)','dry_matter_fraction':'Dry-matter fraction'}
AGES=[13,19,20,22,25]


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def bootstrap_pair(frame,baseline,candidate,draws=5000):
    columns=['plant_uid','genotype','treatment','observed','predicted']
    a=frame[frame.model==baseline][columns].sort_values('plant_uid').reset_index(drop=True)
    b=frame[frame.model==candidate][columns].sort_values('plant_uid').reset_index(drop=True)
    if a.plant_uid.tolist()!=b.plant_uid.tolist() or not np.array_equal(a.observed,b.observed):raise ValueError('Paired comparison requires identical pots and outcomes')
    groups=[g.index.to_numpy() for _,g in a.groupby(['genotype','treatment'])]
    rng=np.random.default_rng(20261028)
    indices=np.column_stack([rng.choice(group,(draws,len(group)),replace=True) for group in groups])
    y=a.observed.to_numpy();pa=a.predicted.to_numpy();pb=b.predicted.to_numpy()
    sst=np.sum((y[indices]-y[indices].mean(axis=1,keepdims=True))**2,axis=1)
    da=np.abs(pa-y);db=np.abs(pb-y)
    mae=(db-da)[indices].mean(axis=1)
    r2=np.sum(((pa-y)**2-(pb-y)**2)[indices],axis=1)/sst
    return dict(n=len(a),draws=draws,mae_delta=float(np.mean(db-da)),mae_low=float(np.percentile(mae,2.5)),mae_high=float(np.percentile(mae,97.5)),
                r2_delta=float(np.sum((pa-y)**2-(pb-y)**2)/np.sum((y-y.mean())**2)),r2_low=float(np.percentile(r2,2.5)),r2_high=float(np.percentile(r2,97.5)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['run','extensions','observations','output']:parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True);figures=a.output/'figures';figures.mkdir(exist_ok=True)
    for root in [a.run,a.extensions]:
        completion=json.loads((root/'completion.json').read_text())
        for name,digest in completion['outputs'].items():
            if sha(root/name)!=digest:raise ValueError('Analysis output changed: '+str(root/name))
    metrics=pd.read_csv(a.run/'metrics.csv');pred=pd.read_csv(a.run/'predictions.csv');ext=pd.read_csv(a.extensions/'metrics.csv');epred=pd.read_csv(a.extensions/'predictions.csv')
    coverage=pd.read_csv(a.run/'coverage.csv');features=pd.read_csv(a.run/'pot_cutoff_features.csv')
    figure_plan=[];deltas=[]
    for (age,cohort,regime,target),frame in pred.groupby(['cutoff_DAS','cohort','regime','target']):
        for baseline,candidate in [('rgb_area_ols','rgb_history'),('rgb_area_ols','rgb_noir_history'),('rgb_history','rgb_noir_history'),('single_view0_area_ols','rgb_area_ols')]:
            if baseline=='single_view0_area_ols' and cohort!='single_view_matched':continue
            deltas.append(dict(cutoff_DAS=age,cohort=cohort,regime=regime,target=target,baseline=baseline,candidate=candidate,**bootstrap_pair(frame,baseline,candidate)))
    delta=pd.DataFrame(deltas);delta.to_csv(a.output/'paired_conditional_deltas.csv',index=False)
    # Phase comparisons preserve each declared phase and matched population.
    phase=[]
    for (age,cohort,regime,target),frame in epred.groupby(['cutoff_DAS','cohort','regime','target']):
        if not cohort.startswith(('phase_available','all_four_phase')):continue
        p=int(cohort.rsplit('_',1)[1])
        phase.append(dict(cutoff_DAS=age,cohort=cohort,regime=regime,target=target,phase=p,
                          **bootstrap_pair(frame,f'phase{p}_area_ols','rgb_area_ols')))
    phase=pd.DataFrame(phase);phase.to_csv(a.output/'phase_conditional_deltas.csv',index=False)
    group=[]
    focal=pred[(pred.cutoff_DAS==25)&pred.cohort.isin(['available_pair','all_pots'])]
    for keys,frame in focal.groupby(['cohort','regime','model','target','treatment']):
        cohort,regime,model,target,t=keys;y=frame.observed.to_numpy();p=frame.predicted.to_numpy()
        group.append(dict(cohort=cohort,regime=regime,model=model,target=target,treatment=t,n=len(frame),mae=float(abs(p-y).mean()),bias=float((p-y).mean()),
                          r2=float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2))))
    pd.DataFrame(group).to_csv(a.output/'within_treatment_errors.csv',index=False)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

    def save(fig,name,group,caption,alt):
        fig.tight_layout();fig.savefig(figures/(name+'.png'),dpi=230);fig.savefig(figures/(name+'.pdf'));plt.close(fig)
        figure_plan.append(dict(id=name,group=group,source=str(figures/(name+'.png')),caption=caption,alt=alt))

    fig=plt.figure(figsize=(11,6));gs=fig.add_gridspec(2,1,height_ratios=[1.2,1.])
    ax=fig.add_subplot(gs[0]);ax.set_xlim(-2,28);ax.set_ylim(-.6,3.2);ax.axis('off')
    for i,t in enumerate(['C','S','P']):
        y=2.5-i
        ax.text(-1,y+.16,dict(C='Control',S='Stress',P='Priming')[t],ha='right',va='center')
        for day in range(28):
            color='#dddcd5' if t=='C' or not (6<=day<=11 and t=='P') and not 14<=day<=19 else ('#b37a22' if day<=11 else '#746395')
            ax.add_patch(plt.Rectangle((day,y),.95,.34,color=color))
        ax.plot(AGES,[y+.17]*5,'ko',ms=4)
    for day in [0,6,11,13,14,19,20,22,25,26]:ax.text(day+.4,-.1,str(day),ha='center',fontsize=10)
    ax.text(13,-.55,'Days after sowing',ha='center');ax.text(26.6,1.5,'Fresh\nharvest',ha='center',va='center',fontsize=10)
    ax.annotate('',xy=(26.4,.35),xytext=(26.4,1.1),arrowprops=dict(arrowstyle='->',color='black'))
    ax.set_title('72 pots · six genotypes · three treatment/run groups · five imaging visits',pad=12)
    ax=fig.add_subplot(gs[1]);ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    boxes=[(.02,.5,.21,.35,'Observed at cutoff\nRGB and NoIR masks\nFixed phase or full rotation'),(.265,.5,.21,.35,'Features available by cutoff\nCurrent projected shape\nArea ratio and slope'),(.51,.5,.21,.35,'Nested group transfer\nHeld-out genotype or run\nBoth groups held out'),(.755,.5,.215,.35,'Harvest outcomes\nFresh and dry shoot mass\nDry-matter fraction')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.006',fc='white',ec='#555'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
    for x in [.235,.48,.725]:ax.annotate('',xy=(x+.025,.675),xytext=(x,.675),arrowprops=dict(arrowstyle='->'))
    ax.text(.5,.15,'One harvest outcome per pot. Multiple images and visits are predictors, not independent plants.',ha='center',fontsize=11)
    save(fig,'F1_study_and_prediction','main','Study design and prediction tasks. Dots indicate imaging at13,19,20,22 and25 DAS. Ochre marks intermittent priming at6–11DAS; purple marks the severe challenge at14–19DAS for P and S. Control remains at25/18°C. The19DAS measurements precede warming. Predictors use observations through the cutoff only; fresh harvest is scheduled at26DAS and shoots are dried at60°C toconstantmass. LAB measurements used separate plants. Grouped validation retains the pot as the observational unit.','Treatment timeline and a four-stage diagram linking cutoff-specific observations to grouped harvest prediction.')

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,cohort,title in zip(axes,['available_pair','common_five_visits'],['Available at each cutoff','All five paired visits observed']):
        bottom=np.zeros(5)
        for t in ['C','P','S']:
            sub=coverage[(coverage.cohort==cohort)&(coverage.treatment==t)].set_index('cutoff_DAS').loc[AGES]
            ax.bar(range(5),sub.n,bottom=bottom,color=COLORS[t],label=t)
            for i,n in enumerate(sub.n):ax.text(i,bottom[i]+n/2,str(n),ha='center',va='center',color='white')
            bottom+=sub.n.to_numpy()
        for i,n in enumerate(bottom):ax.text(i,n+1,str(int(n)),ha='center')
        ax.axhline(72,color='#777',ls=':',lw=1);ax.set_ylim(0,80);ax.set_xticks(range(5),AGES);ax.set_xlabel('Cutoff DAS');ax.set_ylabel('Pots with qualified paired images');ax.set_title(title)
    axes[0].legend(ncol=3,loc='lower left');save(fig,'F2_observation_coverage','main','Observation support changes with cutoff. Current paired-camera eligibility requires at least20 accepted views and four quadrants in each camera, giving63,62,71,65 and68 pots. The common-five-visit sensitivity retains49pots, comprising18C,18P and13S. These populations must not be treated as interchangeable. All72harvest records are retained separately in the missing-image policy analysis.','Stacked bars show treatment-specific paired-camera support at each age and the smaller common five-visit population.')

    def lead_figure(cohort,name,group):
        fig,axes=plt.subplots(2,3,figsize=(11,7),sharex=True)
        for i,target in enumerate(['fresh_mass_g','dry_mass_g']):
            for j,regime in enumerate(['genotype','treatment','joint']):
                ax=axes[i,j]
                for model,color in MODELCOLORS.items():
                    sub=metrics[(metrics.cohort==cohort)&(metrics.regime==regime)&(metrics.target==target)&(metrics.model==model)].sort_values('cutoff_DAS')
                    ax.plot(sub.cutoff_DAS,sub.r2,'o-',color=color,label=LABELS[model],ms=4)
                ax.axhline(0,color='#aaa',ls=':',lw=1);ax.set_title(dict(genotype='Genotype held out',treatment='Treatment/run held out',joint='Both groups held out')[regime]);ax.set_ylabel(TARGETLABELS[target]+'\npooled out-of-fold R²');ax.set_xticks(AGES);ax.grid(alpha=.12)
                if i==1:ax.set_xlabel('Cutoff DAS')
        axes[0,0].legend(fontsize=8)
        save(fig,name,group,'Prediction from observations available by each cutoff. Unequal time intervals are retained. The fresh-harvest lead times are13,7,6,4 and1 days. Each point pools one out-of-fold prediction per selected pot. History fields are omitted at the first visit. Treatment/run and joint-transfer results do not constitute independent-run validation. This panel uses '+cohort.replace('_',' ')+'.','Lead-time curves compare one projected-area predictor with RGB and paired-camera state-and-history models under three disjoint transfer tasks.')
    lead_figure('available_pair','F3_lead_time_transfer','main');lead_figure('common_five_visits','S1_common_population_lead_time','supplement');lead_figure('all_pots','S2_all_pot_policy','supplement')

    fig,axes=plt.subplots(2,3,figsize=(11,7))
    for i,target in enumerate(['fresh_mass_g','dry_mass_g']):
        for j,model in enumerate(['rgb_area_ols','rgb_history','rgb_noir_history']):
            ax=axes[i,j];sub=pred[(pred.cohort=='all_pots')&(pred.cutoff_DAS==25)&(pred.regime=='genotype')&(pred.target==target)&(pred.model==model)].merge(features[features.cutoff_DAS==25][['plant_uid','current_pair_eligible']],on='plant_uid',validate='one_to_one')
            for t in ['C','P','S']:
                for good,marker in [(True,'o'),(False,'^')]:
                    q=sub[(sub.treatment==t)&(sub.current_pair_eligible==good)];ax.scatter(q.observed,q.predicted,color=COLORS[t],s=25 if good else 65,marker=marker,edgecolor='black' if not good else 'none',label=t if good else None)
            lim=[min(ax.get_xlim()[0],ax.get_ylim()[0]),max(ax.get_xlim()[1],ax.get_ylim()[1])];ax.plot(lim,lim,'k:',lw=1)
            q=metrics[(metrics.cohort=='all_pots')&(metrics.cutoff_DAS==25)&(metrics.regime=='genotype')&(metrics.target==target)&(metrics.model==model)].iloc[0]
            ax.set_title(LABELS[model]+f'\nR²={q.r2:.3f}, MAE={q.mae:.3f} g',fontsize=10);ax.set_xlabel('Measured '+TARGETLABELS[target]);ax.set_ylabel('Held-out prediction (g)')
    axes[0,0].legend(fontsize=9);save(fig,'F4_all_pot_harvest_predictions','main','Harvest predictions for all72pots under genotype-held-out evaluation. Triangles mark the four Priming pots without qualified endpoint RGB/NoIR pairs; their missing features are imputed using outer-training data, with training-derived missing indicators. The0.31gplant is retained. These all-pot models test a missing-image policy; an imputed prediction is not a direct image measurement. Ordinary circles identify pots with qualified endpoint pairs.','Measured versus predicted fresh and dry mass for area, RGB history and paired-camera history models; missing endpoint pairs are marked with triangles.')

    fig,axes=plt.subplots(2,3,figsize=(11,7),sharex=True)
    for i,target in enumerate(['fresh_mass_g','dry_mass_g']):
        for j,regime in enumerate(['genotype','treatment','joint']):
            ax=axes[i,j];sub=phase[(phase.cutoff_DAS==25)&phase.cohort.str.startswith('all_four_phase_matched')&(phase.regime==regime)&(phase.target==target)].sort_values('phase')
            ax.errorbar(sub.phase/40*360,sub.r2_delta,yerr=[sub.r2_delta-sub.r2_low,sub.r2_high-sub.r2_delta],fmt='o',color='#27744f',capsize=3)
            ax.axhline(0,color='#777',ls=':');ax.set_xticks([0,90,180,270]);ax.set_ylabel(TARGETLABELS[target]+'\nΔR² rotation minus fixed phase');ax.set_title(dict(genotype='Genotype held out',treatment='Treatment/run held out',joint='Both groups held out')[regime])
            if i==1:ax.set_xlabel('Nominal acquisition phase (degrees)')
    save(fig,'F5_rotation_and_phase','main','Rotation summaries compared with four fixed acquisition phases at25DAS. All four phases and the multiview predictor use exactly the same phase-complete population in each transfer task. No phase was selected using harvest outcomes. Identical nominal phase does not establish a common anatomical azimuth. Whiskers are95% paired whole-pot percentile intervals from5,000 resamples within genotype×treatment strata, conditional on saved out-of-fold predictions; models are not refitted during bootstrap. Positive values favour the rotation median.','Paired differences in R squared compare rotation-median area with each of four fixed phases, with conditional uncertainty intervals.')

    fig,axes=plt.subplots(2,3,figsize=(11,7))
    names=['rgb_area_ols','rgb_noir_history','joint_history_assays','joint_history_appearance','assay_state']
    labels=['One area','Paired history','+ measured assays','+ appearance','Assays only']
    for i,target in enumerate(['fresh_mass_g','dry_mass_g']):
        for j,regime in enumerate(['genotype','treatment','joint']):
            ax=axes[i,j];sub=ext[(ext.cohort=='physiology_available_pair')&(ext.cutoff_DAS==25)&(ext.regime==regime)&(ext.target==target)].set_index('model')
            vals=sub.loc[names].r2;ax.bar(range(len(names)),vals,color=['#777','#724d91','#b37a22','#607e9d','#aaa']);ax.axhline(0,color='#555',lw=.8);ax.set_xticks(range(len(names)),labels,rotation=35,ha='right',fontsize=8);ax.set_ylabel(TARGETLABELS[target]+'\nout-of-fold R²');ax.set_title(regime.replace('_',' ').capitalize())
    save(fig,'F6_functional_and_appearance_controls','main','Selected endpoint controls on the same68paired-image pots. Measured assays are Fv/Fm,PIABS and instrument NIR NDVI, not camera-derived reflectance. Appearance uses five camera-channel summaries per camera. Adding these variables to paired-camera history does not consistently improve the tested ridge predictors and can reduce transfer accuracy. This comparison does not establish that physiology is uninformative or that a calibrated spectral instrument would be redundant.','Bars compare structural and temporal prediction with measured physiological assays and uncalibrated image appearance, retaining negative transfer scores.')

    observation=pd.read_csv(a.observations)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,field,label in zip(axes,['ojip__Pi_Abs','ojip__Fv/Fm'],['PIABS','Fv/Fm']):
        for uid,group in observation[observation.treatment=='C'].groupby('plant_uid'):
            group=group.sort_values('declared_das');ax.plot(group.declared_das,group[field],color='#aaa',lw=.5,alpha=.55)
        means=observation[observation.treatment=='C'].groupby('declared_das')[field].mean()
        ax.plot(means.index,means.values,'o-',color=COLORS['C'],lw=2);ax.set_xlabel('DAS');ax.set_xticks(AGES);ax.set_ylabel(label);ax.set_title('Control: 24 repeated pots')
    save(fig,'S3_control_physiological_change','supplement','Control physiological trajectories. Thin lines show the24pots and the thick line their age-specific mean. PIABS declines overall but not monotonically. Each pot contained165g dried soil; root occupancy, root mass and actual substrate volume were not measured. Restricted rooting space or resource supply is a discussion hypothesis, not an identified cause. LAB sampling used separate plants and did not remove tissue from these pots.','Individual control-pot PIABS and Fv/Fm trajectories show background physiological changes during the imaging period.')
    # Tables selected for readable manuscripts; exhaustive metrics remain data files.
    metrics[(metrics.cutoff_DAS==25)&(metrics.cohort=='available_pair')&metrics.model.isin(['mean','metadata','rgb_area_ols','rgb_area_ridge','rgb_area_allometry','rgb_shape','noir_shape','rgb_noir_shape','rgb_history','rgb_noir_history'])].to_csv(a.output/'endpoint_benchmark_table.csv',index=False)
    ext[ext.cohort.str.startswith('physiology_')].to_csv(a.output/'functional_controls_table.csv',index=False)
    (a.output/'figure_plan.json').write_text(json.dumps(dict(figures=figure_plan),indent=2),encoding='utf-8')
    manifest=dict(status='completed',paired_delta_rows=len(delta),phase_delta_rows=len(phase),figures=len(figure_plan),bootstrap_draws=5000,
                  bootstrap_scope='Fixed out-of-fold predictions, pot resampling within genotype/treatment; no retraining or independent cohort uncertainty',
                  inputs={str(p):sha(p) for p in [a.run/'completion.json',a.extensions/'completion.json',a.observations]},source_sha256=sha(Path(__file__)),
                  outputs={p.relative_to(a.output).as_posix():sha(p) for p in a.output.rglob('*') if p.is_file() and p.name!='synthesis_manifest.json'})
    (a.output/'synthesis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({k:manifest[k] for k in ['status','paired_delta_rows','phase_delta_rows','figures']},indent=2))


if __name__=='__main__':main()
