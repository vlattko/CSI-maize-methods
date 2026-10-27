"""Source-linked white/Times figures and a self-contained prediction report."""
import argparse
import base64
import csv
from datetime import datetime,timezone
import html
import io
import json
import os
from pathlib import Path
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import prediction as p

DOMAINS = ('assay_FvFm','assay_PiAbs','assay_NIR_NDVI')
TITLES = dict(zip(DOMAINS,('Fv/Fm','PIABS','Instrument NIR NDVI')))
MODELS = ('training_mean','treatment_visit_mean','visit_mean','metadata_ridge',
          'combined','metadata_plus_image','dino_rgb','dino_noir','dino_combined',
          'metadata_plus_dino','registered_indices','metadata_plus_indices',
          'registered_camera_dns','metadata_plus_dns','metadata_plus_dns_plus_cross_indices')
LABELS = dict(training_mean='Training mean',treatment_visit_mean='Treatment/visit mean',
    visit_mean='Visit mean',metadata_ridge='Metadata ridge',combined='SAM traits',
    metadata_plus_image='Metadata + SAM',dino_rgb='RGB DINO',dino_noir='NoIR DINO',
    dino_noir_blue='Blue-triplicate NoIR DINO',dino_rgb_plus_noir_blue='RGB + blue-triplicate NoIR DINO',
    dino_combined='RGB + NoIR DINO',metadata_plus_dino='Metadata + DINO',
    registered_indices='27 registered indices',metadata_plus_indices='Metadata + indices',
    registered_camera_dns='Common-overlap DNs',metadata_plus_dns='Metadata + DNs',
    metadata_plus_dns_plus_cross_indices='Metadata + DNs + cross indices',
    sam_traits='SAM traits',sam_traits_plus_indices='SAM + indices',
    camera_dns_plus_cross_indices='DNs + cross indices')


def assert_matched(left,right):
    if left['keys']!=right['keys']:
        raise ValueError('Direct comparison requires identical ordered cohort keys')
    if not np.array_equal(left['observed'],right['observed']):
        raise ValueError('Direct comparison has different observed targets')


def display_choice(rows):
    allowed = [r for r in rows if 'trajectory_null' not in r['model']
               and np.isfinite(float(r['rmse']))]
    return min(allowed,key=lambda r:float(r['rmse'])) if allowed else None


def style():
    times = Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/'times.ttf'
    if times.exists():
        font_manager.fontManager.addfont(str(times))
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':11,'axes.titlesize':13,'axes.labelsize':11,'figure.facecolor':'white',
        'axes.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'xtick.direction':'out','ytick.direction':'out','legend.frameon':False})


def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def catalog(root):
    result = {}
    for marker in sorted(root.glob('*/complete.json')):
        manifest = json.loads(marker.read_text(encoding='utf-8'))
        if manifest['status']!='completed':
            continue
        with np.load(marker.parent/'oof.npz') as data:
            keys = [tuple(k) for k in data['keys']]
            observed = data['observed'].copy()
        result[marker.parent.name] = dict(path=marker.parent,manifest=manifest,
            keys=keys,observed=observed,metrics=read_rows(marker.parent/'metrics.csv'))
    return result


def find(data,cohort,domain,split,family='',sensitivity='source_retained'):
    name = f'{family}{cohort}__{domain}__{sensitivity}__{split}'
    return data.get(name)


def comparison_rows(data,cohort,domain,split):
    left = find(data,cohort,domain,split)
    if left is None:
        return None,{}
    rows = {r['model']:r for r in left['metrics']}
    if cohort=='paired_primary':
        right = find(data,cohort,domain,split,'indices__')
        if right is not None:
            assert_matched(left,right)
            for row in right['metrics']:
                if row['model'] not in rows:
                    rows[row['model']] = row
    return left,rows


def colour(model):
    if 'dino' in model:
        return '#466c8b'
    if 'indices' in model or 'dns' in model:
        return '#a57232'
    if model in ('combined','metadata_plus_image','sam_traits'):
        return '#4f755c'
    return '#505050'


def export(fig,figures,name,caption,records):
    png,pdf = figures/(name+'.png'),figures/(name+'.pdf')
    fig.savefig(png,dpi=220,bbox_inches='tight',pad_inches=.12)
    fig.savefig(pdf,bbox_inches='tight',pad_inches=.12,metadata={'Title':name,'Creator':'CSI maize methods'})
    plt.close(fig)
    records.append(dict(name=name,png=png,pdf=pdf,caption=caption))


def comparison_figures(data,cohort,split,figures,records):
    panels = [comparison_rows(data,cohort,d,split) for d in DOMAINS]
    if not any(source is not None for source,rows in panels):
        return
    models = [m for m in MODELS if any(m in rows for source,rows in panels)]
    label = 'Held-out genotype' if split=='genotype' else 'Held-out treatment'
    for statistic,axis_label in (('rmse','OOF RMSE'),('mean_target_r2','OOF R-squared')):
        fig,axes = plt.subplots(1,3,figsize=(15,max(4.4,.32*len(models)+1.6)),layout='constrained')
        for ax,domain,(source,rows) in zip(axes,DOMAINS,panels):
            ax.set_yticks(range(len(models)),[LABELS.get(m,m) for m in models],fontsize=10)
            ax.invert_yaxis()
            if source is None:
                ax.text(.5,.5,'Waiting for completed experiment',ha='center',va='center',transform=ax.transAxes)
                ax.set_title(TITLES[domain])
                continue
            for i,model in enumerate(models):
                if model in rows and rows[model][statistic] not in ('',None):
                    value = float(rows[model][statistic])
                    ax.barh(i,value,height=.66,color=colour(model),alpha=.86)
            ax.axvline(0,color='black',lw=.65)
            ax.grid(axis='x',color='#dddddd',lw=.5)
            ax.set_axisbelow(True)
            ax.set_title(f'{TITLES[domain]} (n={len(source["keys"])})')
            ax.set_xlabel(axis_label)
        scope = 'all 360 assay rows; metadata controls' if cohort=='all360' else 'matched primary paired cohort'
        fig.suptitle(f'{label}: {scope}',fontsize=15)
        name = f'{cohort}_{split}_{statistic}'
        caption = (f'{label}, {scope}. Every displayed model for a target uses identical '
            'ordered keys and observed targets. Bars show pooled OOF performance; no view is an independent sample. '
            'Registered-index bars appear only after their completed matched-cohort experiment is available.')
        export(fig,figures,name,caption,records)


def oof_figure(data,figures,records):
    panels = [comparison_rows(data,'paired_primary',d,'genotype') for d in DOMAINS]
    if not any(source is not None for source,rows in panels):
        return
    genotypes = sorted({str(k[2]) for source,rows in panels if source for k in source['keys']})
    palette = dict(zip(genotypes,plt.get_cmap('tab10').colors[:len(genotypes)]))
    fig,axes = plt.subplots(1,3,figsize=(14,4.9),layout='constrained')
    for ax,domain,(source,rows) in zip(axes,DOMAINS,panels):
        if source is None:
            ax.text(.5,.5,'Waiting for completed experiment',ha='center',transform=ax.transAxes)
            continue
        best = display_choice(list(rows.values()))
        model = best['model']
        selected = source
        if model not in source['manifest']['models']:
            selected = find(data,'paired_primary',domain,'genotype','indices__')
        with np.load(selected['path']/'oof.npz') as archive:
            y,prediction = archive['observed'][:,0],archive[model][:,0]
        groups = np.array([str(k[2]) for k in selected['keys']])
        for genotype in genotypes:
            keep = groups==genotype
            ax.scatter(y[keep],prediction[keep],s=14,alpha=.64,color=palette[genotype],
                       label=genotype,linewidths=0)
        lo,hi = min(y.min(),prediction.min()),max(y.max(),prediction.max())
        pad = max((hi-lo)*.04,.001)
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='black',lw=.8,ls='--')
        ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel='Observed',ylabel='OOF predicted')
        ax.set_title(f'{TITLES[domain]}\n{LABELS.get(model,model)}; n={len(y)}\n'
                     f'RMSE {float(best["rmse"]):.4g}; R-squared {float(best["mean_target_r2"]):.3f}',fontsize=11)
    handles,labels = next((ax.get_legend_handles_labels() for ax in axes if ax.get_legend_handles_labels()[0]),([],[]))
    fig.legend(handles,labels,loc='outside lower center',ncol=6,fontsize=10)
    fig.suptitle('Internal best-display OOF plots: six held-out-genotype folds',fontsize=14)
    export(fig,figures,'primary_oof_internal_selection',
        'For display only, each target uses the lowest pooled OOF RMSE among completed non-null models '
        'on the same primary cohort. This outer-OOF winner selection is internal and does not give an '
        'unbiased estimate of the selected model. Colours identify six genotype clusters; the dashed line is identity.',records)


def native_figures(data,figures,records):
    preferred = 'paired_primary' if find(data,'paired_primary','NIR','genotype') else 'all360'
    selected = ('training_mean','metadata_ridge','combined','metadata_plus_image',
                'dino_combined','metadata_plus_dino','registered_indices',
                'metadata_plus_dns_plus_cross_indices')
    fig,axes = plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    used = False
    for ax,band in zip(axes,('UVIS','NIR')):
        source,rows = comparison_rows(data,preferred,band,'genotype')
        if source is None:
            ax.text(.5,.5,'Waiting for completed native-grid experiment',ha='center',transform=ax.transAxes)
            continue
        used = True
        tables = read_rows(source['path']/'target_metrics.csv')
        indexed = find(data,preferred,band,'genotype','indices__')
        if indexed:
            assert_matched(source,indexed)
            tables += [r for r in read_rows(indexed['path']/'target_metrics.csv')
                       if r['model'] not in source['manifest']['models']]
        for model in selected:
            curve = [r for r in tables if r['model']==model]
            if curve:
                ax.plot([float(r['target']) for r in curve],[float(r['rmse']) for r in curve],
                    label=LABELS.get(model,model),color=colour(model),
                    ls=':' if model=='training_mean' else '--' if model.startswith('metadata_plus') else '-',lw=1.3)
        ax.set(title=f'{band}: native 256-channel grid (n={len(source["keys"])})',
               xlabel='Source wavelength (instrument header)',ylabel='Native transmittance OOF RMSE')
        ax.grid(color='#e2e2e2',lw=.5)
    if used:
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
        fig.suptitle(f'Six-genotype validation; {preferred} cohort',fontsize=14)
        export(fig,figures,'native_wavelength_errors',
            f'Native instrument-header order and source transmittance values, {preferred} cohort. '
            'The two grids remain separate. Spectral curves are instrument targets and are not spatial hyperspectral images.',records)
    else:
        plt.close(fig)
    full = find(data,preferred,'UVIS','genotype')
    sensitivity = find(data,preferred,'UVIS','genotype',sensitivity='source_flag_sensitivity')
    if full and sensitivity:
        a,b = read_rows(full['path']/'target_metrics.csv'),read_rows(sensitivity['path']/'target_metrics.csv')
        fig,axes=plt.subplots(1,2,figsize=(13,4.3),layout='constrained')
        for model in ('training_mean','metadata_ridge','combined','metadata_plus_dino'):
            left=[r for r in a if r['model']==model]
            right=[r for r in b if r['model']==model]
            if not left or not right:
                continue
            wave=np.array([float(r['target']) for r in left])
            y0=np.array([float(r['rmse']) for r in left])
            y1=np.array([float(r['rmse']) for r in right])
            axes[0].plot(wave,y0,color=colour(model),label=LABELS.get(model,model),lw=1.2)
            axes[0].plot(wave,y1,color=colour(model),ls='--',lw=1.2)
            axes[1].plot(wave,100*(y1/y0-1),color=colour(model),label=LABELS.get(model,model),lw=1.2)
        axes[0].set(title=f'UVIS retained n={len(full["keys"])}; sensitivity n={len(sensitivity["keys"])}',
                    xlabel='Source wavelength (instrument header)',ylabel='OOF RMSE')
        axes[1].set(title='Sensitivity relative to retained-source RMSE',
                    xlabel='Source wavelength (instrument header)',ylabel='RMSE change (%)')
        axes[1].axhline(0,color='black',lw=.7)
        for ax in axes:
            ax.grid(color='#e2e2e2',lw=.5)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
        export(fig,figures,'uvis_source_flag_sensitivity',
            'Solid curves retain all eligible source records, including source-consistency warnings. '
            'Dashed curves omit flagged records as a prespecified sensitivity; this changes the cohort. '
            'No source cell is silently repaired, and a change cannot be interpreted as an independent calibration validation.',records)


def eligibility_figure(args,figures,records):
    path=args.run/'extraction'/'summary.json'
    if not path.exists() or json.loads(path.read_text())['status']!='completed':
        return
    references=p.read_keyed(args.intake/'research_audit'/'reference_master.csv')
    sessions={p.key(r):r for r in p.read_keyed(args.run/'extraction'/'session_traits.csv')}
    rows=[{**r,**sessions[p.key(r)]} for r in references]
    fig,axes=plt.subplots(2,3,figsize=(13.4,8.4),layout='constrained')
    treatments=('C','P','S')
    for ax,cohort,label in zip(axes[0],('rgb_primary','noir_primary','paired_primary'),('RGB','NoIR','Paired')):
        retained=set(p.cohort_indices(rows,cohort))
        matrix=np.empty((3,5))
        for ti,treatment in enumerate(treatments):
            for vi,visit in enumerate(range(1,6)):
                candidates=[i for i,r in enumerate(rows) if r['treatment']==treatment and str(r['measurement'])==str(visit)]
                n=sum(i in retained for i in candidates)
                matrix[ti,vi]=n/max(len(candidates),1)
                ax.text(vi,ti,f'{n}/{len(candidates)}',ha='center',va='center',fontsize=11,
                        color='white' if matrix[ti,vi]>.65 else 'black')
        ax.imshow(matrix,vmin=0,vmax=1,cmap='Greens',aspect='auto')
        ax.set(xticks=range(5),xticklabels=range(1,6),yticks=range(3),yticklabels=treatments,
               xlabel='Recorded visit',ylabel='Treatment',title=f'{label}: accepted / all records')
    retained=set(p.cohort_indices(rows,'paired_primary'))
    rng=np.random.default_rng(20261027)
    for ax,target,domain in zip(axes[1],p.TARGETS,DOMAINS):
        groups=[]
        for state in (True,False):
            values=np.array([p.number(r[target]) for i,r in enumerate(rows) if (i in retained)==state])
            groups.append(values[np.isfinite(values)])
        ax.boxplot(groups,positions=(1,2),widths=.55,showfliers=False,
                   medianprops={'color':'black'},boxprops={'color':'#555555'})
        for x,values in zip((1,2),groups):
            ax.scatter(x+rng.uniform(-.12,.12,len(values)),values,s=8,
                       alpha=.35,color='#4f755c' if x==1 else '#a57232',linewidths=0)
        ax.set(xticks=(1,2),xticklabels=[f'Retained\nn={len(groups[0])}',f'Excluded\nn={len(groups[1])}'],
               ylabel=TITLES[domain],title=TITLES[domain])
        ax.grid(axis='y',color='#e2e2e2',lw=.5)
    fig.suptitle('Coverage selection audit: all 360 assay identities retained in the record',fontsize=14)
    export(fig,figures,'eligibility_selection',
        'Primary eligibility requires at least 20 accepted views and all four quadrants per camera. '
        'Assay distributions compare paired-retained and excluded records descriptively. Treatment/visit '
        'mixtures can differ; this is neither a causal selection effect nor a test of segmentation accuracy. '
        'Stratified counts and distributions are available in the long-form eligibility table.',records)


def contrast_figure(root,data,figures,records):
    path=root/'incremental_contrasts.csv'
    if not path.exists():
        return
    contrasts=read_rows(path)
    planned=(('metadata_ridge','metadata_plus_image'),('metadata_ridge','metadata_plus_dino'),
             ('metadata_ridge','metadata_plus_indices'),('sam_traits','sam_traits_plus_indices'),
             ('metadata_plus_dns','metadata_plus_dns_plus_cross_indices'))
    labels=('Metadata -> + SAM','Metadata -> + DINO','Metadata -> + indices',
            'SAM -> + indices','Metadata + DNs -> + cross indices')
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    used=False
    for axis_row,split in zip(axes,('genotype','treatment')):
        for ax,domain in zip(axis_row,DOMAINS):
            selected=[]
            for row in contrasts:
                source=data.get(row['experiment'])
                if source is None:
                    continue
                manifest=source['manifest']
                if (manifest['cohort']=='paired_primary' and manifest['domain']==domain and
                    manifest['split']==split and '__source_retained__' in row['experiment']):
                    selected.append(row)
            for i,(baseline,augmented) in enumerate(planned):
                row=next((r for r in selected if r['baseline']==baseline and r['augmented']==augmented),None)
                if row:
                    used=True
                    value=float(row['rmse_improvement'])
                    low=float(row['exploratory_cluster_bootstrap_low'])
                    high=float(row['exploratory_cluster_bootstrap_high'])
                    ax.plot([low,high],[i,i],color=colour(augmented),lw=1.3)
                    ax.scatter([value],[i],color=colour(augmented),s=25,zorder=3)
            ax.set(yticks=range(len(labels)),yticklabels=labels,title=f'{TITLES[domain]}: {split}',
                   xlabel='RMSE improvement (baseline - augmented)')
            ax.invert_yaxis()
            ax.axvline(0,color='black',lw=.75,ls='--')
            ax.grid(axis='x',color='#e2e2e2',lw=.5)
    if used:
        fig.suptitle('Paired incremental comparisons on identical primary rows',fontsize=14)
        export(fig,figures,'incremental_paired_contrasts',
            'Positive values favour the augmented representation. Intervals resample six complete genotype '
            'clusters 2,000 times using fixed OOF errors; they do not refit model selection and are exploratory. '
            'The DN/cross-index comparison concerns algebraic representations of the same recorded channels.',records)
    else:
        plt.close(fig)


def table(rows,columns):
    def cell(value):
        if value in ('',None):
            return 'missing'
        if isinstance(value,float):
            return f'{value:.6g}'
        try:
            numeric=float(value)
            if np.isfinite(numeric):
                return f'{numeric:.6g}'
        except (TypeError,ValueError):
            pass
        return html.escape(str(value))
    return '<table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(r.get(c))+'</td>'
        for c in columns)+'</tr>' for r in rows)+'</tbody></table>'


def bundle(root,figures,experiments,args):
    path=root/'prediction_report_bundle.zip'
    temporary=root/'prediction_report_bundle.tmp.zip'
    members=[]
    with zipfile.ZipFile(temporary,'w') as archive:
        candidates=list(root.glob('*.csv'))+list(root.glob('*.md'))+list(root.glob('provenance*.json'))+list(root.glob('audit.json'))
        candidates+=list(root.glob('execution_source*.py'))
        candidates+=list((root/'encoder').glob('availability.csv'))
        candidates+=[root/'encoder'/'operator.json',root/'encoder'/'execution_source.py']
        for experiment in experiments.values():
            folder=experiment['path']
            candidates+=list(folder.glob('*.csv'))+[folder/'complete.json',folder/'features.json']
            if experiment['manifest']['cohort'] in ('all360','paired_primary'):
                candidates.append(folder/'oof.npz')
        candidates+=list(figures.glob('*.png'))+list(figures.glob('*.pdf'))
        for candidate in sorted(set(candidates)):
            if not candidate.exists():
                continue
            relative=candidate.relative_to(root).as_posix()
            archive.write(candidate,relative,compress_type=(zipfile.ZIP_STORED if candidate.suffix in
                          ('.npz','.png','.pdf') else zipfile.ZIP_DEFLATED))
            members.append(relative)
        sources=[(args.intake/'research_audit'/name,
                  'reproduction_inputs/intake/research_audit/'+name)
                 for name in ('reference_master.csv','uvis_consistency_issues.json')]
        sources += [(args.intake/'research_audit'/'tables'/f'{band}_P6_transmitance__List1.csv',
                     f'reproduction_inputs/intake/research_audit/tables/{band}_P6_transmitance__List1.csv')
                    for band in ('UVIS','NIR')]
        sources += [(args.run/'extraction'/name,'reproduction_inputs/run/extraction/'+name)
                    for name in ('session_traits.csv','session_traits_min30.csv','session_traits_min40.csv','summary.json')]
        sources += [(root/'encoder'/name,'reproduction_inputs/run/prediction/encoder/'+name)
                    for name in ('session_embeddings.npz','operator.json','progress.json')]
        sources += [(args.run/'pairing'/'full'/name,'reproduction_inputs/run/pairing/full/'+name)
                    for name in ('session_descriptors.csv','session_descriptors_manifest.json',
                                 'session_camera_dn.csv','session_camera_dn_manifest.json')]
        sources += [(path,'analysis_code/'+path.name) for path in Path(__file__).parent.glob('prediction*.py')]
        for source,name in sources:
            if source.exists():
                archive.write(source,name,compress_type=(zipfile.ZIP_STORED if source.suffix=='.npz'
                                                         else zipfile.ZIP_DEFLATED))
                members.append(name)
    temporary.replace(path)
    return path,members


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('intake','run','repo'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    root=args.run/'prediction'
    figures=root/'figures'
    figures.mkdir(exist_ok=True)
    data=catalog(root)
    if not data:
        raise ValueError('No completed prediction experiments available')
    style()
    records=[]
    comparison_figures(data,'all360','genotype',figures,records)
    for split in ('genotype','treatment'):
        comparison_figures(data,'paired_primary',split,figures,records)
    oof_figure(data,figures,records)
    native_figures(data,figures,records)
    eligibility_figure(args,figures,records)
    contrast_figure(root,data,figures,records)
    numerical=[]
    for experiment in data.values():
        for row in experiment['metrics']:
            numerical.append(dict(experiment=experiment['path'].name,**row))
    p.save_csv(root/'report_metrics_long.csv',numerical)
    bundle_path,members=bundle(root,figures,data,args)
    phase_rows=[]
    for stage,filename,fallback in (('Metadata controls','summary_metadata.json',None),
                           ('Image models','summary_images.json','finalizer_status.json'),
                           ('Registered indices','summary_indices.json','indices_status.json')):
        path=root/filename
        value=json.loads(path.read_text()) if path.exists() else {}
        if not value and fallback and (root/fallback).exists():
            value=json.loads((root/fallback).read_text())
        phase_rows.append(dict(stage=stage,status=value.get('status','waiting / running'),
                               completed_experiments=value.get('experiments','not yet finalized')))
    now=datetime.now(timezone.utc).isoformat(timespec='seconds')
    parts=['<!doctype html><html><head><meta charset="utf-8"><title>CSI maize prediction controls</title>',
        '<style>body{font:18px/1.45 "Times New Roman",Times,serif;background:white;color:#111;max-width:1380px;margin:40px auto;padding:0 24px}h1{font-size:34px}h2{font-size:26px;margin-top:40px}figure{margin:30px 0}img{width:100%;height:auto}figcaption{font-size:16px}table{border-collapse:collapse;font-size:15px;width:100%;margin:16px 0}th,td{border-bottom:1px solid #ccc;text-align:left;padding:6px 9px}th{background:#f4f4f4}a{color:#1b5275}.status{border-left:3px solid #555;padding:10px 18px;background:#f6f6f6}.note{font-size:16px}code{font-size:14px}details{margin:20px 0}</style></head><body>',
        '<h1>Full-cohort prediction controls</h1>',
        f'<p class="note">Investigation: 27 October 2026. Report rebuilt {html.escape(now)}.</p>',
        '<p>We compare recorded metadata, accepted-mask camera traits, frozen DINOv2 image descriptors and registered camera-index representations. '
        'Only completed experiment records enter figures. All 360 assay identities remain in the source record; image-model denominators follow the declared coverage rule.</p>',
        '<div class="status"><strong>Execution status</strong>'+table(phase_rows,('stage','status','completed_experiments'))+
        '<p>Waiting sections contain no invented image or index results. Completed all-360 metadata controls are distinct from the matched image cohort.</p></div>',
        '<h2>Validation and interpretation</h2><p>The primary validation holds out each of six genotypes. Training-only inner folds select ridge alpha and fit imputation and scaling. '
        'A separate challenge holds out each treatment; its metadata model uses visit alone. Candidate treatment/genotype/replicate trajectories remain together. '
        'Treatment transfer can share genotypes and is not a simultaneous unseen-genotype/unseen-treatment test.</p>',
        '<p>Metadata-only success does not establish image physiology. Native transmittance curves are instrument targets, not spatial hyperspectral reconstructions. '
        'The 27 paired formulas transform six recorded channels and cannot add independent spectral measurements. Automatic mask QC does not establish anatomical accuracy. '
        'The bootstrap uses only six genotype clusters and fixed OOF errors, so its intervals are exploratory.</p>',
        '<h2>Figures from completed experiments</h2>']
    index_table=args.run/'pairing'/'full'/'session_descriptors.csv'
    primary=find(data,'paired_primary','assay_FvFm','genotype')
    if index_table.exists() and primary:
        keys=set(primary['keys'])
        available=[r for r in p.read_keyed(index_table) if p.key(r) in keys]
        columns=[n for n in available[0] if n.startswith('paired_index__') and n.endswith('__eligible')]
        any_index=sum(any(p.truth(r[n]) for n in columns) for r in available)
        all_index=sum(all(p.truth(r[n]) for n in columns) for r in available)
        green=sum(p.truth(r['paired_index__CAMSUM6__eligible']) for r in available)
        parts.insert(-1,'<p>Index availability on the matched primary cohort: '+
            f'{any_index}/{len(available)} records have any eligible formula and only '
            f'{all_index}/{len(available)} have all 27. CAMSUM6 and four other green-dependent six-channel '
            f'formulas are eligible in {green}/{len(available)} records. Models retain per-feature missingness '
            'and use training-only imputation. Availability flags are not predictors.</p>')
    for figure in records:
        encoded=base64.b64encode(figure['png'].read_bytes()).decode()
        pdf=figure['pdf'].relative_to(root).as_posix()
        parts.append('<figure><img alt="'+html.escape(figure['name'])+'" src="data:image/png;base64,'+
            encoded+'"><figcaption>'+html.escape(figure['caption'])+
            ' <a href="'+html.escape(pdf)+'" download>Vector PDF</a>.</figcaption></figure>')
    scalar=[r for r in numerical if r['domain'] in DOMAINS and 'trajectory_null' not in r['model'] and
            r['cohort'] in ('all360','paired_primary')]
    parts+=['<h2>Numerical records</h2>',
        '<p>Every numerical table remains available in long form. The bundle includes per-target and per-genotype metrics, '
        'explicit fold keys and null donor mappings, scalar OOF records, primary native OOF arrays and source flags, eligibility distributions and the figure exports. '
        'Rows from different cohorts are labelled and must not be treated as direct modality comparisons.</p>',
        '<p><a href="'+bundle_path.name+'" download>Download the complete report data and figure bundle</a>. '
        '<a href="report_metrics_long.csv" download>Download long-form model metrics</a>.</p>',
        '<details><summary>Scalar assay metric rows from completed all-360 and primary-cohort experiments</summary>'+
        table(scalar,('cohort','domain','split','model','n','rmse','mean_target_r2'))+'</details>',
        '<h2>Sources and limits</h2><p>The executed source snapshots and input hashes accompany each stage. '
        'UVIS source warnings are retained in the main fit and assessed by a separately labelled flagged-record sensitivity. '
        'The frozen encoder uses only pinned local DINOv2-small weights, accepted fixed quarter views and explicit black masked crops; NoIR lies outside normal RGB pretraining. '
        'Physical pot continuity, camera radiometric calibration and independent anatomical truth remain unresolved evidence requirements.</p>',
        '<p class="note">The report is self-contained for reading: figures are embedded. Download links resolve within the accompanying output folder or extracted bundle.</p></body></html>']
    report=root/'prediction_report.html'
    report.write_text('\n'.join(parts),encoding='utf-8')
    # Include the final self-contained page after its data bundle has been built.
    with zipfile.ZipFile(bundle_path,'a',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report,report.name)
    manifest=dict(status='completed',created_utc=now,completed_experiments=len(data),
        figures=len(records),phase_status=phase_rows,source_sha256=p.digest(__file__),
        input_completion_markers={name:p.digest(value['path']/'complete.json') for name,value in data.items()},
        outputs={'html':str(report),'bundle':str(bundle_path)},bundle_members=len(members)+1,
        figure_sources=[dict(name=f['name'],png_sha256=p.digest(f['png']),pdf_sha256=p.digest(f['pdf']),
                             caption=f['caption']) for f in records])
    p.save_json(root/'prediction_report_manifest.json',manifest)
    registry=[]
    for figure in records:
        main=figure['name']=='incremental_paired_contrasts'
        registry.append(dict(section='main_prediction_controls' if main else 'supporting_prediction',
            role='proposed_main' if main else 'supporting',priority=1 if main else 2,
            path=str(figure['png']),pdf_path=str(figure['pdf']),caption=figure['caption'],
            source_data='incremental_contrasts.csv' if main else 'report_metrics_long.csv'))
    p.save_json(root/'report_figure_registry_prediction.json',dict(status='completed',
        proposed_main='incremental_paired_contrasts',figures=registry,
        rationale='Paired improvements show the incremental test and six-genotype uncertainty directly; '
                  'the matched DN comparison separates a representation gain from a new-measurement claim.'))
    print(json.dumps({k:manifest[k] for k in ('status','completed_experiments','figures','phase_status','outputs')},indent=2))


if __name__=='__main__':
    main()
