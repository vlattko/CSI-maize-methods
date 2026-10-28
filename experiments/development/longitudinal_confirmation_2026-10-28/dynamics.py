"""Descriptive same-pot dynamics on the experimenter-declared DAS axis."""
import argparse,base64,csv,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from registry import sha,FIELDS


METRICS=[('ojip__Fv/Fm','Measured Fv/Fm'),('ojip__Pi_Abs','Measured PIABS'),('nir__NDVI','Instrument NIR NDVI'),
         ('rgb__area_fraction__median','RGB projected area / image area'),('noir__area_fraction__median','NoIR projected area / image area'),
         ('rgb__height_fraction__max','Maximum visible RGB height / image height')]
COLORS={'C':'#27744f','P':'#b37a22','S':'#724d91'}


def interval_rates(days,values):
    days=np.asarray(days,float);values=np.asarray(values,float)
    if np.any(np.diff(days)<=0):raise ValueError('Time must increase')
    return np.diff(values)/np.diff(days)


def first_observation_change(values,ratio):
    values=np.asarray(values,float)
    if not np.isfinite(values[0]) or (ratio and values[0]==0):return np.full(values.shape,np.nan)
    return values/values[0] if ratio else values-values[0]


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def key(row):return tuple(str(row[k]) for k in FIELDS)
def number(value):return float(value) if value not in [None,''] else np.nan
def finite(value):return float(value) if np.isfinite(value) else None
def write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def bands(matrix,genotypes,seed=20261028):
    rng=np.random.default_rng(seed);groups=[[i for i,g in enumerate(genotypes) if g==name] for name in sorted(set(genotypes))]
    draws=np.array([np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(2000)])
    with np.errstate(invalid='ignore'):
        values=matrix[draws];counts=np.isfinite(values).sum(axis=1);means=np.divide(np.nansum(values,axis=1),counts,out=np.full(counts.shape,np.nan),where=counts>0)
    return np.nanpercentile(means,[2.5,97.5],axis=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['registry','reference','traits','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--schedule',type=Path,help='Read-only timetable reconciliation overlay')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);registry=json.loads(a.registry.read_text());ids=registry['records'];reference={key(r):r for r in read(a.reference)};traits={key(r):r for r in read(a.traits)}
    if len(reference)!=360 or len(traits)!=360 or {key(r) for r in ids}!=set(reference) or set(traits)!=set(reference):raise ValueError('Registry/reference/trait identity mismatch')
    observations=[];by_pot={}
    for identity in ids:
        k=key(identity);r={**reference[k],**traits[k]};row={**identity}
        for camera in ['rgb','noir']:row[camera+'_eligible']=r[camera+'_eligible']=='True';row[camera+'_accepted_views']=int(r[camera+'_usable_views'])
        for metric,_ in METRICS:
            camera=metric.split('__')[0];row[metric]=finite(number(r[metric])) if camera not in ['rgb','noir'] or row[camera+'_eligible'] else None
        observations.append(row);by_pot.setdefault(row['plant_uid'],[]).append(row)
    rates=[];summaries=[];relative_summaries=[];relative_observations=[];days=[registry['confirmation']['declared_das_by_visit'][str(v)] for v in range(1,6)]
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(12,7));relative_fig,relative_axes=plt.subplots(2,3,figsize=(12,7))
    for ax,rax,(metric,label) in zip(axes.flat,relative_axes.flat,METRICS):
        for treatment,color in COLORS.items():
            pots=[sorted(rr,key=lambda r:int(r['measurement'])) for rr in by_pot.values() if rr[0]['treatment']==treatment]
            if len(pots)!=24 or any(len(rr)!=5 for rr in pots):raise ValueError('Every treatment must retain24 physical pots and five reference visits')
            matrix=np.array([[number(r[metric]) for r in rr] for rr in pots]);lo,hi=bands(matrix,[rr[0]['genotype'] for rr in pots]);mean=np.nanmean(matrix,axis=0)
            ax.fill_between(days,lo,hi,color=color,alpha=.13);ax.plot(days,mean,'o-',color=color,label=registry['confirmation']['treatment_labels'][treatment],ms=4)
            ratio=metric.split('__')[0] in ['rgb','noir'];relative=np.array([first_observation_change(v,ratio) for v in matrix]);rlo,rhi=bands(relative,[rr[0]['genotype'] for rr in pots]);rmean=np.nanmean(relative,axis=0)
            rax.fill_between(days,rlo,rhi,color=color,alpha=.13);rax.plot(days,rmean,'o-',color=color,label=registry['confirmation']['treatment_labels'][treatment],ms=4)
            for j,day in enumerate(days):relative_summaries.append(dict(metric=metric,treatment=treatment,declared_das=day,normalization='ratio_to_13_DAS' if ratio else 'difference_from_13_DAS',n_pots=int(np.isfinite(relative[:,j]).sum()),mean=finite(rmean[j]),bootstrap_low=finite(rlo[j]),bootstrap_high=finite(rhi[j])))
            for i,rr in enumerate(pots):
                for j,day in enumerate(days):relative_observations.append(dict(plant_uid=rr[0]['plant_uid'],treatment=treatment,genotype=rr[0]['genotype'],replicate=rr[0]['replicate'],metric=metric,declared_das=day,normalization='ratio_to_13_DAS' if ratio else 'difference_from_13_DAS',value=finite(relative[i,j])))
            for j,day in enumerate(days):summaries.append(dict(metric=metric,treatment=treatment,declared_das=day,n_pots=int(np.isfinite(matrix[:,j]).sum()),mean=finite(mean[j]),bootstrap_low=finite(lo[j]),bootstrap_high=finite(hi[j])))
            for rr in pots:
                for j,rate in enumerate(interval_rates(days,[number(r[metric]) for r in rr])):
                    rates.append(dict(plant_uid=rr[0]['plant_uid'],treatment=treatment,genotype=rr[0]['genotype'],replicate=rr[0]['replicate'],metric=metric,earlier_das=days[j],later_das=days[j+1],elapsed_days=days[j+1]-days[j],rate_per_declared_day=finite(rate)))
        if a.schedule:
            for panel in [ax,rax]:panel.axvspan(14,19,color='#746395',alpha=.06,zorder=-1)
        ax.set_title(label);ax.set_xticks(days);ax.set_xlabel('Declared days after sowing');ax.grid(alpha=.15)
        rax.set_title(label+('\nratio to first observation' if metric.split('__')[0] in ['rgb','noir'] else '\ndifference from first observation'));rax.set_xticks(days);rax.set_xlabel('Folder-declared days after sowing');rax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=9);fig.tight_layout();fig.savefig(a.output/'same_pot_dynamics.png',dpi=190);fig.savefig(a.output/'same_pot_dynamics.pdf');plt.close(fig)
    relative_axes.flat[0].legend(fontsize=9);relative_fig.tight_layout();relative_fig.savefig(a.output/'first_observation_change.png',dpi=190);relative_fig.savefig(a.output/'first_observation_change.pdf');plt.close(relative_fig)
    write(a.output/'confirmed_pot_observations.csv',observations);write(a.output/'adjacent_visit_rates.csv',rates);write(a.output/'treatment_time_summary.csv',summaries)
    write(a.output/'first_observation_summary.csv',relative_summaries);write(a.output/'first_observation_values.csv',relative_observations)
    payload=json.dumps(dict(rows=observations,metrics=METRICS,days=days),allow_nan=False).replace('<','\\u003c')
    picture=base64.b64encode((a.output/'same_pot_dynamics.png').read_bytes()).decode();confirmation=registry['confirmation']
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Confirmed same-pot dynamics</title><style>body{font:19px/1.5 "Times New Roman",Times,serif;background:white;color:#171717;max-width:1200px;margin:35px auto;padding:0 24px;overflow-wrap:anywhere}h1,h2{font-weight:normal}img,svg{width:100%;height:auto}select{font:inherit;padding:6px}pre{white-space:pre-wrap;font-size:14px}a{color:#275d50}</style><h1>Same-pot trajectories across five imaging visits</h1><p>The experimenter confirmed that matching treatment, genotype and replicate labels identify the same physical pots. The registry contains72 pots and360 visit records. The declared ages are13,19,20,22 and25 DAS. Leaf identities and geometric associations remain hypotheses.</p><p>Control:25/18°C day/night. Stress: abrupt cooling to10/5°C for six days without adaptation. Priming: adaptation starting at12/8°C with a daily ramp towards cold. Photoperiod:16h light and8h dark. Exact ramp increments and alignment of treatment transitions to imaging remain pending the complete method. We do not mark13–19DAS as a confirmed exposure window.</p><img alt="Treatment trajectories with whole-pot bootstrap bands" src="data:image/png;base64,'''+picture+'''"><p>Means retain every available qualified observation; the CSV gives the denominator at each age. Shading is an exploratory95% within-genotype whole-pot bootstrap interval, based on2,000 resamples. The same resampled pots are retained across all five ages. No correction for informative missingness or causal treatment assignment is implied.</p><h2>Inspect a physical pot</h2><label>Pot <select id="pot"></select></label> <label>Measurement <select id="metric"></select></label><div id="chart"></div><p id="caption"></p><h2>Interpretation</h2><p>The uneven day intervals are preserved in every plot and rate calculation. Missing adjacent observations produce missing rates; they are not bridged. Image fractions describe visible projected support on the recorded camera canvas. Changes can reflect plant development, pose, occlusion, truncation and camera geometry; they are not calibrated biomass or leaf-area growth rates. Instrument measurements belong to the pot-labelled acquisition, not to a selected reconstructed leaf.</p><details><summary>Experimenter confirmation and protocol record</summary><pre>'''+html.escape(json.dumps(confirmation,indent=2))+'''</pre></details><p>Data and reference outputs will be made available through Zenodo after publication.</p><script id="payload" type="application/json">'''+payload+'''</script><script>const D=JSON.parse(document.getElementById('payload').textContent);document.getElementById('payload').remove();const $=id=>document.getElementById(id);const pots=[...new Set(D.rows.map(r=>r.plant_uid))];for(const uid of pots){const r=D.rows.find(q=>q.plant_uid===uid),o=document.createElement('option');o.value=uid;o.textContent=r.treatment+' / '+r.genotype+' / replicate '+r.replicate;$('pot').append(o)}for(const[k,label]of D.metrics){const o=document.createElement('option');o.value=k;o.textContent=label;$('metric').append(o)}function render(){const rows=D.rows.filter(r=>r.plant_uid===$('pot').value).sort((a,b)=>a.declared_das-b.declared_das),field=$('metric').value,values=rows.map(r=>r[field]).filter(v=>v!==null);let min=Math.min(...values),max=Math.max(...values);if(!values.length){$('chart').textContent='No qualified observations';return}const pad=Math.max((max-min)*.15,.002);min-=pad;max+=pad;const x=d=>70+(d-13)/12*750,y=v=>330-(v-min)/(max-min)*270;let s='<svg viewBox="0 0 900 390" role="img" aria-label="Same-pot measurement over declared DAS"><path d="M70 40V330H850" fill="none" stroke="#555"/>';for(const day of D.days)s+='<text x="'+x(day)+'" y="360" text-anchor="middle">'+day+' DAS</text>';let segment=[];for(const r of rows){const v=r[field];if(v===null){if(segment.length)s+='<polyline points="'+segment.join(' ')+'" fill="none" stroke="#27684f" stroke-width="3"/>';segment=[];continue}segment.push(x(r.declared_das)+','+y(v));s+='<circle cx="'+x(r.declared_das)+'" cy="'+y(v)+'" r="5" fill="#27684f"/><text x="'+(x(r.declared_das)+8)+'" y="'+(y(v)-8)+'">'+v.toFixed(4)+'</text>'}if(segment.length)s+='<polyline points="'+segment.join(' ')+'" fill="none" stroke="#27684f" stroke-width="3"/>';$('chart').innerHTML=s+'</svg>';$('caption').textContent='Confirmed physical pot: '+rows[0].plant_uid+'; no anatomical leaf identity is inferred.'}$('pot').onchange=render;$('metric').onchange=render;render();</script></html>'''
    relative_picture=base64.b64encode((a.output/'first_observation_change.png').read_bytes()).decode()
    old_protocol=page[page.index('<p>Control:25'):page.index('<img alt="Treatment trajectories')]
    protocol='<p>The protocol record below preserves the supplied treatment information. Exact exposure timing requires a timetable overlay.</p>'
    if a.schedule:
        schedule=json.loads(a.schedule.read_text())
        protocol='<p>The supplied timetable places priming at 6–11 DAS, followed by severe cold at 14–19 DAS in both P and S. At 19 DAS, measurements precede restoration of 25 °C. Recovery images are scheduled at 20, 22 and 25 DAS. Purple shading marks the scheduled severe challenge; control remains at 25/18 °C throughout. The photoperiod is 16 h day and 8 h night. The first image is pre-challenge but follows priming for P.</p><p>Priming uses increasing cold-exposure durations, with 12 °C daytime cooling. The night setting remains unresolved: the cells and experimenter specify 8 °C; the workbook footnote says 5 °C. Treatments occupy different scheduled calendar blocks, so treatment and run may be confounded. Exact execution times, exposure hours and LAB tissue removal remain unresolved. Scheduled dates do not replace camera timestamps.</p>'
    page=page.replace(old_protocol,protocol)
    page=page.replace('<h2>Inspect a physical pot</h2>','<h2>Change relative to each pot\'s first observation</h2><p>Image measurements are divided by the same pot\'s13DAS value; instrument measurements subtract it. A missing first observation stays missing and is never replaced by a later visit. The first visit is a reference measurement, not a confirmed untreated baseline. These comparisons address initial size differences without establishing a causal priming effect.</p><img alt="First-observation relative same-pot changes" src="data:image/png;base64,'+relative_picture+'"><h2>Inspect a physical pot</h2>')
    # Keep prose readable without changing compact machine field names.
    for old,new in [('contains72','contains 72'),('and360','and 360'),('are13,19,20,22 and25','are 13, 19, 20, 22 and 25'),('Control:25','Control: 25'),('to10/5','to 10/5'),('at12/8','at 12/8'),('Photoperiod:16h','Photoperiod: 16 h'),('and8h','and 8 h'),('mark13–19DAS','mark 13–19 DAS'),('exploratory95%','exploratory 95%'),('on2,000','on 2,000'),("pot's13DAS","pot's 13 DAS")]:page=page.replace(old,new)
    (a.output/'same_pot_dynamics.html').write_text(page,encoding='utf-8')
    manifest=dict(status='completed',physical_pots=len(by_pot),reference_visits=len(observations),rate_rows=len(rates),metrics=len(METRICS),bootstrap_unit='whole physical pot within genotype and treatment',inputs={str(q):sha(q) for q in [a.registry,a.reference,a.traits]},source_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in a.output.iterdir() if p.is_file() and p.name not in ['dynamics_manifest.json','workbench.html','workbench_build.log']})
    if a.schedule:manifest['inputs'][str(a.schedule)]=sha(a.schedule)
    (a.output/'dynamics_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({k:manifest[k] for k in ['status','physical_pots','reference_visits','rate_rows','metrics']},indent=2))


if __name__=='__main__':main()
