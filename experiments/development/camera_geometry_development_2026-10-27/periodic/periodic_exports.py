"""Compare shared and angle-conditioned surrogates with true-source examples."""
import sys
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
import argparse
import json
from collections import Counter
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import camera_geometry as g
import periodic_camera as p
from camera_geometry_run import atomic,sha,digest,csv_out,GRID

def load(folder):
    config=json.loads((folder/'configuration.json').read_text(encoding='utf-8'))
    records=[]
    for path in sorted((folder/'records').glob('*.json')):
        r=json.loads(path.read_text(encoding='utf-8'));expected=r.pop('payload_sha256')
        if digest(r)!=expected or r['signature']!=config['signature']:raise ValueError('periodic_record_binding')
        records.append(r)
    if {r['identity']['session'] for r in records}!=set(config['cohort']):raise ValueError('incomplete_periodic_cohort')
    return config,records

def diagnostics(folder,records):
    methods=['phase_shared','periodic_candidate','selected_causal','adaptive_selected_pair_ecc',
             'mismatch_view_periodic','mismatch_other_periodic']
    labels=['Shared','Periodic\ncandidate','Training-CV\nselected','Adaptive\nview ECC','Index+20','Other\nreplicate']
    values=[]
    for method in methods:
        scores=[]
        for record in records:
            ok=[e['iou'] for e in record['evaluation'] if e['method']==method and e['status']=='executed']
            if ok:scores.append(np.median(ok))
        values.append(scores)
    fig,axs=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    axs[0].boxplot(values,tick_labels=labels,showfliers=False)
    axs[0].set(ylabel='Held-out session median silhouette IoU',ylim=(0,1))
    axs[0].tick_params(axis='x',labelsize=7)
    for i,v in enumerate(values,1):axs[0].text(i,.98,f'n={len(v)}',ha='center',va='top',fontsize=7)
    xx=[];yy=[];colors=[]
    for r in records:
        gain=r['fit']['selection'].get('inner_mean_delta')
        if gain is None:continue
        aa={e['rgb_view']:e['iou'] for e in r['evaluation'] if e['method']=='periodic_candidate' and e['status']=='executed'}
        bb={e['rgb_view']:e['iou'] for e in r['evaluation'] if e['method']=='phase_shared' and e['status']=='executed'}
        common=set(aa)&set(bb)
        if not common:continue
        xx.append(gain);yy.append(np.median([aa[i]-bb[i] for i in common]))
        colors.append('#3c865b' if r['fit']['selection']['accepted'] else '#b87845')
    axs[1].scatter(xx,yy,c=colors,s=17,alpha=.65)
    axs[1].axhline(0,color='.6',lw=.8);axs[1].axvline(.02,color='.6',ls=':',lw=.8)
    axs[1].set(xlabel='Inner training-angle CV gain',ylabel='Held-out periodic minus shared IoU')
    statuses=Counter('periodic accepted' if r['fit']['selection']['accepted'] else
                     'shared accepted' if not g.deployment_decision(r['shared_fit'])['deployment_abstain'] else
                     'deployment abstains' for r in records)
    status_colors={'periodic accepted':'#3c865b','shared accepted':'#426e8a','deployment abstains':'#b87845'}
    axs[2].bar(range(len(statuses)),list(statuses.values()),color=[status_colors[s] for s in statuses])
    axs[2].set_xticks(range(len(statuses)),[s.replace(' ','\n') for s in statuses],fontsize=8)
    axs[2].set(ylabel='Acquisitions',title='Training-only deployment decision')
    for i,n in enumerate(statuses.values()):axs[2].text(i,n,str(n),ha='center',va='bottom')
    path=folder/'figures/periodic_heldout_diagnostics.png';path.parent.mkdir(exist_ok=True)
    fig.savefig(path,dpi=170,facecolor='white');plt.close(fig);return path

def examples(folder,records,intake,full_run):
    outputs=[]
    for treatment in ('C','P','S'):
        for visit in ('1','2','3','4','5'):
            pool=sorted([r for r in records if r['identity']['treatment']==treatment and r['identity']['measurement']==visit
                         and r['fit']['model']],key=lambda r:r['identity']['session'])
            if not pool:continue
            chosen=None
            for record in pool:
                ok=[e for e in record['evaluation'] if e['method']=='periodic_candidate' and e['status']=='executed']
                if ok:chosen=(record,min(ok,key=lambda e:e['rgb_view']));break
            if not chosen:continue
            record,row=chosen;view=row['rgb_view'];target=row['noir_view']
            indexed={(x['camera'],x['view']):x for x in record['inputs']}
            arrays=[];masks=[];sources=[]
            for camera,index in [('rgb',view),('noir',target)]:
                inp=indexed[camera,index];source=intake/inp['source'];mask=full_run/inp['mask']
                actual=sha(source)
                if actual!=inp['source_sha256'] or sha(mask)!=inp['actual_mask_sha256']:
                    raise ValueError('source_or_mask_example_hash_mismatch')
                with Image.open(source) as im:
                    im.load();arrays.append(np.array(im.convert('RGB').resize(GRID,Image.Resampling.LANCZOS)))
                with Image.open(mask) as im:masks.append(np.array(im.convert('L').resize(GRID,Image.Resampling.NEAREST))>0)
                sources.append(dict(source=inp['source'],source_sha256=actual,mask_sha256=inp['actual_mask_sha256']))
            a,b=arrays;am,bm=masks;a[~am]=0;b[~bm]=0
            shared=np.array(record['shared_fit']['selected']['matrix'],np.float32)
            periodic=np.array(row['matrix'],np.float32)
            sw=cv2.warpAffine(b,shared,GRID,flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
            pw=cv2.warpAffine(b,periodic,GRID,flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
            overlays=[]
            for matrix in (shared,periodic):
                wn=g.warp_target(bm,matrix,am.shape)
                overlay=np.zeros_like(a);overlay[...,0]=am*240;overlay[...,1]=wn*215;overlay[...,2]=(am&wn)*220
                overlays.append(overlay)
            fig,axs=plt.subplots(2,3,figsize=(10.5,6.2),layout='constrained')
            titles=['RGB foreground','NoIR foreground','Shared affine: NoIR on RGB',
                    'Periodic: NoIR on RGB','Shared mask overlap','Periodic mask overlap']
            for ax,image,title in zip(axs.flat,[a,b,sw,pw,*overlays],titles):
                ax.imshow(image);ax.set_title(title,fontsize=9);ax.axis('off')
            shared_iou=g.iou_warp(am,bm,shared)
            status='periodic accepted' if record['fit']['selection']['accepted'] else 'periodic not selected'
            fig.suptitle(f"{treatment} visit {visit}; withheld index {view}; shared IoU {shared_iou:.3f}, periodic {row['iou']:.3f}; {status}",fontsize=10)
            path=folder/f'figures/source_overlap_{treatment}{visit}.png'
            fig.savefig(path,dpi=150,facecolor='white');plt.close(fig)
            outputs.append(dict(identity=record['identity'],rgb_view=view,noir_view=target,
                shared_iou=shared_iou,periodic_iou=row['iou'],periodic_accepted=record['fit']['selection']['accepted'],
                file=path.relative_to(folder).as_posix(),sha256=sha(path),sources=sources,
                selection='lexicographic acquisition and first available withheld index; no outcome-based case selection'))
    return outputs

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--intake',type=Path,required=True);parser.add_argument('--full-run',type=Path,required=True)
    args=parser.parse_args();config,records=load(args.output)
    enriched=[]
    for record in records:
        for row in record['evaluation']:
            enriched.append({**row,'noir_session':record['other_session'] if row['method'].startswith('mismatch_other') else row['session']})
    csv_out(args.output/'heldout_metrics_with_pair_identity.csv',enriched)
    figure=diagnostics(args.output,records);illustrations=examples(args.output,records,args.intake,args.full_run)
    source=Path(__file__);snapshot=args.output/'postprocessing_source'/sha(source);snapshot.mkdir(parents=True,exist_ok=True)
    (snapshot/source.name).write_bytes(source.read_bytes())
    atomic(args.output/'export_manifest.json',dict(status='completed',configuration_signature=config['signature'],
        export_source_sha256=sha(source),session_rows=len(records),examples=illustrations,
        enriched_metrics_sha256=sha(args.output/'heldout_metrics_with_pair_identity.csv'),
        diagnostics=dict(file=figure.relative_to(args.output).as_posix(),sha256=sha(figure))))
    print(json.dumps(dict(status='completed',examples=len(illustrations))),flush=True)
