"""Contract, diagnostics and true-source overlays for completed camera runs."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import camera_geometry as g
from camera_geometry_run import atomic,sha,digest,csv_out,GRID

def load_records(folder):
    config=json.loads((folder/'configuration.json').read_text(encoding='utf-8'))
    records=[]
    for p in sorted((folder/'records').glob('*.json')):
        record=json.loads(p.read_text(encoding='utf-8')); expected=record.pop('payload_sha256')
        if digest(record)!=expected: raise ValueError('record_payload_mismatch:'+str(p))
        if record['signature']!=config['signature']: raise ValueError('record_configuration_mismatch:'+str(p))
        records.append(record)
    if {r['identity']['session'] for r in records}!=set(config['cohort']):
        raise ValueError('completed_record_cohort_mismatch')
    return config,records

def contract(record):
    fit=record['fit']; selected=fit['selected']; nominal=fit['nominal']
    status=g.deployment_decision(fit)
    gate=fit['phase_gate']
    phase_status=('not_fitted' if selected is None else
                  'alternate_training_accepted' if gate['accepted_alternate'] else
                  'ambiguous_nominal_retained' if gate.get('runner_margin',0)<.015 else
                  'nominal_retained')
    row={**record['identity'],**status,'fit_status':fit['status'],'phase_status':phase_status,
         'selected_sign':selected['sign'] if selected else None,
         'selected_phase':selected['phase'] if selected else None,
         'phase_resolution_indices':4,'nominal_sign':1,'nominal_phase':0,
         'matrix_rgb_to_noir':selected['matrix'] if selected else None,
         'matrix_noir_to_rgb':cv2.invertAffineTransform(np.array(selected['matrix'])).tolist() if selected else None,
         'nominal_matrix_rgb_to_noir':nominal['matrix'] if nominal else None,
         'nominal_moment_matrix_rgb_to_noir':nominal['moment_matrix'] if nominal else None,
         'source_grid_width':GRID[0],'source_grid_height':GRID[1],
         'target_grid_width':GRID[0],'target_grid_height':GRID[1],
         'training_indices':selected['train_indices'] if selected else [],
         'n_training_pairs':selected['n_train'] if selected else 0,
         'training_median_iou':selected['median_iou'] if selected else None,
         'training_only_selection':True,'physical_correspondence_verified':False}
    for method,prefix in [('selected_shared','selected'),('nominal_shared','nominal'),
                          ('nominal_per_view_ecc','adaptive_ecc')]:
        rr=[r for r in record['evaluation'] if r['method']==method and r['status']=='executed']
        row[prefix+'_heldout_n']=len(rr)
        row[prefix+'_heldout_median_iou']=float(np.median([r['iou'] for r in rr])) if rr else None
    return row

def figure_diagnostics(folder,records):
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.3),layout='constrained')
    methods=['nominal_fixed_moment','nominal_shared','nominal_per_view_ecc','mismatch_view_fixed','mismatch_other_fixed']
    vals=[]
    for method in methods:
        values=[]
        for r in records:
            ok=[e['iou'] for e in r['evaluation'] if e['method']==method and e['status']=='executed']
            if ok: values.append(np.median(ok))
        vals.append(values)
    labels=['Fixed\nmoment','Shared\naffine','Adaptive\nview ECC','Index+20\nfixed','Other replicate\nfixed']
    axs[0].boxplot(vals,tick_labels=labels,showfliers=False)
    axs[0].set(ylabel='Held-out session median silhouette IoU',ylim=(0,1))
    axs[0].tick_params(axis='x',labelsize=8)
    for n,v in enumerate(vals,1): axs[0].text(n,.98,f'n={len(v)}',ha='center',va='top',fontsize=8)
    train=[]; test=[]; colors=[]
    for r in records:
        if not r['fit']['selected']: continue
        scores=[e['iou'] for e in r['evaluation'] if e['method']=='selected_shared' and e['status']=='executed']
        if scores:
            train.append(r['fit']['selected']['median_iou']);test.append(np.median(scores))
            colors.append('#468252' if not g.deployment_decision(r['fit'])['deployment_abstain'] else '#c07845')
    axs[1].scatter(train,test,c=colors,s=18,alpha=.7)
    axs[1].plot([0,1],[0,1],color='.6',lw=.8);axs[1].axvline(.65,color='.5',ls=':',lw=1)
    axs[1].set(xlabel='Training median IoU',ylabel='Held-out median IoU',xlim=(0,1),ylim=(0,1))
    phase=Counter(contract(r)['phase_status'] for r in records)
    axs[2].bar(range(len(phase)),list(phase.values()),color='#476d87')
    axs[2].set_xticks(range(len(phase)),[x.replace('_','\n') for x in phase],fontsize=8)
    axs[2].set(ylabel='Acquisitions',title='Training-selected mapping status')
    for i,n in enumerate(phase.values()): axs[2].text(i,n,str(n),ha='center',va='bottom')
    path=folder/'figures/heldout_camera_diagnostics.png';path.parent.mkdir(exist_ok=True)
    fig.savefig(path,dpi=170,facecolor='white');plt.close(fig)
    return path

def source_example(folder,record,intake):
    ok=[e for e in record['evaluation'] if e['method']=='selected_shared' and e['status']=='executed']
    if not ok:return None
    med=np.median([e['iou'] for e in ok]); chosen=min(ok,key=lambda e:(abs(e['iou']-med),e['rgb_view']))
    indexed={(r['camera'],r['view']):r for r in record['inputs']}
    arrays=[]; masks=[]; sources=[]
    for camera,index in [('rgb',chosen['rgb_view']),('noir',chosen['noir_view'])]:
        inp=indexed[camera,index];path=intake/inp['source']
        actual=sha(path)
        if actual!=inp['source_sha256']:raise ValueError('source_hash_mismatch:'+str(path))
        with Image.open(path) as im:
            im.load(); arrays.append(np.array(im.convert('RGB').resize(GRID,Image.Resampling.LANCZOS)))
        # Recover masks from the source-bound extraction path, which is absolute
        # relative to the upstream run declared in the completed configuration.
        masks.append(None);sources.append(dict(source=inp['source'],actual_source_sha256=actual))
    # Reload actual extraction masks via record's root-independent declared path.
    return chosen,arrays,indexed,sources

def export(folder,intake,full_run):
    config,records=load_records(folder)
    rows=[contract(r) for r in records]
    csv_out(folder/'session_camera_lookup.csv',rows)
    atomic(folder/'session_camera_lookup.json',dict(schema=1,status='completed',
        configuration_signature=config['signature'],matrix_direction='RGB pixel to NoIR pixel',
        warp_noir_to_rgb='Use matrix_noir_to_rgb with regular warpAffine, or matrix_rgb_to_noir with WARP_INVERSE_MAP',
        coordinate_grid=GRID,nominal_baseline_retained=True,
        abstention='Training-only quality gate; matrix remains available as a labelled hypothesis when abstaining',
        rows=rows))
    figure=figure_diagnostics(folder,records)
    examples=[]
    for treatment in ('C','P','S'):
        for visit in ('1','2','3','4','5'):
            pool=[r for r in records if r['identity']['treatment']==treatment and r['identity']['measurement']==visit and
                  r['fit']['selected'] and any(e['method']=='selected_shared' and e['status']=='executed' for e in r['evaluation'])]
            if not pool:continue
            # Deterministic source choice independent of assay values. Do not
            # choose a visually favourable acquisition using withheld outcomes.
            record=sorted(pool,key=lambda r:r['identity']['session'])[0]
            result=source_example(folder,record,intake)
            if not result:continue
            selected,arrays,indexed,sources=result
            a,b=arrays;m=np.array(selected['matrix'],np.float32)
            actual_masks=[]
            for camera,index in [('rgb',selected['rgb_view']),('noir',selected['noir_view'])]:
                row=indexed[camera,index];maskpath=full_run/row['mask']
                if sha(maskpath)!=row['actual_mask_sha256']:raise ValueError('illustration_mask_hash_mismatch')
                with Image.open(maskpath) as im:actual_masks.append(np.array(im.convert('L').resize(GRID,Image.Resampling.NEAREST))>0)
            am,bm=actual_masks
            a[~am]=0;b[~bm]=0
            wb=cv2.warpAffine(b,m,GRID,flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
            wn=g.warp_target(bm,m,am.shape)
            overlay=np.zeros_like(a);overlay[...,0]=am*240;overlay[...,1]=wn*215;overlay[...,2]=(am&wn)*220
            fig,axs=plt.subplots(1,4,figsize=(12,3.2),layout='constrained')
            for ax,im,title in zip(axs,[a,b,wb,overlay],['RGB foreground','NoIR foreground','Shared affine: NoIR on RGB','RGB / NoIR mask overlap']):
                ax.imshow(im);ax.set_title(title,fontsize=9);ax.axis('off')
            fig.suptitle(f"{treatment} visit {visit}; test RGB {selected['rgb_view']}, NoIR {selected['noir_view']}; IoU {selected['iou']:.3f}; silhouette surrogate",fontsize=10)
            path=folder/f'figures/source_overlap_{treatment}{visit}.png'
            fig.savefig(path,dpi=150,facecolor='white');plt.close(fig)
            examples.append(dict(identity=record['identity'],rgb_view=selected['rgb_view'],noir_view=selected['noir_view'],
                file=path.relative_to(folder).as_posix(),sha256=sha(path),sources=sources,
                selection='lexicographic acquisition within stratum; test view closest to that acquisition median IoU'))
    sourcepath=Path(__file__)
    sourcecache=folder/'postprocessing_source'/sha(sourcepath)
    sourcecache.mkdir(parents=True,exist_ok=True)
    (sourcecache/sourcepath.name).write_bytes(sourcepath.read_bytes())
    atomic(folder/'export_manifest.json',dict(status='completed',schema=1,
        configuration_signature=config['signature'],export_source_sha256=sha(sourcepath),
        current_core_sha256=sha(Path(g.__file__)),source_cached_in_configuration=config['source_hashes'].get(sourcepath.name),
        session_rows=len(rows),deployment_status=dict(Counter(r['deployment_status'] for r in rows)),
        phase_status=dict(Counter(r['phase_status'] for r in rows)),
        diagnostics=dict(file=figure.relative_to(folder).as_posix(),sha256=sha(figure)),examples=examples,
        outputs={p.name:sha(p) for p in [folder/'session_camera_lookup.json',folder/'session_camera_lookup.csv']}))
    print(json.dumps(dict(status='completed',session_rows=len(rows),examples=len(examples),
          deployment_status=dict(Counter(r['deployment_status'] for r in rows)))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--intake',type=Path,required=True);parser.add_argument('--full-run',type=Path,required=True)
    args=parser.parse_args();export(args.output,args.intake,args.full_run)
