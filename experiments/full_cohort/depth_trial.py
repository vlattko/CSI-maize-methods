"""DA3-SMALL: held-out-view projection with learned and nominal turntable poses.

Three RGB or recorded-NoIR quarter views enter the network. Index30 is never
an input. Prespecified f/W and rotation-sign hypotheses remain separate results.
This is a mask-consistency diagnostic under uncalibrated cameras, not true 3D.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('intake','run','assets-base'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resolution',type=int,default=336);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();out=args.run/'depth_trial';out.mkdir(parents=True,exist_ok=True)
    base=args.assets_base.resolve();vendor=base/'vendor/Depth-Anything-3'
    sys.path[:0]=[str(base/'da3_deps'),str(args.run/'da3_runtime'),str(vendor/'src')]
    sys.path.extend([str(base.parent/'optional_deps'),str(base.parent/'report_deps')])
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import cv2
    import torch
    from PIL import Image
    from depth_anything_3.api import DepthAnything3
    from depth_geometry import turntable,unproject,project,similarity
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12,args.device)
    modelroot=base/'models/DA3-SMALL'
    config=dict(model='DA3-SMALL',assets={p.name:sha(p) for p in modelroot.iterdir() if p.is_file()},
                vendor_git=subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip(),
                vendor_sources={p.relative_to(vendor).as_posix():sha(p) for p in (vendor/'src').rglob('*.py')},
                script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('depth_geometry.py')),
                numpy=np.__version__,torch=torch.__version__,device=args.device,resolution=args.resolution,
                training_indices=[0,10,20],heldout_index=30,nominal_distance_gauge=4.,
                foreground_mask_source='intake_box_diagnostic',calibration=False,
                declared_modes=['learned_pose_aligned_to_training_centres',
                                'nominal_f1.2_sign1','nominal_f1.2_sign-1','nominal_f2.0_sign1','nominal_f2.0_sign-1'])
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();config['signature']=signature
    cp=out/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Changed depth-trial source/config')
    cp.write_text(json.dumps(config,indent=2),encoding='utf-8')
    (out/'execution_source.py').write_bytes(Path(__file__).read_bytes())
    model=DepthAnything3.from_pretrained(str(modelroot),local_files_only=True).to(args.device).eval()
    diagnostic=read(args.intake/'segmentation_triage/view_diagnostics.csv')
    records={(r['treatment'],r['measurement'],r['genotype'],r['replicate'],r['camera'],int(r['view'])):r for r in diagnostic}
    results=[];started=time.monotonic()
    for treatment in 'CPS':
        for visit in range(1,6):
            for camera in ('rgb','noir'):
                lookup=lambda view:records.get((treatment,str(visit),'Os6251/15','1',camera,view))
                images=[];masks=[];source=[]
                for view in (0,10,20,30):
                    row=lookup(view)
                    if not row or not row.get('mask'):break
                    raw=args.intake/row['source']
                    if sha(raw)!=row['source_sha256']:raise ValueError('Depth input changed')
                    with Image.open(raw) as im:
                        im.draft('RGB',(1024,1024));im.load();image=im.convert('RGB');image.thumbnail((1024,1024));array=np.array(image)
                    mask=np.asarray(Image.open(args.intake/row['mask']))>0
                    if mask.shape!=array.shape[:2]:raise ValueError('Depth input/mask mismatch')
                    array[~mask]=0;images.append(Image.fromarray(array));masks.append(mask)
                    source.append(dict(source=row['source'],sha256=row['source_sha256'],mask_sha256=sha(args.intake/row['mask']),qc_pass=row['qc_pass']))
                case=f'{treatment}{visit}_L5_rep1_{camera}'
                for mode in config['declared_modes']:
                    path=out/(case+'__'+mode+'.json')
                    if path.exists():
                        result=json.loads(path.read_text())
                        if result['signature']!=signature:raise ValueError('Cached depth signature mismatch')
                        results.append(result);continue
                    result=dict(case=case,treatment=treatment,measurement=visit,camera=camera,mode=mode,
                                signature=signature,source=source,status='not_run',reason='')
                    if len(images)!=4:result['reason']='missing_diagnostic_mask'
                    elif any(r['qc_pass']!='True' for r in source[:3]):result['reason']='training_mask_QC_failure'
                    else:
                        try:
                            kwargs={};h,w=masks[0].shape
                            if mode.startswith('nominal'):
                                focal=1.2 if 'f1.2' in mode else 2.;sign=-1 if 'sign-1' in mode else 1
                                ext=turntable([0,10,20,30],4.,sign).astype('float32')
                                k=np.array([[focal*w,0,w/2],[0,focal*w,h/2],[0,0,1]],np.float32)
                                kwargs=dict(extrinsics=ext[:3],intrinsics=np.repeat(k[None],3,axis=0))
                            with torch.inference_mode():
                                prediction=model.inference(image=images[:3],process_res=args.resolution,
                                    process_res_method='upper_bound_resize',ref_view_strategy='middle',**kwargs)
                            depth=prediction.depth;es=prediction.extrinsics;ks=prediction.intrinsics
                            if es is None or ks is None or not np.isfinite(depth).all():raise ValueError('Invalid learned geometry')
                            shape=depth.shape[-2:]
                            resized=[cv2.resize(m.astype('uint8'),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0 for m in masks]
                            cloud=np.concatenate([unproject(d,cv2.erode(m.astype('uint8'),np.ones((3,3),np.uint8))>0,k,e,2)
                                                  for d,m,k,e in zip(depth,resized,ks,es)])
                            if mode.startswith('nominal'):
                                target_ext=ext[3];target_k=ks.mean(0)
                            else:
                                nominal=turntable([0,10,20,30],4.,1)
                                centres=np.array([-e[:3,:3].T@e[:3,3] for e in es]);target=np.array([-e[:3,:3].T@e[:3,3] for e in nominal[:3]])
                                scale,rotation,shift=similarity(centres,target);cloud=scale*(cloud@rotation.T)+shift
                                target_ext=nominal[3];target_k=ks.mean(0)
                                result['training_camera_alignment_rmse']=float(np.sqrt(np.mean((scale*(centres@rotation.T)+shift-target)**2)))
                            projected=project(cloud,target_k,target_ext,shape,1);truth=resized[3]
                            intersection=int(np.sum(projected&truth));union=int(np.sum(projected|truth))
                            result.update(status='completed',heldout_iou=intersection/max(union,1),
                                          heldout_precision=intersection/max(int(projected.sum()),1),
                                          heldout_recall=intersection/max(int(truth.sum()),1),
                                          projected_pixels=int(projected.sum()),target_pixels=int(truth.sum()),
                                          heldout_mask_qc_pass=source[3]['qc_pass']=='True',cloud_points=len(cloud),
                                          processed_shape=list(shape))
                            np.savez_compressed(out/(case+'__'+mode+'.npz'),depth=depth,extrinsics=es,intrinsics=ks,
                                                heldout_projection=projected,heldout_mask=truth,world_cloud=cloud[::3])
                            if visit==2:
                                a=np.zeros((*shape,3),np.uint8);a[truth]=[100,190,100];a[projected&~truth]=[230,90,90];a[projected&truth]=[225,225,225]
                                Image.fromarray(a).save(out/(case+'__'+mode+'.png'))
                        except Exception as exc:
                            result.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
                            torch.cuda.empty_cache()
                    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');results.append(result)
                    print(json.dumps({k:v for k,v in result.items() if k not in ('source','signature')}),flush=True)
                (out/'progress.json').write_text(json.dumps(dict(status='running',cases=len(results),planned=150,elapsed_seconds=time.monotonic()-started)),encoding='utf-8')
    fields=list(dict.fromkeys(k for r in results for k in r if k!='source'))
    with (out/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:v for k,v in r.items() if k!='source'} for r in results)
    summary=dict(status='completed',attempt_records=len(results),successful=sum(r['status']=='completed' for r in results),
                 failed=sum(r['status']=='failed' for r in results),skipped=sum(r['status']=='not_run' for r in results),
                 heldout_index=30,nominal_calibration=False,manual_anatomical_validation=False,
                 limitations=['Nominal extrinsics/intrinsics are hypotheses, not measured calibration.',
                              'Held-out target is a SAM foreground, not true anatomical 3D.',
                              'NoIR inputs differ from RGB pretraining.',
                              'Relative depth and supplied camera gauge do not establish physical dimensions.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'progress.json').write_text(json.dumps(dict(status='completed',cases=len(results),planned=150)),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
