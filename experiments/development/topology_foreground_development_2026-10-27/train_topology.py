"""Two fixed5-epoch dense-teacher foreground arms; train/validation only."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='2'
import argparse,csv,datetime,hashlib,json,platform,random,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from topology_core import TinyUNet,segmentation_loss,guard_record,select_validation,teacher_geometry,agreement_metrics,THRESHOLDS,INPUT_SHAPE,PARTITION

SEED=20261027


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)
def write(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def log(out,event,**kw):
    row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**kw)
    with (out/'training_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(row),flush=True)
def state_digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def prepare(args):
    with args.frames.open(encoding='utf-8-sig',newline='') as f:
        # Never open paths outside the authorized two partitions.
        rows=[r for r in csv.DictReader(f) if r['split'] in ['train','val']]
    rows=[r for r in rows if guard_record(r)];rows.sort(key=lambda r:r['id']);cached=[];bindings=[];failures=[];start=time.time()
    for i,r in enumerate(rows,1):
        try:
            image_path=Path(r['image']);mask_path=args.teacher_root/r['mask']
            if digest(image_path)!=r['image_sha256'] or digest(mask_path)!=r['mask_sha256']:raise ValueError('Image/teacher digest mismatch')
            with Image.open(image_path) as im:image=np.asarray(im.convert('RGB').resize((INPUT_SHAPE[1],INPUT_SHAPE[0]),Image.Resampling.BILINEAR)).copy();original_shape=(im.height,im.width)
            with Image.open(mask_path) as im:
                if (im.height,im.width)!=original_shape:raise ValueError('Dense teacher/image shape mismatch')
                truth=np.asarray(im.convert('L'))>0;mask=np.asarray(im.convert('L').resize((INPUT_SHAPE[1],INPUT_SHAPE[0]),Image.Resampling.NEAREST))>0
            geometry=teacher_geometry(truth) if r['split']=='val' else None
            cached.append(dict(id=r['id'],split=r['split'],camera=r['camera'],image=image,mask=mask,truth=truth if r['split']=='val' else None,geometry=geometry,
                identity={k:r[k] for k in ['treatment','measurement','genotype','replicate','session','camera','view']}))
            bindings.append({k:r[k] for k in ['id','split','treatment','measurement','genotype','replicate','camera','view','image','image_sha256','mask','mask_sha256','source_sha256']}|
                dict(resized_image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),resized_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),input_height=INPUT_SHAPE[0],input_width=INPUT_SHAPE[1]))
        except Exception as e:failures.append(dict(id=r['id'],split=r['split'],error=repr(e)))
        if i%300==0:log(args.output,'preflight_cache',completed=i,total=len(rows))
    write(args.output/'data_bindings.csv',bindings);write(args.output/'failed_cases.csv',failures)
    if failures:raise ValueError('Preflight failures retained; training not launched')
    train=[r for r in cached if r['split']=='train'];val=[r for r in cached if r['split']=='val']
    if not train or not val:raise ValueError('Training and validation must both be nonempty')
    dump(args.output/'preflight.json',dict(status='passed',train_frames=len(train),val_frames=len(val),failures=0,
        input_manifest_sha256=digest(args.frames),data_bindings_sha256=digest(args.output/'data_bindings.csv'),test_images_opened=0,
        elapsed_seconds=time.time()-start))
    log(args.output,'preflight_passed',train=len(train),val=len(val),test_images_opened=0)
    return train,val


def tensors(batch,device,targets=True):
    x=torch.from_numpy(np.stack([r['image'].transpose(2,0,1) for r in batch])).to(device,dtype=torch.float32)/255
    y=torch.from_numpy(np.stack([r['mask'] for r in batch])).unsqueeze(1).to(device,dtype=torch.float32) if targets else None
    return x,y


def validate(model,rows,device,batch_size,epoch,arm):
    result=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size];x,_=tensors(batch,device,False);prob=model(x).sigmoid()
            for i,r in enumerate(batch):
                p=F.interpolate(prob[i:i+1],size=r['truth'].shape,mode='bilinear',align_corners=False)[0,0].cpu().numpy()
                for threshold in THRESHOLDS:
                    m=agreement_metrics(r['truth'],p>=threshold,r['geometry']);result.append(dict(arm=arm,epoch=epoch,split='val',threshold=threshold,id=r['id'],camera=r['camera'],**m))
    aggregate=[]
    for threshold in THRESHOLDS:
        rr=[r for r in result if r['threshold']==threshold]
        mean=lambda k:float(np.mean([r[k] for r in rr if r[k] is not None]))
        score=.5*mean('dice')+.25*mean('endpoint_recall_px3')+.25*mean('thin_skeleton_recall_px3')
        aggregate.append(dict(arm=arm,epoch=epoch,split='val',threshold=threshold,score=score,frames=len(rr),mean_iou=mean('iou'),mean_dice=mean('dice'),
            mean_endpoint_recall_px0=mean('endpoint_recall_px0'),mean_endpoint_recall_px3=mean('endpoint_recall_px3'),mean_endpoint_recall_px5=mean('endpoint_recall_px5'),
            mean_thin_skeleton_recall_px0=mean('thin_skeleton_recall_px0'),mean_thin_skeleton_recall_px3=mean('thin_skeleton_recall_px3'),
            endpoint_eligible_frames=sum(r['endpoint_recall_px3'] is not None for r in rr),thin_eligible_frames=sum(r['thin_skeleton_recall_px3'] is not None for r in rr)))
    return aggregate,result


def train_arm(args,arm,train,val,initial):
    torch.manual_seed(SEED);model=TinyUNet(base=8).to('cuda:0');model.load_state_dict(initial['state_dict']);actual=state_digest(model)
    if actual!=initial['state_sha256']:raise ValueError('Arms must start with identical state')
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);allmetrics=[];epochlogs=[];orders=[];folder=args.output/arm;folder.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,6):
        model.train();rng=np.random.default_rng(SEED+epoch);order=rng.permutation(len(train));order_sha=hashlib.sha256('|'.join(train[int(i)]['id'] for i in order).encode()).hexdigest();orders.append(order_sha)
        losses=[];started=time.time()
        for bi,start in enumerate(range(0,len(order),args.batch),1):
            batch=[train[int(i)] for i in order[start:start+args.batch]];x,y=tensors(batch,'cuda:0');opt.zero_grad(set_to_none=True);loss=segmentation_loss(model(x),y,topology=arm=='cldice')
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();opt.step();losses.append(float(loss.detach()))
            if bi%60==0:log(args.output,'training_batch',arm=arm,epoch=epoch,batch=bi,batches=(len(train)+args.batch-1)//args.batch,loss=losses[-1])
        checkpoint=folder/f'epoch_{epoch}.pt';torch.save(dict(state_dict=model.state_dict(),architecture=dict(name='TinyUNet',base=8),input_shape=INPUT_SHAPE,arm=arm,epoch=epoch),checkpoint)
        aggregate,records=validate(model,val,'cuda:0',args.batch,epoch,arm);allmetrics+=aggregate;write(folder/f'validation_epoch_{epoch}.csv',records)
        epochlog=dict(arm=arm,epoch=epoch,training_mean_loss=float(np.mean(losses)),order_sha256=order_sha,checkpoint_sha256=digest(checkpoint),
            elapsed_seconds=time.time()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),validation=aggregate);epochlogs.append(epochlog);dump(folder/f'epoch_{epoch}.json',epochlog);write(folder/'validation_summary.csv',allmetrics)
        log(args.output,'epoch_completed',**epochlog)
    chosen=select_validation(allmetrics);parent=folder/f"epoch_{chosen['epoch']}.pt";data=torch.load(parent,map_location='cpu',weights_only=True);data['chosen_threshold']=chosen['threshold'];data['selection_score']=chosen['score']
    best=folder/'best.pt';torch.save(data,best);shutil.copyfile(folder/'epoch_5.pt',folder/'last.pt')
    frozen=dict(arm=arm,status='frozen',epochs_executed=5,checkpoint=str(best.resolve()),checkpoint_sha256=digest(best),last_checkpoint=str((folder/'last.pt').resolve()),
        last_sha256=digest(folder/'last.pt'),selected_epoch=chosen['epoch'],threshold=chosen['threshold'],validation=chosen,
        initialization_state_sha256=actual,initialization_checkpoint_sha256=digest(args.output/'initialization.pt'),epoch_order_sha256=orders,
        loss='0.5BCE+0.5softDice'+('+0.25soft-clDice(iterations10)' if arm=='cldice' else ''),
        training_frames=len(train),validation_frames=len(val),teacher_scope='Operational SAM foreground fidelity, not human accuracy or botanical topology',
        dense_mask_vs_yolo_confound='Dense original SAM masks retain raster holes/thin structures; YOLO polygon teachers differ',
        input_policy='Unmasked full-scene RGB/NoIR processed images; PIL bilinear resize640x481; DN/255; FP32; no augmentation or channel reordering',
        output_policy='Single foreground probability; bilinear probability resize to requested original processing grid before validation-selected threshold',
        primary_reference='Shit etal.CVPR2021 clDice, https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf')
    dump(folder/'model_card.json',frozen);dump(folder/'selection.json',frozen);del model,opt;torch.cuda.empty_cache();return frozen


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',type=Path,required=True);p.add_argument('--teacher-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',type=Path,required=True);p.add_argument('--batch',type=int,default=8);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.cuda.set_device(0);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
    sources={}
    for f in ['topology_core.py','train_topology.py','infer_topology.py','test_topology.py']:
        src=Path(__file__).parent/f;target=a.output/'execution_sources'/f;target.parent.mkdir(exist_ok=True);target.write_bytes(src.read_bytes());sources[f]=digest(src)
    config=dict(status='declared_before_training',seed=SEED,architecture=dict(name='TinyUNet',base=8),input_shape=INPUT_SHAPE,epochs_per_arm=5,batch=a.batch,
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),thresholds=THRESHOLDS,partition=PARTITION,device='cuda:0',precision='FP32',cpu_threads=2,
        loss_arms=dict(baseline='0.5BCE+0.5Dice',cldice='0.5BCE+0.5Dice+0.25softclDice10'),
        validation_selection='0.5Dice+0.25endpoint-support@3px+0.25thin-skeleton-support@3px; original processing1024 grid; ties earliest epoch then higher threshold',
        source_sha256=sources,frame_manifest_sha256=digest(a.frames),test_opened=False,runtime=dict(torch=torch.__version__,python=platform.python_version(),GPU=torch.cuda.get_device_name(0)))
    dump(a.output/'configuration.json',config);train,val=prepare(a)
    model=TinyUNet(base=8);initial=dict(state_dict=model.state_dict(),architecture=dict(name='TinyUNet',base=8),input_shape=INPUT_SHAPE,state_sha256=state_digest(model));torch.save(initial,a.output/'initialization.pt');del model
    log(a.output,'training_launched',initialization_sha256=digest(a.output/'initialization.pt'),initial_state_sha256=initial['state_sha256'],parameters=sum(v.numel() for v in initial['state_dict'].values()),train=len(train),val=len(val),device='cuda:0')
    results=[train_arm(a,arm,train,val,initial) for arm in ['baseline','cldice']]
    if results[0]['epoch_order_sha256']!=results[1]['epoch_order_sha256']:raise ValueError('Training orders differ between arms')
    if any(digest(Path(__file__).parent/f)!=h for f,h in sources.items()):raise ValueError('Execution source changed while training')
    frozen=dict(status='frozen',models=results,configuration_sha256=digest(a.output/'configuration.json'),preflight_sha256=digest(a.output/'preflight.json'),
        inference_module=str(Path(__file__).with_name('infer_topology.py').resolve()),inference_module_sha256=digest(Path(__file__).with_name('infer_topology.py')),
        test_opened=False,selection_frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    dump(a.output/'frozen_model_selection.json',frozen)
    dump(a.gate,dict(status='frozen',owner='full_architecture_mentor',reason='Both dense U-Net arms completed5epochs with identical initialization/order; epoch and threshold frozen using validation only.',
        configuration_sha256=frozen['configuration_sha256'],checkpoint_sha256={m['arm']:m['checkpoint_sha256'] for m in results},models=results,
        frozen_selection=str((a.output/'frozen_model_selection.json').resolve()),frozen_selection_sha256=digest(a.output/'frozen_model_selection.json'),
        inference_module=frozen['inference_module'],inference_module_sha256=frozen['inference_module_sha256'],test_opened=False))
    log(a.output,'both_arms_frozen',models=[dict(arm=m['arm'],checkpoint_sha256=m['checkpoint_sha256'],epoch=m['selected_epoch'],threshold=m['threshold']) for m in results],gate=str(a.gate))


if __name__=='__main__':main()
