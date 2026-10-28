"""Bounded offline YOLO26 whole-plant teacher distillation on one assigned GPU."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import time
import foreground_core as core

EXECUTION_SOURCE=Path(__file__).read_bytes()
EXECUTION_SHA256=__import__('hashlib').sha256(EXECUTION_SOURCE).hexdigest()


def check_membership(paths,rows,split,mode):
    lookup={r['image']:r for r in rows}
    if len(paths)!=len(set(paths)):
        raise ValueError('Duplicate training image')
    for path in paths:
        r=lookup.get(path)
        if r is None or r['split']!=split or str(r['teacher_accepted']).lower()!='true' or (mode=='rgb' and r['camera']!='rgb'):
            raise ValueError('Training/validation list violates frozen split or teacher acceptance')


def train_one(args,mode,epochs,phase):
    destination=args.output/'training'/f'{mode}_{phase}_e{epochs}'
    manifest=destination/'run_manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['status']=='completed' and core.digest(old['best_checkpoint'])==old['best_sha256']:
            print('reusing completed',destination,flush=True)
            return old
        raise ValueError('Completed checkpoint integrity mismatch')
    dataset_root=args.output/'all_views' if phase.startswith('all40') else args.output
    data=dataset_root/'dataset'/f'{mode}.yaml'
    if phase.startswith('all40'):
        data=dataset_root/'dataset/joint_checkpoint_validation.yaml'
    preparation=json.loads((dataset_root/'preparation.json').read_text())
    if preparation['status']!='completed':
        raise ValueError('Teacher dataset preparation is incomplete')
    frame_table=dataset_root/('frame_inventory.csv' if phase.startswith('all40') else 'quarter_frames.csv')
    if core.digest(frame_table)!=preparation['frame_table_sha256']:
        raise ValueError('Prepared frame inventory changed')
    inventory=core.read_csv(frame_table)
    lookup={r['image']:r for r in inventory}
    for split in ('train','val'):
        list_name='joint_checkpoint_validation.txt' if phase.startswith('all40') and split=='val' else f'{mode}_{split}.txt'
        paths=(dataset_root/'dataset'/list_name).read_text().splitlines()
        check_membership(paths,inventory,split,mode)
        for path in paths:
            r=lookup[path]
            if core.digest(path)!=r['image_sha256'] or core.digest(r['label'])!=r['label_sha256']:
                raise ValueError('Prepared training image or teacher label changed')
    from ultralytics import YOLO
    import torch,ultralytics
    source=args.initial if args.initial else args.output/'yolo26n-seg.pt'
    if args.initial and (not args.initial.resolve().is_relative_to(args.output.resolve()) or core.digest(args.initial)!=args.initial_sha256):
        raise ValueError('Additional training needs an intact output-owned parent checkpoint')
    model=YOLO(str(source))
    kwargs=dict(data=str(data),epochs=epochs,imgsz=640,batch=8,workers=2 if phase!='pilot' else 0,device=args.device,
        project=str(args.output/'training'),name=destination.name,exist_ok=True,
        seed=core.SEED,deterministic=True,optimizer='AdamW',lr0=.001,
        patience=epochs+1,amp=True,cache='disk' if phase!='pilot' else False,mosaic=0.,mixup=0.,copy_paste=0.,
        degrees=0.,translate=.02,scale=.10,fliplr=.5,flipud=0.,
        hsv_h=0.,hsv_s=0.,hsv_v=.10,plots=True,save=True,save_period=5,
        verbose=False,close_mosaic=0)
    # Keep spectral channel order intact; only modest brightness/geometric augmentation.
    torch.cuda.set_device(args.device)  # initialize the explicitly assigned device before memory accounting
    start=time.perf_counter(); torch.cuda.reset_peak_memory_stats(args.device)
    core.write_json(destination/'requested_configuration.json',dict(kwargs=kwargs,
        training_source_sha256=EXECUTION_SHA256,preparation_sha256=core.digest(dataset_root/'preparation.json'),
        pretrained_sha256=core.digest(source),label_scope='whole plant foreground; teacher agreement only'))
    (destination/'execution_train_foreground.py').write_bytes(EXECUTION_SOURCE)
    shutil.copyfile(Path(core.__file__),destination/'execution_foreground_core.py')
    result=model.train(**kwargs)
    best=destination/'weights/best.pt'
    record=dict(status='completed',mode=mode,phase=phase,epochs_requested=epochs,
        elapsed_seconds=time.perf_counter()-start,kwargs=kwargs,
        best_checkpoint=str(best),best_sha256=core.digest(best),last_checkpoint=str(destination/'weights/last.pt'),
        last_sha256=core.digest(destination/'weights/last.pt'),pretrained_sha256=core.digest(source),
        data_yaml_sha256=core.digest(data),preparation_sha256=core.digest(dataset_root/'preparation.json'),
        initialization_checkpoint=str(source),additional_parent_checkpoint=bool(args.initial),
        code_sha256=EXECUTION_SHA256,core_sha256=core.digest(Path(core.__file__)),
        metrics={k:float(v) for k,v in result.results_dict.items()},
        runtime=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,python=platform.python_version(),
                     GPU=torch.cuda.get_device_name(args.device),device=args.device,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device)),
        limits='Checkpoint chosen by validation teacher labels; no expert or leaf-instance accuracy claim')
    core.write_json(manifest,record)
    print('completed',json.dumps(record),flush=True)
    del model
    torch.cuda.empty_cache()
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--phase',choices=['pilot','full','all40','all40_extension'],required=True)
    p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--modes',default='rgb,joint')
    p.add_argument('--device',default='1')
    p.add_argument('--initial',type=Path)
    p.add_argument('--initial-sha256')
    args=p.parse_args()
    args.device=int(args.device)
    if args.device != 1 or args.epochs<1 or args.epochs>40:
        raise ValueError('This run is bounded to GPU1 and1..40epochs')
    modes=args.modes.split(',')
    if any(x not in ('rgb','joint') for x in modes):
        raise ValueError('Unrecognized training condition')
    if args.phase!='pilot':
        gate=json.loads((args.output/'evaluation/pilot_validation/viability.json').read_text())
        if not gate['viable']:
            raise ValueError('Pilot did not satisfy the declared teacher-agreement viability rule')
    _,policy=core.configure_offline(args.repo,args.output)
    core.write_json(args.output/'offline_policy.json',policy)
    for mode in modes:
        train_one(args,mode,args.epochs,args.phase)


if __name__=='__main__':
    main()
