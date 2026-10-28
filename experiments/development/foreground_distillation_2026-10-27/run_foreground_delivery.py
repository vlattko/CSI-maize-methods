"""Publish final model candidates, actual runtime evidence and whole-cohort UI masks."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import foreground_core as core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','intake','teacher-run','output'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args(); scripts=Path(__file__).parent
    common=['--repo',str(args.repo),'--output',str(args.output)]
    extra=['--intake',str(args.intake),'--teacher-run',str(args.teacher_run)]
    def run(name,script,arguments):
        marker=args.output/'pipeline'/f'{name}.json'
        if marker.exists() and json.loads(marker.read_text()).get('returncode')==0:
            return
        command=[sys.executable,str(scripts/script),*arguments]
        log=args.output/'logs'/f'{name}_{time.strftime("%Y%m%d_%H%M%S")}.log'
        record=dict(status='running',step=name,command=command,log=str(log))
        core.write_json(args.output/'pipeline/delivery_state.json',record)
        print('starting',name,flush=True)
        with open(log,'w',encoding='utf8') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        record.update(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode)
        core.write_json(marker,record); core.write_json(args.output/'pipeline/delivery_state.json',record)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {log}')
    print('waiting for completed locked all-angle test',flush=True)
    while True:
        path=args.output/'pipeline/comparison_all_test.json'
        if path.exists() and json.loads(path.read_text()).get('returncode')==0:
            break
        time.sleep(10)
    run('publish_final','publish_foreground.py',['--output',str(args.output),'--stage','final'])
    validation=args.output/'evaluation/comparison_validation'
    selection=json.loads((validation/'selection.json').read_text())
    group=core.read_csv(validation/'group_metrics.csv')
    choices=[]
    for name in ('joint','joint_all40','joint_all40_960'):
        threshold=selection['thresholds'][name]
        row=next(r for r in group if r['model']==name and r['group']=='overall'
                 and r['teacher_accepted'].lower()=='true' and float(r['confidence'])==threshold)
        choices.append(dict(name=name,mean_iou=float(row['mean_iou']),endpoint3=float(row['mean_endpoint_recall_px3'])))
    default=max(choices,key=lambda r:(r['mean_iou'],r['endpoint3'],r['name']))
    card_path=args.output/'deployment'/f'final_{default["name"]}'/'model_card.json'
    card=json.loads(card_path.read_text())
    binding=dict(status='model_selected_cache_waiting',model_card=str(card_path),model_name=card['name'],
        weights=card['weights'],weights_sha256=card['weights_sha256'],confidence=card['confidence'],
        selection='maximum accepted-validation mean IoU among joint quarter640,all40640,all40960; endpoint3 then name break ties',
        validation_choices=choices,test_used_for_model_selection=False,
        cache_root=str(args.output/'student_cache'/card['name']),
        cache_status=str(args.output/'student_cache'/card['name']/'status.json'),
        frames_csv=str(args.output/'student_cache'/card['name']/'frames.csv'))
    core.write_json(args.output/'viewer_model_binding.json',binding)
    inventory=core.read_csv(args.output/'all_views/frame_inventory.csv')
    samples={}
    for row in inventory:
        if row['split']=='val' and row['teacher_accepted'].lower()=='true':
            key=(row['treatment'],row['measurement'],row['camera'])
            old=samples.get(key)
            if old is None or (int(row['view']),int(row['replicate']),row['source']) < (int(old['view']),int(old['replicate']),old['source']):
                samples[key]=row
    sources=[str(args.intake/row['source']) for _,row in sorted(samples.items())]
    parallel=args.output/'pipeline/parallel_cache_state.json'
    if parallel.exists():
        while json.loads(parallel.read_text()).get('status') not in ('completed','failed'):
            core.write_json(args.output/'pipeline/delivery_state.json',dict(status='waiting_for_parallel_gpu0_cache',
                reason='Cache is shared with final delivery; avoid duplicate inference and benchmark only after competing work completes.'))
            time.sleep(10)
        if json.loads(parallel.read_text()).get('status')=='failed':
            raise RuntimeError('Parallel cache failed; retain its checkpoint and repair before delivery')
    runtime=[]
    for device,label in [('1','gpu'),('cpu','cpu')]:
        destination=args.output/f'runtime_benchmark_{label}'
        run('benchmark_'+label,'infer_foreground.py',common[:-2]+['--output',str(destination),
            '--weights',card['weights'],'--model-card',str(card_path),'--source',*sources,'--device',device])
        record=json.loads((destination/'inference_manifest.json').read_text())
        item=dict(device=device,frames=len(record['rows']),model_load_seconds=record['model_load_seconds'],
                  sampling='one accepted validation frame per treatment/visit/camera; lowest available view and replicate',
                  concurrent_training=False)
        for metric in ('decode_ms','predict_ms','end_to_end_ms'):
            values=[r[metric] for r in record['rows']]
            item[metric+'_median']=float(np.median(values)); item[metric+'_p95']=float(np.quantile(values,.95))
        runtime.append(item)
    core.write_json(args.output/'runtime_benchmark.json',dict(status='completed',models=runtime,
        scope='same30stratifiedvalidation sources; model warmup excluded; end-to-end includes raw source hash/decode and mask/black-PNG output writes; no SAM speedup inferred'))
    run('final_summary','summarize_foreground.py',common+extra)
    run('final_student_cache','cache_foreground.py',common+extra+['--model-card',str(card_path),'--batch-size','8','--decode-workers','4'])
    cache_status=json.loads(Path(binding['cache_status']).read_text())
    if cache_status['status']!='completed' or cache_status['cached_frames']!=27126 or cache_status['reference_records']!=360:
        raise ValueError('Final viewer cache does not retain the complete available cohort')
    binding.update(status='completed',cache_frames=cache_status['cached_frames'],
                   cache_manifest_sha256=core.digest(binding['cache_status']))
    core.write_json(args.output/'viewer_model_binding.json',binding)
    source_folder=args.output/'source_snapshot_final'; source_folder.mkdir(exist_ok=True)
    for path in scripts.iterdir():
        if path.suffix in ('.py','.md'):
            shutil.copyfile(path,source_folder/path.name)
    package=args.output/'foreground_model_bundle.zip'
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in (args.output/'deployment',source_folder):
            for path in directory.rglob('*'):
                if path.is_file():
                    archive.write(path,path.relative_to(args.output))
        for name in ('foreground_summary.json','test_summary.csv','runtime_benchmark.json','viewer_model_binding.json','evaluation_protocol.json','all40_selection.json'):
            archive.write(args.output/name,name)
    core.write_json(args.output/'pipeline/delivery_state.json',dict(status='completed',
        bundle=str(package),bundle_sha256=core.digest(package),viewer_binding_sha256=core.digest(args.output/'viewer_model_binding.json'),
        reference_records=360,available_frame_predictions=27126,model_card=str(card_path)))
    print('completed foreground system',flush=True)


if __name__=='__main__':
    main()
