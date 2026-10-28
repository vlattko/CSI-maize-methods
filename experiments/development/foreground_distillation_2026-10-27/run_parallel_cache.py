"""Cache the validation-selected YOLO on releasedGPU0 while the locked test runsGPU1."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import foreground_core as core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','intake','teacher-run','output'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args(); scripts=Path(__file__).parent
    state=args.output/'pipeline/parallel_cache_state.json'
    core.write_json(state,dict(status='waiting_validation_freeze',device=0))
    while not (args.output/'evaluation/final_protocol_ready.json').exists():
        time.sleep(10)
    core.write_json(state,dict(status='waiting_gpu0_release',device=0))
    released=args.output/'gpu0_cache_available.json'
    while not released.exists() or json.loads(released.read_text()).get('status')!='released':
        time.sleep(10)
    selection_path=args.output/'evaluation/comparison_validation/selection.json'
    selection=json.loads(selection_path.read_text())
    protocol=json.loads((args.output/'evaluation/final_protocol_ready.json').read_text())
    if core.digest(selection_path)!=protocol['confidence_selection_sha256']:
        raise ValueError('Validation selection changed after protocol freeze')
    publish_log=args.output/'logs/publish_frozen_parallel_cache.log'
    with open(publish_log,'w',encoding='utf8') as stream:
        result=subprocess.run([sys.executable,str(scripts/'publish_foreground.py'),'--output',str(args.output),'--stage','frozen'],stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        core.write_json(state,dict(status='failed',stage='publish_frozen',log=str(publish_log),returncode=result.returncode))
        raise RuntimeError('Frozen candidate publication failed')
    groups=core.read_csv(args.output/'evaluation/comparison_validation/group_metrics.csv')
    candidates=[]
    for name in ('joint','joint_all40','joint_all40_960'):
        confidence=selection['thresholds'][name]
        row=next(r for r in groups if r['model']==name and r['group']=='overall' and r['teacher_accepted'].lower()=='true' and float(r['confidence'])==confidence)
        candidates.append(dict(name=name,mean_iou=float(row['mean_iou']),endpoint3=float(row['mean_endpoint_recall_px3'])))
    chosen=max(candidates,key=lambda r:(r['mean_iou'],r['endpoint3'],r['name']))
    card=args.output/'deployment'/f'final_{chosen["name"]}'/'model_card.json'
    metadata=json.loads(card.read_text())
    command=[sys.executable,str(scripts/'cache_foreground.py'),'--repo',str(args.repo),
        '--intake',str(args.intake),'--teacher-run',str(args.teacher_run),'--output',str(args.output),
        '--model-card',str(card),'--device','0','--batch-size','8','--decode-workers','4']
    log=args.output/'logs'/f'parallel_gpu0_student_cache_{time.strftime("%Y%m%d_%H%M%S")}.log'
    record=dict(status='running',device=0,command=command,log=str(log),model_card=str(card),
        validation_candidates=candidates,chosen=chosen,selection_uses_test=False,
        parallel_with_locked_gpu1_test=True,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    core.write_json(state,record)
    core.write_json(args.output/'viewer_model_binding.json',dict(status='choices_frozen_test_and_cache_pending',
        model_card=str(card),model_name=metadata['name'],weights=metadata['weights'],weights_sha256=metadata['weights_sha256'],
        confidence=metadata['confidence'],imgsz=metadata['imgsz'],test_used_for_model_selection=False,
        cache_root=str(args.output/'student_cache'/metadata['name']),cache_status=str(args.output/'student_cache'/metadata['name']/'status.json'),
        frames_csv=str(args.output/'student_cache'/metadata['name']/'frames.csv')))
    with open(log,'w',encoding='utf8') as stream:
        process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    record.update(status='completed' if process.returncode==0 else 'failed',returncode=process.returncode,
                  ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    if process.returncode==0:
        cache=args.output/'student_cache'/metadata['name']/'status.json'
        cache_status=json.loads(cache.read_text())
        if cache_status['status']!='completed' or cache_status['cached_frames']!=27126 or cache_status['device']!=0:
            raise ValueError('GPU0cache did not retain the complete available cohort')
        record['cache_manifest']=str(cache); record['cache_manifest_sha256']=core.digest(cache)
    core.write_json(state,record)
    core.write_json(args.output/'pipeline/final_student_cache.json',record)
    if process.returncode:
        raise RuntimeError('Parallel student cache failed')
    print('completed parallelGPU0whole-cohort cache',flush=True)


if __name__=='__main__':
    main()
