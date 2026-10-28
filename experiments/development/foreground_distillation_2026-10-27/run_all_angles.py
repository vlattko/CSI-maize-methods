"""Validation-gated all-angle continuation and the locked final genotype test."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import foreground_core as core


def selected_summary(root,model):
    selection=json.loads((root/'selection.json').read_text())
    threshold=selection['thresholds'][model]
    rows=core.read_csv(root/'group_metrics.csv')
    row=next(r for r in rows if r['model']==model and r['group']=='overall'
             and r['teacher_accepted'].lower()=='true' and float(r['confidence'])==threshold)
    return dict(confidence=threshold,n=int(row['n']),mean_iou=float(row['mean_iou']),
        endpoint_recall_px3=float(row['mean_endpoint_recall_px3']),
        thin_skeleton_recall_px0=float(row['mean_thin_skeleton_recall_px0']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','intake','teacher-run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(); scripts=Path(__file__).parent
    common=['--repo',str(args.repo),'--output',str(args.output)]
    extra=['--intake',str(args.intake),'--teacher-run',str(args.teacher_run)]
    def run(name,script,arguments):
        marker=args.output/'pipeline'/f'{name}.json'
        if marker.exists() and json.loads(marker.read_text()).get('returncode')==0:
            return
        command=[sys.executable,str(scripts/script),*arguments]
        log=args.output/'logs'/f'{name}_{time.strftime("%Y%m%d_%H%M%S")}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        record=dict(status='running',step=name,command=command,log=str(log))
        core.write_json(args.output/'pipeline/all_angles_state.json',record)
        print('starting',name,flush=True)
        with open(log,'w',encoding='utf8') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        record.update(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode)
        core.write_json(marker,record); core.write_json(args.output/'pipeline/all_angles_state.json',record)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {log}')
    print('waiting for quarter full validation and all-angle preparation',flush=True)
    while True:
        full=args.output/'pipeline/full_validation.json'
        prepared=args.output/'all_views/preparation.json'
        if full.exists() and prepared.exists() and json.loads(full.read_text()).get('returncode')==0 and json.loads(prepared.read_text()).get('status')=='completed':
            break
        time.sleep(10)
    dataset=args.output/'all_views/dataset'
    inventory=core.read_csv(args.output/'all_views/frame_inventory.csv')
    validation=[r['image'] for r in inventory if r['split']=='val' and r['teacher_accepted'].lower()=='true' and int(r['view']) in core.QUARTERS]
    (dataset/'joint_checkpoint_validation.txt').write_text('\n'.join(validation)+'\n',encoding='utf8')
    (dataset/'joint_checkpoint_validation.yaml').write_text(f'path: {dataset.as_posix()}\ntrain: joint_train.txt\nval: joint_checkpoint_validation.txt\ntest: joint_test.txt\nnames:\n  0: plant_foreground\n',encoding='utf8')
    core.write_json(args.output/'all40_validation_schedule.json',dict(status='declared_before_all40_training',
        per_epoch_validation='accepted fixed-quarter views of validation genotype',per_epoch_frames=len(validation),
        postfit_decision_validation='all available angles of the same validation genotype',test='all available angles of held-out test genotype',
        data_yaml_sha256=core.digest(dataset/'joint_checkpoint_validation.yaml'),list_sha256=core.digest(dataset/'joint_checkpoint_validation.txt')))
    run('all40_train5','train_foreground.py',common+['--phase','all40','--epochs','5','--modes','joint'])
    run('all40_validation5','evaluate_foreground.py',common+extra+['--phase','all40','--epochs','5','--split','val','--all-angles'])
    root=args.output/'evaluation/all40_validation'
    full=selected_summary(root,'joint_all40'); quarter=selected_summary(root,'joint_quarter')
    extend=(full['mean_iou']>=quarter['mean_iou']-.01 and (full['endpoint_recall_px3']<.90 or full['mean_iou']<.85))
    decision=dict(extend=extend,all40_validation=full,quarter_validation=quarter,
        rule='meanIoU>=quartermeanIoU-.01 AND(endpointrecall3<.90 ORmeanIoU<.85)',
        protocol_sha256=core.digest(args.output/'evaluation_protocol.json'),test_used=False)
    core.write_json(args.output/'all40_extension_decision.json',decision)
    manifest=json.loads((args.output/'training/joint_all40_e5/run_manifest.json').read_text())
    candidates=[dict(cumulative_all_angle_epochs=5,checkpoint=manifest['best_checkpoint'],sha256=manifest['best_sha256'],validation=full)]
    if extend:
        run('all40_additional5','train_foreground.py',common+['--phase','all40_extension','--epochs','5','--modes','joint',
            '--initial',manifest['best_checkpoint'],'--initial-sha256',manifest['best_sha256']])
        run('all40_extension_validation','evaluate_foreground.py',common+extra+['--phase','all40_extension','--epochs','5','--split','val','--all-angles'])
        manifest=json.loads((args.output/'training/joint_all40_extension_e5/run_manifest.json').read_text())
        candidates.append(dict(cumulative_all_angle_epochs=10,checkpoint=manifest['best_checkpoint'],sha256=manifest['best_sha256'],
            validation=selected_summary(args.output/'evaluation/all40_extension_validation','joint_all40')))
    choice=max(candidates,key=lambda x:(x['validation']['mean_iou'],x['validation']['endpoint_recall_px3']))
    core.write_json(args.output/'all40_selection.json',dict(**choice,candidates=candidates,
        selection='maximum accepted-validation meanIoU; ties favor higher endpoint support; test untouched'))
    run('comparison_all_validation','evaluate_foreground.py',common+extra+['--phase','comparison','--epochs','25','--split','val','--all-angles'])
    selection=args.output/'evaluation/comparison_validation/selection.json'
    core.write_json(args.output/'evaluation/final_protocol_ready.json',dict(status='frozen_before_test',
        protocol_sha256=core.digest(args.output/'evaluation_protocol.json'),all40_selection_sha256=core.digest(args.output/'all40_selection.json'),
        confidence_selection_sha256=core.digest(selection),test_genotype='Os7798/2',
        comparison_checkpoints=json.loads(selection.read_text())['models']))
    run('comparison_all_test','evaluate_foreground.py',common+extra+['--phase','comparison','--epochs','25','--split','test','--all-angles'])
    core.write_json(args.output/'pipeline/all_angles_state.json',dict(status='completed',test_evaluated=True))


if __name__=='__main__':
    main()
