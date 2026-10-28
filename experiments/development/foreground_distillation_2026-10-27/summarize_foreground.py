"""Usable model summary and deterministic black-background failure panels."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image
import foreground_core as core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','intake','teacher-run','output'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    root=args.output/'evaluation/comparison_test'
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='completed' or not manifest['all_angles']:
        raise ValueError('Completed all-angle final test is required')
    records=core.read_csv(root/'frame_metrics.csv')
    inventory={r['id']:r for r in core.read_csv(args.output/'all_views/frame_inventory.csv')}
    groups=defaultdict(list)
    for record in records:
        for camera in ('both',record['camera']):
            model='legacy_camera_matched' if record['model'].startswith('legacy_weak_') else record['model']
            groups[(model,camera,record['teacher_accepted'])].append(record)
    summaries=[]
    for (model,camera,accepted),rows in groups.items():
        item=dict(model=model,camera=camera,teacher_accepted=accepted,frames=len(rows),
            acquisitions=len({inventory[r['id']]['session'] for r in rows}),
            genotype_count=len({r['genotype'] for r in rows}),
            nonempty_fraction=float(np.mean([int(r['predicted_pixels'])>0 for r in rows])),
            mean_iou=float(np.mean([float(r['iou']) for r in rows])),median_iou=float(np.median([float(r['iou']) for r in rows])),
            mean_dice=float(np.mean([float(r['dice']) for r in rows])),
            inference_p50_ms=float(np.median([float(r.get('foreground_ms',r['predict_ms'])) for r in rows])),
            inference_p95_ms=float(np.quantile([float(r.get('foreground_ms',r['predict_ms'])) for r in rows],.95)))
        for metric in ('endpoint_recall_px0','endpoint_recall_px3','endpoint_recall_px5','thin_skeleton_recall_px0','thin_skeleton_recall_px3'):
            values=[float(r[metric]) for r in rows if r[metric]!='']
            item['mean_'+metric]=float(np.mean(values)) if values else None
        summaries.append(item)
    core.write_csv(args.output/'test_summary.csv',summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    panels=args.output/'failure_panels'; panels.mkdir(parents=True,exist_ok=True)
    by_condition=defaultdict(list)
    for r in records:
        if r['model']=='joint_all40':
            by_condition[(r['treatment'],r['measurement'],r['camera'],r['teacher_accepted'])].append(r)
    lookup={(r['model'],r['id']):r for r in records}
    test_inventory=[r for r in inventory.values() if r['split']=='test']
    coverage=[]
    for treatment in ('C','P','S'):
        for visit in range(1,6):
            for camera in ('rgb','noir'):
                rows=[r for r in test_inventory if r['treatment']==treatment and int(r['measurement'])==visit and r['camera']==camera]
                accepted=sum(r['teacher_accepted'].lower()=='true' for r in rows)
                coverage.append(dict(treatment=treatment,measurement=visit,camera=camera,planned_reference_acquisitions=4,
                    planned_frames=160,available_frames=len(rows),accepted_teacher_frames=accepted,
                    rejected_teacher_frames=len(rows)-accepted,missing_frames=160-len(rows),
                    primary_panel_status='available' if accepted else 'no accepted source example'))
    core.write_csv(args.output/'test_condition_coverage.csv',coverage)
    panel_records=[]
    for condition,rows in sorted(by_condition.items()):
        chosen=min(rows,key=lambda r:(float(r['iou']),r['id']))
        r=inventory[chosen['id']]
        rgb=core.processing_image(args.intake/r['source'])
        teacher=np.array(Image.open(args.teacher_run/r['mask']))>0
        panel_models=[name for name in ('rgb','joint','joint_all40','joint_all40_960','unet_baseline','unet_cldice') if (name,r['id']) in lookup]
        predictions={name:np.array(Image.open(root/'masks'/name/(r['id']+'.png')))>0 for name in panel_models}
        union=teacher.copy()
        for mask in predictions.values():
            union|=mask
        yy,xx=np.nonzero(union)
        box=(max(0,int(xx.min())-20),max(0,int(yy.min())-20),min(rgb.shape[1],int(xx.max())+21),min(rgb.shape[0],int(yy.max())+21)) if len(xx) else (0,0,rgb.shape[1],rgb.shape[0])
        def crop(image):
            x0,y0,x1,y1=box
            return image[y0:y1,x0:x1]
        teacher_image=rgb.copy(); teacher_image[~teacher]=0
        display=[('Source',rgb),('Accepted SAM teacher' if r['teacher_accepted'].lower()=='true' else 'Rejected SAM diagnostic',teacher_image)]
        labels=dict(rgb='RGB-only quarter25',joint='Joint quarter25',joint_all40='Joint all40@640',
                    joint_all40_960='Same all40@960',unet_baseline='Dense BCE+Dice',unet_cldice='Dense +soft-clDice')
        for name in panel_models:
            label=labels[name]
            masked=rgb.copy(); masked[~predictions[name]]=0
            metric=lookup[(name,r['id'])]
            display.append((f'{label}\nIoU {float(metric["iou"]):.3f}; endpoint3 {float(metric["endpoint_recall_px3"]):.2f}' if metric['endpoint_recall_px3'] else f'{label}\nIoU {float(metric["iou"]):.3f}',masked))
        errors=np.zeros_like(rgb); errors[teacher & predictions['joint_all40']]=[80,160,120]
        errors[teacher & ~predictions['joint_all40']]=[255,70,70]
        errors[~teacher & predictions['joint_all40']]=[70,140,255]
        display.append(('All40 overlap / lost / extra\ngreen / red / blue',errors))
        columns=3; nrows=int(np.ceil(len(display)/columns))
        fig,axes=plt.subplots(nrows,columns,figsize=(13,4*nrows),squeeze=False)
        for axis,(title,image) in zip(axes.ravel(),display):
            axis.imshow(crop(image)); axis.set_title(title,fontsize=10); axis.axis('off')
        for axis in axes.ravel()[len(display):]:
            axis.axis('off')
        t,v,c,accepted=condition
        fig.suptitle(f'{t}, recorded visit {v}, {c}; lowest all40 teacher IoU in this condition\n'
            f'{r["session"]}, view {r["view"]}; teacher accepted={accepted}; shared union crop with20px padding',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.89))
        path=panels/f'{t}_visit{v}_{c}_teacher{accepted}.png'
        fig.savefig(path,dpi=150,facecolor='white'); plt.close(fig)
        panel_records.append(dict(treatment=t,measurement=v,camera=c,teacher_accepted=accepted,
            id=r['id'],source=r['source'],panel=str(path),condition_frames=len(rows),
            selection='minimum final all40 teacher IoU within condition; fixed tie-break by id',
            crop_xyxy=box,all40_iou=float(chosen['iou'])))
    core.write_json(args.output/'failure_panels_manifest.json',dict(status='completed',panels=panel_records,
        planned_test_conditions=30,conditions_without_accepted_teacher=[r for r in coverage if not r['accepted_teacher_frames']],
        limits='Worst teacher-agreement examples after model choices were frozen. Rejected SAM masks are diagnostics, not truth.'))
    accepted=[r for r in summaries if r['teacher_accepted'].lower()=='true']
    text=['<!doctype html><meta charset="utf-8"><title>Maize foreground student</title>',
          '<style>body{max-width:1200px;margin:35px auto;font:16px Arial;line-height:1.5}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}img{width:100%}</style>',
          '<h1>Whole-plant foreground student</h1><p>YOLO26 models trained from automatic SAM masks. All displayed test frames come from one genotype excluded from training and validation. These scores measure teacher agreement; they do not measure expert foreground or leaf-instance accuracy.</p>',
          '<table><tr><th>Model</th><th>Camera</th><th>Frames</th><th>Acquisitions</th><th>Mean IoU</th><th>Mean Dice</th><th>Endpoint support3px</th><th>Thin support0px</th><th>Nonempty</th></tr>']
    for row in sorted(accepted,key=lambda r:(r['camera'],r['model'])):
        text.append('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in
            [row['model'],row['camera'],row['frames'],row['acquisitions'],f"{row['mean_iou']:.3f}",f"{row['mean_dice']:.3f}",
             f"{row['mean_endpoint_recall_px3']:.3f}" if row['mean_endpoint_recall_px3'] is not None else 'missing',
             f"{row['mean_thin_skeleton_recall_px0']:.3f}" if row['mean_thin_skeleton_recall_px0'] is not None else 'missing',
             f"{row['nonempty_fraction']:.1%}"] )+'</tr>')
    text.append('</table><p>Frames and repeated views are dependent technical observations. Endpoint and thin-region scores refer to an unreviewed teacher skeleton, including spurs and boundary irregularities; they are not biological tip recall. Legacy foreground is a union from two camera-specific weak instance models, trained on89images from the validation genotype; it does not establish biological leaf tracking.</p><p><a href="runtime_benchmark.json">Actual CPU/GPU runtime evidence</a> | <a href="test_condition_coverage.csv">All30condition denominators, including missing source frames</a></p><h2>Condition disagreements</h2>')
    for condition in coverage:
        if not condition['accepted_teacher_frames']:
            text.append(f'<p>{condition["treatment"]}, visit{condition["measurement"]}, {condition["camera"]}: no accepted source example; available{condition["available_frames"]}/160planned frames. No primary panel is fabricated.</p>')
    for panel in panel_records:
        text.append(f'<h3>{html.escape(panel["treatment"])} / visit{panel["measurement"]} / {panel["camera"]} / teacher accepted{panel["teacher_accepted"]}</h3><img src="{Path(panel["panel"]).relative_to(args.output).as_posix()}" alt="Source, SAM teacher and three student foregrounds with lost and extra pixels">')
    (args.output/'foreground_lab.html').write_text('\n'.join(text),encoding='utf8')
    core.write_json(args.output/'foreground_summary.json',dict(status='completed',test_summary=summaries,
        failure_panels=len(panel_records),test_manifest_sha256=core.digest(root/'manifest.json'),
        test_frame_metrics_sha256=core.digest(root/'frame_metrics.csv'),summary_source_sha256=core.digest(__file__),
        primary_test_genotype=manifest['evaluated_genotypes'],all_angles=manifest['all_angles']))
    print('summary and failure panels completed',flush=True)


if __name__=='__main__':
    main()
