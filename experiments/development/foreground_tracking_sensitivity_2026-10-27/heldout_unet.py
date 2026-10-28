"""Replay frozen U-Net test masks through the same genotype-held-out tracking subset."""
import argparse,csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
from tracking_sensitivity import sha,read,dump,table,truth


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['foreground','full-run','teacher-tracking','tracking-source','full-source','yolo-tracking','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);test=a.foreground/'evaluation/comparison_test';manifest=json.loads((test/'manifest.json').read_text())
    if manifest['status']!='completed' or not manifest['all_angles'] or sha(test/'frame_metrics.csv')!=manifest['frame_metrics_sha256']:raise ValueError('Completed frozen all-angle test required')
    inventory=read(a.foreground/'all_views/frame_inventory.csv');rows=[r for r in inventory if r['split']=='test' and r['camera']=='rgb' and truth(r['teacher_accepted'])]
    genotypes={r['genotype'] for r in rows}
    if genotypes!={'Os7798/2'} or len({r['id'] for r in rows})!=len(rows):raise ValueError('Unexpected held-out identity population')
    metrics={(r['model'],r['id']):r for r in read(test/'frame_metrics.csv')};teacher_table=a.full_run/'extraction/view_traits.csv';summaries=[]
    for arm in ['unet_baseline','unet_cldice']:
        cache=a.output/'caches'/arm;cache.mkdir(parents=True,exist_ok=True);(cache/'masks').mkdir(exist_ok=True)
        spec=manifest['models'][arm]
        if sha(Path(spec['path']))!=spec['sha256']:raise ValueError('Frozen model bytes changed')
        operator=dict(model=arm,model_sha256=spec['sha256'],teacher_table_sha256=sha(teacher_table),test_manifest_sha256=sha(test/'manifest.json'),scope='Frozen genotype-held-out test predictions; no new inference or selection')
        operator['signature']=hashlib.sha256(json.dumps(operator,sort_keys=True).encode()).hexdigest();bound=[]
        for r in rows:
            source=test/'masks'/arm/(r['id']+'.png');teacher=a.full_run/r['mask']
            if sha(teacher)!=r['mask_sha256']:raise ValueError('Teacher mask bytes changed')
            with Image.open(source) as im:pred=np.asarray(im.convert('L'))>0
            with Image.open(teacher) as im:target=np.asarray(im.convert('L'))>0
            if target.shape!=pred.shape:raise ValueError('Prediction grid mismatch')
            recorded=metrics[(arm,r['id'])];intersection=int((pred&target).sum());union=int((pred|target).sum());iou=intersection/union if union else 1.
            if int(recorded['predicted_pixels'])!=int(pred.sum()) or int(recorded['teacher_pixels'])!=int(target.sum()) or abs(float(recorded['iou'])-iou)>1e-12:raise ValueError('Stored test mask disagrees with frozen metrics')
            path=cache/'masks'/(r['id']+'.png');shutil.copy2(source,path)
            bound.append(dict(r,signature=operator['signature'],teacher_mask=r['mask'],student_mask=path.relative_to(cache).as_posix(),student_mask_sha256=sha(path),nonempty=bool(pred.any()),processing_width=pred.shape[1],processing_height=pred.shape[0]))
        table(cache/'frames.csv',bound);dump(cache/'operator.json',operator);dump(cache/'status.json',dict(status='completed',expected_frames=len(bound),cached_frames=len(bound),frames_sha256=sha(cache/'frames.csv'),operator_signature=operator['signature']))
        destination=a.output/'tracking'/arm
        command=[sys.executable,str(Path(__file__).with_name('tracking_sensitivity.py')),'--full-run',str(a.full_run),'--teacher-tracking',str(a.teacher_tracking),'--cache',str(cache),'--output',str(destination),'--tracking-source',str(a.tracking_source),'--full-source',str(a.full_source),'--genotype','Os7798/2']
        with (a.output/(arm+'.log')).open('w',encoding='utf-8') as stream:run=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if run.returncode:raise RuntimeError('Tracking subprocess failed: '+arm)
        summary=json.loads((destination/'summary.json').read_text());summaries.append(dict(model=arm,**summary))
    yolo=[r for r in read(a.yolo_tracking/'sessions.csv') if r['genotype']=='Os7798/2'];source_counts={tuple(r[k] for k in ['treatment','genotype','replicate','visit']):int(r['matched_source_views']) for r in yolo}
    for arm in ['unet_baseline','unet_cldice']:
        actual=read(a.output/'tracking'/arm/'sessions.csv')
        if {tuple(r[k] for k in ['treatment','genotype','replicate','visit']):int(r['matched_source_views']) for r in actual}!=source_counts:raise ValueError('Matched source support differs between models')
    result=dict(status='passed',scope='Fixed same-source held-out-genotype downstream mask sensitivity; no anatomical accuracy or model reselection',genotype='Os7798/2',acquisitions=len(yolo),matched_source_views=sum(source_counts.values()),
        teacher_nodes=sum(int(r['teacher_nodes']) for r in yolo),yolo_nodes=sum(int(r['student_nodes']) for r in yolo),unet=summaries,test_manifest_sha256=sha(test/'manifest.json'),source_sha256=sha(Path(__file__)),stored_prediction_metric_checks=len(rows)*2)
    dump(a.output/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
