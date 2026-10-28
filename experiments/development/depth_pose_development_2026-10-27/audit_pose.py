"""Separate learned training consistency from nominal-camera extrapolation."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('intake','run','operators','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.operators));from depth_geometry import unproject,project
    diagnostics=rows(args.intake/'segmentation_triage/view_diagnostics.csv')
    lookup={(r['treatment'],r['measurement'],r['camera'],int(r['view'])):r for r in diagnostics if r['genotype']=='Os6251/15' and r['replicate']=='1'}
    results=[];pose=[];files=[]
    for camera in ('rgb','noir'):
        for treatment in 'CPS':
            for visit in range(1,6):
                case=f'{treatment}{visit}_L5_rep1_{camera}'
                path=args.run/'depth_trial'/(case+'__learned_pose_aligned_to_training_centres.npz')
                if not path.exists():continue
                config=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                with np.load(path,allow_pickle=False) as data:
                    depth,extrinsics,intrinsics=data['depth'],data['extrinsics'],data['intrinsics']
                masks=[]
                for j,view in enumerate((0,10,20)):
                    row=lookup[(treatment,str(visit),camera,view)];maskpath=args.intake/row['mask']
                    if sha(maskpath)!=config['source'][j]['mask_sha256']:raise ValueError('Depth diagnostic mask changed')
                    with Image.open(maskpath) as im:mask=np.array(im)>0
                    masks.append(cv2.resize(mask.astype('uint8'),(depth.shape[2],depth.shape[1]),interpolation=cv2.INTER_NEAREST)>0)
                clouds=[unproject(d,cv2.erode(m.astype('uint8'),np.ones((3,3),np.uint8))>0,k,e,2) for d,m,k,e in zip(depth,masks,intrinsics,extrinsics)]
                for a in range(3):
                    for b in range(3):
                        prediction=project(clouds[a],intrinsics[b],extrinsics[b],masks[b].shape,1);truth=masks[b]
                        intersection=int(np.sum(prediction&truth));union=int(np.sum(prediction|truth))
                        results.append(dict(case=case,camera=camera,treatment=treatment,visit=visit,source_view=[0,10,20][a],target_view=[0,10,20][b],same_camera=a==b,iou=intersection/max(union,1),precision=intersection/max(int(prediction.sum()),1),recall=intersection/max(int(truth.sum()),1)))
                centres=np.array([-e[:3,:3].T@e[:3,3] for e in extrinsics])
                for a,b in [(0,1),(1,2),(0,2)]:
                    rotation=extrinsics[a,:3,:3]@extrinsics[b,:3,:3].T
                    angle=float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))))
                    pose.append(dict(case=case,camera=camera,source_view=[0,10,20][a],target_view=[0,10,20][b],relative_rotation_degrees=angle,nominal_rotation_degrees=90 if b-a==1 else 180,centre_distance=float(np.linalg.norm(centres[a]-centres[b]))))
                files.append(dict(path=str(path),sha256=sha(path)))
    for name,data in [('training_reprojection.csv',results),('camera_pose_diagnostics.csv',pose)]:
        with (args.output/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    summary=dict(cases=len(files),same_camera_median_iou={c:float(np.median([r['iou'] for r in results if r['camera']==c and r['same_camera']])) for c in ('rgb','noir')},
        cross_training_camera_median_iou={c:float(np.median([r['iou'] for r in results if r['camera']==c and not r['same_camera']])) for c in ('rgb','noir')},
        adjacent_learned_rotation_median_degrees={c:float(np.median([r['relative_rotation_degrees'] for r in pose if r['camera']==c and r['nominal_rotation_degrees']==90])) for c in ('rgb','noir')},
        source_sha256=sha(Path(__file__)),operator_sha256=sha(args.operators/'depth_geometry.py'),input_files=files,
        scope='Debug audit of three-view learned geometry. Training reprojection is not heldout accuracy. Nominal90degree intervals remain assumptions; no camera calibration or anatomy truth.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if k!='input_files'},indent=2))


if __name__=='__main__':main()
