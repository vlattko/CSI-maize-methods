"""Test frozen harmonic terminal point scaffolds against real instance support.

The known external orthographic camera fixes inversion from sinusoidal image
coordinates to scale-free 3D points. Distances are to observed instance point
support, not independently annotated anatomical tips. IDs enter scoring only.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from architecture_analysis import read,write,dump,sha

def terminal_point(track,gauge,render_axis=192,render_base=360,render_scale=300):
    b=np.asarray(track['x_coefficient']);scale=gauge['height']
    return np.array([b[1]*scale/render_scale,b[2]*scale/render_scale,(track['height']*scale-gauge['base']+render_base)/render_scale])

def run(args):
    start=time.time();out=args.run/'architecture'/'external_pheno4d'/'scaffold';out.mkdir(parents=True,exist_ok=True);rendered=args.run/'architecture'/'external_pheno4d'/'rendered';mapping=read(rendered/'rotational_track_oracle_purity.csv');manifest=json.loads((args.external/'download_manifest.json').read_text(encoding='utf-8'));rows=[];case_rows=[];example=None
    for record in manifest['files']:
        path=args.external/record['path'];stem=path.stem;modelpath=rendered/f'frozen_{stem}.json';model=json.loads(modelpath.read_text(encoding='utf-8'))
        if sha(path)!=record['sha256']:raise ValueError('External source hash mismatch')
        cloud=np.loadtxt(path,dtype='float32');foreground=cloud[:,4]>0;p=cloud[foreground,:3];labels=cloud[foreground,4].astype(int);low=np.percentile(p[:,2],2);basal=p[p[:,2]<=np.percentile(p[:,2],8)];origin=np.r_[np.median(basal[:,:2],axis=0),low];height=float(p[:,2].max()-low);points=(p-origin)/height;ids=sorted(int(i) for i in np.unique(labels));trees={label:cKDTree(points[labels==label]) for label in ids};case_points=[]
        for track in model['tracks']:
            evaluation=next(r for r in mapping if r['scan']==stem and int(r['track_id'])==track['track_id']);label=int(evaluation['mapped_id_evaluation_only']);point=terminal_point(track,model);correct=float(trees[label].query(point)[0]) if label in trees else None;wrong=min([float(tree.query(point)[0]) for other,tree in trees.items() if other!=label] or [np.nan]);expected_intercept=(192-model['axis'])/model['height'];residual=track['x_coefficient'][0]-expected_intercept
            rows.append(dict(plant=path.parent.name,scan=stem,track_id=track['track_id'],mapped_id_evaluation_only=label,point_x_native_horizontal=float(point[0]),point_y_native_depth=float(point[1]),point_z_native_up=float(point[2]),correct_instance_support_distance=correct,nearest_wrong_instance_support_distance=wrong if np.isfinite(wrong) else None,correct_instance_closer=bool(correct is not None and correct<wrong),training_oracle_purity=float(evaluation['training_oracle_purity']),training_xy_rmse=track['training_rmse'],horizontal_intercept_residual_train_height=float(residual),input_model_sha256=sha(modelpath),source_sha256=record['sha256'],distance_units='Rendered scan height gauge; observed point support distance, not anatomical tip error'))
            case_points.append((label,point))
        case_rows.append(dict(plant=path.parent.name,scan=stem,true_tip_instances=len(ids),terminal_scaffold_points=len(model['tracks'])))
        if stem=='M01_0324_a':example=(points,labels,case_points)
    write(out/'terminal_scaffold_point_support.csv',rows);write(out/'scaffold_cases.csv',case_rows)
    correct=[r['correct_instance_support_distance'] for r in rows if r['correct_instance_support_distance'] is not None];wrong=[r['nearest_wrong_instance_support_distance'] for r in rows if r['nearest_wrong_instance_support_distance'] is not None];summary=dict(scans=len(case_rows),plants=len(set(r['plant'] for r in case_rows)),true_tip_instances=sum(r['true_tip_instances'] for r in case_rows),terminal_scaffold_points=len(rows),mapped_points_with_correct_instance_support=len(correct),correct_instance_closer_points=sum(r['correct_instance_closer'] for r in rows),median_correct_instance_support_distance=float(np.median(correct)),p90_correct_instance_support_distance=float(np.percentile(correct,90)),median_nearest_wrong_instance_support_distance=float(np.median(wrong)),median_abs_intercept_residual_train_height=float(np.median([abs(r['horizontal_intercept_residual_train_height']) for r in rows])),input_manifest_sha256=sha(args.external/'download_manifest.json'),script_sha256=sha(__file__),elapsed_seconds=time.time()-start,claim='Frozen training-only harmonic model inverted under known orthographic render camera. Scale-free3D terminal hypothesis points; distances to observed instance support, not true anatomical tip errors. No meshes, insertions, physicalunits or hidden surfaces inferred.')
    dump(out/'external_scaffold_summary.json',summary)
    if example:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':11});fig,axes=plt.subplots(1,3,figsize=(11,4));points,labels,scaffold=example;rng=np.random.default_rng(2026);ix=rng.choice(len(points),min(len(points),12000),replace=False)
        for ax,(a,b,name) in zip(axes,[(0,2,'Horizontal / up'),(1,2,'Depth / up'),(0,1,'Horizontal / depth')]):
            ax.scatter(points[ix,a],points[ix,b],c=labels[ix],s=.3,cmap='tab10',alpha=.35,rasterized=True)
            for label,p in scaffold:ax.plot(p[a],p[b],'k*',markersize=9);ax.annotate(str(label),(p[a],p[b]),xytext=(4,4),textcoords='offset points')
            ax.set_title(name);ax.set_aspect('equal');ax.set_xlabel('Scan-height gauge')
        fig.suptitle('Frozen terminal scaffold on real M01_0324 geometry\nStars: recovered points; colours/labels are oracle scoring annotations, not fitting inputs',fontsize=12);fig.tight_layout(rect=[0,0,1,.87]);fig.savefig(out/'terminal_scaffold_known_geometry.png',dpi=180);plt.close(fig)
    # CSI counterparts remain explicitly conditional, with no oracle distances.
    csi=args.run/'architecture'/args.output_name/'tracking/rotational_track_hypotheses.csv'
    if csi.exists():
        hypotheses=[]
        for r in read(csi):hypotheses.append({k:r[k] for k in ['treatment','measurement','genotype','replicate','session','track_id']}|dict(x=float(r['x_cos']),up=float(r['height']),depth=float(r['x_sin']),horizontal_bias=float(r['x_intercept']),units='Training silhouette-height orthographic gauge',status='Conditional terminal point scaffold; camera, physical identity and organ labels unvalidated'))
        write(args.run/'architecture'/args.output_name/'tracking/conditional_terminal_scaffold_points.csv',hypotheses)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--external',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output-name',default='');run(p.parse_args())
