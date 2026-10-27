"""Cross-date association of automatically recovered external terminal scaffolds.

Unlike the provided-instance association benchmark, these objects come from the
silhouette-only terminal tracker. Known rendering cameras, foreground semantics
and physical plant continuity remain supplied. Labels are scoring-only.
"""
import argparse,json
from pathlib import Path
import numpy as np
from architecture_analysis import read,write,dump,sha
from architecture_external_pheno4d import associate,score

def source_gauge(cloud):
    points=cloud[cloud[:,4]>0,:3];low=np.percentile(points[:,2],2);basal=points[points[:,2]<=np.percentile(points[:,2],8)];return np.r_[np.median(basal[:,:2],axis=0),low],float(points[:,2].max()-low)

def run(args):
    out=args.run/'architecture'/'external_pheno4d'/'scaffold';rows=read(out/'terminal_scaffold_point_support.csv');manifest=json.loads((args.external/'download_manifest.json').read_text(encoding='utf-8'));by={}
    for f in manifest['files']:by.setdefault(Path(f['path']).parent.name,[]).append(f)
    results=[];linksrows=[]
    for plant,pair in sorted(by.items()):
        pair.sort(key=lambda r:r['path']);clouds=[];gauges=[]
        for f in pair:
            p=args.external/f['path']
            if sha(p)!=f['sha256']:raise ValueError('External source hash mismatch')
            c=np.loadtxt(p,dtype='float32');clouds.append(c);gauges.append(source_gauge(c))
        earlier_origin,earlier_height=gauges[0];feature_sets=[];truth_sets=[];token_sets=[]
        for record,(origin,height) in zip(pair,gauges):
            rr=[r for r in rows if r['plant']==plant and r['scan']==Path(record['path']).stem];xyz=np.array([[float(r[k]) for k in ['point_x_native_horizontal','point_y_native_depth','point_z_native_up']] for r in rr]);xyz=xyz.reshape(-1,3);shared=(xyz*height+origin-earlier_origin)/earlier_height;feature_sets.append(shared);truth_sets.append([int(r['mapped_id_evaluation_only']) for r in rr]);token_sets.append([int(r['track_id']) for r in rr])
        # Geometry-only crossdate engine never receives numeric instance IDs.
        cost=np.linalg.norm(feature_sets[0][:,None,:]-feature_sets[1][None,:,:],axis=2)/.2;links,a,b=associate(cost,.75);metrics=score(links,a,b,*truth_sets);all_common=set(clouds[0][clouds[0][:,4]>0,4].astype(int))&set(clouds[1][clouds[1][:,4]>0,4].astype(int));results.append(dict(plant=plant,earlier_recovered_tracks=len(feature_sets[0]),later_recovered_tracks=len(feature_sets[1]),all_true_common_instances=len(all_common),**metrics,end_to_end_recall_given_known_render_camera_and_foreground=metrics['correct_links']/max(len(all_common),1)))
        for i,j in links:linksrows.append(dict(plant=plant,earlier_track=token_sets[0][i],later_track=token_sets[1][j],geometry_cost=float(cost[i,j]),earlier_truth_for_scoring=truth_sets[0][i],later_truth_for_scoring=truth_sets[1][j],correct=truth_sets[0][i]==truth_sets[1][j]))
    write(out/'automatic_scaffold_crossdate_scores.csv',results);write(out/'automatic_scaffold_crossdate_links.csv',linksrows);correct=sum(r['correct_links'] for r in results);pred=sum(r['predicted_links'] for r in results);recovered=sum(r['true_common_instances'] for r in results);alltrue=sum(r['all_true_common_instances'] for r in results);summary=dict(plant_pairs=len(results),correct_crossdate_links=correct,predicted_crossdate_links=pred,recovered_true_common_instances=recovered,all_true_common_instances=alltrue,conditional_precision=correct/max(pred,1),recall_among_common_recovered_instances=correct/max(recovered,1),recall_among_all_true_common_instances=correct/max(alltrue,1),script_sha256=sha(__file__),scaffold_input_sha256=sha(out/'terminal_scaffold_point_support.csv'),scope='Objects automatically recovered from rendered silhouettes under known static geometry/cameras and supplied foreground semantics. True IDs used for scoring only. This joins two component stages, not validation of CSI RGB/SAM/unknowncamera or plantcontinuity.')
    dump(out/'automatic_scaffold_crossdate_summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--external',type=Path,required=True);p.add_argument('--run',type=Path,required=True);run(p.parse_args())
