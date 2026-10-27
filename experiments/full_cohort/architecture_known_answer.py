"""Synthetic known-answer stress tests, separate from plant-data validation."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from architecture_analysis import write,dump,sha,visual_hull,project_hull
from architecture_tracking import fit_tracks,predict,design

def terminal_case(seed,scenario):
    rng=np.random.default_rng(seed);truth=[dict(coef=[0,.32,0],height=.85),dict(coef=[0,-.22,.13],height=.57),dict(coef=[0,.12,-.20],height=.30)]
    if scenario=='rank_ambiguous':truth[1]=dict(truth[0])
    candidates={};truthpoints={}
    for v in range(40):
        points=[];t=v*2*np.pi/40
        for leaf,q in enumerate(truth):
            x=float(design([v])[0]@q['coef']);y=q['height']
            if scenario=='perspective_mismatch':
                depth=-q['coef'][1]*np.sin(t)+q['coef'][2]*np.cos(t);x=x/(1-.7*depth);y=y/(1-.7*depth)
            truthpoints[(v,leaf)]=np.array([x,y])
            if scenario in ['occlusion_noise','distractors','rank_ambiguous'] and rng.random()<.30:continue
            noise=0 if scenario=='clean' else .008
            points.append(dict(x=x+rng.normal(0,noise),y=y+rng.normal(0,noise),root_length=y+.3,truth=leaf))
        if scenario=='distractors':
            for _ in range(3):points.append(dict(x=rng.uniform(-.5,.5),y=rng.uniform(.1,1),root_length=rng.uniform(.3,1.4),truth=-1))
        candidates[v]=points
    train={v:cc for v,cc in candidates.items() if v%4!=2};tracks=fit_tracks(train)
    cost=np.array([[np.mean([np.linalg.norm(predict(q,v)-truthpoints[(v,j)]) for v in range(40)]) for j in range(len(truth))] for q in tracks])
    matches=[]
    if len(tracks):
        ii,jj=linear_sum_assignment(cost)
        matches=[(int(i),int(j),float(cost[i,j])) for i,j in zip(ii,jj) if cost[i,j]<.07]
    purity=[]
    for q in tracks:
        labels=[train[a['view']][a['candidate_id']]['truth'] for a in q['assignments']];counts=[labels.count(i) for i in range(len(truth))];purity.append(max(counts)/len(labels))
    heldout=[]
    for i,j,d in matches:
        heldout.extend(float(np.linalg.norm(predict(tracks[i],v)-truthpoints[(v,j)])) for v in range(40) if v%4==2)
    return dict(scenario=scenario,seed=seed,true_trajectories=len(truth),recovered_hypotheses=len(tracks),matched_truth=len(matches),precision=len(matches)/max(len(tracks),1),recall=len(matches)/len(truth),median_assignment_purity=float(np.median(purity)) if purity else None,median_heldout_true_tip_error=float(np.median(heldout)) if heldout else None,ambiguous_geometry=scenario=='rank_ambiguous',interpretation='Known synthetic oracle terminals; no image segmentation or botanical validation')

def hull_case(name,resolution=64):
    xy=np.linspace(-.8,.8,resolution);yy=np.linspace(0,1.15,resolution);x,y,z=np.meshgrid(xy,yy,xy,indexing='ij');grid=(x,y,z)
    if name=='solid_cylinder':truth=(x*x+z*z<.23**2)&(y>.04)&(y<1.0)
    else:
        # Three thin, overlapping geometric ribbons plus a central stem. These
        # are finite voxel solids solely to expose hull-envelope ambiguity.
        truth=(x*x+z*z<.025**2)&(y<.9)
        for az,height,length in [(0,.3,.45),(1.9,.5,.5),(3.8,.7,.4)]:
            along=x*np.cos(az)+z*np.sin(az);across=-x*np.sin(az)+z*np.cos(az)
            truth|=(along>0)&(along<length)&(abs(across)<.025)&(abs(y-height-.3*along)<.018)
    shape=(160,192);axis=96;base=150;scale=130
    masks={v:project_hull(truth,grid,v,shape,axis,base,scale) for v in range(40) if v%4!=2}
    hull,hg=visual_hull(masks,axis,base,scale,resolution=resolution);iou3d=float((hull&truth).sum()/max((hull|truth).sum(),1));scores=[]
    for v in range(40):
        if v%4!=2:continue
        target=project_hull(truth,grid,v,shape,axis,base,scale);pred=project_hull(hull,hg,v,shape,axis,base,scale);scores.append(float((target&pred).sum()/max((target|pred).sum(),1)))
    return dict(scene=name,resolution=resolution,true_voxels=int(truth.sum()),hull_voxels=int(hull.sum()),voxel_iou=iou3d,median_heldout_silhouette_iou=float(np.median(scores)),volume_ratio=float(hull.sum()/max(truth.sum(),1)),interpretation='Discrete known-answer conditional orthographic projection; silhouette fit is not surface/organ truth')

def run(args):
    started=time.time();out=args.run/'architecture'/'known_answer';out.mkdir(parents=True,exist_ok=True)
    execution_sources={}
    for name in ['architecture_known_answer.py','architecture_analysis.py','architecture_tracking.py']:
        body=(Path(__file__).parent/name).read_bytes();(out/('execution_'+name)).write_bytes(body)
        import hashlib
        execution_sources[name]=hashlib.sha256(body).hexdigest()
    rows=[terminal_case(seed,scenario) for scenario in ['clean','occlusion_noise','distractors','perspective_mismatch','rank_ambiguous'] for seed in range(args.seeds)];write(out/'synthetic_terminal_trajectories.csv',rows)
    summaries=[]
    for scenario in sorted(set(r['scenario'] for r in rows)):
        rr=[r for r in rows if r['scenario']==scenario];s=dict(scenario=scenario,seeds=len(rr))
        for metric in ['precision','recall','median_assignment_purity','median_heldout_true_tip_error']:
            vals=[r[metric] for r in rr if r[metric] is not None];s[metric+'_median']=float(np.median(vals)) if vals else None;s[metric+'_min']=float(min(vals)) if vals else None
        summaries.append(s)
    hulls=[hull_case(name) for name in ['solid_cylinder','thin_ribbons']];write(out/'known_answer_hulls.csv',hulls);write(out/'synthetic_scenario_summary.csv',summaries)
    dump(out/'known_answer_summary.json',dict(terminal_scenarios=summaries,hulls=hulls,elapsed_seconds=time.time()-started,execution_source_sha256=execution_sources,scope='Synthetic assumptions and degeneracies only. Rank-ambiguous trajectories can fit perfectly while physical organ identity remains unidentified. Birth/loss unit test represents injected visible states, not inferred deaths/emergence in maize.'))
    # Exportable scientific figure, not a claim of real-data accuracy.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(10,4.1));labels=[s['scenario'].replace('_','\n') for s in summaries];xx=np.arange(len(labels));axes[0].bar(xx-.17,[s['precision_median'] for s in summaries],.34,label='hypothesis precision');axes[0].bar(xx+.17,[s['recall_median'] for s in summaries],.34,label='oracle trajectory recall');axes[0].set_xticks(xx,labels,fontsize=8);axes[0].set_ylim(0,1.05);axes[0].set_ylabel('Synthetic known-answer fraction');axes[0].legend(fontsize=8);axes[0].set_title('Unassisted terminal trajectories')
    xx=np.arange(2);axes[1].bar(xx-.17,[r['median_heldout_silhouette_iou'] for r in hulls],.34,label='held-out silhouette IoU');axes[1].bar(xx+.17,[r['voxel_iou'] for r in hulls],.34,label='true 3D voxel IoU');axes[1].set_xticks(xx,['Solid cylinder','Thin ribbons']);axes[1].set_ylim(0,1.05);axes[1].legend(fontsize=8);axes[1].set_title('Silhouette prediction vs 3D truth');fig.tight_layout();fig.savefig(out/'known_answer_stress_tests.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(summaries=summaries,hulls=hulls),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--seeds',type=int,default=30);run(p.parse_args())
