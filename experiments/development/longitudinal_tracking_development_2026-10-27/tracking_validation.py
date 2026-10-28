"""Independent denominator/support audit and synthetic perturbation experiments."""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from tracking_graph_core import fit_cycles,associate_visits,point,build_candidate_graph
from tracking_development import read,write,dump,sha,log


def external_audit(output,external):
    out=output/'external';manifest=json.loads((external/'download_manifest.json').read_text());cases=read(out/'cases.csv');points=read(out/'conditional_points.csv');scores=read(out/'heldout_scores.csv')
    supports=[];findings=[];true_total=0;common={};mapped_total=0;observable_mapped=0;purities=[]
    for rec in manifest['files']:
        path=external/rec['path'];stem=path.stem
        if sha(path)!=rec['sha256']:raise ValueError('Source mismatch')
        cloud=np.loadtxt(path,dtype='float32');fg=cloud[cloud[:,4]>0];p=fg[:,:3];labels=fg[:,4].astype(int);ids=sorted(set(labels));true_total+=len(ids);common.setdefault(path.parent.name,[]).append(set(ids))
        low=np.percentile(p[:,2],2);basal=p[p[:,2]<=np.percentile(p[:,2],8)];origin=np.r_[np.median(basal[:,:2],axis=0),low];height=float(p[:,2].max()-low);xyz=(p-origin)/height
        trees={i:cKDTree(xyz[labels==i]) for i in ids};case=next(r for r in cases if r['scan']==stem);observable=set(json.loads(case['observable_ids_evaluation_only']))
        mapping=json.loads((out/'scoring_maps'/f'{stem}.json').read_text());frozen=json.loads((out/'frozen'/f'{stem}.json').read_text());counts=np.array(mapping['training_ownership_counts']);used=list(mapping['mapping'].values())
        if len(used)!=len(set(used)):findings.append('Nonunique map '+stem)
        mapped_total+=len(used);observable_mapped+=len(set(used)&observable)
        if any(v%4==2 for v in frozen['train_views']):findings.append('Training leakage '+stem)
        for r in [r for r in points if r['scan']==stem]:
            q=np.array([float(r['x']),float(r['depth']),float(r['up'])]);label=int(r['mapped_id_evaluation_only']);ti=int(r['node_id'].split(':t')[-1]);track=frozen['tracks'][ti]
            correct=float(trees[label].query(q)[0]) if label in trees else None
            wrong=min(float(t.query(q)[0]) for i,t in trees.items() if i!=label)
            purity=float(counts[ti,mapping['true_ids'].index(label)]/track['training_support']) if label else 0.;purities.append(purity)
            supports.append(dict(**r,training_owner_purity=purity,correct_instance_support_distance=correct,
                nearest_wrong_instance_support_distance=wrong,correct_support_closer=bool(correct is not None and correct<wrong)))
    if any(int(r['view'])%4!=2 for r in scores):findings.append('Nonheldout scoring view')
    if len({(r['node_id'],r['view']) for r in scores})!=len(scores):findings.append('Duplicate evaluation rows')
    correct=[r['correct_instance_support_distance'] for r in supports if r['correct_instance_support_distance'] is not None]
    fullcommon=sum(len(a&b) for a,b in common.values())
    result=dict(status='passed' if not findings else 'failed',findings=findings,scans=len(cases),plants=len(common),
        true_instance_dates_from_original_labels=true_total,true_common_from_original_labels=fullcommon,
        distinct_mapped_instance_dates=mapped_total,mapped_terminal_observable_instance_dates=observable_mapped,
        mapped_track_training_purity_atleast_point8=sum(p>=.8 for p in purities),total_track_hypotheses=len(purities),
        points_with_correct_instance_support=len(correct),correct_support_closer=sum(r['correct_support_closer'] for r in supports),
        median_correct_point_support_distance=float(np.median(correct)),p90_correct_point_support_distance=float(np.percentile(correct,90)),
        maximum_correct_point_support_distance=float(max(correct)),
        units='Known rendering scan-height gauge; observed point support distance, not anatomical tip error',
        script_sha256=sha(__file__))
    write(out/'point_support_audit.csv',supports);dump(out/'independent_denominator_support_audit.json',result);log(output,'independent_external_audit',**result)
    if findings:raise ValueError(findings)
    return result


BASE=np.array([[.32,.13,.9],[-.31,.09,.64],[.12,-.26,.39],[-.13,-.12,.19]])


def synthetic_observations(points,phase=0,occlusion=False,noise=0,clutter=False):
    rng=np.random.default_rng(812);theta=np.radians(phase);rot=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    p=np.array(points).copy();p[:,:2]=p[:,:2]@rot.T;out={}
    for v in range(40):
        if v%4==2:continue
        angle=v*2*np.pi/40;out[v]=[]
        for i,(x,z,y) in enumerate(p):
            if occlusion and i==1 and v not in {0,3,7,11,15,19,23,27,31,35}:continue
            out[v].append(dict(x=x*np.cos(angle)+z*np.sin(angle)+rng.normal(0,noise),y=y+rng.normal(0,noise),root_length=float(np.linalg.norm([x,z,y]))))
        if clutter:out[v].append(dict(x=rng.uniform(-.5,.5),y=rng.uniform(.05,1.1),root_length=rng.uniform(.3,1.3)))
    return out,p


def score_mapping(nodes,truth_points,ids):
    if not nodes:return []
    cost=np.linalg.norm(np.array([point(q) for q in nodes])[:,None,:]-truth_points[None,:,:],axis=2)
    ii,jj=linear_sum_assignment(cost);mapping={int(i):ids[int(j)] for i,j in zip(ii,jj) if cost[i,j]<.08}
    return [mapping.get(i,0) for i in range(len(nodes))]


def synthetic_run(output):
    out=output/'synthetic';out.mkdir(parents=True,exist_ok=True);rows=[]
    for name,phase,occlusion,noise,clutter,hidden,new in [
        ('clean',0,False,0,False,False,False),('phase90',90,False,0,False,False,False),
        ('phase171',171,False,0,False,False,False),('occluded20of30',0,True,0,False,False,False),
        ('phase99_noisy_clutter',99,True,.008,True,False,False),('unobserved_later',0,False,0,False,True,False),
        ('newly_observed',0,False,0,False,False,True),('phase90_unobserved_later',90,True,.003,False,True,False)]:
        ca,pa=synthetic_observations(BASE);later=BASE.copy();ids_a=[1,2,3,4];ids_b=ids_a[:]
        if hidden:later=later[[0,2,3]];ids_b=[1,3,4]
        if new:later=np.vstack([later,[.42,.1,.5]]);ids_b.append(5)
        cb,pb=synthetic_observations(later,phase,occlusion,noise,clutter);a=fit_cycles(ca);b=fit_cycles(cb)
        for i,q in enumerate(a):q['node_id']=f'{name}:a{i}'
        for i,q in enumerate(b):q['node_id']=f'{name}:b{i}'
        ta=score_mapping(a,pa,ids_a);tb=score_mapping(b,pb,ids_b);r=associate_visits(a,b)
        aa={q['node_id']:t for q,t in zip(a,ta)};bb={q['node_id']:t for q,t in zip(b,tb)}
        correct=sum(aa[e['source']]>0 and aa[e['source']]==bb[e['target']] for e in r['accepted_links']);expected=len(set(ids_a)&set(ids_b));pred=len(r['accepted_links'])
        row=dict(case=name,phase_injected=phase,best_phase=r['best_phase_degrees'],earlier_truth=len(ids_a),later_truth=len(ids_b),earlier_tracks=len(a),later_tracks=len(b),true_common=expected,
            predicted_links=pred,correct_links=correct,precision=correct/max(pred,1),recall_full=correct/max(expected,1),retained_phases=len(r['retained_phases_degrees']))
        rows.append(row);dump(out/f'{name}.json',dict(summary=row,earlier_frozen=a,later_frozen=b,association=r,truth_used_for_scoring_only=dict(earlier=ta,later=tb)))
    # Structural missing state with successful gap-two recovery.
    nodes=fit_cycles(synthetic_observations(BASE)[0])
    for i,q in enumerate(nodes):q['node_id']=f'gap:a{i}'
    later=[dict(q,node_id=f'gap:b{i}') for i,q in enumerate(nodes)]
    g=build_candidate_graph('synthetic_missing_middle',[dict(visit=1,nodes=nodes),dict(visit=2,nodes=[]),dict(visit=3,nodes=later)])
    dump(out/'missing_middle_graph.json',g);write(out/'scores.csv',rows)
    summary=dict(cases=len(rows),known_common_links=sum(r['true_common'] for r in rows),predicted_links=sum(r['predicted_links'] for r in rows),
        correct_links=sum(r['correct_links'] for r in rows),missing_middle_selected_gap_links=sum(e['selected'] and e['visit_gap']==2 for e in g['edges']),
        script_sha256=sha(__file__),scope='Designed numeric rotation/phase/occlusion controls, not independent biological validation')
    dump(out/'summary.json',summary);log(output,'synthetic_controls_completed',**summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--external',type=Path,required=True);a=p.parse_args();external_audit(a.output,a.external);synthetic_run(a.output)
