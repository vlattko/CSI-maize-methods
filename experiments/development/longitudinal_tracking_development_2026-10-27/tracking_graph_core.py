"""Rotation-supported association hypotheses, with explicit alignment alternatives.

This module never consumes true organ IDs. Scores and stability fractions are
algorithmic sensitivity measures, not calibrated probabilities. Coordinates are
conditional on the supplied projection/gauge. Absence is never interpreted as
death; a newly observed path does not establish organ birth.
"""
import os
for _k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_k]='2'
import itertools
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
_THREAD_LIMITER=threadpool_limits(limits=2)


CONFIG=dict(cycle_gate=.055,minimum_support=8,minimum_support_fraction=.25,
            minimum_quadrants=3,max_tracks=12,assignment_gap=.75,
            phase_tolerance=.12,minimum_phase_link_fraction=.9,
            minimum_assignment_margin=.10,maximum_link_cost=1.2)


def design(views):
    t=np.asarray(views)*2*np.pi/40
    return np.column_stack([np.ones(len(t)),np.cos(t),np.sin(t)])


def predict(track,view):
    return np.array([design([view])[0]@np.asarray(track['x_coefficient']),track['height']])


def _assign(candidates,tracks,gate):
    assigned=[[] for _ in tracks]
    for v,cc in sorted(candidates.items()):
        if not cc or not tracks: continue
        xy=np.array([[c['x'],c['y']] for c in cc]); pp=np.array([predict(t,v) for t in tracks])
        d=np.linalg.norm(pp[:,None,:]-xy[None,:,:],axis=2)
        links,_,_,_=partial_assignment(d/gate,gap=.5)
        for ti,ci in links:
            if d[ti,ci]<gate: assigned[ti].append((v,ci,cc[ci],float(d[ti,ci])))
    return assigned


def _refit(selected):
    a=design([s[0] for s in selected]); xy=np.array([[s[2]['x'],s[2]['y']] for s in selected])
    coef=np.linalg.lstsq(a,xy[:,0],rcond=None)[0];height=float(np.median(xy[:,1]))
    for _ in range(3):
        residual=np.sqrt((a@coef-xy[:,0])**2+(height-xy[:,1])**2)
        med=float(np.median(residual)); mad=float(np.median(abs(residual-med)))
        weight=np.minimum(1.,max(.012,med+2.5*mad)/np.maximum(residual,1e-8))
        coef=np.linalg.lstsq(a*np.sqrt(weight[:,None]),xy[:,0]*np.sqrt(weight),rcond=None)[0]
        height=float(np.average(xy[:,1],weights=weight))
    return dict(x_coefficient=coef.tolist(),height=height)


def fit_cycles(candidates,gate=CONFIG['cycle_gate'],max_tracks=CONFIG['max_tracks']):
    """Propose on separated angles, then jointly reassign/refit all retained paths.

    Held-out v=2 mod4 is an API error. Candidate IDs here are extractor indices,
    not organ IDs. Empty training views remain part of the support denominator.
    """
    if any(int(v)%4==2 for v in candidates): raise ValueError('Held-out views cannot fit cycles')
    views=sorted(candidates); nonempty=[v for v in views if candidates[v]]
    minimum=max(CONFIG['minimum_support'],math.ceil(CONFIG['minimum_support_fraction']*len(views)))
    if len(nonempty)<minimum:return []
    remaining={v:list(enumerate(candidates[v])) for v in views}; tracks=[]
    for _ in range(max_tracks):
        available=[v for v in views if remaining[v]]; proposals=[];anchor_sets=set()
        if len(available)<minimum: break
        for offset in range(0,12,2):
            targets=[offset,(offset+13)%40,(offset+27)%40]
            anchors=tuple(min(available,key=lambda v:min(abs(v-t),40-abs(v-t))) for t in targets)
            if len(set(anchors))<3 or anchors in anchor_sets:continue
            anchor_sets.add(anchors);aa=design(anchors)
            if np.linalg.cond(aa)>12: continue
            for combo in itertools.product(*[remaining[v] for v in anchors]):
                xy=np.array([[c['x'],c['y']] for i,c in combo])
                if np.ptp(xy[:,1])>gate*2: continue
                coef=np.linalg.solve(aa,xy[:,0]); h=float(np.median(xy[:,1]))
                if np.hypot(*coef[1:])>1.5 or abs(coef[0])>.5:continue
                if any(np.linalg.norm(np.r_[coef[1:]-np.asarray(t['x_coefficient'])[1:],h-t['height']])<.025 for t in tracks):continue
                proposals.append((coef,h))
        if not proposals:break
        bx=np.array([p[0] for p in proposals]); by=np.array([p[1] for p in proposals]);pp=design(views)@bx.T
        support=np.zeros(len(proposals),int);loss=np.zeros(len(proposals))
        for j,v in enumerate(views):
            if not remaining[v]:loss+=gate;continue
            xy=np.array([[c['x'],c['y']] for i,c in remaining[v]])
            dd=np.sqrt((pp[j,:,None]-xy[None,:,0])**2+(by[:,None]-xy[None,:,1])**2);best=dd.min(1)
            support+=best<gate;loss+=np.minimum(best,gate)
        best=int(np.lexsort((loss,-support))[0]);coef,h=proposals[best]
        if support[best]<minimum: break
        selected=[]
        for v in views:
            if not remaining[v]:continue
            dist=[np.linalg.norm(np.array([design([v])[0]@coef,h])-[c['x'],c['y']]) for i,c in remaining[v]]
            j=int(np.argmin(dist))
            if dist[j]<gate:selected.append((v,remaining[v][j][0],remaining[v][j][1],dist[j]))
        if len(selected)<minimum:break
        t=_refit(selected);tracks.append(t)
        for v,i,c,d in selected:remaining[v]=[(j,q) for j,q in remaining[v] if j!=i]
    for _ in range(5):
        selected_sets=_assign(candidates,tracks,gate)
        tracks=[_refit(ss) if len(ss)>=3 else t for t,ss in zip(tracks,selected_sets)]
    assignments=_assign(candidates,tracks,gate);output=[]
    for t,selected in zip(tracks,assignments):
        if len(selected)<minimum:continue
        vv=[s[0] for s in selected];aa=design(vv);quadrants=len(set(v//10 for v in vv))
        if quadrants<CONFIG['minimum_quadrants'] or np.linalg.cond(aa)>12:continue
        # Recompute distances to the final fitted model, not stale proposal fits.
        resid=np.array([np.linalg.norm(predict(t,v)-[c['x'],c['y']]) for v,i,c,d in selected])
        coef=np.asarray(t['x_coefficient']);variance=float(np.mean(resid**2));cov=np.linalg.pinv(aa.T@aa)*variance
        alternatives=[]
        for sector in range(4):
            subset=[s for s in selected if s[0]//10!=sector]
            if len(subset)>=5 and np.linalg.cond(design([s[0] for s in subset]))<20:
                q=_refit(subset);alternatives.append(q['x_coefficient'][1:]+[q['height']])
        point=np.r_[coef[1:],t['height']]
        spread=np.max(np.linalg.norm(np.asarray(alternatives)-point,axis=1)) if alternatives else None
        output.append(dict(**t,track_id=len(output),radius=float(np.hypot(*coef[1:])),
            root_length=float(np.median([s[2].get('root_length',np.linalg.norm(point)) for s in selected])),
            training_support=len(selected),training_total_views=len(views),training_rmse=float(np.sqrt(variance)),
            training_quadrants=quadrants,design_condition=float(np.linalg.cond(aa)),coefficient_covariance=cov.tolist(),
            leave_sector_max_point_shift=None if spread is None else float(spread),
            point_alternatives_training_only=alternatives,
            assignments=[dict(view=v,candidate_id=i,distance=float(d)) for (v,i,c,old),d in zip(selected,resid)],
            status='rotation_supported_terminal_hypothesis'))
    return output


def partial_assignment(cost,gap=.75,forbidden=None):
    cost=np.asarray(cost,float);n,m=cost.shape
    aug=np.full((n+m,n+m),1e6);aug[:n,:m]=cost
    if forbidden is not None:aug[forbidden[0],forbidden[1]]=1e6
    for i in range(n):aug[i,m+i]=gap
    for j in range(m):aug[n+j,j]=gap
    aug[n:,m:]=0
    if n+m:
        left,right=linear_sum_assignment(aug);total=float(aug[left,right].sum())
        links=[(int(i),int(j)) for i,j in zip(left,right) if i<n and j<m]
    else:links=[];total=0.
    return links,[i for i in range(n) if not any(i==a for a,b in links)],[j for j in range(m) if not any(j==b for a,b in links)],total


def point(node):return np.array([node['x_coefficient'][1],node['x_coefficient'][2],node['height']])


def association_cost(earlier,later,phase_degrees=0,method='phase_geometry'):
    n,m=len(earlier),len(later)
    if not n or not m:return np.empty((n,m))
    a=np.array([point(q) for q in earlier]);b=np.array([point(q) for q in later]);r=np.radians(phase_degrees)
    rot=np.array([[np.cos(r),-np.sin(r)],[np.sin(r),np.cos(r)]]);b[:,:2]=b[:,:2]@rot.T
    if method=='centroid':return np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)/.2
    dz=(a[:,None,2]-b[None,:,2])/.18
    dr=np.array([[x['radius']-y['radius'] for y in later] for x in earlier])/.2
    dl=np.array([[x['root_length']-y['root_length'] for y in later] for x in earlier])/.35
    if method=='invariant_geometry':return np.sqrt((dz**2+dr**2+dl**2)/3)
    dh=np.linalg.norm(a[:,None,:2]-b[None,:,:2],axis=2)/.25
    return np.sqrt((2*dh**2+2*dz**2+dr**2+dl**2)/6)


def associate_visits(earlier,later,phases=None,method='phase_geometry'):
    phases=list(range(0,360,9)) if phases is None else list(phases)
    if not phases:raise ValueError('At least one phase is required')
    profiles=[];models=[]
    for phase in phases:
        cost=association_cost(earlier,later,phase,method);links,ua,ub,total=partial_assignment(cost,CONFIG['assignment_gap'])
        objective=total/max(len(earlier),len(later),1)
        profiles.append(dict(phase_degrees=float(phase),objective=float(objective),predicted_links=len(links)))
        models.append((cost,links,ua,ub,total))
    best=int(np.argmin([p['objective'] for p in profiles]));threshold=profiles[best]['objective']+CONFIG['phase_tolerance']
    retained=[k for k,p in enumerate(profiles) if p['objective']<=threshold+1e-12]
    cost,links,ua,ub,total=models[best];edges=[];accepted=[]
    for i,a in enumerate(earlier):
        for j,b in enumerate(later):
            votes=sum((i,j) in models[k][1] for k in retained);fraction=votes/max(len(retained),1)
            if cost[i,j]>2.5 and not votes:continue
            margin=partial_assignment(cost,CONFIG['assignment_gap'],forbidden=(i,j))[3]-total if (i,j) in links else 0.
            stable=bool((i,j) in links and fraction>=CONFIG['minimum_phase_link_fraction'] and margin>=CONFIG['minimum_assignment_margin'] and cost[i,j]<=CONFIG['maximum_link_cost'])
            edge=dict(source=a['node_id'],target=b['node_id'],cost=float(cost[i,j]),
                phase_assignment_fraction=float(fraction),assignment_margin=float(margin),
                best_alignment_selected=(i,j) in links,selected=stable,
                status='stable_geometry_hypothesis' if stable else 'ambiguous',
                calibration='Sensitivity fraction and objective margin; not identity probability')
            edges.append(edge)
            if stable:accepted.append(edge)
    accepted_a={e['source'] for e in accepted};accepted_b={e['target'] for e in accepted}
    earlier_states=[dict(node_id=a['node_id'],state='linked_hypothesis' if a['node_id'] in accepted_a else 'unobserved_after_or_unmatched') for a in earlier]
    later_states=[dict(node_id=b['node_id'],state='linked_hypothesis' if b['node_id'] in accepted_b else 'newly_observed_birth_unresolved') for b in later]
    return dict(method=method,best_phase_degrees=profiles[best]['phase_degrees'],
        retained_phases_degrees=[profiles[k]['phase_degrees'] for k in retained],phase_profiles=profiles,
        phase_identifiable=len(retained)<=5 and min(len(earlier),len(later))>=2,
        edges=edges,accepted_links=accepted,earlier_states=earlier_states,later_states=later_states,
        best_alignment_predicted_links=len(links),alignment_objective=profiles[best]['objective'])


def build_candidate_graph(candidate_id,sessions,phases=None):
    sessions=sorted(sessions,key=lambda s:s['visit']);pairs=[];edges=[]
    for a,b in itertools.combinations(sessions,2):
        gap=b['visit']-a['visit']
        if gap not in [1,2]:continue
        result=associate_visits(a['nodes'],b['nodes'],phases=phases)
        pair_id=f"{candidate_id}:{a['visit']}>{b['visit']}"
        pairs.append(dict(pair_id=pair_id,earlier_visit=a['visit'],later_visit=b['visit'],visit_gap=gap,
            **{k:v for k,v in result.items() if k not in ['edges','accepted_links']}))
        for e in result['edges']:edges.append(dict(**e,pair_id=pair_id,visit_gap=gap,state='adjacent_visit_candidate' if gap==1 else 'possible_reappearance'))
    # Gap-two proposals enter a selected chain only if adjacent selected links
    # have not already used either endpoint. All alternatives stay in the graph.
    used_out={e['source'] for e in edges if e['selected'] and e['visit_gap']==1}
    used_in={e['target'] for e in edges if e['selected'] and e['visit_gap']==1}
    for e in sorted([e for e in edges if e['visit_gap']==2],key=lambda e:e['cost']):
        if e['selected'] and (e['source'] in used_out or e['target'] in used_in):
            e['selected']=False;e['status']='gap_conflicts_with_adjacent_hypothesis'
        elif e['selected']:used_out.add(e['source']);used_in.add(e['target'])
    nodes=[n for s in sessions for n in s['nodes']];parent={n['node_id']:n['node_id'] for n in nodes}
    def find(k):
        while parent[k]!=k:k=parent[k]
        return k
    for e in edges:
        if e['selected']:parent[find(e['target'])]=find(e['source'])
    components={}
    for n in nodes:components.setdefault(find(n['node_id']),[]).append(n['node_id'])
    chains=[dict(chain_id=f'{candidate_id}:chain{i}',node_ids=nn,status='provisional_geometry_chain') for i,nn in enumerate(components.values())]
    return dict(schema_version='1.0',candidate_id=candidate_id,physical_identity_status='metadata_candidate_unverified',
        coordinate_status='Conditional nominal orthographic training-height gauge; no physical units',
        sessions=sessions,nodes=nodes,edges=edges,pairs=pairs,chains=chains,
        limitations=['Skeleton terminal hypotheses are not anatomical leaves.','Missing observations do not establish loss.','New observations do not establish birth.','Phase-profile stability is not calibrated identity confidence.'])
