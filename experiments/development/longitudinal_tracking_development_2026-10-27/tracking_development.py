"""Execute source-linked longitudinal tracking development, CPU only.

Old full-cohort numerical outputs and source remain read-only. This runner
records newly executed fits and component tests in an independent output root.
"""
import os
for _k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[_k]='2'
import argparse,csv,datetime,hashlib,itertools,json,math,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tracking_graph_core import CONFIG,fit_cycles,predict,point,partial_assignment,association_cost,associate_visits,build_candidate_graph
from tracking_observations import extract_paths,normalize_paths

OLD=Path(__file__).resolve().parent.parent/'full_cohort_2026-10-27'
if not (OLD/'architecture_analysis.py').is_file():
    OLD=Path(__file__).resolve().parents[2]/'full_cohort'
if not (OLD/'architecture_analysis.py').is_file():
    raise ImportError('Tracking requires the sibling experiments/full_cohort source directory')
sys.path.insert(0,str(OLD))
from architecture_analysis import load_mask,coverage,accepted
from architecture_external_rendered import render_points,owner_at
from architecture_tracking import fit_tracks as prior_fit_tracks


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.floating):return float(value) if np.isfinite(value) else None
    if isinstance(value,float) and not math.isfinite(value):return None
    return value
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
def log(out,event,**values):
    row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**clean(values))
    with (out/'experiment_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def gauge_from_masks(masks):
    aa=[];bb=[];hh=[]
    for v,mask in masks.items():
        if v%4==2:continue
        yy,xx=np.where(mask)
        if len(xx):aa.append(float(np.median(xx[yy>=np.percentile(yy,95)])));bb.append(float(yy.max()));hh.append(float(np.ptp(yy)+1))
    if not aa:raise ValueError('No training foreground for gauge')
    return dict(axis=float(np.median(aa)),base=float(np.median(bb)),height=float(np.median(hh)))


def make_nodes(tracks,token,visit,candidates,gauge,frames):
    result=[]
    for t in tracks:
        node=dict(t);node['node_id']=f"{token}:v{visit}:t{t['track_id']}";node['visit']=visit
        node['point']={'x':float(t['x_coefficient'][1]),'up':float(t['height']),'depth':float(t['x_coefficient'][2])}
        node['coordinate_status']='conditional_nominal_orthographic_training_height_gauge'
        node['uncertainty_status']='training residual/sector sensitivity only; no calibrated anatomical interval'
        assignments={a['view']:a for a in t['assignments']};evidence=[]
        for v in range(40):
            frame=frames.get(v);pp=predict(t,v);cc=candidates.get(v,[])
            dd=[np.linalg.norm(pp-[c['x'],c['y']]) for c in cc];ci=int(np.argmin(dd)) if dd else None
            observed=ci is not None and dd[ci]<.07
            if frame is None:status='source_missing'
            elif not frame.get('qc_pass',True):status='mask_rejected'
            elif v in assignments:status='training_assigned';ci=assignments[v]['candidate_id'];observed=True
            elif observed:status='heldout_compatible_terminal' if v%4==2 else 'training_nearby_unassigned'
            elif frame.get('censored_terminal_candidates',0):status='censored_candidates_present_terminal_unobserved'
            else:status='terminal_unobserved_or_incompatible'
            pixel=[gauge['axis']+gauge['height']*pp[0],gauge['base']-gauge['height']*pp[1]]
            width=frame.get('processing_width',256) if frame else 256;height=frame.get('processing_height',192) if frame else 192
            row=dict(view=v,nominal_angle_degrees=v*9,split='heldout' if v%4==2 else 'train',status=status,
                terminal_observed=bool(observed),visibility_interpretation='Terminal image support only, not complete organ visibility',
                censored_candidate_count=frame.get('censored_terminal_candidates',0) if frame else 0,
                predicted_xy_normalized=pp.tolist(),predicted_pixel_processing=pixel,
                predicted_image_fraction=[float(pixel[0]/width),float(pixel[1]/height)],candidate_count=len(cc),
                nearest_terminal_distance=float(min(dd)) if dd else None)
            if ci is not None and frame is not None and frame.get('qc_pass',True):
                c=cc[ci];row['observed_candidate']=dict(candidate_index=ci,source_candidate_id=c['original_candidate_id'],
                    terminal_pixel_processing=[c['px'],c['py']],path_processing_px=c['path_processing_px'])
            if frame is not None:row['source']=frame
            evidence.append(row)
        node['angle_evidence']=evidence;result.append(node)
    return result


def heldout_rows(nodes,candidates,identity,owners=None,mapping=None):
    rows=[]
    for node in nodes:
        for v in range(2,40,4):
            if v not in candidates:continue
            cc=candidates[v];pp=predict(node,v);dd=[float(np.linalg.norm(pp-[c['x'],c['y']])) for c in cc]
            ci=int(np.argmin(dd)) if dd else None;distance=min(dd) if dd else None
            row=dict(**identity,node_id=node['node_id'],track_id=node['track_id'],view=v,candidate_count=len(cc),
                nearest_terminal_distance=distance,compatible_at_gate=bool(distance is not None and distance<.07))
            if owners is not None:
                label=mapping.get(node['track_id'],0);nearest=owner_at(owners[v],cc[ci]['px'],cc[ci]['py']) if ci is not None else 0
                row.update(mapped_id_evaluation_only=label,nearest_id_evaluation_only=nearest,
                    target_terminal_available=bool(label and any(owner_at(owners[v],c['px'],c['py'])==label for c in cc)),
                    correct_mapped_id_and_gate=bool(label and label==nearest and distance is not None and distance<.07))
            rows.append(row)
    return rows


def run_csi(args):
    out=args.output/'csi';out.mkdir(parents=True,exist_ok=True);source=args.full_run/'extraction/view_traits.csv'
    rows=read(source);reference=read(args.full_run/'architecture/full_reference_architecture_coverage.csv')
    groups={};registry={}
    for r in rows:
        if r['camera']=='rgb':groups.setdefault((r['treatment'],r['genotype'],r['replicate'],int(r['measurement'])),[]).append(r)
    for r in reference:
        if r['camera']=='rgb':registry.setdefault((r['treatment'],r['genotype'],r['replicate']),[]).append(int(r['measurement']))
    cases=[];all_nodes=[];all_edges=[];all_pairs=[];evalrows=[];index=[];errors=[]
    for k,(identity,visits) in enumerate(sorted(registry.items()),1):
        tr,gen,rep=identity;candidate_id='|'.join(identity);token=hashlib.sha256(candidate_id.encode()).hexdigest()[:16];sessions=[]
        for visit in sorted(set(visits)):
            group=groups.get((*identity,visit),[]);frames={}
            for r in group:
                frames[int(r['view'])]=dict(source_path=r['source'],source_sha256=r['source_sha256'],mask_path=r.get('mask'),
                    mask_sha256=r.get('mask_sha256'),preview_path=r.get('preview'),qc_pass=accepted(r),qc_issues=r.get('qc_issues',''),
                    processing_width=256,processing_height=round(float(r['processing_height'])*256/float(r['processing_width'])),
                    native_width=int(r['native_width']),native_height=int(r['native_height']))
            good=[r for r in group if accepted(r)];cv=coverage([int(r['view']) for r in good]);session=dict(visit=visit,nodes=[],frames=[dict(view=v,**f) for v,f in sorted(frames.items())],coverage=cv)
            status='missing_source' if not group else 'insufficient_coverage'
            if cv['eligible'] and sum(int(r['view'])%4!=2 for r in good)>=15:
                try:
                    masks={int(r['view']):load_mask(r,args.full_run,width=256) for r in good};gauge=gauge_from_masks(masks)
                    paths={v:extract_paths(mask,min_length=args.min_branch) for v,mask in masks.items()}
                    for v,cc in paths.items():frames[v].update(total_extracted_candidates=len(cc),censored_terminal_candidates=sum(c['censored'] for c in cc))
                    session['frames']=[dict(view=v,**f) for v,f in sorted(frames.items())]
                    candidates={v:normalize_paths(cc,**gauge) for v,cc in paths.items()};train={v:cc for v,cc in candidates.items() if v%4!=2}
                    tracks=fit_cycles(train);nodes=make_nodes(tracks,token,visit,candidates,gauge,frames)
                    # Persist a training-only frozen scientific model before evaluation.
                    dump(out/'frozen'/f'{token}_v{visit}.json',dict(identity=dict(treatment=tr,genotype=gen,replicate=rep,visit=visit),gauge=gauge,
                        train_views=sorted(train),tracks=tracks,input_table_sha256=sha(source)))
                    session.update(nodes=nodes,gauge=gauge,all_extracted_path_count=sum(len(v) for v in paths.values()),
                        censored_path_count=sum(c['censored'] for cc in paths.values() for c in cc))
                    evalrows.extend(heldout_rows(nodes,candidates,dict(treatment=tr,genotype=gen,replicate=rep,visit=visit)))
                    status='executed_with_hypotheses' if nodes else 'executed_no_retained_hypothesis'
                except Exception as e:
                    errors.append(dict(candidate_id=candidate_id,visit=visit,error=repr(e)));status='failed'
            session['status']=status;sessions.append(session)
            cases.append(dict(treatment=tr,genotype=gen,replicate=rep,visit=visit,status=status,accepted_views=cv['accepted_views'],nodes=len(session['nodes'])))
        graph=build_candidate_graph(candidate_id,sessions)
        graph.update(identity=dict(treatment=tr,genotype=gen,replicate=rep),source_roots=dict(intake=str(args.intake) if args.intake else None,derived=str(args.full_run)),
            provenance=dict(view_traits_sha256=sha(source),configuration=CONFIG,min_branch_processing_pixels=args.min_branch))
        relative=f'graphs/{token}.json';dump(out/relative,graph)
        index.append(dict(candidate_id=candidate_id,identity=graph['identity'],path=relative,visits=len(sessions),nodes=len(graph['nodes']),
            selected_adjacent_links=sum(e['selected'] and e['visit_gap']==1 for e in graph['edges']),
            selected_gap_links=sum(e['selected'] and e['visit_gap']==2 for e in graph['edges'])))
        for n in graph['nodes']:all_nodes.append(dict(candidate_id=candidate_id,**{key:value for key,value in n.items() if key not in ['angle_evidence','assignments','coefficient_covariance','point_alternatives_training_only']}))
        for e in graph['edges']:all_edges.append(dict(candidate_id=candidate_id,**e))
        for p in graph['pairs']:all_pairs.append(dict(candidate_id=candidate_id,**p))
        if k%8==0:log(args.output,'csi_candidate_progress',completed=k,total=len(registry),nodes=len(all_nodes))
    write(out/'sessions.csv',cases);write(out/'nodes.csv',all_nodes);write(out/'edges.csv',all_edges);write(out/'pair_summaries.csv',all_pairs);write(out/'heldout_compatibility.csv',evalrows);write(out/'errors.csv',errors)
    valid=[r for r in evalrows if r['nearest_terminal_distance'] is not None]
    summary=dict(candidate_identities=len(index),reference_visit_slots=len(cases),planned_adjacent_pairs=sum(p['visit_gap']==1 for p in all_pairs),
        planned_gap_two_pairs=sum(p['visit_gap']==2 for p in all_pairs),executed_sessions=sum(c['status'].startswith('executed') for c in cases),
        sessions_with_nodes=sum(c['nodes']>0 for c in cases),node_hypotheses=len(all_nodes),
        candidate_edges=len(all_edges),selected_adjacent_hypotheses=sum(e['selected'] and e['visit_gap']==1 for e in all_edges),
        selected_possible_reappearances=sum(e['selected'] and e['visit_gap']==2 for e in all_edges),heldout_track_view_rows=len(evalrows),
        heldout_candidate_available=len(valid),heldout_compatible_rows=sum(r['compatible_at_gate'] for r in evalrows),
        heldout_median_nearest_terminal_distance=float(np.median([r['nearest_terminal_distance'] for r in valid])) if valid else None,
        errors=len(errors),physical_identity_status='Metadata candidate identities only; selected links are geometry hypotheses, not authenticated leaves',
        prior_reference_track_hypotheses=377,input_table_sha256=sha(source))
    dump(out/'summary.json',summary);dump(out/'index.json',dict(schema_version='1.0',summary=summary,candidates=index));log(args.output,'csi_completed',**summary)
    return summary


def external_scan(args,record):
    path=args.external/record['path']
    if sha(path)!=record['sha256']:raise ValueError('External source digest mismatch')
    cloud=np.loadtxt(path,dtype='float32');points=cloud[cloud[:,4]>0,:3];labels=cloud[cloud[:,4]>0,4].astype('int16')
    low=np.percentile(points[:,2],2);basal=points[points[:,2]<=np.percentile(points[:,2],8)];origin=np.r_[np.median(basal[:,:2],axis=0),low];height=float(points[:,2].max()-low);xyz=(points-origin)/height
    cells=np.floor(xyz/.003).astype(np.int32);_,ix=np.unique(cells,axis=0,return_index=True);xyz=xyz[ix];labels=labels[ix]
    masks={};owners={};paths={};frames={};out=args.output/'external';stem=path.stem
    for v in range(40):
        masks[v],owners[v]=render_points(xyz,labels,v);paths[v]=extract_paths(masks[v],min_length=args.min_branch)
        rel=f'renders/{stem}/view_{v:02d}.png';p=out/rel;p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray((masks[v]*255).astype('uint8')).save(p)
        frames[v]=dict(source_path=record['path'],source_sha256=record['sha256'],render_path=rel,mask_sha256=sha(p),
            qc_pass=True,processing_width=384,processing_height=384,nominal_known_render_camera=True,
            total_extracted_candidates=len(paths[v]),censored_terminal_candidates=sum(c['censored'] for c in paths[v]))
    gauge=gauge_from_masks(masks);candidates={v:normalize_paths(cc,**gauge) for v,cc in paths.items()};train={v:cc for v,cc in candidates.items() if v%4!=2}
    tracks=fit_cycles(train);prior_tracks=prior_fit_tracks(train)
    dump(out/'frozen'/f'{stem}.json',dict(scan=stem,gauge=gauge,train_views=sorted(train),tracks=tracks,source_sha256=record['sha256']))
    ids=sorted(int(i) for i in np.unique(labels));counts=np.zeros((len(tracks),len(ids)),int)
    for i,t in enumerate(tracks):
        for a in t['assignments']:
            c=train[a['view']][a['candidate_id']];label=owner_at(owners[a['view']],c['px'],c['py'])
            if label in ids:counts[i,ids.index(label)]+=1
    mapping={}
    if len(tracks):
        ii,jj=linear_sum_assignment(-counts);mapping={int(i):ids[int(j)] for i,j in zip(ii,jj) if counts[i,j]>0}
    dump(out/'scoring_maps'/f'{stem}.json',dict(mapping=mapping,training_ownership_counts=counts.tolist(),true_ids=ids,
        rule='One-to-one maximum TRAIN ownership count, fixed before opening heldout labels. Never used by fit/association.'))
    nodes=make_nodes(tracks,stem,0,candidates,gauge,frames);truth=[];point_rows=[]
    for n in nodes:
        t=n['track_id'];label=mapping.get(t,0);truth.append(label)
        # Invert only the known rendering camera into scan-height coordinates.
        q=np.array([n['x_coefficient'][1]*gauge['height']/300,n['x_coefficient'][2]*gauge['height']/300,
            (n['height']*gauge['height']-gauge['base']+360)/300])
        n['known_render_point']=q.tolist();n['training_ownership_purity_evaluation_only']=float(counts[t,ids.index(label)]/n['training_support']) if label else 0.
        point_rows.append(dict(scan=stem,node_id=n['node_id'],mapped_id_evaluation_only=label,x=float(q[0]),depth=float(q[1]),up=float(q[2])))
    observable={
        label:sum(any(owner_at(owners[v],c['px'],c['py'])==label for c in candidates[v]) for v in train) for label in ids}
    evaluation=heldout_rows(nodes,candidates,dict(plant=path.parent.name,scan=stem),owners,mapping)
    return dict(plant=path.parent.name,scan=stem,nodes=nodes,truth=truth,true_ids=ids,origin=origin,height=height,
        original_path=record['path'],source_sha256=record['sha256'],gauge=gauge,evaluation=evaluation,
        observable_instances=sum(v>=8 for v in observable.values()),observable_ids=[k for k,v in observable.items() if v>=8],point_rows=point_rows,
        prior_fit_on_same_new_candidates=len(prior_tracks),candidate_count=sum(len(c) for c in candidates.values()))


def shared_nodes(scan,earlier):
    out=[]
    for n in scan['nodes']:
        p=(np.asarray(n['known_render_point'])*scan['height']+scan['origin']-earlier['origin'])/earlier['height'];q=dict(n)
        q['source_projection_model']=dict(x_coefficient=n['x_coefficient'],height=n['height'])
        q.update(x_coefficient=[0,float(p[0]),float(p[1])],height=float(p[2]),radius=float(np.linalg.norm(p[:2])),root_length=n['root_length']*scan['height']/earlier['height'],
            point=dict(x=float(p[0]),depth=float(p[1]),up=float(p[2])),coordinate_status='known_render_shared_earlier_scan_height_gauge')
        out.append(q)
    return out


def benchmark_pair(a,b,truth_a,truth_b,all_common,plant,scenario,method):
    if method in ['centroid_fixed_phase','simple_geometry_invariant']:
        cost=association_cost(a,b,0,'centroid' if method=='centroid_fixed_phase' else 'invariant_geometry');links,ua,ub,total=partial_assignment(cost,.75)
        chosen=[dict(source=a[i]['node_id'],target=b[j]['node_id'],cost=float(cost[i,j])) for i,j in links];phase=0.;retained=1
    else:
        result=associate_visits(a,b,phases=[0] if method=='graph_known_pose' else None);chosen=result['accepted_links'];phase=result['best_phase_degrees'];retained=len(result['retained_phases_degrees'])
    lookup_a={q['node_id']:t for q,t in zip(a,truth_a)};lookup_b={q['node_id']:t for q,t in zip(b,truth_b)}
    correct=sum(lookup_a[e['source']]>0 and lookup_a[e['source']]==lookup_b[e['target']] for e in chosen)
    recovered_common=len((set(truth_a)&set(truth_b))-{0})
    row=dict(plant=plant,scenario=scenario,method=method,earlier_nodes=len(a),later_nodes=len(b),predicted_links=len(chosen),correct_links=correct,
        recovered_true_common=recovered_common,all_true_common=all_common,precision=correct/max(len(chosen),1),
        recall_recovered=correct/max(recovered_common,1),recall_full=correct/max(all_common,1),best_phase_degrees=phase,retained_phases=retained)
    linksrows=[dict(plant=plant,scenario=scenario,method=method,**e,earlier_id_evaluation_only=lookup_a[e['source']],later_id_evaluation_only=lookup_b[e['target']],
        correct=bool(lookup_a[e['source']]>0 and lookup_a[e['source']]==lookup_b[e['target']])) for e in chosen]
    return row,linksrows


def run_external(args):
    out=args.output/'external';out.mkdir(parents=True,exist_ok=True);manifest=json.loads((args.external/'download_manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed' or len(manifest['files'])!=14:raise ValueError('Expected original complete 14-scan subset')
    scans=[];cases=[];evalrows=[];points=[];by={}
    for i,record in enumerate(manifest['files'],1):
        scan=external_scan(args,record);scans.append(scan);by.setdefault(scan['plant'],[]).append(scan);evalrows+=scan['evaluation'];points+=scan['point_rows']
        cases.append(dict(plant=scan['plant'],scan=scan['scan'],true_instance_dates=len(scan['true_ids']),observable_instance_dates=scan['observable_instances'],
            observable_ids_evaluation_only=scan['observable_ids'],
            retained_tracks=len(scan['nodes']),distinct_scored_instances=len(set(scan['truth'])-{0}),unmapped_tracks=scan['truth'].count(0),
            prior_fit_same_new_candidates=scan['prior_fit_on_same_new_candidates'],extracted_candidates=scan['candidate_count']))
        log(args.output,'external_scan',completed=i,total=14,scan=scan['scan'],nodes=len(scan['nodes']),true_instances=len(scan['true_ids']))
    pairrows=[];linksrows=[];index=[]
    for plant,pair in sorted(by.items()):
        pair.sort(key=lambda q:q['scan']);a,b=pair;aa=shared_nodes(a,a);bb=shared_nodes(b,a);common=len(set(a['true_ids'])&set(b['true_ids']))
        sessions=[dict(visit=1,nodes=aa,scan=a['scan']),dict(visit=2,nodes=bb,scan=b['scan'])];graph=build_candidate_graph(plant,sessions)
        graph['physical_identity_status']='known_external_plant_continuity';graph['coordinate_status']='known orthographic render camera; shared earlier scan-height gauge'
        graph['true_ids_evaluation_only']={n['node_id']:t for scan in pair for n,t in zip(scan['nodes'],scan['truth'])}
        dump(out/'graphs'/f'{plant}.json',graph);index.append(dict(candidate_id=plant,path=f'graphs/{plant}.json',nodes=len(graph['nodes'])))
        for scenario in ['original','phase90','later_node_hidden','earlier_node_hidden','phase90_later_hidden']:
            x=[dict(q) for q in aa];y=[dict(q) for q in bb];ta=a['truth'][:];tb=b['truth'][:]
            if 'phase90' in scenario:
                for q in y:q['x_coefficient']=[0,q['x_coefficient'][2],-q['x_coefficient'][1]]
            if 'later' in scenario and y:y=y[1:];tb=tb[1:]
            if 'earlier' in scenario and x:x=x[1:];ta=ta[1:]
            for method in ['centroid_fixed_phase','simple_geometry_invariant','graph_known_pose','graph_unknown_pose']:
                r,ll=benchmark_pair(x,y,ta,tb,common,plant,scenario,method);pairrows.append(r);linksrows+=ll
    # Same numeric IDs in different physical plants never make a correct link.
    wrong=[]
    for a,b in itertools.permutations([sorted(pair,key=lambda q:q['scan']) for pair in by.values()],2):
        x=shared_nodes(a[0],a[0]);y=shared_nodes(b[1],b[0]);r=associate_visits(x,y)
        wrong.append(dict(earlier_plant=a[0]['plant'],later_plant=b[1]['plant'],possible_links=min(len(x),len(y)),accepted_false_links=len(r['accepted_links'])))
    aggregates=[]
    for scenario,method in itertools.product(sorted(set(r['scenario'] for r in pairrows)),sorted(set(r['method'] for r in pairrows))):
        rr=[r for r in pairrows if r['scenario']==scenario and r['method']==method];correct=sum(r['correct_links'] for r in rr);pred=sum(r['predicted_links'] for r in rr);full=sum(r['all_true_common'] for r in rr);recovered=sum(r['recovered_true_common'] for r in rr)
        aggregates.append(dict(scenario=scenario,method=method,plant_pairs=len(rr),correct_links=correct,predicted_links=pred,all_true_common=full,
            recovered_true_common=recovered,precision=correct/max(pred,1),recall_full=correct/max(full,1),recall_recovered=correct/max(recovered,1)))
    write(out/'cases.csv',cases);write(out/'heldout_scores.csv',evalrows);write(out/'conditional_points.csv',points);write(out/'pair_scores.csv',pairrows);write(out/'predicted_links.csv',linksrows);write(out/'aggregate_scores.csv',aggregates);write(out/'wrong_physical_plant_controls.csv',wrong)
    summary=dict(scans=len(cases),plants=len(by),true_instance_dates=sum(r['true_instance_dates'] for r in cases),observable_instance_dates=sum(r['observable_instance_dates'] for r in cases),
        node_hypotheses=sum(r['retained_tracks'] for r in cases),distinct_scored_instance_dates=sum(r['distinct_scored_instances'] for r in cases),
        unmapped_tracks=sum(r['unmapped_tracks'] for r in cases),prior_fit_same_new_candidates=sum(r['prior_fit_same_new_candidates'] for r in cases),
        heldout_track_rows=len(evalrows),heldout_correct_id_and_gate=sum(r['correct_mapped_id_and_gate'] for r in evalrows),
        heldout_target_terminal_available=sum(r['target_terminal_available'] for r in evalrows),wrong_plant_comparisons=len(wrong),
        accepted_wrong_plant_links=sum(r['accepted_false_links'] for r in wrong),possible_wrong_plant_links=sum(r['possible_links'] for r in wrong),
        results=aggregates,input_manifest_sha256=sha(args.external/'download_manifest.json'),
        scope='Known cameras, supplied foreground semantics, registered original point geometry and known physical continuity; labels used for scoring only. CSI anatomy and physical identity remain unvalidated.')
    dump(out/'summary.json',summary);dump(out/'index.json',dict(schema_version='1.0',summary=summary,candidates=index));log(args.output,'external_completed',**summary)
    return summary


def audit(args):
    findings=[];graphs=0;nodes=0;adjacent=0;observations=0
    for branch in ['external','csi']:
        index_path=args.output/branch/'index.json'
        if not index_path.exists():continue
        index=json.loads(index_path.read_text(encoding='utf-8'))
        for r in index['candidates']:
            g=json.loads((args.output/branch/r['path']).read_text(encoding='utf-8'));graphs+=1;nodes+=len(g['nodes'])
            ids=[n['node_id'] for n in g['nodes']]
            if len(ids)!=len(set(ids)):findings.append('Duplicate node IDs in '+g['candidate_id'])
            for n in g['nodes']:
                if any(a['view']%4==2 for a in n['assignments']):findings.append('Training leakage '+n['node_id'])
                if len(n['angle_evidence'])!=40:findings.append('Missing angular states '+n['node_id'])
                observations+=len(n['angle_evidence'])
            selected=[e for e in g['edges'] if e['selected']]
            if len({e['source'] for e in selected})!=len(selected):findings.append('Multiple outgoing selected edges '+g['candidate_id'])
            if len({e['target'] for e in selected})!=len(selected):findings.append('Multiple incoming selected edges '+g['candidate_id'])
            adjacent+=sum(p['visit_gap']==1 for p in g['pairs'])
            if branch=='csi' and (len(g['sessions'])!=5 or sum(p['visit_gap']==1 for p in g['pairs'])!=4):findings.append('Incomplete candidate reference slots '+g['candidate_id'])
    result=dict(status='passed' if not findings else 'failed',graphs=graphs,nodes=nodes,angle_evidence_rows=observations,adjacent_pairs=adjacent,findings=findings,
        checks=['Unique nodes','No heldout training assignments','40 explicit angle states per node','One-to-one selected chains','All five CSI reference slots/four adjacent pairs'])
    dump(args.output/'audit.json',result);log(args.output,'audit_completed',**result)
    if findings:raise ValueError('Graph audit failed')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--full-run',type=Path,required=True);p.add_argument('--external',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--intake',type=Path);p.add_argument('--mode',choices=['external','csi','all','audit'],default='all');p.add_argument('--min-branch',type=int,default=6);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);sources={}
    for name in ['tracking_graph_core.py','tracking_observations.py','tracking_development.py','test_tracking_graph.py']:
        src=Path(__file__).parent/name;data=src.read_bytes();sources[name]=sha(src);target=args.output/'execution_sources'/name;target.parent.mkdir(exist_ok=True);target.write_bytes(data)
    dependencies={str(OLD/name):sha(OLD/name) for name in ['architecture_analysis.py','architecture_external_rendered.py','architecture_tracking.py']}
    dump(args.output/'configuration.json',dict(config=CONFIG,min_branch=args.min_branch,source_hashes=sources,readonly_dependency_hashes=dependencies,
        full_run=str(args.full_run),external=str(args.external),intake=str(args.intake) if args.intake else None,thread_limit=2,mode=args.mode))
    log(args.output,'started',mode=args.mode,configuration=CONFIG,min_branch=args.min_branch)
    if args.mode in ['external','all']:run_external(args)
    if args.mode in ['csi','all']:run_csi(args)
    audit(args)
    changed=[name for name,digest in sources.items() if sha(Path(__file__).parent/name)!=digest]
    if changed:raise RuntimeError('Execution source changed during run: '+str(changed))
    log(args.output,'completed',mode=args.mode,source_changed_during_run=False)


if __name__=='__main__':main()
