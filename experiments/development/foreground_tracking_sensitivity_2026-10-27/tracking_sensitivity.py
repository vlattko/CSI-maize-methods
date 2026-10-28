"""Matched teacher/student foreground substitution through frozen cyclic tracking."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,json,math,shutil,sys,time
from pathlib import Path
import numpy as np
from PIL import Image

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def table(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)

def truth(value):return str(value).lower() in ('true','1')


def selected_acquisition(identity,status,genotype=None):
    return status.startswith('executed') and (genotype is None or identity['genotype']==genotype)

def contained(root,name):
    root=Path(root).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Escaping input path')
    return path

def cache_lookup(cache):
    status=json.loads((cache/'status.json').read_text());operator=json.loads((cache/'operator.json').read_text())
    if status['status']!='completed' or status['cached_frames']!=status['expected_frames']:raise ValueError('Completed full cache required')
    if sha(cache/'frames.csv')!=status['frames_sha256']:raise ValueError('Cache table digest mismatch')
    rows=read(cache/'frames.csv')
    if len(rows)!=status['expected_frames']:raise ValueError('Incomplete cache table')
    lookup={}
    for row in rows:
        key=(row['session'],row['camera'],int(row['view']))
        if key in lookup:raise ValueError('Duplicate source-view cache key')
        if row['signature']!=operator['signature'] or row['signature']!=status['operator_signature']:raise ValueError('Operator signature mismatch')
        lookup[key]=row
    return lookup,status,operator

def bind_frame(identity,visit,view,frame,lookup):
    # The source path is the authoritative acquisition, not a basename or guessed metadata ID.
    source=frame['source_path'];session=source.rsplit('/',2)[0];row=lookup[(session,'rgb',view)]
    if row['source']!=source or row['source_sha256']!=frame['source_sha256']:raise ValueError('Teacher/student source mismatch')
    if any(str(row[k])!=str(identity[k]) for k in ('treatment','genotype','replicate')) or int(row['measurement'])!=visit:raise ValueError('Four-key cache mismatch')
    if not truth(row['teacher_accepted']):raise ValueError('Teacher view-support convention disagrees')
    return row

def load_mask(root,name,digest):
    path=contained(root,name)
    if sha(path)!=digest:raise ValueError('Mask digest mismatch')
    with Image.open(path) as im:return np.asarray(im.convert('L').resize((256,round(im.height*256/im.width)),Image.Resampling.NEAREST))>0

def set_distance(a,b,scale):
    a=np.asarray(a,float).reshape(-1,2);b=np.asarray(b,float).reshape(-1,2)
    if not len(a) or not len(b):return dict(state='both_empty' if not len(a) and not len(b) else 'teacher_empty' if not len(a) else 'student_empty',teacher_count=len(a),student_count=len(b),symmetric_mean=None,symmetric_p90=None)
    distance=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=-1)/scale
    nearest=np.r_[distance.min(axis=1),distance.min(axis=0)]
    return dict(state='both_nonempty',teacher_count=len(a),student_count=len(b),symmetric_mean=float(np.mean(nearest)),symmetric_p90=float(np.quantile(nearest,.9)))

def pixels(nodes,gauge,view,predict):
    return [[gauge['axis']+gauge['height']*predict(n,view)[0],gauge['base']-gauge['height']*predict(n,view)[1]] for n in nodes]

def nearest(point,candidates,scale):
    return float(min(np.linalg.norm(np.asarray(point)-np.array([c['px'],c['py']])) for c in candidates)/scale) if candidates else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('full-run','teacher-tracking','cache','output','tracking-source','full-source'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-cases',type=int);p.add_argument('--genotype');args=p.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty output root')
    out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(args.full_source));sys.path.insert(0,str(args.tracking_source))
    import tracking_development as t
    lookup,status,operator=cache_lookup(args.cache)
    source=args.full_run/'extraction/view_traits.csv';index_path=args.teacher_tracking/'csi/index.json';index=json.loads(index_path.read_text())
    if operator['teacher_table_sha256']!=sha(source) or operator['teacher_table_sha256']!=index['summary']['input_table_sha256']:raise ValueError('Teacher extraction source mismatch')
    provenance=dict(status='running',cache_status_sha256=sha(args.cache/'status.json'),cache_frames_sha256=sha(args.cache/'frames.csv'),cache_operator=operator,teacher_index_sha256=sha(index_path),source_table_sha256=sha(source),configuration=t.CONFIG,min_branch=6,outer_views=list(range(2,40,4)),maximum_cases=args.max_cases,source_sha256={p.name:sha(p) for folder in (Path(__file__).parent,args.tracking_source) for p in folder.glob('*.py')},scope='Teacher-matched foreground sensitivity, not anatomical accuracy')
    provenance['genotype_filter']=args.genotype;dump(out/'configuration.json',provenance)
    snapshot=out/'execution_sources';snapshot.mkdir()
    for folder in (Path(__file__).parent,args.tracking_source):
        target=snapshot/folder.name;target.mkdir()
        for path in folder.glob('*.py'):shutil.copyfile(path,target/path.name)
    sessions=[];heldout=[];sets=[];errors=[];graphindex=[];processed=0;teacher_replays=0;start=time.time();matchedframes=0;emptyframes=0;graphs=[];eligible_selected=0
    for entry in index['candidates']:
        graphpath=contained(args.teacher_tracking/'csi',entry['path']);graph=json.loads(graphpath.read_text());identity=graph['identity'];candidate=graph['candidate_id'];token=Path(entry['path']).stem;new_sessions=[]
        for session in graph['sessions']:
            visit=session['visit'];base=dict(**identity,visit=visit,candidate_id=candidate);eligible=selected_acquisition(identity,session['status'],args.genotype);eligible_selected+=eligible;selected=eligible and (args.max_cases is None or processed<args.max_cases)
            new=dict(visit=visit,nodes=[],status='outside_pilot' if eligible else 'teacher_not_executed',teacher_status=session['status'])
            if args.genotype and identity['genotype']!=args.genotype:new['status']='outside_selected_genotype'
            if selected:
                try:
                    frames={r['view']:r for r in session['frames'] if truth(r['qc_pass'])};student_masks={};teacher_masks={};source_frames={}
                    for view,frame in sorted(frames.items()):
                        row=bind_frame(identity,visit,view,frame,lookup)
                        student_masks[view]=load_mask(args.cache,row['student_mask'],row['student_mask_sha256']);teacher_masks[view]=load_mask(args.full_run,frame['mask_path'],frame['mask_sha256'])
                        if student_masks[view].shape!=teacher_masks[view].shape:raise ValueError('Canvas mismatch')
                        matchedframes+=1;emptyframes+=not student_masks[view].any()
                        source_frames[view]=dict(frame,teacher_mask_path=frame['mask_path'],mask_path=row['student_mask'],mask_sha256=row['student_mask_sha256'],mask_root=str(args.cache),student_empty=not bool(student_masks[view].any()),mask_operator_signature=row['signature'])
                    tg=t.gauge_from_masks(teacher_masks)
                    if not all(math.isclose(tg[k],session['gauge'][k],abs_tol=1e-12) for k in tg):raise ValueError('Frozen teacher gauge replay mismatch')
                    teacher_paths={v:t.extract_paths(m,min_length=6) for v,m in teacher_masks.items()};tc={v:t.normalize_paths(cc,**tg) for v,cc in teacher_paths.items()}
                    teacher_nodes=session['nodes'];student_paths={v:t.extract_paths(m,min_length=6) for v,m in student_masks.items()}
                    if teacher_replays<12:
                        replay=t.fit_cycles({v:cc for v,cc in tc.items() if v%4!=2})
                        if len(replay)!=len(teacher_nodes):raise ValueError('Frozen teacher node-count replay mismatch')
                        for a,b in zip(replay,teacher_nodes):np.testing.assert_allclose(a['x_coefficient']+[a['height']],b['x_coefficient']+[b['height']],atol=1e-12,rtol=1e-12)
                        teacher_replays+=1
                    try:sg=t.gauge_from_masks(student_masks)
                    except ValueError:sg=None
                    sc={v:t.normalize_paths(cc,**sg) for v,cc in student_paths.items()} if sg else {v:[] for v in student_masks}
                    # Raw pixel evidence remains observable even if the training gauge abstains.
                    student_pixel_candidates={v:[dict(px=c['terminal_x_px'],py=c['terminal_y_px']) for c in cc if not c['censored']] for v,cc in student_paths.items()}
                    tracks=t.fit_cycles({v:cc for v,cc in sc.items() if v%4!=2}) if sg else []
                    student_nodes=t.make_nodes(tracks,'student:'+token,visit,sc,sg,source_frames) if sg else []
                    new.update(nodes=student_nodes,gauge=sg,status='executed_with_hypotheses' if tracks else 'executed_no_retained_hypothesis',coverage=session['coverage'])
                    for view in range(40):
                        tp=pixels(teacher_nodes,tg,view,t.predict);sp=pixels(student_nodes,sg,view,t.predict) if sg else []
                        comparison=set_distance(tp,sp,tg['height']);available=view in frames
                        raw=set_distance([[c['px'],c['py']] for c in tc.get(view,[])],[[c['px'],c['py']] for c in student_pixel_candidates.get(view,[])],tg['height'])
                        sets.append(dict(base,view=view,split='heldout' if view%4==2 else 'train',source_available=available,source=frames[view]['source_path'] if available else '',**comparison,terminal_set_state=raw['state'],terminal_set_symmetric_mean=raw['symmetric_mean'],terminal_set_symmetric_p90=raw['symmetric_p90']))
                        if view%4==2:
                            for method,nodes,gauge,own,other in [('teacher',teacher_nodes,tg,tc,student_pixel_candidates),('student',student_nodes,sg,sc,tc)]:
                                for node,point in zip(nodes,pixels(nodes,gauge,view,t.predict)):
                                    distance=nearest(point,own.get(view,[]),gauge['height']) if available else None;cross=nearest(point,other.get(view,[]),tg['height']) if available else None
                                    heldout.append(dict(base,method=method,node_id=node['node_id'],view=view,source_available=available,candidate_available=bool(own.get(view,[])),nearest_own_terminal_distance=distance,compatible=distance is not None and distance<.07,nearest_other_operator_terminal_distance=cross,cross_operator_compatible=cross is not None and cross<.07))
                    sessions.append(dict(base,status=new['status'],teacher_nodes=len(teacher_nodes),student_nodes=len(tracks),matched_source_views=len(frames),student_nonempty_views=sum(m.any() for m in student_masks.values()),teacher_training_height=tg['height'],student_training_height=sg['height'] if sg else None))
                    dump(out/'frozen'/f'{token}_v{visit}.json',dict(identity=base,train_views=sorted(v for v in sc if v%4!=2),gauge=sg,tracks=tracks,mask_operator=operator['signature']))
                except Exception as exc:new['status']='failed';errors.append(dict(base,error=repr(exc)));sessions.append(dict(base,status='failed',teacher_nodes=len(session['nodes']),student_nodes=0))
                processed+=1
                print(json.dumps(dict(event='progress',processed=processed,errors=len(errors),elapsed_seconds=time.time()-start)),flush=True)
            new_sessions.append(new)
        newgraph=t.build_candidate_graph(candidate,new_sessions);newgraph.update(identity=identity,mask_operator=operator,source_roots=dict(student_cache=str(args.cache),teacher_run=str(args.full_run)),teacher_graph_sha256=sha(graphpath))
        dump(out/'graphs'/f'{token}.json',newgraph);graphs.append(newgraph);graphindex.append(dict(candidate_id=candidate,identity=identity,path=f'graphs/{token}.json',nodes=len(newgraph['nodes'])))
    table(out/'sessions.csv',sessions);table(out/'heldout.csv',heldout);table(out/'setwise.csv',sets);table(out/'errors.csv',errors)
    summary=dict(status='completed' if not errors else 'completed_with_failures',scope='diagnostic_first_cases' if args.max_cases else 'full_fixed_teacher_eligible_cohort',teacher_reference_slots=360,fixed_teacher_eligible_acquisitions=index['summary']['executed_sessions'],processed_acquisitions=processed,matched_source_views=matchedframes,student_empty_matched_views=int(emptyframes),teacher_nodes=sum(r['teacher_nodes'] for r in sessions),student_nodes=sum(r['student_nodes'] for r in sessions),teacher_fit_replays=teacher_replays,teacher_sessions_with_nodes=sum(r['teacher_nodes']>0 for r in sessions),student_sessions_with_nodes=sum(r['student_nodes']>0 for r in sessions),heldout_planned_rows_by_method={m:sum(r['method']==m for r in heldout) for m in ('teacher','student')},heldout_source_available_by_method={m:sum(r['method']==m and r['source_available'] for r in heldout) for m in ('teacher','student')},heldout_compatible_by_method={m:sum(r['method']==m and r['compatible'] for r in heldout) for m in ('teacher','student')},setwise_state_counts={state:sum(r['state']==state for r in sets) for state in sorted({r['state'] for r in sets})},setwise_median_symmetric_mean=float(np.median([r['symmetric_mean'] for r in sets if r['symmetric_mean'] is not None])) if any(r['symmetric_mean'] is not None for r in sets) else None,selected_adjacent_links=sum(e['selected'] and e['visit_gap']==1 for g in graphs for e in g['edges']),selected_gap_links=sum(e['selected'] and e['visit_gap']==2 for g in graphs for e in g['edges']),errors=len(errors),elapsed_seconds=time.time()-start,limits=['Metadata candidate identity is not verified physical continuity.','Differences measure sensitivity to foreground masks, not leaf-tracking accuracy.','Empty-set distances are unavailable; both-empty is not perfect agreement.','Coordinates remain conditional nominal orthographic training-height gauges.'])
    outer=[r for r in sets if r['split']=='heldout' and r['symmetric_mean'] is not None]
    summary['setwise_heldout_both_nonempty_rows']=len(outer)
    summary['setwise_heldout_median_symmetric_mean']=float(np.median([r['symmetric_mean'] for r in outer])) if outer else None
    summary['selected_genotype']=args.genotype;summary['selected_expected_acquisitions']=eligible_selected
    if args.genotype:summary['scope']='fixed_teacher_eligible_heldout_genotype'
    if not args.max_cases and processed!=eligible_selected:raise ValueError('Fixed-cohort denominator changed')
    if sha(args.cache/'frames.csv')!=provenance['cache_frames_sha256']:raise ValueError('Cache table changed during run')
    dump(out/'summary.json',summary);dump(out/'index.json',dict(summary=summary,candidates=graphindex));print(json.dumps(summary,indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
