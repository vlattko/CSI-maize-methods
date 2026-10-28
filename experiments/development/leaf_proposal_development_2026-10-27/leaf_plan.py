"""Bind real track hypotheses, paired-camera surrogates and source prompts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import leaf_proposals as p

TRACK_SOURCE=Path(__file__).parent.parent/'longitudinal_tracking_development_2026-10-27'
sys.path.insert(0,str(TRACK_SOURCE))
from tracking_observations import extract_paths

KEYS=('treatment','measurement','genotype','replicate')
ANGLES=(0,20)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)
def yes(v):return str(v).lower() in ('true','1')
def key(row):return tuple(str(row[k]) for k in KEYS)

def load_parent(row,full_run):
    if row is None:raise ValueError('extraction_record_unavailable')
    if not yes(row['qc_pass']):raise ValueError('parent_foreground_qc_rejected')
    path=full_run/row['mask'];blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=row['mask_sha256']:raise ValueError('parent_mask_hash_mismatch')
    with Image.open(io.BytesIO(blob)) as im:
        if im.size!=(int(row['processing_width']),int(row['processing_height'])):raise ValueError('parent_mask_dimensions')
        return np.asarray(im.convert('L').resize((256,192),Image.Resampling.NEAREST))>0

def observations(node,view):
    evidence=next((e for e in node.get('angle_evidence',[]) if e['view']==view),None)
    if not evidence or not evidence['terminal_observed'] or not evidence.get('observed_candidate'):return None
    return evidence['observed_candidate']

def case_pool(tracking,frame_lookup,cameras):
    index=read(tracking/'csi/index.json');result=[]
    for item in index['candidates']:
        graph_path=tracking/'csi'/item['path'];graph=read(graph_path)
        for session in graph['sessions']:
            identity={**graph['identity'],'measurement':str(session['visit'])}
            identity={k:str(identity[k]) for k in KEYS}
            rr=frame_lookup.get((*key(identity),'rgb',0))
            if rr is None:continue
            camera=cameras.get(rr['session'])
            nodes=session['nodes']
            count=[sum(observations(n,v) is not None for n in nodes) for v in ANGLES]
            paired=all(frame_lookup.get((*key(identity),c,v)) is not None and
                       yes(frame_lookup[(*key(identity),c,v)]['qc_pass']) for c in ('rgb','noir') for v in ANGLES)
            result.append(dict(identity={**identity,'session':rr['session']},nodes=nodes,session=session,
                 graph_path=graph_path,graph_sha256=sha(graph_path),camera=camera,
                 eligible=bool(nodes and min(count)>0 and paired and camera and not camera['deployment_abstain']),
                 observed_nodes_per_angle=count,paired_usable=paired))
    return result

def build_frame(case,camera,view,frame_lookup,full_run):
    identity=case['identity'];row=frame_lookup.get((*key(identity),camera,view))
    frame=dict(identity=identity,camera=camera,view=view,status='planned',proposals=[],failures=[],
       graph_path=str(case['graph_path']),graph_sha256=case['graph_sha256'],
       region_status='unreviewed_track_supported_region_not_anatomical_ground_truth',
       camera_surrogate_selection=case['camera'].get('periodic_selection') if case['camera'] else None)
    if row:frame['source_record']={k:row[k] for k in ('source','source_sha256','mask','mask_sha256',
         'processing_width','processing_height','native_width','native_height','qc_pass','qc_issues','configuration_signature')}
    try:parent=load_parent(row,full_run)
    except (ValueError,OSError) as exc:
        frame.update(status='unavailable',failure_reason=str(exc));return frame
    with Image.open(full_run/row['mask']) as im:parent_processing=np.asarray(im.convert('L'))>0
    local=extract_paths(parent,min_length=6)
    if not local:
        frame.update(status='no_local_distal_branch',failure_reason='no_recomputed_skeleton_paths');return frame
    ranked=sorted(case['nodes'],key=lambda n:(-n['training_support'],n['node_id']))
    observed=[n for n in ranked if observations(n,view)]
    frame.update(observed_track_nodes=len(observed),configured_track_node_cap=4,
                 omitted_by_cap=max(0,len(observed)-4),source_grid=[256,192])
    matched=[]
    for node in observed[:4]:
        obs=observations(node,view);terminal=obs['terminal_pixel_processing']
        candidate=dict(candidate_id=node['node_id'],track_id=node['track_id'],training_support=node['training_support'],
            graph_observation_status='terminal_observed',rgb_terminal_grid=terminal,
            candidate_identity_unverified=True,region_review_status='unreviewed',camera=camera,view=view)
        try:
            if camera=='rgb':
                match=p.match_terminal(terminal,local,2.)
                candidate['prompt_origin']='observed_RGB_track_terminal_and_own_skeleton_branch'
            else:
                cam=case['camera']
                if not cam or cam['deployment_abstain']:raise ValueError('camera_surrogate_abstains_for_NoIR_prompt_transfer')
                m=np.asarray(cam['view_matrices_rgb_to_noir'][str(view)])
                mapped=m@np.r_[terminal,1]
                height=max(c['mask_height_px'] for c in local)
                match=p.match_terminal(mapped,local,max(6,.08*height))
                candidate.update(prompt_origin='weak_camera_transfer_then_observed_NoIR_skeleton_branch',
                    mapped_rgb_terminal_grid=mapped.tolist(),mapping_method=cam['deployment_method'],
                    mapping_matrix_rgb_to_noir=m.tolist(),transfer_is_tissue_correspondence=False)
            candidate['terminal_match']={k:v for k,v in match.items() if k!='candidate'}
            chosen=match['candidate']
            if chosen is None:raise ValueError('observed_terminal_not_within_match_gate')
            branch=chosen['visible_branch_over_height']*chosen['mask_height_px']
            others=[[c['terminal_x_px'],c['terminal_y_px']] for c in local]
            prompt=p.make_prompt(chosen['path_processing_px'],[chosen['terminal_x_px'],chosen['terminal_y_px']],branch,parent,others)
            prompt_processing=p.processing_prompt(prompt,parent_processing)
            candidate.update(status='ready',local_source_candidate_id=chosen['candidate_id'],
                prompt_grid=prompt,prompt_processing=prompt_processing,
                local_terminal_grid=[chosen['terminal_x_px'],chosen['terminal_y_px']])
            matched.append(chosen['candidate_id']);frame['proposals'].append(candidate)
        except (ValueError,KeyError) as exc:
            candidate.update(status='failed_prompt',reason=str(exc));frame['failures'].append(candidate)
    duplicates={k for k,n in Counter(matched).items() if n>1}
    for candidate in frame['proposals']:
        candidate['duplicate_local_terminal_match']=candidate['local_source_candidate_id'] in duplicates
    frame['status']='ready' if frame['proposals'] else 'no_ready_prompt'
    return frame

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tracking',type=Path,required=True)
    parser.add_argument('--camera-lookup',type=Path,required=True);parser.add_argument('--full-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    table=rows(args.full_run/'extraction/view_traits.csv')
    lookup={(*key(r),r['camera'].lower(),int(r['view'])):r for r in table}
    if len(lookup)!=len(table):raise ValueError('duplicate_four_key_camera_index')
    cameras={r['session']:r for r in read(args.camera_lookup)['rows']}
    pool=case_pool(args.tracking,lookup,cameras);chosen=[];missing=[]
    for treatment in ('C','P','S'):
        for visit in ('1','2','3','4','5'):
            group=sorted([r for r in pool if r['identity']['treatment']==treatment and r['identity']['measurement']==visit],
                         key=lambda r:r['identity']['session'])
            eligible=[r for r in group if r['eligible']]
            if not group:missing.append(dict(treatment=treatment,measurement=visit,reason='no_graph_case'));continue
            chosen.append((eligible or group)[0])
    frames=[build_frame(case,camera,view,lookup,args.full_run) for case in chosen for camera in ('rgb','noir') for view in ANGLES]
    source=Path(__file__).parent
    plan=dict(schema=1,pilot=True,planned_strata=15,planned_camera_frames=60,
       actual_strata=len(chosen),actual_frame_tasks=len(frames),angles=ANGLES,cameras=['rgb','noir'],policy=p.POLICY,
       case_selection='lexicographic acquisition per treatment/visit meeting both observed RGB tracks and accepted camera transfer; fallback retained explicitly',
       cases=[dict(identity=c['identity'],eligible=c['eligible'],observed_nodes_per_angle=c['observed_nodes_per_angle'],
                   graph_sha256=c['graph_sha256']) for c in chosen],missing_strata=missing,frames=frames,
       frame_status=dict(Counter(f['status'] for f in frames)),ready_prompts=sum(len(f['proposals']) for f in frames),
       failed_prompts=sum(len(f['failures']) for f in frames),
       provenance=dict(view_traits_sha256=sha(args.full_run/'extraction/view_traits.csv'),
           tracking_index_sha256=sha(args.tracking/'csi/index.json'),camera_lookup_sha256=sha(args.camera_lookup),
           tracking_observations_sha256=sha(TRACK_SOURCE/'tracking_observations.py'),
           source_hashes={x.name:sha(x) for x in source.glob('*.py')}))
    plan['signature']=digest(plan);atomic(args.output/'pilot_plan.json',plan)
    print(json.dumps({k:plan[k] for k in ('actual_strata','actual_frame_tasks','ready_prompts','failed_prompts','frame_status','signature')}),flush=True)

if __name__=='__main__':main()
