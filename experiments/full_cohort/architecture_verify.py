"""Independent full-run denominator, held-out split and bounded-domain checks."""
import argparse,collections,json,math
from pathlib import Path
import numpy as np
from architecture_analysis import read,write,dump,sha,accepted,load_mask

def run(args):
    out=args.run/'architecture';s=json.loads((out/'architecture_summary.json').read_text(encoding='utf-8'));frames=read(args.run/'extraction/view_traits.csv');coverage=read(out/'full_reference_architecture_coverage.csv');harm=read(out/'harmonic_descriptors.csv');maskaudit=read(out/'mask_hash_audit.csv');cases=read(out/'tracking/tracking_cases.csv');tracks=read(out/'tracking/rotational_track_hypotheses.csv');held=read(out/'tracking/heldout_terminal_compatibility.csv')
    assert s['mode']=='full_extraction' and s['input_sha256']==sha(args.run/'extraction/view_traits.csv')
    assert len(frames)==s['input_rows'];assert sum(accepted(r) for r in frames)==s['accepted_input_rows']==len(maskaudit)
    assert len(coverage)==720 and len({tuple(r[k] for k in ['treatment','measurement','genotype','replicate','camera']) for r in coverage})==720
    primary=sum(r['primary_eligible'].lower()=='true' for r in coverage);all40=sum(r['all40_accepted_budget_eligible'].lower()=='true' for r in coverage);assert len([r for r in harm if r['status']=='fit'])==primary*8
    phase_counts=collections.Counter()
    for r in read(out/'view_budget_stability.csv'):phase_counts[(r['session'],r['camera'],r['metric'],r['budget'])]+=1
    assert all(n==40 for n in phase_counts.values());assert len(phase_counts)==all40*8*4
    assert len(held)==len({(r['session'],r['track_id'],r['view']) for r in held}) and all(int(r['view'])%4==2 for r in held)
    assert len(tracks)==len({(r['session'],r['track_id']) for r in tracks})
    for p in (out/'tracking').glob('frozen_*.json'):
        model=json.loads(p.read_text(encoding='utf-8'));assert all(v%4!=2 for v in model['train_views'])
        assert all(a['view']%4!=2 for t in model['tracks'] for a in t['assignments'])
    gcases=read(out/'geometry_diagnostics/geometry_diagnostic_cases.csv');assert len(gcases)==15;trajectories=collections.Counter((r['treatment'],r['genotype'],r['replicate']) for r in gcases);assert sorted(trajectories.values())==[5,5,5]
    geometry=read(out/'geometry_diagnostics/numerical_and_camera_scores.csv');selected=read(out/'geometry_diagnostics/train_selected_camera_diagnostic.csv')
    for chosen in selected:
        models={}
        for r in geometry:
            if r['case']==chosen['case'] and r['experiment']=='perspective_family':models.setdefault((float(r['distance_over_height']),float(r['roll_deg']),float(r['elevation_deg'])),[]).append(r)
        best=min(models,key=lambda k:(-np.median([float(r['iou']) for r in models[k] if r['split']=='train']),k));assert best==(float(chosen['distance_over_height']),float(chosen['roll_deg']),float(chosen['elevation_deg']))
    domain=[]
    for case in gcases:
        rr=[r for r in frames if r['session']==case['session'] and r['camera']=='rgb' and accepted(r) and int(r['view'])%4!=2];bounds=[];axes=[];bases=[];heights=[]
        for row in rr:
            m=load_mask(row,args.run,160);y,x=np.where(m);axes.append(float(np.median(x[y>=np.percentile(y,95)])));bases.append(float(y.max()));heights.append(float(np.ptp(y)+1));bounds.append((float(x.min()),float(x.max()),float(y.min()),float(y.max())))
        axis=float(np.median(axes));base=float(np.median(bases));scale=float(np.median(heights));horizontal=max(max(abs(x0-axis),abs(x1-axis))/scale for x0,x1,y0,y1 in bounds);top=max((base-y0)/scale for x0,x1,y0,y1 in bounds);bottom=min((base-y1)/scale for x0,x1,y0,y1 in bounds)
        domain.append({k:case[k] for k in ['treatment','measurement','genotype','replicate','session']}|dict(max_abs_horizontal_over_height=horizontal,max_up_over_height=top,min_up_over_height=bottom,nominal_orthographic_horizontal_domain_exceeded=horizontal>.8,nominal_orthographic_top_domain_exceeded=top>1.15,note='Audit of fixed[-.8,.8]horizontal/[0,1.15]up prior at nominalorthography; alternativeperspectiveconesdiffer. No anatomical inference.'))
    write(out/'geometry_diagnostics/declared_domain_support_audit.csv',domain)
    result=dict(status='passed',input_rows=len(frames),accepted_hash_bound_masks=len(maskaudit),reference_camera_rows=len(coverage),primary_camera_rows=primary,all40_budget_camera_rows=all40,phase_groups=len(phase_counts),phase_rows=sum(phase_counts.values()),track_hypotheses=len(tracks),executed_rgb_cases=sum(r['status']=='executed' for r in cases),rgb_cases_with_tracks=sum(r['status']=='executed' and int(r['track_hypotheses'])>0 for r in cases),heldout_unique_rows=len(held),deeper_cases=len(gcases),deeper_candidate_trajectories=len(trajectories),camera_selection_training_only='Verified from full alternatives for every selected camera',nominal_horizontal_domain_exceeded_cases=sum(r['nominal_orthographic_horizontal_domain_exceeded'] for r in domain),nominal_top_domain_exceeded_cases=sum(r['nominal_orthographic_top_domain_exceeded'] for r in domain),script_sha256=sha(__file__))
    dump(out/'architecture_full_run_verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);run(p.parse_args())
