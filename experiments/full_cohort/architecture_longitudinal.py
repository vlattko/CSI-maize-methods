"""Full-reference coverage and repeated-label visit differences in image traits."""
import argparse,itertools,json,math
from pathlib import Path
import numpy as np
from architecture_analysis import read,write,dump,sha,accepted,coverage,number,METRICS
FOUR=['treatment','measurement','genotype','replicate']

def aggregate_reference(master,frames,verified,mode):
    groups={}
    for r in frames:groups.setdefault(tuple(str(r[k]) for k in FOUR)+(r['camera'],),[]).append(r)
    output=[]
    for reference in master:
        key=tuple(str(reference[k]) for k in FOUR)
        for camera in ['rgb','noir']:
            rr=groups.get(key+(camera,),[]);good=[r for r in rr if accepted(r) and key+(camera,int(r['view'])) in verified];cov=coverage([r['view'] for r in good]);out={**dict(zip(FOUR,key)),'camera':camera,'source_rows':len(rr),'raw_qc_accepted_views':sum(accepted(r) for r in rr),'verified_accepted_views':len(good),'quadrants':cov['quadrants'],'primary_eligible':cov['eligible'],'strict_eligible':cov['strict_eligible'],'all40_accepted_budget_eligible':len(good)==40,'record_state':'observed' if rr else ('not_in_partial_snapshot' if mode!='full_extraction' else 'no_decodable_frame'),'physical_continuity':'unconfirmed_repeated_label_candidate'}
            for metric in METRICS:
                values=np.array([number(r.get(metric)) for r in good]);values=values[np.isfinite(values)]
                out[metric+'_median']=float(np.median(values)) if len(values) and cov['eligible'] else None
            output.append(out)
    return output

def run(args):
    out=args.run/'architecture'/args.output_name;summary=json.loads((out/'architecture_summary.json').read_text(encoding='utf-8'));frames=read(summary['input_table']);verified={tuple(r[k] for k in FOUR)+(r['camera'],int(r['view'])) for r in read(out/'mask_hash_audit.csv')};master=read(args.intake/'research_audit/reference_master.csv')
    if len(master)!=360:raise ValueError('Full reference denominator must be360')
    table=aggregate_reference(master,frames,verified,summary['mode']);write(out/'full_reference_architecture_coverage.csv',table);trajectories={}
    for r in table:trajectories.setdefault((r['treatment'],r['genotype'],r['replicate'],r['camera']),{})[int(r['measurement'])]=r
    changes=[];coverage_rows=[]
    for identity,visits in sorted(trajectories.items()):
        base=dict(zip(['treatment','genotype','replicate','camera'],identity));coverage_rows.append({**base,'expected_visits':5,'observed_visits':sum(r['source_rows']>0 for r in visits.values()),'primary_eligible_visits':sum(r['primary_eligible'] for r in visits.values()),'strict_eligible_visits':sum(r['strict_eligible'] for r in visits.values()),'all_five_primary_eligible':all(v in visits and visits[v]['primary_eligible'] for v in range(1,6))})
        for earlier in range(1,5):
            a=visits.get(earlier);b=visits.get(earlier+1);eligible=bool(a and b and a['primary_eligible'] and b['primary_eligible'])
            for metric in METRICS:
                av=a.get(metric+'_median') if a else None;bv=b.get(metric+'_median') if b else None;changes.append({**base,'earlier_visit':earlier,'later_visit':earlier+1,'metric':metric,'paired_primary_eligible':eligible,'earlier_value':av,'later_value':bv,'later_minus_earlier':bv-av if eligible and av is not None and bv is not None else None,'interpretation':'Candidate repeated-label difference in image-frame descriptors, not physical growth or rate'})
    write(out/'candidate_trajectory_coverage.csv',coverage_rows);write(out/'candidate_visit_differences.csv',changes)
    strata=[]
    for treatment,visit,camera in itertools.product(['C','P','S'],range(1,6),['rgb','noir']):
        rr=[r for r in table if r['treatment']==treatment and int(r['measurement'])==visit and r['camera']==camera];strata.append(dict(treatment=treatment,visit=visit,camera=camera,reference_rows=len(rr),observed_rows=sum(r['source_rows']>0 for r in rr),primary_eligible=sum(r['primary_eligible'] for r in rr),strict_eligible=sum(r['strict_eligible'] for r in rr),all40_accepted_budget_eligible=sum(r['all40_accepted_budget_eligible'] for r in rr)))
    write(out/'coverage_selection_by_treatment_visit_camera.csv',strata)
    result=dict(mode=summary['mode'],reference_assay_rows=len(master),reference_camera_rows=len(table),observed_camera_rows=sum(r['source_rows']>0 for r in table),primary_eligible_camera_rows=sum(r['primary_eligible'] for r in table),strict_eligible_camera_rows=sum(r['strict_eligible'] for r in table),all40_accepted_camera_rows=sum(r['all40_accepted_budget_eligible'] for r in table),candidate_camera_trajectories=len(coverage_rows),all_five_primary_eligible_camera_trajectories=sum(r['all_five_primary_eligible'] for r in coverage_rows),candidate_difference_rows=len(changes),eligible_candidate_difference_rows=sum(r['paired_primary_eligible'] for r in changes),script_sha256=sha(__file__),scope='Visit-indexed normalized image measurements only; repeated labels are candidate continuity. Missing values remain missing;360references retained.')
    dump(out/'longitudinal_architecture_summary.json',result)
    dump(out/'additional_organ_counterfactual_eligibility.json',dict(new_validated_organ_geometry_cases=0,new_true_organ_geometry_counterfactuals=0,reason='No additional anatomically reviewed organ identity and independently supported geometry. Visual-hull envelopes and skeleton paths cannot justify removing/repositioning a physical leaf. Existing source-assisted pilots remain separately scoped; no seeds transferred.'))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--intake',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output-name',default='');run(p.parse_args())
