"""Independent denominator, one-to-one map and exposure-table consistency audit."""
import argparse,collections,json,math
from pathlib import Path
from architecture_analysis import read,dump,sha

def run(args):
    out=args.run/'architecture'/'external_pheno4d';rendered=out/'rendered';tracks=read(rendered/'rotational_track_oracle_purity.csv');obs=read(rendered/'oracle_organ_observability.csv');held=read(rendered/'heldout_oracle_organ_compatibility.csv');support=read(out/'scaffold/terminal_scaffold_point_support.csv');cross=read(out/'scaffold/automatic_scaffold_crossdate_links.csv');cases=read(rendered/'rendered_cases.csv')
    mapped=[(r['scan'],int(r['mapped_id_evaluation_only'])) for r in tracks if int(r['mapped_id_evaluation_only'])>0]
    assert len(mapped)==len(set(mapped))==len(tracks),'Duplicate/unmapped true instance-date tracks'
    observable=collections.Counter();all_instances=set()
    for r in obs:
        k=(r['scan'],int(r['true_leaf_id_for_scoring']));all_instances.add(k)
        if r['split']=='train' and int(r['oracle_terminal_candidates'])>0:observable[k]+=1
    eligible={k for k,n in observable.items() if n>=8};assert set(mapped).issubset(eligible)
    assert len(held)==len({(r['scan'],r['track_id'],r['view']) for r in held})
    assert all(int(r['view'])%4==2 for r in held)
    assert len(support)==len(tracks)==len({(r['scan'],r['track_id']) for r in support})
    for case in cases:
        scan=case['scan'];model=rendered/f'frozen_{scan}.json';mapping=json.loads((rendered/f'frozen_scoring_map_{scan}.json').read_text(encoding='utf-8'))
        assert mapping['input_frozen_model_sha256']==sha(model)
        assert all(v%4!=2 for v in mapping['train_views'])
        assert len(mapping['mapping'].values())==len(set(mapping['mapping'].values()))
        for token,label in mapping['mapping'].items():
            row=mapping['training_ownership_counts'][int(token)];col=mapping['true_ids_for_scoring'].index(label);assert row[col]==max(row)
            r=next(r for r in tracks if r['scan']==scan and int(r['track_id'])==int(token));assert int(r['mapped_id_evaluation_only'])==label
    raw_gt_instances=None;raw_gt_common=None
    if args.external:
        import numpy as np
        manifest=json.loads((args.external/'download_manifest.json').read_text(encoding='utf-8'));rawsets={}
        for f in manifest['files']:
            p=args.external/f['path'];labels=np.loadtxt(p,dtype='int16',usecols=[4]);actual={int(i) for i in np.unique(labels) if i>0};rawsets[p.stem]=actual
            expected={label for scan,label in all_instances if scan==p.stem};assert actual==expected,'Rendered denominator lost an original instance'
        raw_gt_instances=sum(len(s) for s in rawsets.values());raw_gt_common=sum(len(rawsets[f'M{i:02d}_0324_a']&rawsets[f'M{i:02d}_0325_a']) for i in range(1,8))
    for plant in set(r['plant'] for r in cross):
        rr=[r for r in cross if r['plant']==plant];assert len(rr)==len(set(r['earlier_track'] for r in rr))==len(set(r['later_track'] for r in rr))
        for r in rr:
            first=next(q for q in tracks if q['plant']==plant and '_0324_' in q['scan'] and q['track_id']==r['earlier_track']);later=next(q for q in tracks if q['plant']==plant and '_0325_' in q['scan'] and q['track_id']==r['later_track']);assert first['mapped_id_evaluation_only']==r['earlier_truth_for_scoring'];assert later['mapped_id_evaluation_only']==r['later_truth_for_scoring']
    visibility=[r for r in read(out/'exposure/fixed_receiver_visibility.csv') if r['experiment']=='primary'];edges=[r for r in read(out/'exposure/fixed_shape_neighbour_removal_edges.csv') if r['experiment']=='primary'];rawcases=[r for r in read(out/'exposure/exposure_cases.csv') if int(r['resolution'])==256];fields=['plant','date','scheme','view','receiver_id'];lookup={tuple(r[k] for k in fields):r for r in visibility}
    assert len(lookup)==len(visibility)
    assert len(visibility)==sum(int(r['removable_regions'])*40 for r in rawcases)
    assert len(edges)==sum(int(r['removable_regions'])*(int(r['removable_regions'])-1)*40 for r in rawcases)
    keyset=set();gain_sums=collections.Counter()
    for r in edges:
        key=tuple(r[k] for k in fields);v=lookup[key];full=key+(r['removed_id'],);assert full not in keyset;keyset.add(full)
        den=int(r['potential_pixels']);before=int(r['baseline_visible_pixels']);after=int(r['after_removal_visible_pixels']);gain=int(r['gain_pixels']);assert den==int(v['potential_pixels']);assert before==int(v['baseline_visible_pixels']);assert gain==after-before>=0
        assert int(r['removed_id'])!=int(r['receiver_id'])
        if r['scheme']=='leaf_collar':assert int(r['removed_id'])>=2 and int(r['receiver_id'])>=2
        if den:assert math.isclose(float(r['gain_fraction']),gain/den,abs_tol=1e-12)
        gain_sums[key]+=gain
    for key,r in lookup.items():
        assert int(r['potential_pixels'])==int(r['baseline_visible_pixels'])+int(r['exclusively_blocked_pixels'])+int(r['redundantly_blocked_pixels'])
        assert gain_sums[key]<=int(r['exclusively_blocked_pixels'])
    result=dict(status='passed',scans=len(cases),true_instance_dates=len(all_instances),observable_true_instance_dates=len(eligible),recovered_unique_true_instance_dates=len(mapped),duplicate_recovered_instance_dates=0,mapped_points_outside_observability=0,heldout_unique_rows=len(held),nonheldout_rows=0,scoring_mapping='One-to-one TRAIN ownership-count assignment; persisted before heldout scoring. Every mapped label attains its track row maximum. No numeric label enters model features or hypothesis selection.',gt_assistance='Supplied foreground semantics and known cameras for rendering; true IDs for training-only scoring-map and heldout ownership/support scoring. Supplied instances only in the separately labelled instance-correspondence benchmark; true region geometry/labels in exposure interventions.',crossdate_predicted_links=len(cross),crossdate_correct_links=sum(r['correct'].lower()=='true' for r in cross),primary_exposure_states=len(visibility),primary_exposure_edges=len(edges),fixed_receiver_and_conservation_checks='Every primary edge/state independently verified, including stem retention and nonnegative gain.',script_sha256=sha(__file__),input_hashes={str(p.relative_to(out)):sha(p) for p in [rendered/'rotational_track_oracle_purity.csv',rendered/'oracle_organ_observability.csv',rendered/'heldout_oracle_organ_compatibility.csv',out/'scaffold/terminal_scaffold_point_support.csv',out/'scaffold/automatic_scaffold_crossdate_links.csv',out/'exposure/fixed_shape_neighbour_removal_edges.csv']})
    result['raw_original_tip_instance_dates']=raw_gt_instances;result['raw_original_common_tip_instances']=raw_gt_common
    dump(out/'external_denominator_audit.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--external',type=Path);run(p.parse_args())
