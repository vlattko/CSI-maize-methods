"""Recompute denominators and inspect local camera conditioning from frozen fits."""
import argparse,ast,gzip,json
from pathlib import Path
import numpy as np
from perspective_points import marginal_camera_information
from perspective_run import read,write,dump,sha,log


def audit(output,full_run):
    cache=json.loads(gzip.decompress((output/'observations.json.gz').read_bytes()));selection=json.loads((output/'frozen_camera_selection.json').read_text());summary=json.loads((output/'summary.json').read_text());config=json.loads((output/'configuration.json').read_text());findings=[]
    if cache['input_view_table_sha256']!=sha(full_run/'extraction/view_traits.csv'):findings.append('Input view table changed')
    for name,digest in config['execution_source_sha256'].items():
        if sha(output/'execution_sources'/name)!=digest:findings.append('Execution snapshot digest mismatch '+name)
    previous=ast.parse((output/'execution_sources/perspective_points.py').read_text());current=ast.parse(Path(__file__).with_name('perspective_points.py').read_text())
    old_functions={n.name:ast.dump(n) for n in previous.body if isinstance(n,ast.FunctionDef)};current_functions={n.name:ast.dump(n) for n in current.body if isinstance(n,ast.FunctionDef)}
    if any(current_functions.get(k)!=v for k,v in old_functions.items()):findings.append('Fitting function changed after execution')
    full=[];inner=[];by={c['case_id']:c for c in cache['cases']}
    for case in cache['cases']:
        for t in case['full_tracks']:
            full.append(t['node_id'])
            if any(v%4==2 for v in t['views']):findings.append('Outer leakage in full fit')
        for t in case['inner_tracks']:
            inner.append(t['node_id'])
            if any(v%4 in [1,2] for v in t['views']):findings.append('Inner leakage')
    if len(full)!=642 or len(full)!=len(set(full)):findings.append('Full node denominator or uniqueness')
    if len(inner)!=492 or len(inner)!=len(set(inner)):findings.append('Inner node denominator or uniqueness')
    if selection['input_observation_cache_sha256']!=sha(output/'observations.json.gz'):findings.append('Selection cache digest mismatch')
    families=read(output/'inner_family_summary.csv');best=min(families,key=lambda r:(float(r['inner_case_median']),r['family']))
    if best['family']!=selection['selected']:findings.append('Selection differs from inner objective')
    if len(families)!=30 or any(int(r['point_fits'])!=492 for r in families):findings.append('Family fit denominator')
    outer=read(output/'outer_predictions.csv');methods=sorted(set(r['model'] for r in outer));stats=[];caseerrors={}
    if len({(r['model'],r['node_id'],r['view']) for r in outer})!=len(outer):findings.append('Duplicate outer rows')
    if any(int(r['view'])%4!=2 for r in outer):findings.append('Outer split contains training angles')
    for method in methods:
        rr=[r for r in outer if r['model']==method];nodes={r['node_id'] for r in rr}
        if nodes!=set(full) or len(rr)!=6420:findings.append('Model denominator '+method)
        available=[r for r in rr if r['nearest_terminal_distance']!=''];median=float(np.median([float(r['nearest_terminal_distance']) for r in available]));scores={}
        for r in rr:
            if r['clipped_error_for_camera_selection']!='':scores.setdefault(r['case_id'],[]).append(float(r['clipped_error_for_camera_selection']))
        caseerrors[method]={k:float(np.sqrt(np.mean(np.square(v)))) for k,v in scores.items()}
        observed=next(r for r in summary['outer_results'] if r['method']==method)
        if abs(median-observed['median_nearest_terminal_distance'])>1e-12:findings.append('Median mismatch '+method)
        stats.append(dict(method=method,planned_rows=len(rr),candidate_available=len(available),node_count=len(nodes)))
    projected=read(output/'selected_family_projection_40views.csv')
    if len(projected)!=642*40 or len({(r['node_id'],r['view']) for r in projected})!=642*40:findings.append('40-view projection denominator')
    selected=selection['selected'];orthographic=selection['selected_orthographic'];conditioning=[];global_rows={selected:[],orthographic:[]};modelcount=0;pointconditions=[]
    for path in sorted((output/'session_models').glob('*.json')):
        record=json.loads(path.read_text());case=by[path.stem];tracks={t['node_id']:t for t in case['full_tracks']}
        for family in [selected,orthographic]:
            rows=[];camera=None
            for entry in record['models']:
                model=entry['model']
                if model['family']['family']!=family:continue
                camera=model['family'];t=tracks[entry['node_id']]
                rows.append(dict(point=model['point'],views=t['views'],offset=model['offset'],scale=case['full_scale']))
                if family==selected:
                    modelcount+=1;pointconditions.append(model['point_jacobian_condition'])
                    if model['status']!='fitted':findings.append('Selected fit failure')
                    if camera['kind']=='perspective' and model['minimum_camera_depth']<.149:findings.append('Inadequate selected depth')
            if rows:
                info=marginal_camera_information(rows,camera);global_rows[family]+=rows
                conditioning.append(dict(case_id=case['case_id'],family=family,**info))
    globals_out={}
    for family,rr in global_rows.items():
        f=next(r for r in selection['families'] if r['family']==family)
        globals_out[family]=marginal_camera_information(rr,f)
    paired=[]
    for caseid,base in caseerrors['constant_height_harmonic'].items():
        paired.append(dict(case_id=caseid,baseline_case_rmse=base,selected_case_rmse=caseerrors[selected][caseid],
            baseline_minus_selected=base-caseerrors[selected][caseid],unrestricted_2D_case_rmse=caseerrors['unrestricted_2D_harmonic'][caseid]))
    values=[r['baseline_minus_selected'] for r in paired]
    result=dict(status='passed' if not findings else 'failed',findings=findings,eligible_acquisitions=len(cache['cases']),full_nodes=len(full),inner_nodes=len(inner),
        inner_point_fits=30*len(inner),retained_family_full_point_fits=len(selection['retained'])*len(full),outer_model_count=len(methods),outer_rows=len(outer),
        selected_projection_rows=len(projected),selected_fitted_points=modelcount,retained_camera_families=len(selection['retained']),
        median_selected_point_jacobian_condition=float(np.median(pointconditions)),
        paired_cases=len(paired),cases_selected_clipped_rmse_better=sum(v>0 for v in values),cases_selected_clipped_rmse_worse=sum(v<0 for v in values),
        median_paired_baseline_minus_selected=float(np.median(values)),p10_paired_difference=float(np.percentile(values,10)),p90_paired_difference=float(np.percentile(values,90)),
        global_conditional_camera_information=globals_out,
        split_rules=dict(inner_training='view%4 in[0,3]',inner_validation='view%4==1',outer_validation='view%4==2'),
        inner_detection_and_gauge_refitted=True,original_fitting_functions_unchanged=True,
        conditioning_function_appended_after_fit=True,diagnostic_source_sha256=sha(__file__),current_core_sha256=sha(Path(__file__).with_name('perspective_points.py')),
        caveat='Local full-rank information does not override the20 retained projection families or validate inferred organ identity, fixed origins or rigid-point assumptions.')
    write(output/'conditioning_by_case.csv',conditioning);write(output/'paired_outer_case_differences.csv',paired);dump(output/'independent_audit.json',result);log(output,'independent_audit_completed',**result)
    if findings:raise ValueError(findings)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--full-run',type=Path,required=True);a=p.parse_args();audit(a.output,a.full_run)
