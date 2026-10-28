"""Independent table/graph denominator audit after a matched foreground substitution."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import scipy,skimage

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def median(rows,key):
    values=[float(r[key]) for r in rows if r[key]!=''];return float(np.median(values)) if values else None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--teacher-tracking',type=Path,required=True);args=p.parse_args();root=args.output
    summary=json.loads((root/'summary.json').read_text());index=json.loads((root/'index.json').read_text());teacher=json.loads((args.teacher_tracking/'csi/index.json').read_text())['summary']
    sessions=read(root/'sessions.csv');heldout=read(root/'heldout.csv');sets=read(root/'setwise.csv');errors=read(root/'errors.csv')
    assert summary['status']=='completed' and len(sessions)==339 and not errors
    assert len({(r['treatment'],r['genotype'],r['replicate'],int(r['visit'])) for r in sessions})==339
    assert sum(int(r['teacher_nodes']) for r in sessions)==teacher['node_hypotheses']==642
    assert len(sets)==339*40 and len({(r['candidate_id'],r['visit'],r['view']) for r in sets})==339*40
    baseline=[r for r in heldout if r['method']=='teacher'];student=[r for r in heldout if r['method']=='student']
    assert len(baseline)==6420
    assert sum(r['source_available']=='True' for r in baseline)==teacher['heldout_track_view_rows']==6338
    assert sum(r['candidate_available']=='True' for r in baseline)==teacher['heldout_candidate_available']==5966
    assert sum(r['compatible']=='True' for r in baseline)==teacher['heldout_compatible_rows']==3852
    for row in heldout:
        assert int(row['view'])%4==2
        if row['source_available']=='False':assert row['nearest_own_terminal_distance']=='' and row['nearest_other_operator_terminal_distance']=='' and row['compatible']=='False'
    for row in sets:
        if row['state']!='both_nonempty':assert row['symmetric_mean']=='' and row['symmetric_p90']==''
    nodes=0;slots=0;links=0;gaps=0;angular=0;observed=0
    for entry in index['candidates']:
        graph=json.loads((root/entry['path']).read_text());assert graph['physical_identity_status']=='metadata_candidate_unverified';slots+=len(graph['sessions']);nodes+=len(graph['nodes'])
        links+=sum(e['selected'] and e['visit_gap']==1 for e in graph['edges']);gaps+=sum(e['selected'] and e['visit_gap']==2 for e in graph['edges'])
        for node in graph['nodes']:
            assert len(node['angle_evidence'])==40 and {r['view'] for r in node['angle_evidence']}==set(range(40))
            assert all(a['view']%4!=2 for a in node['assignments'])
            for angle in node['angle_evidence']:
                angular+=1
                if 'source' in angle:
                    source=angle['source'];assert source['mask_root'].endswith('final_joint_all40_960') and source['mask_operator_signature'];assert source['source_sha256'] and source['mask_sha256'];observed+=1
    assert slots==360 and len(index['candidates'])==72 and nodes==summary['student_nodes']==608
    assert links==summary['selected_adjacent_links'] and gaps==summary['selected_gap_links']
    methods=[]
    for method,rows in [('teacher',baseline),('student',student)]:
        available=[r for r in rows if r['source_available']=='True'];own=[r for r in rows if r['nearest_own_terminal_distance']!=''];other=[r for r in rows if r['nearest_other_operator_terminal_distance']!='']
        methods.append(dict(method=method,planned=len(rows),source_available=len(available),own_candidate_available=len(own),compatible=sum(r['compatible']=='True' for r in rows),median_own_distance_conditional=median(rows,'nearest_own_terminal_distance'),other_operator_candidate_available=len(other),cross_operator_compatible=sum(r['cross_operator_compatible']=='True' for r in rows),median_other_distance_conditional=median(rows,'nearest_other_operator_terminal_distance')))
    outer=[r for r in sets if r['split']=='heldout'];ratios=[float(r['student_training_height'])/float(r['teacher_training_height']) for r in sessions if r['student_training_height']!='']
    report=dict(status='passed',acquisitions=339,candidates=72,reference_visit_slots=slots,student_nodes=nodes,student_angular_records=angular,student_source_bound_angular_records=observed,unchanged_teacher_baseline=dict(nodes=642,planned_heldout=6420,source_available=6338,candidate_available=5966,compatible=3852),methods=methods,heldout_setwise_rows=len(outer),heldout_setwise_states={state:sum(r['state']==state for r in outer) for state in sorted({r['state'] for r in outer})},heldout_projected_point_distance_median_conditional=median(outer,'symmetric_mean'),heldout_raw_terminal_distance_median_conditional=median([r for r in outer if r['source_available']=='True'],'terminal_set_symmetric_mean'),training_height_student_teacher_ratio_median=float(np.median(ratios)),training_height_student_teacher_ratio_p10_p90=np.quantile(ratios,[.1,.9]).tolist(),versions=dict(numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__),auditor_sha256=sha(Path(__file__)),inputs_sha256={name:sha(root/name) for name in ('summary.json','configuration.json','sessions.csv','heldout.csv','setwise.csv','index.json')},scope='Numerical/operator sensitivity and source evidence, not anatomical validation.')
    (root/'independent_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
