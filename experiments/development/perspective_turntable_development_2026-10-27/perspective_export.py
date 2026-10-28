"""Export four-key point/projection lookup without altering scientific fits."""
import argparse,json
from pathlib import Path
from perspective_run import read,dump,sha,log


def export(output,tracking):
    selection=json.loads((output/'frozen_camera_selection.json').read_text());summary=json.loads((output/'summary.json').read_text());index=json.loads((tracking/'index.json').read_text());predictions=read(output/'selected_family_projection_40views.csv');by={}
    for r in predictions:by.setdefault(r['node_id'],{})[r['view']]={k:float(r[k]) for k in ['pixel_x','pixel_y','image_x_fraction','image_y_fraction']}
    sessionmodels={}
    for path in (output/'session_models').glob('*.json'):
        record=json.loads(path.read_text());key=tuple(str(record['identity'][k]) for k in ['treatment','measurement','genotype','replicate']);sessionmodels[key]=record
    rows=[];total=0
    for row in index['candidates']:
        graph=json.loads((tracking/row['path']).read_text())
        for session in graph['sessions']:
            identity={**graph['identity'],'measurement':str(session['visit'])};key=tuple(str(identity[k]) for k in ['treatment','measurement','genotype','replicate']);record=sessionmodels.get(key);nodes=[]
            if record:
                for original in session['nodes']:
                    variants=[e['model'] for e in record['models'] if e['node_id']==original['node_id']];selected=next(m for m in variants if m['family']['family']==selection['selected']);p=selected['point']
                    nodes.append(dict(node_id=original['node_id'],track_id=original['track_id'],point=dict(x=p[0],up=p[1],depth=p[2]),
                        fit_status=selected['status'],training_rmse_image_width=selected['training_rmse'],point_jacobian_condition=selected['point_jacobian_condition'],
                        minimum_camera_depth=selected['minimum_camera_depth'],point_parameter_covariance=selected['point_parameter_covariance'],
                        projection_by_view=by[original['node_id']],
                        retained_family_points=[dict(family=m['family']['family'],point=dict(x=m['point'][0],up=m['point'][1],depth=m['point'][2])) for m in variants],
                        source_projection_evidence_ref=dict(candidate_graph=row['path'],node_id=original['node_id']),
                        uncertainty='Conditional point-fit and finite camera-family sensitivity, not anatomical confidence'))
            total+=len(nodes);rows.append(dict(**identity,status='conditional_points_available' if nodes else 'no_retained_point_hypothesis',nodes=nodes,
                original_coverage=session['coverage'],gauge=record['gauge'] if record else session.get('gauge'),
                horizontal_offset_image_width=next((e['model']['offset'] for e in record['models'] if e['model']['family']['family']==selection['selected']),None) if record else None))
    if len(rows)!=360 or total!=642:raise ValueError('Lookup denominator mismatch')
    lookup=dict(status='completed',schema=1,model_kind='conditional_rigid_turntable_point',rows=rows,reference_rows=360,node_hypotheses=total,
        selected_family=next(f for f in selection['families'] if f['family']==selection['selected']),retained_family_names=selection['retained'],
        source_grid_width=256,source_grid_height=192,parameter_order=['x','up','depth'],
        point_coordinate_units='Per-acquisition perspective camera distance D=1 convention; cross-visit physical distance equivalence unverified; orthographic alternatives use image-width gauge',
        selected_elevation_at_prespecified_grid_boundary=abs(next(f for f in selection['families'] if f['family']==selection['selected'])['elevation'])==5,
        camera_calibration_verified=False,anatomical_identity_verified=False,measured_servo_angles_available=False,
        nominal_angle_degrees='9 * view_index',outer_selection_used=False,
        qualification='Camera family selected on inner training angles;20/30families retained. No leaf meshes, physical scale or uniquely calibrated3D geometry.',
        frozen_camera_selection_sha256=sha(output/'frozen_camera_selection.json'),script_sha256=sha(__file__))
    dump(output/'conditional_point_lookup.json',lookup);log(output,'point_lookup_exported',reference_rows=len(rows),node_hypotheses=total,path=str(output/'conditional_point_lookup.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--tracking',type=Path,required=True);a=p.parse_args();export(a.output,a.tracking)
