"""Leakage and cohort invariants for the full-cohort prediction producer."""
import csv
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import prediction as p


def fixture_rows():
    return [dict(treatment=t, measurement=str(v), genotype=g, replicate=str(r),
                 rgb_eligible='True', noir_eligible='True',
                 rgb_quadrants='4', noir_quadrants='4',
                 rgb_usable_views='40', noir_usable_views='40',
                 rgb__area__median=str(v+r), noir__area__median=str(v-r))
            for t in ['C', 'S'] for g in ['a', 'b', 'c', 'd', 'e', 'f']
            for r in [1, 2] for v in [1, 2]]


def test_missing_cells_stay_missing_and_duplicate_keys_fail(tmp_path):
    f = tmp_path/'x.csv'
    f.write_text('treatment,measurement,genotype,replicate,x\nC,1,a,1,\n')
    rows = p.read_keyed(f)
    assert np.isnan(p.number(rows[0]['x']))
    with f.open('a') as out:
        out.write('C,1,a,1,2\n')
    with pytest.raises(ValueError, match='Duplicate'):
        p.read_keyed(f)


def test_direct_modality_models_share_paired_rows():
    rows = fixture_rows()
    rows[0]['noir_eligible'] = 'False'
    rows[1]['rgb_eligible'] = 'False'
    assert p.cohort_indices(rows, 'paired_primary').tolist() == list(range(2, 48))
    assert p.cohort_indices(rows, 'rgb_primary').tolist() == [0]+list(range(2, 48))
    rows[3]['rgb_usable_views'] = '30'
    assert 3 not in p.cohort_indices(rows, 'paired_strict40')


def test_treatment_transfer_keeps_whole_candidate_and_does_not_invent_means():
    rows = fixture_rows()
    for train, test, label in p.partitions(rows, 'treatment'):
        assert {rows[i]['treatment'] for i in train}.isdisjoint(
            {rows[i]['treatment'] for i in test})
        with pytest.raises(ValueError, match='support'):
            p.metadata_predict(rows, np.arange(len(rows))[:,None], train, test,
                               ('treatment', 'measurement'))


def test_genotype_outer_folds_and_nested_target_isolation():
    rows = fixture_rows()
    x = np.array([[int(r['measurement']), int(r['replicate'])] for r in rows], float)
    y = (3*x[:,0]+x[:,1])[:,None]
    train, test, _ = next(p.partitions(rows, 'genotype'))
    a, record = p.fit_outer(rows, x, y, train, test, seed=8)
    corrupted = y.copy()
    corrupted[test] = 1e9
    b, _ = p.fit_outer(rows, x, corrupted, train, test, seed=8)
    np.testing.assert_array_equal(a, b)
    assert record['inner_group'] == 'genotype'
    assert record['inner_folds'] == 5
    assert len(list(p.partitions(rows, 'genotype'))) == 6


def test_negative_labels_preserve_entire_trajectories_and_training_membership():
    rows = fixture_rows()
    train, test, _ = next(p.partitions(rows, 'genotype'))
    y = np.array([[100*i+int(r['measurement'])] for i,r in enumerate(rows)], float)
    permuted, mapping = p.permute_trajectories(rows, y, train, 17)
    assert set(mapping) == set(train)
    assert set(mapping.values()) == set(train)
    assert any(a != b for a,b in mapping.items())
    for i,j in mapping.items():
        assert rows[i]['treatment'] == rows[j]['treatment']
        assert rows[i]['measurement'] == rows[j]['measurement']
        np.testing.assert_array_equal(permuted[i], y[j])
    for a,b in zip(train[:-1],train[1:]):
        if p.candidate(rows[a]) == p.candidate(rows[b]):
            assert p.candidate(rows[mapping[a]]) == p.candidate(rows[mapping[b]])


def test_metadata_training_mean_is_hand_checked():
    rows = [dict(treatment='C',measurement='1') for _ in range(3)]
    pred = p.metadata_predict(rows, np.array([[2.],[4.],[100.]]), [0,1], [2], ())
    assert pred[0,0] == 3


def test_native_grid_keeps_source_order_and_flags_without_exclusion(tmp_path):
    tables = tmp_path/'research_audit'/'tables'
    tables.mkdir(parents=True)
    (tables/'UVIS_P6_transmitance__List1.csv').write_text(
        'Treatment,Measurement,GEN,REP,327,325.1\nC,1,a,1,-0.1,0.2\n')
    (tmp_path/'research_audit'/'uvis_consistency_issues.json').write_text(
        '[{"key":["C",1,"a",1],"reason":"nonpositive_transmittance"}]')
    waves, values, flags = p.read_spectra(tmp_path, 'UVIS')
    assert waves.tolist() == [327,325.1]
    assert values[('C','1','a','1')].tolist() == [-0.1,0.2]
    assert flags[('C','1','a','1')] == ['nonpositive_transmittance']


def test_oof_output_covers_every_row_and_resumes_with_bound_provenance(tmp_path):
    rows = fixture_rows()
    y = np.array([[int(r['measurement'])] for r in rows],float)
    arguments = dict(mode='genotype',cohort='all360',domain='assay_synthetic',
                     image=False,flags={},signature='synthetic',nulls=False)
    result = p.run_experiment(tmp_path,rows,y,['known'],**arguments)
    assert result['n'] == 48
    oof = np.load(tmp_path/'oof.npz')
    assert oof['observed'].shape == (48,1)
    assert np.max(np.abs(oof['visit_mean']-y)) == 0
    assert set(oof['outer_fold']) == {'a','b','c','d','e','f'}
    assert p.run_experiment(tmp_path,rows,y,['known'],**arguments)['status'] == 'completed'
    arguments['signature'] = 'different_source'
    with pytest.raises(ValueError,match='provenance mismatch'):
        p.run_experiment(tmp_path,rows,y,['known'],**arguments)


def test_image_feature_whitelist_cannot_import_assay_targets():
    rows = fixture_rows()
    for r in rows:
        r['ojip__Fv/Fm'] = '0.8'
        r['nir__NDVI'] = '0.5'
    sets = p.feature_sets(rows,True,'genotype','paired_primary')
    assert sets['rgb'][1] == ['rgb__area__median']
    assert sets['combined'][0].shape == (48,2)
    assert all('Fv/Fm' not in n and 'NDVI' not in n for _,names in sets.values() for n in names)


def test_audit_catches_train_test_key_overlap_and_corrupted_oof(tmp_path):
    import prediction_audit as audit
    rows = fixture_rows()
    y = np.array([[int(r['measurement'])] for r in rows],float)
    p.run_experiment(tmp_path,rows,y,['known'],mode='genotype',cohort='all360',
                     domain='assay_synthetic',image=False,flags={},signature='test',nulls=False)
    assert audit.audit_experiment(tmp_path)['rows'] == 48
    path = tmp_path/'folds.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:
        folds = list(csv.DictReader(f))
    original = folds[0]['train_keys']
    folds[0]['train_keys'] = folds[0]['test_keys']
    p.save_csv(path,folds)
    with pytest.raises(ValueError,match='overlap'):
        audit.audit_experiment(tmp_path)
    folds[0]['train_keys'] = original
    p.save_csv(path,folds)
    oof = dict(np.load(tmp_path/'oof.npz'))
    oof['visit_mean'][0,0] = 100
    np.savez_compressed(tmp_path/'oof.npz',**oof)
    with pytest.raises(ValueError,match='mean|metric'):
        audit.audit_experiment(tmp_path)


def test_encoder_crop_removes_background_and_preserves_aspect_and_blue_ablation():
    import prediction_encoders as encoder
    image = np.full((20,30,3),255,np.uint8)
    mask = np.zeros((20,30),bool)
    mask[5:15,12:17] = True
    image[mask] = [10,50,90]
    rgb, box = encoder.make_crop(image,mask,'recorded')
    blue, _ = encoder.make_crop(image,mask,'blue_triplicate')
    assert rgb.size == (224,224)
    assert tuple(np.array(rgb)[0,0]) == (0,0,0)
    assert tuple(np.array(rgb)[112,112]) == (10,50,90)
    assert tuple(np.array(blue)[112,112]) == (90,90,90)
    assert box == [11,4,18,16]
    with pytest.raises(ValueError,match='Empty'):
        encoder.make_crop(image,np.zeros_like(mask),'recorded')


def test_encoder_aggregation_retains_missing_sessions_and_camera_separation():
    import prediction_encoders as encoder
    rows = fixture_rows()[:2]
    records = [dict(**{k:rows[0][k] for k in p.KEYS},camera='rgb',variant='recorded',
                    view=0,feature=np.full(384,2.)),
               dict(**{k:rows[0][k] for k in p.KEYS},camera='rgb',variant='recorded',
                    view=10,feature=np.full(384,4.)),
               dict(**{k:rows[0][k] for k in p.KEYS},camera='noir',variant='recorded',
                    view=0,feature=np.full(384,8.))]
    arrays, counts = encoder.aggregate(rows,records)
    assert arrays['rgb'].shape == (2,384)
    assert arrays['rgb'][0,0] == 3
    assert arrays['noir'][0,0] == 8
    assert np.isnan(arrays['rgb'][1]).all()
    assert counts['rgb'].tolist() == [2,0]


def test_encoder_models_use_same_rows_and_keep_metadata_increment_separate():
    rows = fixture_rows()
    for r in rows:
        for camera in ('rgb','noir','noir_blue'):
            for j in range(2):
                r[f'dino_{camera}__{j:03}'] = str(j+1)
    sets = p.feature_sets(rows,True,'genotype','paired_primary')
    assert sets['dino_rgb'][0].shape == (48,2)
    assert sets['dino_combined'][0].shape == (48,4)
    assert sets['metadata_plus_dino'][0].shape == (48,19)
    assert sets['dino_rgb_plus_noir_blue'][0].shape == (48,4)


def test_declared_eligibility_cannot_override_missing_quadrants_or_counts():
    rows = fixture_rows()
    rows[0]['rgb_quadrants'] = '3'
    rows[1]['noir_usable_views'] = '19'
    assert p.cohort_indices(rows,'paired_primary').tolist() == list(range(2,48))
    rows[2]['rgb_usable_views'] = '25'
    assert 2 not in p.cohort_indices(rows,'paired_strict30')


def test_partial_or_changed_extraction_is_not_accepted_for_full_prediction(tmp_path):
    import json
    table = tmp_path/'session_traits.csv'
    table.write_text('synthetic')
    summary = dict(status='running',reference_samples=360,
                   minimum_accepted_views=20,required_quadrants=4,
                   session_table_sha256=p.digest(table))
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='complete'):
        p.validate_extraction(table)
    summary['status']='completed'
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    assert p.validate_extraction(table)['status']=='completed'
    table.write_text('altered')
    with pytest.raises(ValueError,match='hash'):
        p.validate_extraction(table)


def test_finalizer_waits_for_both_completed_inputs(tmp_path):
    import json
    import prediction_finalize as finalize
    (tmp_path/'extraction').mkdir()
    (tmp_path/'prediction'/'encoder').mkdir(parents=True)
    assert finalize.ready(tmp_path) is False
    (tmp_path/'extraction'/'summary.json').write_text(json.dumps({'status':'completed'}))
    encoder = tmp_path/'prediction'/'encoder'/'progress.json'
    encoder.write_text(json.dumps({'status':'streaming'}))
    assert finalize.ready(tmp_path) is False
    encoder.write_text(json.dumps({'status':'completed'}))
    assert finalize.ready(tmp_path) is True


def test_independent_frame_aggregation_audit_catches_selection_or_summary_changes():
    import prediction_audit as audit
    row=dict(treatment='C',measurement='1',genotype='a',replicate='1')
    frames=[dict(**row,camera='rgb',view=2*i,qc_pass=True,area=i) for i in range(20)]
    session=dict(**row,rgb_decodable_views=20,rgb_usable_views=20,rgb_quadrants=4,
                 rgb_eligible=True,noir_decodable_views=0,noir_usable_views=0,
                 noir_quadrants=0,noir_eligible=False,paired_eligible=False,
                 rgb__area__median=9.5,rgb__area__max=19,rgb__area__iqr=9.5,
                 noir__area__median='',noir__area__max='',noir__area__iqr='')
    assert audit.audit_aggregation([session],frames)['sessions']==1
    session['rgb_eligible']=False
    with pytest.raises(ValueError,match='eligibility'):
        audit.audit_aggregation([session],frames)
    session['rgb_eligible']=True
    session['rgb__area__median']=100
    with pytest.raises(ValueError,match='aggregate'):
        audit.audit_aggregation([session],frames)


def test_extraction_preflight_rejects_changed_key_config_and_mask(tmp_path):
    import hashlib,json
    import prediction_audit as audit
    intake,run,repo=(tmp_path/n for n in ('intake','run','repo'))
    for directory in (intake,run/'extraction'/'records',run/'extraction'/'masks',
                      run/'extraction'/'sources',repo/'src'/'csi_maize'/'_methods'):
        directory.mkdir(parents=True)
    row=dict(treatment='C',measurement='1',genotype='a',replicate='1')
    session='IMAGES/Control1/capture_session_C1_1-1'
    source=session+'/rgb/image_rgb_000.jpg'
    p.save_csv(intake/'session_decode_qc.csv',[dict(**row,session=session,
                                                known_reference_identity=True)])
    p.save_csv(intake/'jpeg_decode_manifest.csv',[dict(path=source,sha256='0'*64,decode_ok=True)])
    extraction_source=run/'extraction'/'sources'/'source.py'
    extraction_source.write_text('frozen')
    source_hash=p.digest(extraction_source)
    extraction_source.rename(extraction_source.with_name('extraction_'+source_hash+'.py'))
    config=dict(shards=1,tasks=1,task_digest=hashlib.sha256(json.dumps(
        [(source,'0'*64)]).encode()).hexdigest(),operator_hashes={},
        extraction_source_sha256=source_hash)
    config['signature']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    p.save_json(run/'extraction'/'configuration_shard_0.json',config)
    mask=run/'extraction'/'masks'/'one.png'
    mask.write_bytes(b'unchanged_mask')
    record=dict(**row,session=session,camera='rgb',view=0,source=source,
                source_sha256='0'*64,shard=0,configuration_signature=config['signature'],
                qc_pass=True,mask='extraction/masks/one.png',mask_sha256=p.digest(mask))
    path=run/'extraction'/'records'/(hashlib.sha256(source.encode()).hexdigest()[:24]+'.json')
    p.save_json(path,record)
    assert audit.audit_preflight(intake,run,repo)['frames']==1
    record['genotype']='wrong'
    p.save_json(path,record)
    with pytest.raises(ValueError,match='identity'):
        audit.audit_preflight(intake,run,repo)
    record['genotype']='a'
    record['configuration_signature']='wrong'
    p.save_json(path,record)
    with pytest.raises(ValueError,match='configuration'):
        audit.audit_preflight(intake,run,repo)
    record['configuration_signature']=config['signature']
    p.save_json(path,record)
    mask.write_bytes(b'altered_mask')
    with pytest.raises(ValueError,match='mask hash'):
        audit.audit_preflight(intake,run,repo)


def test_index_predictors_are_exact_whitelist_and_ignore_availability_control_fields():
    import prediction_indices as indices
    rows = fixture_rows()
    for row in rows:
        for name in indices.INDEX_COLUMNS+indices.DN_COLUMNS:
            row[name] = '' if 'noir_G' in name else '1.25'
        row['paired_index__WRONG_PAIR__median']='999'
        row['paired_index__ExG__eligible']='True'
        row['ECC_fit_score']='999'
    sets = indices.designs(rows,'genotype')
    assert sets['registered_indices'][0].shape == (48,54)
    assert sets['cross_camera_indices'][0].shape == (48,30)
    assert sets['registered_camera_dns'][0].shape == (48,12)
    assert sets['metadata_plus_dns_plus_cross_indices'][0].shape == (48,57)
    assert np.isnan(sets['registered_indices'][0]).any()
    assert not any('WRONG_PAIR' in n or 'eligible' in n or 'ECC_fit' in n
                   for _,names in sets.values() for n in names)


def test_custom_index_design_executes_nested_oof_without_changing_cohort(tmp_path):
    rows = fixture_rows()
    y = np.array([[int(r['replicate'])] for r in rows],float)
    meta,names = p.metadata_features(rows)
    design = dict(metadata_ridge=(meta,names),
                  registered_indices=(y.copy(),['synthetic_index']))
    result = p.run_experiment(tmp_path,rows,y,['known'],mode='genotype',
        cohort='paired_primary',domain='assay_synthetic',image=True,flags={},
        signature='custom_design',nulls=False,designs=design)
    metrics = {r['model']:r for r in result['summary']}
    assert metrics['registered_indices']['rmse'] < .01
    assert metrics['treatment_visit_mean']['rmse'] == .5
    with np.load(tmp_path/'oof.npz') as data:
        assert data['registered_indices'].shape == data['metadata_ridge'].shape == (48,1)


def test_selection_audit_keeps_excluded_targets_missing_instead_of_zero():
    import prediction_audit as audit
    rows = fixture_rows()[:2]
    for i,row in enumerate(rows):
        for target in p.TARGETS:
            row[target] = '.8' if i==0 else ''
    rows[1]['rgb_eligible']='False'
    rows[1]['rgb_usable_views']='10'
    result = audit.selection_audit(rows,rows)
    overall = [r for r in result if r['cohort']=='paired_primary' and
               r['grouping']=='overall' and r['target']==p.TARGETS[0]]
    excluded = next(r for r in overall if r['eligible'] is False)
    included = next(r for r in overall if r['eligible'] is True)
    assert excluded['sessions']==1 and excluded['observed_targets']==0
    assert excluded['mean'] is None
    assert included['mean']==.8


def test_incremental_oof_contrast_uses_paired_errors_on_identical_rows(tmp_path):
    import prediction_audit as audit
    rows=fixture_rows()
    y=np.array([[int(r['replicate'])] for r in rows],float)
    np.savez_compressed(tmp_path/'oof.npz',keys=np.array([p.key(r) for r in rows]),
                        observed=y,baseline=np.full_like(y,1.5),augmented=y.copy())
    result=audit.incremental_contrast(tmp_path,'baseline','augmented')
    assert result['rows']==48
    assert result['rmse_improvement']==.5
    assert result['exploratory_cluster_bootstrap_low']==.5
    assert result['exploratory_cluster_bootstrap_high']==.5


def test_report_never_combines_different_cohort_keys_for_direct_comparison():
    import prediction_report as report
    left=dict(keys=[('C','1','a','1')],observed=np.array([[.8]]))
    right=dict(keys=[('C','1','b','1')],observed=np.array([[.8]]))
    with pytest.raises(ValueError,match='cohort'):
        report.assert_matched(left,right)
    right['keys']=left['keys']
    report.assert_matched(left,right)
    right['observed']=np.array([[.9]])
    with pytest.raises(ValueError,match='target'):
        report.assert_matched(left,right)


def test_report_best_display_is_explicitly_selected_from_completed_non_null_models():
    import prediction_report as report
    metrics=[dict(model='training_mean',rmse=2.),dict(model='metadata_ridge',rmse=1.),
             dict(model='metadata_plus_image_trajectory_null_1979',rmse=.1)]
    assert report.display_choice(metrics)['model']=='metadata_ridge'


def test_index_phase_can_follow_direct_completed_image_cli_without_finalizer_worker(tmp_path):
    import json
    from types import SimpleNamespace
    import prediction_indices as indices
    (tmp_path/'extraction').mkdir()
    (tmp_path/'prediction').mkdir()
    paths=[tmp_path/'extraction'/'summary.json',tmp_path/'index.json',tmp_path/'dn.json',
           tmp_path/'prediction'/'summary_images.json']
    for path in paths:
        path.write_text(json.dumps({'status':'completed'}))
    args=SimpleNamespace(run=tmp_path,index_manifest=paths[1],dn_manifest=paths[2])
    assert indices.ready(args) is True
