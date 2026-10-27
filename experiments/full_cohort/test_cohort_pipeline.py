from pathlib import Path

import pytest

from cohort_pipeline import build_plan, validate_layout, input_hashes, output_hashes, verify_outputs, run_lock, claim_folder, retire_interrupted


def config(tmp_path):
    return {k: str(tmp_path / k) for k in ('raw','intake','run','repo','models','dino','external','da3_assets')}


def test_complete_plan_orders_scientific_dependencies(tmp_path):
    stages = build_plan(config(tmp_path), Path(__file__).parent, 'full')
    names = [s['name'] for s in stages]
    assert names.index('triage') < names.index('extract') < names.index('extraction_preflight')
    assert names.index('extraction_preflight') < names.index('aggregate') < names.index('extraction_tables')
    assert names.index('common_dn') < names.index('index_predictions')
    assert names.index('image_predictions') < names.index('index_predictions')
    assert names.index('external_scaffold') < names.index('external_tracking') < names.index('external_audit')
    assert names[-1] == 'full_report'
    assert not any('--download' in s['args'] for s in stages)


def test_cohort_route_has_no_hidden_external_or_depth_dependency(tmp_path):
    settings = config(tmp_path)
    del settings['external'], settings['da3_assets']
    stages = build_plan(settings, Path(__file__).parent, 'cohort')
    assert not any(s['name'].startswith('external_') or s['name']=='depth' for s in stages)
    assert stages[-1]['name'] == 'prediction_report'


@pytest.mark.parametrize('field', ['intake','run'])
def test_generated_output_cannot_enter_original_archive(tmp_path, field):
    settings = config(tmp_path)
    settings[field] = str(Path(settings['raw'])/'generated')
    with pytest.raises(ValueError, match='separate'):
        validate_layout(settings)


def test_cannot_nest_run_and_intake(tmp_path):
    settings = config(tmp_path)
    settings['run'] = str(Path(settings['intake'])/'run')
    with pytest.raises(ValueError, match='separate'):
        validate_layout(settings)


def test_full_route_requires_supplied_external_and_depth_assets(tmp_path):
    settings = config(tmp_path)
    del settings['external']
    with pytest.raises(ValueError, match='external'):
        build_plan(settings, Path(__file__).parent, 'full')


def test_rejects_device_ignored_by_fixed_cuda_zero_producers(tmp_path):
    settings=config(tmp_path);settings['device']='cuda:1'
    with pytest.raises(ValueError,match='cuda:0'):
        build_plan(settings,Path(__file__).parent,'cohort')


def test_inputs_bind_actual_bytes_not_only_paths(tmp_path):
    settings=config(tmp_path)
    raw=Path(settings['raw']);(raw/'IMAGES').mkdir(parents=True)
    image=raw/'IMAGES/frame.jpg';image.write_bytes(b'original')
    (raw/'assay.xlsx').write_bytes(b'workbook')
    sam=Path(settings['models'])/'sam2.1-hiera-small';sam.mkdir(parents=True)
    (sam/'model.safetensors').write_bytes(b'SAM')
    dino=Path(settings['dino']);dino.mkdir();(dino/'model.safetensors').write_bytes(b'DINO')
    before=input_hashes(settings,'cohort')
    image.write_bytes(b'changed!')
    assert input_hashes(settings,'cohort')!=before
    image.write_bytes(b'original');(dino/'model.safetensors').write_bytes(b'OTHER')
    assert input_hashes(settings,'cohort')!=before


def test_resume_checks_data_outputs_beyond_one_summary(tmp_path):
    settings=config(tmp_path);out=Path(settings['run']);out.mkdir()
    (Path(settings['intake'])).mkdir()
    (out/'summary.json').write_text('{"status":"completed"}')
    mask=out/'mask.png';mask.write_bytes(b'mask')
    recorded=output_hashes(settings,{})
    mask.unlink()
    with pytest.raises(ValueError,match='missing|changed'):
        verify_outputs(recorded)


def test_output_last_writer_allows_legitimate_later_audit_update(tmp_path):
    settings=config(tmp_path);out=Path(settings['run']);out.mkdir();Path(settings['intake']).mkdir()
    audit=out/'audit.json';audit.write_text('{"n":72}')
    first=output_hashes(settings,{})
    audit.write_text('{"n":84,"final":true}')
    final=output_hashes(settings,first)
    verify_outputs(final)
    assert first[str(audit)]['sha256']!=final[str(audit)]['sha256']


def test_concurrent_driver_cannot_acquire_run(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError,match='Another coordinator'):
            with run_lock(tmp_path):raise AssertionError('Must not launch')
    with run_lock(tmp_path):pass


def test_preexisting_intake_cannot_be_claimed_or_retired(tmp_path):
    settings=config(tmp_path);folder=Path(settings['intake']);folder.mkdir();(folder/'user.txt').write_text('retain')
    state={}
    with pytest.raises(ValueError,match='Preexisting'):
        claim_folder(folder,state)
    with pytest.raises(ValueError,match='unowned'):
        retire_interrupted(settings,'intake',state)
    assert (folder/'user.txt').read_text()=='retain'


def test_owned_interrupted_intake_is_preserved_for_clean_retry(tmp_path):
    settings=config(tmp_path);folder=Path(settings['intake']);state={}
    claim_folder(folder,state);folder.mkdir();(folder/'partial.txt').write_text('evidence')
    retire_interrupted(settings,'intake',state)
    assert not folder.exists()
    assert (Path(state['retired_partial_outputs'][0])/'partial.txt').read_text()=='evidence'


def test_interrupted_overwrite_is_conservatively_rejected(tmp_path):
    settings=config(tmp_path);out=Path(settings['run']);out.mkdir();Path(settings['intake']).mkdir()
    report=out/'report.html';report.write_text('original report')
    recorded=output_hashes(settings,{})
    report.write_text('partially rewritten')
    with pytest.raises(ValueError,match='fresh work directory'):
        verify_outputs(recorded)
