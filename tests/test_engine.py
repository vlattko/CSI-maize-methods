"""A real two-command pipeline must execute, resume and detect tampering."""

from pathlib import Path
import json

import pytest

from csi_maize.config import RunConfig
from csi_maize.engine import execute, ordered_stages, verify_run
from csi_maize.workspace import create_workspace


def test_dependency_cycle_and_unknown_stage_rejected():
    with pytest.raises(ValueError, match="cycle"):
        ordered_stages({"a": {"dependencies": ["b"]}, "b": {"dependencies": ["a"]}}, ["a"])
    with pytest.raises(ValueError, match="Unknown"):
        ordered_stages({}, ["missing"])


def fixture(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "source.txt").write_text("raw")
    cfg = RunConfig(data_root=data, work_dir=tmp_path / "run", source_folders=())
    create_workspace(cfg)
    base = cfg.work_dir / "prototype/research_v2"
    (base / "step_one.py").write_text(
        "from pathlib import Path\nPath('outputs/one.json').write_text('{\"value\":7}')\n"
    )
    (base / "step_two.py").write_text(
        "from pathlib import Path\nimport json\na=json.loads(Path('outputs/one.json').read_text())\n"
        "Path('outputs/two.json').write_text(json.dumps({'value':a['value']*2}))\n"
    )
    stages = {
        "one": dict(
            dependencies=[],
            commands=[["step_one.py"]],
            outputs=["prototype/research_v2/outputs/one.json"],
        ),
        "two": dict(
            dependencies=["one"],
            commands=[["step_two.py"]],
            outputs=["prototype/research_v2/outputs/two.json"],
        ),
    }
    return cfg, stages


def test_execute_real_commands_and_resume_without_rewriting(tmp_path):
    cfg, stages = fixture(tmp_path)
    result = execute(cfg, ["two"], stages=stages)
    assert result["status"] == "completed"
    path = cfg.work_dir / "prototype/research_v2/outputs/two.json"
    assert json.loads(path.read_text()) == {"value": 14}
    previous_time = path.stat().st_mtime_ns
    execute(cfg, ["two"], stages=stages, resume=True)
    assert path.stat().st_mtime_ns == previous_time


def test_resume_rejects_output_tampering(tmp_path):
    cfg, stages = fixture(tmp_path)
    execute(cfg, ["two"], stages=stages)
    (cfg.work_dir / "prototype/research_v2/outputs/one.json").write_text('{"value":900}')
    with pytest.raises(ValueError, match="changed"):
        execute(cfg, ["two"], stages=stages, resume=True)


def test_resume_rejects_modified_materialized_source(tmp_path):
    cfg, stages = fixture(tmp_path)
    execute(cfg, ["two"], stages=stages)
    (cfg.work_dir / "prototype/research_v2/step_two.py").write_text(
        "raise RuntimeError('tampered')"
    )
    with pytest.raises(ValueError, match="changed"):
        execute(cfg, ["two"], stages=stages, resume=True)


def test_failure_is_recorded_and_not_reported_complete(tmp_path):
    cfg, stages = fixture(tmp_path)
    stages["one"]["commands"] = [["step_two.py"]]
    with pytest.raises(RuntimeError, match="failed"):
        execute(cfg, ["one"], stages=stages)
    state = json.loads((cfg.work_dir / "run_manifest.json").read_text())
    assert state["status"] == "failed"
    assert state["stages"]["one"]["status"] == "failed"


def test_stage_cannot_execute_outside_materialized_code(tmp_path):
    cfg, stages = fixture(tmp_path)
    stages["one"]["commands"] = [["../../../outside.py"]]
    with pytest.raises((ValueError, RuntimeError)):
        execute(cfg, ["one"], stages=stages)


def test_truncated_event_history_is_rejected(tmp_path):
    cfg, stages = fixture(tmp_path)
    execute(cfg, ["two"], stages=stages)
    (cfg.work_dir / "events.jsonl").write_text("")
    with pytest.raises(ValueError, match="Event"):
        verify_run(cfg.work_dir)


def test_materialized_configuration_change_is_rejected(tmp_path):
    cfg, stages = fixture(tmp_path)
    execute(cfg, ["two"], stages=stages)
    p = cfg.work_dir / "run_config.json"
    value = json.loads(p.read_text())
    value["seed"] = 7
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed"):
        execute(cfg, ["two"], stages=stages, resume=True)


def test_summary_failure_cannot_be_promoted_by_successful_process_exit(tmp_path):
    cfg, stages = fixture(tmp_path)
    p = cfg.work_dir / "prototype/research_v2/step_one.py"
    p.write_text(
        "from pathlib import Path\nPath('outputs/one.json').write_text('{\"status\":\"failed\"}')\n"
    )
    with pytest.raises(RuntimeError, match="failed"):
        execute(cfg, ["one"], stages=stages)


def test_resume_rejects_changed_runtime(tmp_path, monkeypatch):
    import csi_maize.engine as engine

    cfg, stages = fixture(tmp_path)
    monkeypatch.setattr(engine, "_versions", lambda: {"environment": "A"})
    execute(cfg, ["two"], stages=stages)
    monkeypatch.setattr(engine, "_versions", lambda: {"environment": "B"})
    with pytest.raises(ValueError, match="Runtime"):
        execute(cfg, ["two"], stages=stages, resume=True)


def test_standalone_verification_checks_materialized_model_weights(tmp_path):
    from csi_maize.provenance import digest, write_json

    cfg, stages = fixture(tmp_path)
    weight = cfg.work_dir / "prototype/research_v2/checkpoint.bin"
    weight.write_bytes(b"weights")
    write_json(
        cfg.work_dir / "metadata/model_assets.json",
        {
            "models": {
                "fixture": {
                    "files": {
                        "checkpoint.bin": {
                            "workspace_path": "prototype/research_v2/checkpoint.bin",
                            "sha256": digest(weight),
                        }
                    }
                }
            }
        },
    )
    execute(cfg, ["two"], stages=stages)
    weight.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_run(cfg.work_dir)


def test_generated_root_is_validated_before_a_stage_can_write(tmp_path):
    from csi_maize.workspace import _bind_directory

    cfg, stages = fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    _bind_directory(outside, cfg.work_dir / "logs")
    with pytest.raises(ValueError):
        execute(cfg, ["one"], stages=stages)
    assert not list(outside.iterdir())


def test_registered_input_binding_topology_is_verified(tmp_path):
    cfg, stages = fixture(tmp_path)
    execute(cfg, ["two"], stages=stages)
    # A registered source must resolve to the declared raw directory, not a new local copy.
    from csi_maize.provenance import write_json

    outside = tmp_path / "raw_images"
    outside.mkdir()
    local = cfg.work_dir / "images"
    local.mkdir()
    write_json(
        cfg.work_dir / "input_bindings.json",
        {"bindings": [{"logical_path": "images", "source_path": str(outside), "kind": "symlink"}]},
    )
    # Verify first-run binding validation too: an edited descriptor is never authority to relabel bytes.
    from csi_maize.engine import verify_input_bindings

    with pytest.raises(ValueError):
        verify_input_bindings(cfg)


def test_first_execution_rejects_changed_copied_workbook(tmp_path):
    cfg, stages = fixture(tmp_path)
    (cfg.data_root / "assay.xlsx").write_bytes(b"original")
    (cfg.work_dir / "assay.xlsx").write_bytes(b"edited copy")
    with pytest.raises(ValueError, match="workbook"):
        execute(cfg, ["one"], stages=stages)
