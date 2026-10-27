"""Safety and provenance guarantees of the public execution interface."""

import json
from pathlib import Path

import pytest

from csi_maize.config import RunConfig
from csi_maize.provenance import digest, write_json, verify_files
from csi_maize.workspace import create_workspace, contained_output


def test_workspace_preserves_sources_and_copies_workbooks(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    images = data / "images"
    images.mkdir()
    (images / "original.jpg").write_bytes(b"input")
    (data / "assay.xlsx").write_bytes(b"workbook")
    run = tmp_path / "run"
    cfg = RunConfig(data_root=data, work_dir=run, source_folders=("images",))
    create_workspace(cfg)
    assert (run / "images/original.jpg").read_bytes() == b"input"
    assert (run / "assay.xlsx").read_bytes() == b"workbook"
    assert (data / "assay.xlsx").read_bytes() == b"workbook"
    assert (
        json.loads((run / "input_bindings.json").read_text())["bindings"][0]["logical_path"]
        == "images"
    )


def test_refuse_nonempty_unowned_workspace(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    (run / "valuable.txt").write_text("keep")
    cfg = RunConfig(data_root=data, work_dir=run, source_folders=())
    with pytest.raises(ValueError, match="nonempty"):
        create_workspace(cfg)
    assert (run / "valuable.txt").read_text() == "keep"


def test_nested_workbooks_are_copied_and_original_locations_are_fingerprinted(tmp_path):
    from csi_maize.engine import raw_hashes

    data = tmp_path / "data"
    nested = data / "assays"
    nested.mkdir(parents=True)
    (data / "Genotype_list.xlsx").write_bytes(b"genotypes")
    (nested / "OJIP.xlsx").write_bytes(b"assays")
    cfg = RunConfig(
        data_root=data, work_dir=tmp_path / "run", source_folders=(), workbook_folders=("assays",)
    )
    create_workspace(cfg)
    assert (cfg.work_dir / "OJIP.xlsx").read_bytes() == b"assays"
    before = raw_hashes(cfg)
    assert before["assays/OJIP.xlsx"] == digest(nested / "OJIP.xlsx")
    (nested / "OJIP.xlsx").write_bytes(b"changed assays")
    assert raw_hashes(cfg) != before
    restored = RunConfig.load(cfg.work_dir / "run_config.json")
    assert restored.workbook_folders == ("assays",)


def test_ambiguous_workbook_basenames_rejected_before_workspace_is_created(tmp_path):
    data = tmp_path / "data"
    nested = data / "assays"
    nested.mkdir(parents=True)
    (data / "same.xlsx").write_bytes(b"root version")
    (nested / "same.xlsx").write_bytes(b"nested version")
    cfg = RunConfig(
        data_root=data, work_dir=tmp_path / "run", source_folders=(), workbook_folders=("assays",)
    )
    with pytest.raises(ValueError, match="[Dd]uplicate|[Aa]mbiguous"):
        create_workspace(cfg)
    assert not cfg.work_dir.exists()


def test_workbook_folder_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        RunConfig(data_root=tmp_path, work_dir=tmp_path / "run", workbook_folders=("../outside",))


def test_case_colliding_workbooks_cannot_overwrite_each_other(tmp_path):
    data = tmp_path / "data"
    nested = data / "assays"
    nested.mkdir(parents=True)
    (data / "SAME.xlsx").write_bytes(b"root")
    (nested / "same.xlsx").write_bytes(b"nested")
    cfg = RunConfig(
        data_root=data, work_dir=tmp_path / "run", source_folders=(), workbook_folders=("assays",)
    )
    with pytest.raises(ValueError, match="[Dd]uplicate|[Aa]mbiguous"):
        create_workspace(cfg)
    assert not cfg.work_dir.exists()


def test_legacy_config_without_empty_workbook_option_remains_verifiable(tmp_path):
    from csi_maize.engine import verify_input_bindings

    data = tmp_path / "data"
    data.mkdir()
    cfg = RunConfig(data_root=data, work_dir=tmp_path / "run", source_folders=())
    create_workspace(cfg)
    path = cfg.work_dir / "run_config.json"
    stored = json.loads(path.read_text())
    stored.pop("workbook_folders")
    path.write_text(json.dumps(stored))
    verify_input_bindings(cfg)


@pytest.mark.parametrize(
    "relative", ["../escape", "/absolute", "prototype/research_v2/outputs/../../../../escape"]
)
def test_output_traversal_rejected(tmp_path, relative):
    with pytest.raises(ValueError):
        contained_output(tmp_path, relative)


def test_source_folder_traversal_rejected(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(ValueError):
        RunConfig(data_root=data, work_dir=tmp_path / "run", source_folders=("../elsewhere",))


def test_hash_verification_detects_changed_file(tmp_path):
    path = tmp_path / "result.csv"
    path.write_text("original")
    expected = {"result.csv": digest(path)}
    verify_files(tmp_path, expected)
    path.write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        verify_files(tmp_path, expected)


def test_strict_json_rejects_infinity(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"invalid": float("inf")})


def test_json_write_is_complete(tmp_path):
    path = tmp_path / "record.json"
    write_json(path, {"status": "complete", "value": 0})
    assert json.loads(path.read_text()) == {"status": "complete", "value": 0}


def test_result_symlink_outside_workspace_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "result").write_text("same")
    root = tmp_path / "run"
    root.mkdir()
    from csi_maize.workspace import _bind_directory

    _bind_directory(outside, root / "linked")
    with pytest.raises(ValueError):
        verify_files(root, {"linked/result": digest(outside / "result")})


@pytest.mark.parametrize("folder", ["metadata", "reports", "logs", "runtime", "."])
def test_generated_namespaces_cannot_bind_raw_sources(tmp_path, folder):
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(ValueError):
        RunConfig(data_root=data, work_dir=tmp_path / "run", source_folders=(folder,))


def test_overlapping_bindings_are_rejected(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(ValueError, match="overlap"):
        RunConfig(
            data_root=data, work_dir=tmp_path / "run", source_folders=("images", "images/child")
        )


def test_workspace_cannot_live_inside_bound_source(tmp_path):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    with pytest.raises(ValueError):
        RunConfig(data_root=data, work_dir=data / "images/run", source_folders=("images",))
