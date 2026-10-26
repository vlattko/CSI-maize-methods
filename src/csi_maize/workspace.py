"""Bind immutable inputs to a disposable, explicitly owned computation root."""

from pathlib import Path
import os
import shutil
import subprocess

from .config import RunConfig
from .provenance import digest, write_json


def contained_output(workspace: Path, relative: str) -> Path:
    workspace = Path(workspace).absolute()
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or ":" in str(relative):
        raise ValueError("Output path must be relative to the workspace")
    target = workspace / path
    resolved = target.resolve()
    if resolved == workspace.resolve() or not resolved.is_relative_to(workspace.resolve()):
        raise ValueError("Output escapes the computational workspace")
    return target


def _bind_directory(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source, target_is_directory=True)
        return "symlink"
    except OSError:
        if os.name != "nt":
            raise
    # Windows junctions do not require developer mode; no command text is constructed.
    result = subprocess.run(
        ["cmd.exe", "/c", "mklink", "/J", str(target), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or not target.is_dir():
        raise OSError("Could not bind the source directory; use a writable local work directory")
    return "junction"


def create_workspace(config: RunConfig) -> Path:
    root = config.work_dir
    if not config.data_root.is_dir():
        raise ValueError("Raw data directory does not exist")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Refusing to reuse a nonempty workspace; use an empty directory or resume")
    root.mkdir(parents=True, exist_ok=True)
    base = root / "prototype/research_v2"
    base.mkdir(parents=True, exist_ok=True)
    (base / "outputs").mkdir()
    (root / "research_audit").mkdir()
    source = Path(__file__).parent / "_methods"
    shutil.copytree(
        Path(__file__).parent,
        root / "runtime/csi_maize",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix in {".py", ".json", ".html", ".js", ".css"}:
            shutil.copyfile(path, base / path.name)
    from .report import build_atlas_template

    (base / "spatial_fusion_viewer.html").write_text(
        build_atlas_template(), encoding="utf-8", newline="\n"
    )
    bindings = []
    for relative in config.source_folders:
        raw = config.data_root / relative
        if not raw.exists():
            continue
        if not raw.is_dir():
            raise ValueError(f"Source folder is not a directory: {relative}")
        kind = _bind_directory(raw.absolute(), root / relative)
        bindings.append(dict(logical_path=relative, source_path=str(raw.absolute()), kind=kind))
    about = config.data_root / "about"
    if about.is_dir():
        kind = _bind_directory(about.absolute(), root / "about")
        bindings.append(dict(logical_path="about", source_path=str(about.absolute()), kind=kind))
    for raw in sorted(config.data_root.glob("*.xlsx")):
        shutil.copyfile(raw, root / raw.name)
        assert digest(raw) == digest(root / raw.name)
    mapping = config.data_root / "prototype/identity_mapping.csv"
    if mapping.is_file():
        shutil.copyfile(mapping, root / "prototype/identity_mapping.csv")
    settings = dict(
        reference_table="research_audit/reference_master.csv",
        source_folders=list(config.source_folders),
        identity_mapping="prototype/identity_mapping.csv",
        allowed_views_per_camera=[30, 40],
        feature_views_per_camera=4,
        thumbnail_width=640,
        ridge_penalty=10,
        targets=["ojip__Fv/Fm", "ojip__Pi_Abs", "nir__NDVI"],
        spectral_sources=["NIR_P6_transmitance.xlsx", "UVIS_P6_transmitance.xlsx"],
        exclude_spectral_keys=[["S", "2", "Mo17", "4"], ["P", "4", "Os6251/15", "4"]],
        case="publication" if config.profile != "example" else "example",
        expected_acquisitions=config.expected_acquisitions,
    )
    write_json(base / "config.json", settings)
    write_json(root / "prototype/config.json", settings)
    write_json(root / "input_bindings.json", {"bindings": bindings})
    write_json(root / "run_config.json", config.to_dict())
    write_json(root / ".csi_workspace.json", {"schema_version": 1, "owner": "csi-maize-methods"})
    return root
