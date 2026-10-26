"""Dependency-boundary tests without requiring GPU or optional model loaders."""

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest

from csi_maize import assets


def adapter():
    try:
        return importlib.import_module("csi_maize.ultralytics_offline")
    except ModuleNotFoundError:
        pytest.fail("The checked offline Ultralytics adapter is missing")


@pytest.fixture
def model_inputs(tmp_path, monkeypatch):
    base = tmp_path / "run/prototype/research_v2"
    base.mkdir(parents=True)
    payload = b"pinned AMP self-check bytes"
    specification = {
        "cache_path": "yolo26n.pt",
        "workspace_path": "prototype/research_v2/yolo26n.pt",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps(
            {
                "models": {
                    "yolo26n-amp-check": {"files": [specification]},
                    "yolo26n-seg": {"files": [dict(specification)]},
                }
            }
        )
    )
    monkeypatch.setattr(assets, "MANIFEST_PATH", manifest)
    monkeypatch.setenv(
        "YOLO_CONFIG_DIR", str(base / "outputs/multimodal_leaf/2026-10-24/runtime_config")
    )
    monkeypatch.setenv("MPLCONFIGDIR", str(base / "outputs/matplotlib"))
    monkeypatch.setenv("YOLO_OFFLINE", "false")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "true")
    return base, payload, specification


def dependency_stub(base, monkeypatch, *, global_settings=False):
    names = [
        "ultralytics",
        "ultralytics.utils",
        "ultralytics.utils.checks",
        "ultralytics.utils.downloads",
        "ultralytics.data",
        "ultralytics.data.utils",
        "ultralytics.utils.plotting",
        "ultralytics.nn",
        "ultralytics.nn.tasks",
    ]
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def incidental_network(*args, **kwargs):
        pytest.fail("Incidental dependency download was attempted")

    for name in [
        "ultralytics.utils.checks",
        "ultralytics.data.utils",
        "ultralytics.utils.plotting",
    ]:
        modules[name].check_font = incidental_network
    for name in ["ultralytics.utils.downloads", "ultralytics.nn.tasks"]:
        modules[name].attempt_download_asset = incidental_network
    modules["ultralytics.utils.downloads"].safe_download = incidental_network
    modules["ultralytics.utils.downloads"].download = incidental_network
    modules["ultralytics"].download = incidental_network
    config_dir = base / "outputs/multimodal_leaf/2026-10-24/runtime_config/Ultralytics"
    if global_settings:
        config_dir = base.parents[2] / "global-user-settings"
    modules["ultralytics.utils"].USER_CONFIG_DIR = config_dir
    modules["ultralytics.utils"].SETTINGS_FILE = config_dir / "settings.json"
    settings = {
        "weights_dir": "wrong",
        "runs_dir": "wrong",
        "datasets_dir": "wrong",
        "sync": True,
        "wandb": True,
        "tensorboard": True,
        "mlflow": True,
        "clearml": True,
        "comet": True,
        "dvc": True,
        "neptune": True,
        "raytune": True,
    }
    modules["ultralytics.utils"].SETTINGS = settings
    modules["ultralytics"].settings = settings
    modules["ultralytics"].__version__ = "8.4.187"
    return modules


def test_missing_amp_checkpoint_fails_before_importing_optional_dependency(
    model_inputs, monkeypatch
):
    base, _, _ = model_inputs
    module = adapter()
    original = importlib.import_module

    def no_dependency(name, *args, **kwargs):
        if name.startswith("ultralytics"):
            pytest.fail("Optional dependency imported before missing pinned AMP asset was checked")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", no_dependency)
    with pytest.raises(FileNotFoundError, match="AMP|yolo26n.pt"):
        module.configure_ultralytics(base, require_amp=True)


def test_checked_bundled_font_and_pinned_amp_keep_training_offline(model_inputs, monkeypatch):
    base, payload, specification = model_inputs
    (base / "yolo26n.pt").write_bytes(payload)
    modules = dependency_stub(base, monkeypatch)
    policy = adapter().configure_ultralytics(base, require_amp=True)
    font = modules["ultralytics.data.utils"].check_font("Arial.ttf")
    assert Path(font).name == "DejaVuSans.ttf"
    assert Path(font).is_file()
    assert modules["ultralytics.utils.plotting"].check_font("Arial.Unicode.ttf") == font
    assert modules["ultralytics.utils.checks"].check_font("missing-font.ttf") == font
    assert modules["ultralytics.utils"].WEIGHTS_DIR == base
    assert modules["ultralytics"].settings["weights_dir"] == str(base)
    assert not modules["ultralytics"].settings["sync"]
    assert not modules["ultralytics"].settings["wandb"]
    assert os.environ["YOLO_OFFLINE"] == "true"
    assert policy["amp_checkpoint_sha256"] == specification["sha256"]
    assert policy["font"]["family"] == "DejaVu Sans"
    assert not list(base.rglob("Arial*.ttf"))
    assert (
        json.loads(
            (base / "outputs/multimodal_leaf/2026-10-24/offline_runtime_policy.json").read_text()
        )["font"]["sha256"]
        == hashlib.sha256(Path(font).read_bytes()).hexdigest()
    )


def test_download_entry_points_require_existing_local_files(model_inputs, monkeypatch):
    base, payload, _ = model_inputs
    modules = dependency_stub(base, monkeypatch)
    adapter().configure_ultralytics(base, require_amp=False)
    existing = base / "trained.pt"
    existing.write_bytes(payload)
    assert Path(modules["ultralytics.utils.downloads"].attempt_download_asset(existing)) == existing
    assert Path(modules["ultralytics.nn.tasks"].attempt_download_asset(existing)) == existing
    for call in [
        lambda: modules["ultralytics.utils.downloads"].attempt_download_asset("missing.pt"),
        lambda: modules["ultralytics.utils.downloads"].safe_download(
            "https://example.invalid/font.ttf"
        ),
        lambda: modules["ultralytics"].download("https://example.invalid/weights.pt"),
    ]:
        with pytest.raises(FileNotFoundError, match="[Oo]ffline|[Dd]ownload|[Ll]ocal"):
            call()


def test_corrupted_amp_checkpoint_is_rejected(model_inputs, monkeypatch):
    base, payload, _ = model_inputs
    (base / "yolo26n.pt").write_bytes(payload[:-1] + b"!")
    dependency_stub(base, monkeypatch)
    with pytest.raises(ValueError, match="checksum"):
        adapter().configure_ultralytics(base, require_amp=True)


def test_existing_global_settings_are_never_updated(model_inputs, monkeypatch):
    base, _, _ = model_inputs
    modules = dependency_stub(base, monkeypatch, global_settings=True)
    with pytest.raises(RuntimeError, match="settings|before import"):
        adapter().configure_ultralytics(base, require_amp=False)
    assert modules["ultralytics"].settings["weights_dir"] == "wrong"


def test_inference_checkpoint_integrity_is_checked_before_model_loader(model_inputs):
    base, payload, specification = model_inputs
    checkpoint = base / "trained.pt"
    checkpoint.write_bytes(payload)
    module = adapter()
    assert (
        module.require_checkpoint(checkpoint, base=base, expected_sha256=specification["sha256"])
        == checkpoint
    )
    checkpoint.write_bytes(payload[:-1] + b"!")
    with pytest.raises(ValueError, match="checksum"):
        module.require_checkpoint(checkpoint, base=base, expected_sha256=specification["sha256"])
    with pytest.raises(FileNotFoundError, match="checkpoint|asset"):
        module.require_checkpoint(
            base / "missing.pt", base=base, expected_sha256=specification["sha256"]
        )


def test_training_entrypoints_install_adapter_before_optional_import_and_preserve_recipe():
    methods = Path(__file__).resolve().parents[1] / "src/csi_maize/_methods"
    for filename in [
        "train_multimodal_leaf.py",
        "train_noir_blue_leaf.py",
        "infer_multimodal_leaf.py",
    ]:
        tree = ast.parse((methods / filename).read_text())
        main = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        configured = [
            node
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "configure_ultralytics"
        ]
        assert configured, f"{filename} does not configure the offline dependency boundary"
        first_dependency = min(
            node.lineno
            for node in ast.walk(main)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and (
                isinstance(node, ast.Import)
                and any(name.name == "ultralytics" for name in node.names)
                or isinstance(node, ast.ImportFrom)
                and node.module == "ultralytics"
            )
        )
        assert configured[0].lineno < first_dependency
        if filename.startswith("train_"):
            training = next(
                node
                for node in calls
                if isinstance(node.func, ast.Attribute) and node.func.attr == "train"
            )
            args = {keyword.arg: keyword.value for keyword in training.keywords}
            assert isinstance(args["amp"], ast.Constant) and args["amp"].value is True
            assert isinstance(args["imgsz"], ast.Constant) and args["imgsz"].value == 768
            assert isinstance(args["seed"], ast.Constant) and args["seed"].value == 20261024
            assert isinstance(args["batch"], ast.Constant) and args["batch"].value == 8


def test_weak_training_help_does_not_require_weights_or_optional_dependency(tmp_path):
    import subprocess

    entrypoint = (
        Path(__file__).resolve().parents[1] / "src/csi_maize/_methods/train_multimodal_leaf.py"
    )
    environment = dict(os.environ, CSI_MAIZE_WORKSPACE=str(tmp_path / "empty-run"))
    result = subprocess.run(
        [sys.executable, str(entrypoint), "--help"], capture_output=True, text=True, env=environment
    )
    assert result.returncode == 0, result.stderr
    assert "--allow-weak-labels" in result.stdout
