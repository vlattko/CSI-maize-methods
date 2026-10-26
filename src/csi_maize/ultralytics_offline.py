"""Checked offline adapter for the pinned optional Ultralytics dependency.

The numerical AMP check and training recipes are retained. Plotting uses the
open DejaVu font already shipped by Matplotlib rather than fetching a font.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import re
import sys

from . import assets
from .provenance import digest, write_json
from .workspace import contained_output

PINNED_VERSION = "8.4.187"


def require_checkpoint(
    path: Path | str, *, base: Path, model_id: str | None = None, expected_sha256: str | None = None
) -> Path:
    """Require a contained local checkpoint with an independently bound hash."""
    base = Path(base).absolute()
    target = Path(path)
    if not target.is_absolute():
        target = base / target
    if not target.resolve().is_relative_to(base.resolve()):
        raise ValueError("Checkpoint escapes the numerical workspace")
    if not target.is_file():
        raise FileNotFoundError(
            f"Missing local checkpoint asset {target.name}; prepare the verified model assets first"
        )
    expected_size = None
    if model_id is not None:
        manifest = json.loads(assets.MANIFEST_PATH.read_text(encoding="utf-8-sig"))
        if model_id not in manifest["models"]:
            raise ValueError(f"Unknown declared checkpoint model: {model_id}")
        files = manifest["models"][model_id]["files"]
        if len(files) != 1:
            raise ValueError("This checkpoint requires a single declared weight file")
        expected_size = files[0]["bytes"]
        declared = files[0]["sha256"]
        if expected_sha256 is not None and expected_sha256 != declared:
            raise ValueError("Checkpoint hash conflicts with the asset declaration")
        expected_sha256 = declared
    if expected_sha256 is None or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("A checkpoint checksum must be supplied before model loading")
    if (
        expected_size is not None
        and target.stat().st_size != expected_size
        or digest(target) != expected_sha256
    ):
        raise ValueError(f"Local checkpoint checksum mismatch: {target.name}")
    return target


def _replace_aliases(attribute, original, replacement):
    for name, module in list(sys.modules.items()):
        if module is not None and (name == "ultralytics" or name.startswith("ultralytics.")):
            if getattr(module, attribute, None) is original:
                setattr(module, attribute, replacement)


def configure_ultralytics(base: Path, *, require_amp: bool = True) -> dict:
    """Install the supported local-file boundary before importing YOLO models."""
    base = Path(base).absolute()
    amp = (
        require_checkpoint(base / "yolo26n.pt", base=base, model_id="yolo26n-amp-check")
        if require_amp
        else None
    )
    relative = "outputs/multimodal_leaf/2026-10-24"
    runtime = contained_output(base, relative)
    config_dir = contained_output(base, relative + "/runtime_config")
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    mpl_config = contained_output(base, relative + "/matplotlib_config")
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config)

    import matplotlib
    from matplotlib.ft2font import FT2Font

    font_root = Path(matplotlib.get_data_path()) / "fonts/ttf"
    font = font_root / "DejaVuSans.ttf"
    license_file = font_root / "LICENSE_DEJAVU"
    if (
        not font.is_file()
        or not license_file.is_file()
        or FT2Font(font).family_name != "DejaVu Sans"
    ):
        raise FileNotFoundError("Matplotlib's bundled open DejaVu Sans font/license is unavailable")

    ultra = importlib.import_module("ultralytics")
    utils = importlib.import_module("ultralytics.utils")
    if ultra.__version__ != PINNED_VERSION:
        raise RuntimeError(f"Offline adapter requires ultralytics=={PINNED_VERSION}")
    for location in (utils.USER_CONFIG_DIR, utils.SETTINGS_FILE):
        if not Path(location).resolve().is_relative_to(config_dir.resolve()):
            raise RuntimeError(
                "Ultralytics has non-run settings; configure the adapter before importing the dependency in a fresh process"
            )
    settings = utils.SETTINGS
    settings.update(
        {
            "weights_dir": str(base),
            "datasets_dir": str(runtime / "weak_datasets"),
            "runs_dir": str(runtime / "training"),
            **{
                key: False
                for key in (
                    "sync",
                    "wandb",
                    "tensorboard",
                    "mlflow",
                    "clearml",
                    "comet",
                    "dvc",
                    "neptune",
                    "raytune",
                )
                if key in settings
            },
        }
    )
    utils.WEIGHTS_DIR = base
    utils.DATASETS_DIR = runtime / "weak_datasets"
    utils.RUNS_DIR = runtime / "training"
    utils.ONLINE = False
    downloads = importlib.import_module("ultralytics.utils.downloads")
    checks = importlib.import_module("ultralytics.utils.checks")
    checks.ONLINE = False

    def local_asset(file, *args, **kwargs):
        if "://" in str(file):
            raise FileNotFoundError(
                "Offline model loading requires a verified local checkpoint; downloads are refused"
            )
        target = Path(file)
        if not target.is_absolute():
            target = base / target
        if not target.resolve().is_relative_to(base.resolve()):
            raise ValueError("Checkpoint escapes the numerical workspace")
        if not target.is_file():
            raise FileNotFoundError(
                f"Offline checkpoint is missing: {target.name}; run the explicit assets command first"
            )
        return str(target)

    def no_download(*args, **kwargs):
        raise FileNotFoundError(
            "Offline dependency download refused; prepare declared assets explicitly"
        )

    def bundled_font(*args, **kwargs):
        return font

    # Replace cached aliases as well as provider entry points. Late imports then
    # receive these same local-only functions without changing check_amp itself.
    for attribute, replacement in (
        ("attempt_download_asset", local_asset),
        ("safe_download", no_download),
        ("download", no_download),
    ):
        original = getattr(downloads, attribute)
        _replace_aliases(attribute, original, replacement)
        setattr(downloads, attribute, replacement)
    original_font = checks.check_font
    _replace_aliases("check_font", original_font, bundled_font)
    checks.check_font = bundled_font
    data_utils = importlib.import_module("ultralytics.data.utils")
    plotting = importlib.import_module("ultralytics.utils.plotting")
    data_utils.check_font = bundled_font
    plotting.check_font = bundled_font
    policy = {
        "ultralytics_version": PINNED_VERSION,
        "incidental_downloads": "refused",
        "settings_scope": relative + "/runtime_config/Ultralytics",
        "weights_scope": "workspace/prototype/research_v2",
        "amp_recipe": "unchanged; upstream check uses pinned local detection checkpoint"
        if require_amp
        else "not requested for inference",
        "amp_checkpoint_sha256": digest(amp) if amp is not None else None,
        "font": {
            "family": "DejaVu Sans",
            "source": "Matplotlib bundled fonts/ttf/DejaVuSans.ttf",
            "sha256": digest(font),
            "license": "DejaVu Fonts License (Bitstream Vera/Arev; DejaVu changes public domain)",
            "license_sha256": digest(license_file),
        },
    }
    write_json(contained_output(base, relative + "/offline_runtime_policy.json"), policy)
    return policy
