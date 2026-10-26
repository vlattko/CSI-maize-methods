"""Verify pinned pretrained assets without importing or executing model code."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import tempfile
import urllib.request
from urllib.parse import urlsplit

from .config import RunConfig
from .provenance import digest, write_json

MANIFEST_PATH = Path(__file__).with_name("assets_manifest.json")


def _overlaps_source(path: Path, package: Path) -> bool:
    resolved = path.resolve()
    return resolved.is_relative_to(package) or package.is_relative_to(resolved)


def _owned_runtime_snapshot(workspace: Path, package: Path) -> bool:
    expected = workspace.resolve() / "runtime/csi_maize"
    if package != expected:
        return False
    marker = _contained(workspace, ".csi_workspace.json")
    if not marker.is_file():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError):
        return False
    return (
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and value.get("owner") == "csi-maize-methods"
    )


def _relative(value: str) -> str:
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or PureWindowsPath(value).drive
        or ".." in path.parts
        or ":" in value
    ):
        raise ValueError("Asset path must be relative and contained")
    return path.as_posix()


def _contained(root: Path, relative: str) -> Path:
    path = root / _relative(relative)
    if path.resolve() == root.resolve() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Asset path escapes outside its declared root")
    return path


def _verify(path: Path, file: dict, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing model asset: {label}")
    if path.stat().st_size != file["bytes"] or digest(path) != file["sha256"]:
        raise ValueError(f"Model asset checksum mismatch: {label}")


def _validate_file(model: dict, file: dict) -> None:
    cache = _relative(file["cache_path"])
    _relative(file["workspace_path"])
    for fallback in file.get("fallback_inputs", []):
        _relative(fallback)
    if (
        isinstance(file["bytes"], bool)
        or not isinstance(file["bytes"], int)
        or file["bytes"] <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", file["sha256"])
    ):
        raise ValueError("Asset size and SHA256 must be pinned")
    filename = PurePosixPath(cache).name
    if model["provider"] == "huggingface":
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]) or not re.fullmatch(
            r"facebook/[A-Za-z0-9_.-]+", model["model"]
        ):
            raise ValueError("Hugging Face model requires a pinned official revision")
        expected = f"https://huggingface.co/{model['model']}/resolve/{model['revision']}/{filename}"
    elif model["provider"] == "github_release":
        if model.get("repository") != "ultralytics/assets" or not re.fullmatch(
            r"v[0-9]+\.[0-9]+\.[0-9]+", model["revision"]
        ):
            raise ValueError("Only pinned official Ultralytics release assets are accepted")
        expected = f"https://github.com/ultralytics/assets/releases/download/{model['revision']}/{filename}"
    else:
        raise ValueError("Unknown pretrained asset provider")
    if file["url"] != expected:
        raise ValueError("Asset download URL must match its pinned official upstream revision")


def _download(file: dict, destination: Path, label: str) -> None:
    """Stream official bytes into an untrusted temporary file, then verify."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=destination.name + ".", suffix=".download", dir=destination.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            request = urllib.request.Request(
                file["url"], headers={"User-Agent": "csi-maize-methods/0.1"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                if hasattr(response, "geturl"):
                    final = urlsplit(response.geturl())
                    host = final.hostname or ""
                    allowed = host in {
                        "huggingface.co",
                        "github.com",
                        "release-assets.githubusercontent.com",
                        "objects.githubusercontent.com",
                    } or host.endswith((".hf.co", ".huggingface.co"))
                    if final.scheme != "https" or not allowed:
                        raise ValueError("Asset download redirected outside official HTTPS storage")
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > file["bytes"]:
                        raise ValueError(f"Model asset checksum/size mismatch: {label}")
                    stream.write(chunk)
        _verify(temporary, file, label)
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_models(
    config: RunConfig,
    workspace: Path,
    allow_download: bool = False,
    required: tuple[str, ...] | None = None,
) -> dict:
    """Verify explicit cache inputs, optionally fetch missing files, and stage.

    Offline is the default. Existing altered assets are always rejected rather
    than silently replaced. Only selected assets are considered; historical
    reconstruction checkpoints are outside this manifest and cannot be fetched.
    """
    workspace = Path(workspace).absolute()
    package = Path(__file__).resolve().parent
    if package.is_relative_to(workspace.resolve()) and not _owned_runtime_snapshot(
        workspace, package
    ):
        raise ValueError("Model workspace must not contain the package source tree")
    cache = (
        Path(config.models_root)
        if config.models_root is not None
        else workspace / "prototype/research_v2/models"
    )
    if _overlaps_source(cache, package):
        raise ValueError("Model cache must stay outside package source files")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported asset manifest schema")
    chosen = (
        list(required)
        if required is not None
        else list(manifest["default_required"])
        + list(manifest.get("profile_additions", {}).get(config.profile, []))
    )
    chosen = list(dict.fromkeys(chosen))
    unknown = set(chosen) - manifest["models"].keys()
    if unknown:
        raise ValueError("Unknown requested model: " + ", ".join(sorted(unknown)))
    plan, cache_paths, destinations = [], set(), set()
    for name in chosen:
        model = manifest["models"][name]
        for file in model["files"]:
            _validate_file(model, file)
            if file["cache_path"] in cache_paths or file["workspace_path"] in destinations:
                raise ValueError("Duplicate model asset destination")
            cache_paths.add(file["cache_path"])
            destinations.add(file["workspace_path"])
            cached = _contained(cache, file["cache_path"])
            target = _contained(workspace, file["workspace_path"])
            if _overlaps_source(cached, package) or _overlaps_source(target, package):
                raise ValueError("Model asset targets must not overlap package source files")
            if target.exists():
                _verify(target, file, name + "/workspace")
            source = cached if cached.exists() else None
            if source is None:
                for fallback in file.get("fallback_inputs", []):
                    candidate = _contained(config.data_root, fallback)
                    if candidate.exists():
                        source = candidate
                        break
            if source is not None:
                _verify(source, file, name + "/" + file["cache_path"])
            plan.append((name, model, file, source, cached, target))
    missing = [
        name + "/" + file["cache_path"] for name, _, file, source, _, _ in plan if source is None
    ]
    if missing and not allow_download:
        raise FileNotFoundError(
            "Offline model assets are missing: "
            + ", ".join(missing)
            + "; populate models_root or explicitly allow the models-stage download"
        )
    # No run output is materialized until every requested cache input verifies.
    resolved = []
    for name, model, file, source, cached, target in plan:
        if source is None:
            _download(file, cached, name + "/" + file["cache_path"])
            source = cached
        resolved.append((name, model, file, source, target))
    result = {
        "status": "verified",
        "manifest_sha256": digest(MANIFEST_PATH),
        "runtime_models_root": "prototype/research_v2/models",
        "models": {},
    }
    for name, model, file, source, target in resolved:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve() and not target.exists():
            with tempfile.NamedTemporaryFile(
                prefix=target.name + ".", suffix=".staging", dir=target.parent, delete=False
            ) as stream:
                staging = Path(stream.name)
            try:
                shutil.copyfile(source, staging)
                _verify(staging, file, name + "/staging")
                staging.replace(target)
            finally:
                staging.unlink(missing_ok=True)
        _verify(target, file, name + "/workspace")
        entry = result["models"].setdefault(
            name,
            {
                "model": model["model"],
                "revision": model["revision"],
                "license": model["license"],
                "files": {},
            },
        )
        entry["files"][file["cache_path"]] = {
            "bytes": file["bytes"],
            "sha256": file["sha256"],
            "workspace_path": file["workspace_path"],
            "source_url": file["url"],
        }
    write_json(_contained(workspace, "metadata/model_assets.json"), result)
    return result
