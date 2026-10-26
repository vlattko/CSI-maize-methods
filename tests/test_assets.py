"""Asset integrity and explicit network-boundary tests with tiny real files."""

import hashlib
import importlib
import io
import json
from pathlib import Path

import pytest

from csi_maize.config import RunConfig


def api(fixture, monkeypatch):
    try:
        module = importlib.import_module("csi_maize.assets")
    except ModuleNotFoundError:
        pytest.fail("Verified model-asset preparation has not been implemented")
    monkeypatch.setattr(module, "MANIFEST_PATH", fixture[4])
    return module, *fixture


@pytest.fixture
def tiny_assets(tmp_path, monkeypatch):
    data, cache, workspace = (tmp_path / name for name in ["raw", "cache", "run"])
    data.mkdir()
    cache.mkdir()
    workspace.mkdir()
    payload = b"known pretrained bytes"
    revision = "a" * 40
    entry = {
        "cache_path": "sam/config.json",
        "workspace_path": "prototype/research_v2/models/sam/config.json",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": f"https://huggingface.co/facebook/sam/resolve/{revision}/config.json",
    }
    manifest = {
        "schema_version": 1,
        "default_required": ["sam"],
        "profile_additions": {"learning": ["dino"]},
        "models": {
            "sam": {
                "provider": "huggingface",
                "model": "facebook/sam",
                "revision": revision,
                "license": "Apache-2.0",
                "files": [entry],
            },
            "dino": {
                "provider": "huggingface",
                "model": "facebook/dino",
                "revision": revision,
                "license": "Apache-2.0",
                "files": [
                    dict(
                        entry,
                        cache_path="dino/config.json",
                        workspace_path="prototype/research_v2/models/dino/config.json",
                        url=f"https://huggingface.co/facebook/dino/resolve/{revision}/config.json",
                    )
                ],
            },
        },
    }
    path = tmp_path / "assets.json"
    path.write_text(json.dumps(manifest))
    config = RunConfig(data_root=data, work_dir=workspace, models_root=cache, source_folders=())
    return config, workspace, cache, manifest, path, payload


def seed(cache, path, payload):
    target = cache / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def forbid_network(*args, **kwargs):
    pytest.fail("Offline asset preparation attempted network access")


def test_missing_asset_fails_offline_without_materializing_partial_outputs(
    tiny_assets, monkeypatch
):
    module, config, workspace, cache, _, _, _ = api(tiny_assets, monkeypatch)
    monkeypatch.setattr(module.urllib.request, "urlopen", forbid_network)
    with pytest.raises(FileNotFoundError, match="[Oo]ffline.*sam|sam.*[Oo]ffline"):
        module.prepare_models(config, workspace)
    assert not (workspace / "prototype").exists()
    assert not list(cache.iterdir())


def test_local_bytes_are_verified_and_copied_without_loading_model_code(tiny_assets, monkeypatch):
    module, config, workspace, cache, _, _, payload = api(tiny_assets, monkeypatch)
    original = seed(cache, "sam/config.json", payload)
    monkeypatch.setattr(module.urllib.request, "urlopen", forbid_network)
    result = module.prepare_models(config, workspace)
    target = workspace / "prototype/research_v2/models/sam/config.json"
    assert target.read_bytes() == payload
    assert original.read_bytes() == payload
    assert result["status"] == "verified"
    assert (
        result["models"]["sam"]["files"]["sam/config.json"]["sha256"]
        == hashlib.sha256(payload).hexdigest()
    )
    record = json.loads((workspace / "metadata/model_assets.json").read_text())
    assert str(cache) not in json.dumps(record)
    assert str(workspace) not in json.dumps(record)


def test_tampered_cache_is_rejected_even_with_download_permission(tiny_assets, monkeypatch):
    module, config, workspace, cache, _, _, payload = api(tiny_assets, monkeypatch)
    seed(cache, "sam/config.json", payload[:-1] + b"!")
    monkeypatch.setattr(module.urllib.request, "urlopen", forbid_network)
    with pytest.raises(ValueError, match="checksum|digest"):
        module.prepare_models(config, workspace, allow_download=True)
    assert not (workspace / "prototype").exists()


def test_workspace_copy_cannot_be_silently_reused_after_tampering(tiny_assets, monkeypatch):
    module, config, workspace, cache, _, _, payload = api(tiny_assets, monkeypatch)
    seed(cache, "sam/config.json", payload)
    monkeypatch.setattr(module.urllib.request, "urlopen", forbid_network)
    module.prepare_models(config, workspace)
    (workspace / "prototype/research_v2/models/sam/config.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum|digest"):
        module.prepare_models(config, workspace)


def test_requested_subset_does_not_require_or_fetch_optional_models(tiny_assets, monkeypatch):
    module, config, workspace, cache, _, _, payload = api(tiny_assets, monkeypatch)
    seed(cache, "sam/config.json", payload)
    monkeypatch.setattr(module.urllib.request, "urlopen", forbid_network)
    result = module.prepare_models(config, workspace, required=("sam",))
    assert set(result["models"]) == {"sam"}
    assert not (workspace / "prototype/research_v2/models/dino").exists()
    with pytest.raises(ValueError, match="[Uu]nknown.*model"):
        module.prepare_models(config, workspace, required=("DA3",))


@pytest.mark.parametrize(
    "key,value",
    [
        ("cache_path", "../escape.pt"),
        ("workspace_path", "../escape.pt"),
        ("workspace_path", "/absolute.pt"),
        ("workspace_path", "C:/absolute.pt"),
        ("cache_path", r"..\escape.pt"),
        ("url", "https://attacker.invalid/config.json"),
        ("url", "https://huggingface.co/facebook/sam/resolve/main/config.json"),
    ],
)
def test_uncontained_paths_and_unpinned_or_unofficial_downloads_are_rejected(
    tiny_assets, key, value, monkeypatch
):
    module, config, workspace, cache, manifest, path, payload = api(tiny_assets, monkeypatch)
    seed(cache, "sam/config.json", payload)
    manifest["models"]["sam"]["files"][0][key] = value
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(module.urllib.request, "urlopen", forbid_network)
    with pytest.raises(ValueError):
        module.prepare_models(config, workspace, allow_download=True)


def test_destination_symlink_cannot_redirect_model_writes_outside_workspace(
    tiny_assets, monkeypatch
):
    module, config, workspace, cache, _, _, payload = api(tiny_assets, monkeypatch)
    seed(cache, "sam/config.json", payload)
    external = workspace.parent / "external"
    external.mkdir()
    target = workspace / "prototype/research_v2/models"
    target.parent.mkdir(parents=True)
    try:
        target.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks unavailable")
    with pytest.raises(ValueError, match="escape|outside|contain"):
        module.prepare_models(config, workspace)
    assert not list(external.iterdir())


def test_explicit_download_verifies_transport_bytes_before_using_them(tiny_assets, monkeypatch):
    module, config, workspace, cache, _, _, payload = api(tiny_assets, monkeypatch)
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload)
    )
    result = module.prepare_models(config, workspace, allow_download=True)
    assert result["status"] == "verified"
    assert (cache / "sam/config.json").read_bytes() == payload
    assert (workspace / "prototype/research_v2/models/sam/config.json").read_bytes() == payload


def test_corrupt_download_is_never_promoted_to_a_cached_checkpoint(tiny_assets, monkeypatch):
    module, config, workspace, cache, _, _, _ = api(tiny_assets, monkeypatch)
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong bytes")
    )
    with pytest.raises(ValueError, match="checksum|digest"):
        module.prepare_models(config, workspace, allow_download=True)
    assert not (cache / "sam/config.json").exists()
    assert not list(cache.rglob("*.download"))
    assert not (workspace / "prototype").exists()


def test_learning_profile_requires_dino_without_guessing_another_revision(tiny_assets, monkeypatch):
    module, config, workspace, cache, _, _, payload = api(tiny_assets, monkeypatch)
    config = RunConfig(
        data_root=config.data_root,
        work_dir=workspace,
        models_root=cache,
        source_folders=(),
        profile="learning",
    )
    seed(cache, "sam/config.json", payload)
    monkeypatch.setattr(module.urllib.request, "urlopen", forbid_network)
    with pytest.raises(FileNotFoundError, match="dino"):
        module.prepare_models(config, workspace)
    assert not (workspace / "prototype").exists()
    seed(cache, "dino/config.json", payload)
    assert set(module.prepare_models(config, workspace)["models"]) == {"sam", "dino"}


def test_cache_symlink_cannot_read_an_undeclared_external_checkpoint(tiny_assets, monkeypatch):
    module, config, workspace, cache, _, _, payload = api(tiny_assets, monkeypatch)
    external = workspace.parent / "outside-cache"
    external.mkdir()
    (external / "config.json").write_bytes(payload)
    try:
        (cache / "sam").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks unavailable")
    monkeypatch.setattr(module.urllib.request, "urlopen", forbid_network)
    with pytest.raises(ValueError, match="escape|outside|contain"):
        module.prepare_models(config, workspace)
    assert not (workspace / "prototype").exists()


def test_declared_legacy_weight_input_is_verified_without_mutating_raw_source(
    tiny_assets, monkeypatch
):
    module, config, workspace, cache, manifest, path, payload = api(tiny_assets, monkeypatch)
    entry = manifest["models"]["sam"]["files"][0]
    entry["fallback_inputs"] = ["original-weights/config.json"]
    path.write_text(json.dumps(manifest))
    original = seed(config.data_root, "original-weights/config.json", payload)
    monkeypatch.setattr(module.urllib.request, "urlopen", forbid_network)
    module.prepare_models(config, workspace)
    assert original.read_bytes() == payload
    assert (workspace / "prototype/research_v2/models/sam/config.json").read_bytes() == payload
    assert not list(cache.iterdir())


def test_publication_manifest_includes_pinned_amp_support_without_proprietary_font():
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "src/csi_maize/assets_manifest.json").read_text()
    )
    assert "yolo26n-amp-check" in manifest["default_required"]
    record = manifest["models"]["yolo26n-amp-check"]
    assert (
        record["files"][0]["sha256"]
        == "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
    )
    assert record["files"][0]["workspace_path"] == "prototype/research_v2/yolo26n.pt"
    assert all(
        "arial" not in file["cache_path"].lower()
        for model in manifest["models"].values()
        for file in model["files"]
    )


def test_materialized_runtime_hook_prepares_verified_assets_in_owned_workspace(
    tiny_assets, monkeypatch
):
    import os
    import subprocess
    import sys
    from csi_maize.workspace import create_workspace

    module, config, workspace, cache, manifest, _, payload = api(tiny_assets, monkeypatch)
    seed(cache, "sam/config.json", payload)
    create_workspace(config)
    runtime_manifest = workspace / "runtime/csi_maize/assets_manifest.json"
    runtime_manifest.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, "-m", "csi_maize.hooks", "models", "--workspace", str(workspace)],
        cwd=workspace / "prototype/research_v2",
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(workspace / "runtime")),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "verified"
    assert (workspace / "prototype/research_v2/models/sam/config.json").read_bytes() == payload
    assert (workspace / "metadata/model_assets.json").is_file()


def test_unowned_code_tree_cannot_be_a_model_workspace(tiny_assets, monkeypatch):
    module, config, workspace, cache, _, _, payload = api(tiny_assets, monkeypatch)
    seed(cache, "sam/config.json", payload)
    source = workspace / "checkout/csi_maize"
    source.mkdir(parents=True)
    original = source / "assets.py"
    original.write_text("protected source")
    monkeypatch.setattr(module, "__file__", str(original))
    with pytest.raises(ValueError, match="source|package|runtime"):
        module.prepare_models(config, workspace)
    assert original.read_text() == "protected source"
    assert not (workspace / "prototype").exists()


def test_only_owned_exact_runtime_snapshot_may_live_below_workspace(tiny_assets, monkeypatch):
    module, config, workspace, cache, _, _, payload = api(tiny_assets, monkeypatch)
    seed(cache, "sam/config.json", payload)
    source = workspace / "runtime/csi_maize"
    source.mkdir(parents=True)
    original = source / "assets.py"
    original.write_text("protected source")
    monkeypatch.setattr(module, "__file__", str(original))
    with pytest.raises(ValueError, match="source|package|runtime"):
        module.prepare_models(config, workspace)
    (workspace / ".csi_workspace.json").write_text(
        json.dumps({"schema_version": 1, "owner": "csi-maize-methods"})
    )
    assert module.prepare_models(config, workspace)["status"] == "verified"
    assert original.read_text() == "protected source"


def test_materialized_runtime_is_not_a_permitted_asset_write_target(tiny_assets, monkeypatch):
    module, config, workspace, cache, manifest, path, payload = api(tiny_assets, monkeypatch)
    seed(cache, "sam/config.json", payload)
    source = workspace / "runtime/csi_maize"
    source.mkdir(parents=True)
    original = source / "assets.py"
    original.write_text("protected source")
    monkeypatch.setattr(module, "__file__", str(original))
    (workspace / ".csi_workspace.json").write_text(
        json.dumps({"schema_version": 1, "owner": "csi-maize-methods"})
    )
    manifest["models"]["sam"]["files"][0]["workspace_path"] = "runtime/csi_maize/model.pt"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source|package|runtime"):
        module.prepare_models(config, workspace)
    assert not (source / "model.pt").exists()


def test_model_cache_must_not_contain_the_materialized_package(tiny_assets, monkeypatch):
    module, config, workspace, cache, _, _, payload = api(tiny_assets, monkeypatch)
    source = workspace / "runtime/csi_maize"
    source.mkdir(parents=True)
    original = source / "assets.py"
    original.write_text("protected source")
    monkeypatch.setattr(module, "__file__", str(original))
    (workspace / ".csi_workspace.json").write_text(
        json.dumps({"schema_version": 1, "owner": "csi-maize-methods"})
    )
    config = RunConfig(
        data_root=config.data_root,
        work_dir=workspace,
        models_root=workspace / "runtime",
        source_folders=(),
    )
    seed(config.models_root, "sam/config.json", payload)
    with pytest.raises(ValueError, match="source|package|runtime"):
        module.prepare_models(config, workspace)
    assert original.read_text() == "protected source"
