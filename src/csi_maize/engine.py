"""Execute and verify the declared scientific stage graph without hidden caches."""

from importlib import metadata
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

from .config import RunConfig
from .provenance import append_event, digest, file_manifest, utc, verify_files, write_json
from .workspace import contained_output, create_workspace


def ordered_stages(stages: dict, targets: list[str]) -> list[str]:
    done, active, order = set(), set(), []

    def visit(name):
        if name not in stages:
            raise ValueError(f"Unknown stage: {name}")
        if name in active:
            raise ValueError(f"Dependency cycle at {name}")
        if name in done:
            return
        active.add(name)
        for parent in stages[name]["dependencies"]:
            visit(parent)
        active.remove(name)
        done.add(name)
        order.append(name)

    for name in targets:
        visit(name)
    return order


def source_hashes():
    package = Path(__file__).parent
    return {
        p.relative_to(package).as_posix(): digest(p)
        for p in sorted(package.rglob("*"))
        if p.is_file()
        and p.suffix in {".py", ".json", ".html", ".css", ".js"}
        and "__pycache__" not in p.parts
    }


def raw_hashes(config: RunConfig):
    files = {p for p in config.data_root.glob("*.xlsx") if p.is_file()}
    for folder in config.source_folders:
        parent = config.data_root / folder
        if parent.is_dir():
            files.update(p for p in parent.rglob("*") if p.is_file())
    about = config.data_root / "about"
    if about.is_dir():
        files.update(
            p
            for p in about.rglob("*")
            if p.is_file() and p.suffix.lower() in {".docx", ".json", ".ipynb"}
        )
    mapping = config.data_root / "prototype/identity_mapping.csv"
    if mapping.is_file():
        files.add(mapping)
    return {p.relative_to(config.data_root).as_posix(): digest(p) for p in sorted(files)}


def _generated(root):
    files = []
    for relative in ("prototype/research_v2/outputs", "research_audit", "metadata", "reports"):
        folder = root / relative
        if folder.is_dir():
            files.extend(p for p in folder.rglob("*") if p.is_file())
    return files


def _versions():
    result = dict(python=sys.version, platform=platform.platform(), executable=sys.executable)
    for name in (
        "numpy",
        "scipy",
        "Pillow",
        "openpyxl",
        "torch",
        "torchvision",
        "transformers",
        "ultralytics",
        "opencv-python",
        "matplotlib",
        "scikit-image",
    ):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _save_state(path, state, events):
    lines = events.read_bytes().splitlines() if events.exists() else []
    state["event_count"] = len(lines)
    state["event_tail_sha256"] = hashlib.sha256(lines[-1]).hexdigest() if lines else None
    write_json(path, state)


def _verify_events(root, state):
    path = root / "events.jsonl"
    lines = path.read_bytes().splitlines() if path.exists() else []
    if len(lines) != state["event_count"]:
        raise ValueError("Event history length changed")
    previous = None
    for number, line in enumerate(lines, 1):
        event = json.loads(line)
        if event["sequence"] != number or event["previous_sha256"] != previous:
            raise ValueError("Event history chain changed")
        previous = hashlib.sha256(line).hexdigest()
    if previous != state["event_tail_sha256"]:
        raise ValueError("Event history tail changed")


def _check_completion(path):
    if path.suffix != ".json":
        return
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    candidates = [value] if isinstance(value, dict) else []
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        candidates += [r for r in value["results"] if isinstance(r, dict)]
    for row in candidates:
        status = str(row.get("status", "")).lower()
        if any(word in status for word in ("fail", "partial", "interrupted")):
            raise RuntimeError(
                f"Declared result reports an incomplete or failed analysis: {path.name} ({status})"
            )


def _verify_assets(root):
    manifest = root / "metadata/model_assets.json"
    if not manifest.exists():
        return
    values = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {
        entry["workspace_path"]: entry["sha256"]
        for model in values["models"].values()
        for entry in model["files"].values()
    }
    verify_files(root, expected)


def _validate_write_roots(root, stage=None):
    paths = [
        "prototype/research_v2/outputs",
        "research_audit",
        "metadata",
        "reports",
        "logs",
        "runtime",
    ]
    if stage:
        paths += stage["outputs"]
    for relative in paths:
        target = contained_output(root, relative)
        if target.resolve() != root.resolve() / relative:
            raise ValueError(f"Generated path is redirected through a link: {relative}")


def verify_input_bindings(config):
    values = json.loads((config.work_dir / "input_bindings.json").read_text(encoding="utf-8"))
    allowed = set(config.source_folders) | {"about"}
    seen = set()
    for entry in values["bindings"]:
        name = entry["logical_path"]
        if name not in allowed or name in seen:
            raise ValueError("Input binding is not an authorized unique source folder")
        seen.add(name)
        expected = (config.data_root / name).resolve()
        actual = config.work_dir / name
        if Path(entry["source_path"]).resolve() != expected or actual.resolve() != expected:
            raise ValueError(f"Registered input binding changed: {name}")
    for name in config.source_folders:
        if (config.data_root / name).exists() and name not in seen:
            raise ValueError(f"Required source folder is not bound: {name}")
    for raw in config.data_root.glob("*.xlsx"):
        copy = config.work_dir / raw.name
        if not copy.is_file() or digest(raw) != digest(copy):
            raise ValueError(f"Materialized workbook changed or is missing: {raw.name}")
    stored = json.loads((config.work_dir / "run_config.json").read_text(encoding="utf-8"))
    if stored != config.to_dict():
        raise ValueError("Materialized run configuration changed")
    base = config.work_dir / "prototype/research_v2"
    if digest(base / "config.json") != digest(config.work_dir / "prototype/config.json"):
        raise ValueError("The two compatibility configurations disagree")


def execute(config: RunConfig, targets: list[str], *, stages=None, resume=False):
    if stages is None:
        from .stages import STAGES

        stages = STAGES
    order = ordered_stages(stages, targets)
    root = config.work_dir
    events = root / "events.jsonl"
    marker = root / ".csi_workspace.json"
    if not marker.exists():
        create_workspace(config)
    if json.loads(marker.read_text())["owner"] != "csi-maize-methods":
        raise ValueError("Workspace ownership marker is invalid")
    _validate_write_roots(root)
    verify_input_bindings(config)
    record = root / "run_manifest.json"
    sources = source_hashes()
    inputs = raw_hashes(config)
    context = dict(config=config.to_dict(), code_sha256=sources, raw_sha256=inputs)
    fingerprint = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    if record.exists():
        if not resume:
            raise ValueError("A run already exists; use --resume or a new empty work directory")
        state = json.loads(record.read_text(encoding="utf-8"))
        if state["fingerprint"] != fingerprint:
            raise ValueError("Inputs, code or configuration changed; start a new run")
        if state["runtime"] != _versions():
            raise ValueError("Runtime environment changed; start a new run")
        verify_files(root, state["staged_code_sha256"])
        verify_files(root, state["materialized_input_sha256"])
        verify_files(root, state.get("output_sha256", {}))
        _verify_events(root, state)
        _verify_assets(root)
    else:
        base_code = [
            p
            for p in (root / "prototype/research_v2").iterdir()
            if p.is_file() and p.suffix in {".py", ".json", ".html", ".css", ".js"}
        ]
        runtime_code = [
            p
            for p in (root / "runtime").rglob("*")
            if p.is_file() and p.suffix in {".py", ".json", ".html", ".css", ".js"}
        ]
        state = dict(
            schema_version=1,
            created_utc=utc(),
            mode=config.mode,
            status="initialized",
            fingerprint=fingerprint,
            context=context,
            runtime=_versions(),
            stages={},
            output_sha256={},
        )
        state["staged_code_sha256"] = file_manifest(root, base_code + runtime_code)
        copies = list(root.glob("*.xlsx")) + [
            root / "run_config.json",
            root / "input_bindings.json",
            root / ".csi_workspace.json",
            root / "prototype/config.json",
        ]
        state["materialized_input_sha256"] = file_manifest(root, copies)
        _save_state(record, state, events)
    base = root / "prototype/research_v2"
    env = dict(
        os.environ,
        CSI_MAIZE_WORKSPACE=str(root),
        CSI_MAIZE_MODELS=str(base / "models"),
        PYTHONUTF8="1",
        PYTHONPATH=str(root / "runtime"),
    )
    if config.device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = config.device.split(":", 1)[1]
    elif config.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    for name in order:
        stage = stages[name]
        _validate_write_roots(root, stage)
        previous = state["stages"].get(name)
        if previous and previous["status"] == "completed" and name != "models":
            if previous["commands"] != stage["commands"]:
                raise ValueError(f"Stage definition changed: {name}")
            print(f"VERIFIED {name}", flush=True)
            continue
        started = time.monotonic()
        log = root / "logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        state["status"] = "running"
        state["stages"][name] = dict(
            status="running",
            started_utc=utc(),
            commands=stage["commands"],
            log=log.relative_to(root).as_posix(),
        )
        append_event(events, "stage_started", {"stage": name})
        _save_state(record, state, events)
        print(f"RUN {name}", flush=True)
        try:
            with log.open("a", encoding="utf-8", newline="\n") as stream:
                for command in stage["commands"]:
                    if not command:
                        raise ValueError("Empty stage command")
                    if command[0].startswith("@"):
                        arguments = [
                            sys.executable,
                            "-m",
                            "csi_maize.hooks",
                            command[0][1:],
                            "--workspace",
                            str(root),
                        ]
                    else:
                        script = Path(command[0])
                        if script.is_absolute() or ".." in script.parts or script.suffix != ".py":
                            raise ValueError("Stage scripts must be materialized Python files")
                        path = base / script
                        if not path.is_file() or not path.resolve().is_relative_to(base.resolve()):
                            raise ValueError("Stage script is missing or outside materialized code")
                        arguments = [
                            sys.executable,
                            "-X",
                            "utf8",
                            "-m",
                            "csi_maize.method_runner",
                            str(script),
                            *command[1:],
                        ]
                    outcome = subprocess.run(
                        arguments,
                        cwd=base,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    if outcome.returncode:
                        raise RuntimeError(
                            f"Stage {name} failed with exit {outcome.returncode}; see {log.relative_to(root)}"
                        )
            required = [contained_output(root, p) for p in stage["outputs"]]
            for path in required:
                if not path.exists():
                    raise RuntimeError(
                        f"Stage {name} failed to create declared output: {path.relative_to(root)}"
                    )
                if path.is_file():
                    _check_completion(path)
            state["stages"][name].update(
                status="completed",
                completed_utc=utc(),
                elapsed_seconds=round(time.monotonic() - started, 3),
                declared_output_sha256=file_manifest(root, required),
            )
            append_event(
                events,
                "stage_completed",
                {"stage": name, "elapsed_seconds": state["stages"][name]["elapsed_seconds"]},
            )
        except BaseException as error:
            state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            state["stages"][name].update(status=state["status"], error=str(error), ended_utc=utc())
            state["output_sha256"] = file_manifest(root, _generated(root))
            append_event(events, "stage_" + state["status"], {"stage": name, "error": str(error)})
            _save_state(record, state, events)
            raise
        state["output_sha256"] = file_manifest(root, _generated(root))
        _save_state(record, state, events)
        print(f"DONE {name} ({state['stages'][name]['elapsed_seconds']:.1f}s)", flush=True)
    state.update(status="completed", completed_utc=utc(), targets=targets)
    state["output_sha256"] = file_manifest(root, _generated(root))
    _save_state(record, state, events)
    return state


def verify_run(root: Path, *, verify_inputs=True):
    root = Path(root).absolute()
    state = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if state["status"] != "completed":
        raise ValueError(f"Run is {state['status']}; it is not a completed pipeline")
    verify_files(root, state["output_sha256"])
    verify_files(root, state["staged_code_sha256"])
    verify_files(root, state["materialized_input_sha256"])
    _verify_events(root, state)
    if verify_inputs:
        _verify_assets(root)
        config = RunConfig(**state["context"]["config"])
        verify_input_bindings(config)
        if raw_hashes(config) != state["context"]["raw_sha256"]:
            raise ValueError("Original raw inputs changed or are unavailable")
    return dict(
        status="passed",
        mode=state["mode"],
        stages=len(state["stages"]),
        verified_outputs=len(state["output_sha256"]),
        original_inputs_verified=verify_inputs,
        fingerprint=state["fingerprint"],
    )
