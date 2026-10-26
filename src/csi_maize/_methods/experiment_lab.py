"""Small durable experiment register; no biological conclusions are inferred here."""

from common import workspace_relative

from bootstrap import BASE, ROOT, OUT
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

RUN = OUT / "additive_experiments/2026-10-24"
EXPERIMENTS = {
    "E01": ("e01_angular", "Directional optical and architectural profiles"),
    "E02": ("e02_spatial", "Spatial tissue-appearance atlas"),
    "E03": ("e03_discordance", "Appearance and physiology discordance"),
    "E04": ("e04_development", "Architectural stage versus chronological round"),
    "E05": ("e05_mechanics", "Rigid-model residuals and mechanics acquisition"),
    "E06": ("e06_recovery", "Cohort changes and recovery-memory protocol"),
    "E07": ("e07_light", "Conditional light interception and self-shading"),
    "E08": ("e08_ambiguity", "Alternative image-compatible organ hypotheses"),
    "E09": ("e09_views", "Acquisition budget and observation value"),
    "E10": ("e10_spectral", "Spectral compression and sensor-information limits"),
    "E11": ("e11_learning", "Cross-view self-supervised representation trial"),
    "E12": ("e12_benchmark", "Ambiguity and quality-control benchmark"),
}
STATUSES = {"completed_exploratory", "partial_requires_new_data", "requires_new_data", "failed"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_event(path, experiment, event, details):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.read_bytes().splitlines() if path.exists() else []
    payload = {
        "sequence": len(previous) + 1,
        "recorded_utc": utc(),
        "experiment": experiment,
        "event": event,
        "previous_event_sha256": sha256_bytes(previous[-1]) if previous else None,
        "details": details,
    }
    line = json.dumps(payload, ensure_ascii=True, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
        stream.flush()
    return payload


def validate_result(result):
    required = {
        "id",
        "title",
        "status",
        "question",
        "methods",
        "metrics",
        "findings",
        "limitations",
        "next_requirements",
        "input_sha256",
        "artifacts",
    }
    if required - set(result):
        raise ValueError("Missing result fields: " + str(sorted(required - set(result))))
    if result["id"] not in EXPERIMENTS or result["status"] not in STATUSES:
        raise ValueError("Unknown experiment or status")
    if (
        result["status"] in {"requires_new_data", "partial_requires_new_data"}
        and not result["next_requirements"]
    ):
        raise ValueError("A pending acquisition must explain what is missing")
    if not isinstance(result["input_sha256"], dict):
        raise ValueError("Source hashes must map filenames to digests")
    json.dumps(result, allow_nan=False)
    return result


def verify_hashes(mapping, root=ROOT):
    for relative, expected in mapping.items():
        path = Path(root) / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("Input hash mismatch: " + str(relative))


def initialize():
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / "events.jsonl"
    if not path.exists():
        append_event(
            path,
            "ALL",
            "run_initialized",
            {
                "python": platform.python_version(),
                "executable": sys.executable,
                "seed": 20261024,
                "interpretation": "Exploratory archive experiments; new acquisition requirements remain pending",
                "configuration_sha256": sha256_file(BASE / "config.json"),
                "method_source_sha256": sha256_file(BASE / "experiment_lab.py"),
            },
        )
        for key, (_, title) in EXPERIMENTS.items():
            append_event(path, key, "registered", {"title": title})
    update_register()


def update_register():
    records = []
    for key, (folder, title) in EXPERIMENTS.items():
        path = RUN / folder / "summary.json"
        if path.exists():
            result = validate_result(read_json(path))
            records.append(
                {
                    "id": key,
                    "title": title,
                    "status": result["status"],
                    "summary": workspace_relative(path),
                    "summary_sha256": sha256_file(path),
                }
            )
        else:
            records.append({"id": key, "title": title, "status": "planned", "summary": None})
    (RUN / "register.json").write_text(
        json.dumps({"updated_utc": utc(), "experiments": records}, indent=2), encoding="utf-8"
    )
    return records


if __name__ == "__main__":
    initialize()
