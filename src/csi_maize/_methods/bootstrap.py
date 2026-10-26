"""Explicit filesystem context for the portable numerical methods.

The stage runner copies these modules into the declared workspace before
execution. Installed Python environments supply dependencies; source folders
never double as package environments.
"""

from pathlib import Path
import json
import os
import ast
import hashlib

SOURCE = Path(__file__).absolute().parent
if os.environ.get("CSI_MAIZE_WORKSPACE"):
    ROOT = Path(os.environ["CSI_MAIZE_WORKSPACE"]).absolute()
    BASE = ROOT / "prototype" / "research_v2"
    OUT = BASE / "outputs"
else:
    ROOT = SOURCE.parents[2]
    BASE = SOURCE
    OUT = ROOT / ".test_outputs"


def method_config():
    path = BASE / "config.json"
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def model_directory(name):
    """Resolve a checkpoint in the explicitly configured pretrained cache."""
    cache = Path(os.environ.get("CSI_MAIZE_MODELS", str(BASE / "models")))
    return cache / name


def preferred_device():
    """Use the first visible CUDA device, or CPU when CUDA is masked/unavailable."""
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def publication_case():
    return method_config().get("case", "publication") == "publication"


def validate_rotation_records(rows):
    """Require paired 40-view rotations; do not reshape incomplete new data."""
    groups = {}
    for row in rows:
        key = (row["session"], row["camera"])
        view = int(row["view"])
        if row["camera"] not in ("rgb", "noir") or view in groups.setdefault(key, set()):
            raise ValueError("Duplicate or unknown camera/view: " + str((key, view)))
        groups[key].add(view)
    sessions = {key[0] for key in groups}
    if not sessions:
        raise ValueError("No complete paired camera rotations supplied")
    for session in sessions:
        for camera in ("rgb", "noir"):
            if groups.get((session, camera)) != set(range(40)):
                raise ValueError("Method requires indices 0..39 for both cameras: " + session)
    expected = method_config().get("expected_acquisitions", 41) if publication_case() else None
    if expected is not None and len(sessions) != int(expected):
        raise ValueError(f"Publication case expects {expected} acquisitions, found {len(sessions)}")
    return len(sessions)


def method_source_hashes(*entrypoints):
    """Hash the full local import closure, including imports inside functions."""
    from common import workspace_relative

    pending = [BASE / name for name in entrypoints]
    found = {}
    while pending:
        path = pending.pop()
        key = workspace_relative(path)
        if key in found:
            continue
        raw = path.read_bytes()
        found[key] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw.decode("utf-8-sig"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
        pending.extend(
            BASE / (name + ".py") for name in modules if (BASE / (name + ".py")).is_file()
        )
    return dict(sorted(found.items()))


def read_csv(path):
    import csv

    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fields=None):
    import csv

    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
