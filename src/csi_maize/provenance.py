"""Strict records, streaming digests and verified output manifests."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def verify_files(root: Path, expected: dict[str, str]):
    root = Path(root).absolute()
    for name, expected_digest in expected.items():
        path = (root / name).absolute()
        if not path.is_relative_to(root) or ".." in Path(name).parts or not path.is_file():
            raise ValueError(f"Missing or unsafe result path: {name}")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Result path escapes the workspace: {name}")
        if digest(path) != expected_digest:
            raise ValueError(f"Result changed: {name}")


def file_manifest(root: Path, paths):
    root = Path(root).absolute()
    records = {}
    for item in paths:
        item = Path(item)
        files = sorted(p for p in item.rglob("*") if p.is_file()) if item.is_dir() else [item]
        for path in files:
            if not path.is_file():
                raise ValueError(f"Declared output is missing: {path.relative_to(root)}")
            records[path.absolute().relative_to(root).as_posix()] = digest(path)
    return records


def append_event(path: Path, event: str, details: dict):
    path = Path(path)
    previous = path.read_bytes().splitlines() if path.exists() else []
    record = dict(
        sequence=len(previous) + 1,
        utc=utc(),
        event=event,
        details=details,
        previous_sha256=hashlib.sha256(previous[-1]).hexdigest() if previous else None,
    )
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload + "\n")
    return record
