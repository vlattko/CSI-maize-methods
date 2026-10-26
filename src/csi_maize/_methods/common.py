"""Paths, tables and explicit sample identity for the local research prototype."""

from pathlib import Path, PureWindowsPath
import csv
import json
import re

from bootstrap import ROOT, BASE, OUT

GENOTYPES = {
    "1": "Os024445",
    "2": "Mo17",
    "3": "B73",
    "4": "Os7798/2",
    "5": "Os6251/15",
    "6": "Os622448",
}
KEYS = ("treatment", "measurement", "genotype", "replicate")


def config():
    return json.loads((BASE / "config.json").read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        raise ValueError("Explicit columns required for an empty table")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_key(row):
    return tuple(str(row[k]).strip() for k in KEYS)


def filename_key(name):
    match = re.fullmatch(r"capture_session_([CSP])(\d+)_(\d+)-(\d+)", name)
    if not match:
        return None
    treatment, measurement, genotype, replicate = match.groups()
    if genotype not in GENOTYPES:
        raise ValueError("Unknown genotype code: " + genotype)
    return treatment, str(int(measurement)), GENOTYPES[genotype], str(int(replicate))


def safe_source(relative):
    """Resolve workspace inputs, allowing only explicitly registered junctions.

    Check lexical containment before resolution so an intentional data junction
    is accepted without permitting traversal into an unregistered directory.
    """
    relative = str(relative).replace("\\", "/")
    logical = Path(relative)
    if logical.is_absolute() or PureWindowsPath(relative).drive or ".." in logical.parts:
        raise ValueError("Source path escapes study folder: " + relative)
    lexical = ROOT / logical
    path = lexical.resolve()
    binding_path = ROOT / "input_bindings.json"
    data = (
        json.loads(binding_path.read_text(encoding="utf-8-sig")) if binding_path.is_file() else {}
    )
    # A registered logical prefix must keep its registered target even if a
    # replacement link happens to point back inside the computational root.
    bindings = sorted(
        data.get("bindings", []),
        key=lambda entry: len(Path(str(entry["logical_path"])).parts),
        reverse=True,
    )
    for binding in bindings:
        prefix = Path(str(binding["logical_path"]).replace("\\", "/"))
        if (
            not prefix.parts
            or prefix.is_absolute()
            or PureWindowsPath(str(prefix)).drive
            or ".." in prefix.parts
        ):
            raise ValueError("Invalid input binding logical path: " + str(prefix))
        if logical.is_relative_to(prefix):
            source_path = Path(binding["source_path"])
            if not source_path.is_absolute():
                raise ValueError("Input binding source must be absolute: " + str(prefix))
            source = source_path.resolve()
            expected = source / logical.relative_to(prefix)
            if path == expected.resolve() and path.is_relative_to(source):
                return lexical
            raise ValueError("Registered input binding changed or escapes its source: " + relative)
    if path.is_relative_to(ROOT.resolve()):
        return lexical
    raise ValueError("Source path escapes registered inputs: " + relative)


def workspace_relative(path):
    """Serialize a validated lexical workspace identity, never a raw target.

    Resolving a registered junction is necessary to validate containment but
    destroys its logical workspace identity. Keep those two operations separate.
    Already-resolved external paths are rejected rather than reverse-mapped to
    a possibly ambiguous binding alias.
    """
    value = str(path).replace("\\", "/")
    candidate = Path(value)
    if ".." in candidate.parts or (PureWindowsPath(value).drive and not candidate.is_absolute()):
        raise ValueError("Path is not a contained lexical workspace identity: " + value)
    if candidate.is_absolute():
        try:
            logical = candidate.relative_to(ROOT)
        except ValueError as error:
            raise ValueError("Path is outside the lexical workspace: " + value) from error
    else:
        logical = candidate
    safe_source(logical)
    return logical.as_posix()


def reference():
    rows = read_csv(safe_source(config()["reference_table"]))
    indexed = {sample_key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate reference sample keys")
    return indexed


def load_mapping():
    path = safe_source(config()["identity_mapping"])
    if not path.exists():
        return {}
    result = {}
    for row in read_csv(path):
        session = row["session"].replace("\\", "/")
        populated = [bool(row[k].strip()) for k in KEYS]
        if not any(populated):
            continue
        if not all(populated) or not row["mapping_evidence"].strip():
            raise ValueError("Partial identity or missing evidence: " + session)
        if session in result:
            raise ValueError("Duplicate session mapping: " + session)
        safe_source(session)
        result[session] = sample_key(row)
    return result


def complete_views(rgb, noir, allowed):
    if len(rgb) != len(noir) or len(rgb) not in allowed:
        return False
    try:
        a = [int(p.stem.rsplit("_", 1)[1]) for p in rgb]
        b = [int(p.stem.rsplit("_", 1)[1]) for p in noir]
    except ValueError:
        return False
    return len(set(a)) == len(a) and set(a) == set(b)


def build_index():
    cfg = config()
    refs = reference()
    mapping = load_mapping()
    records = []
    seen = {}
    paths = set()
    for folder in cfg["source_folders"]:
        parent = safe_source(folder)
        if parent.exists():
            paths.update(p for p in parent.rglob("capture_session*") if p.is_dir())
    for path in sorted(paths):
        relative = workspace_relative(path)
        named = filename_key(path.name)
        mapped = mapping.get(relative)
        if named and mapped and named != mapped:
            raise ValueError("Filename conflicts with identity mapping: " + relative)
        key = mapped or named
        counts = {}
        empty = {}
        nonempty = {}
        for camera in ("rgb", "noir"):
            files = list((path / camera).glob("*.jpg"))
            counts[camera] = sum(p.stat().st_size > 0 for p in files)
            empty[camera] = sum(p.stat().st_size == 0 for p in files)
            nonempty[camera] = [p for p in files if p.stat().st_size > 0]
        status = (
            "unresolved_identity"
            if key is None
            else "unknown_reference_key"
            if key not in refs
            else "identified"
        )
        if key in seen:
            raise ValueError(
                "Multiple sessions claim sample " + str(key) + ": " + seen[key] + " and " + relative
            )
        if key is not None:
            seen[key] = relative
        complete = complete_views(
            nonempty["rgb"], nonempty["noir"], cfg["allowed_views_per_camera"]
        ) and not sum(empty.values())
        records.append(
            {
                "session": relative,
                **dict(zip(KEYS, key or ("", "", "", ""))),
                "identity_status": status,
                "rgb_nonempty": counts["rgb"],
                "noir_nonempty": counts["noir"],
                "empty_files": sum(empty.values()),
                "count_complete": complete,
                "eligible": status == "identified" and complete,
            }
        )
    write_csv(OUT / "sessions.csv", records)
    missing = [dict(zip(KEYS, key)) for key in refs if key not in seen]
    write_csv(OUT / "missing_reference_sessions.csv", missing, list(KEYS))
    unresolved = [
        {"session": r["session"], **dict.fromkeys(KEYS, ""), "mapping_evidence": ""}
        for r in records
        if r["identity_status"] == "unresolved_identity"
    ]
    write_csv(
        OUT / "identity_mapping_template.csv", unresolved, ["session", *KEYS, "mapping_evidence"]
    )
    return records
