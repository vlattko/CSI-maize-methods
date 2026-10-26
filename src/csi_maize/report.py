"""Self-contained architecture report and readable supporting-atlas template.

Only declared scientific fields enter the public document. Paths and complete
internal manifests are deliberately not copied into its embedded payload.
"""

from __future__ import annotations

import csv
import html
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re


WEB = Path(__file__).with_name("web")
RUN = Path("leaf_dynamics/2026-10-24")


def build_atlas_template() -> str:
    """Inline the authoring assets while retaining scientific population tokens."""
    return (
        (WEB / "atlas.html")
        .read_text(encoding="utf-8")
        .replace("__ATLAS_CSS__", (WEB / "atlas.css").read_text(encoding="utf-8"))
        .replace("__ATLAS_JS__", (WEB / "atlas.js").read_text(encoding="utf-8"))
    )


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def _csv(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, ValueError):
        return []


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value) -> str:
    number = _number(value)
    return "Unavailable" if number is None else f"{number:.4f}"


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f'<th scope="col">{html.escape(label)}</th>' for label in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows
    )
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _missing(title: str, explanation: str) -> str:
    return (
        f'<div class="missing"><h3>{html.escape(title)}</h3><p>{html.escape(explanation)}</p></div>'
    )


def _mesh(outputs: Path):
    paths = sorted(outputs.glob("organ_inverse_triangles/*/browser_mesh.json"))
    if not paths:
        return None
    source = _json(paths[0])
    try:
        vertices = source["vertices"]
        if not vertices or not all(
            len(v) == 3 and all(_number(c) is not None for c in v) for v in vertices
        ):
            return None
        vertices = [[float(c) for c in v] for v in vertices]
        faces = source["faces"]
        if not faces or not all(
            len(f) == 3 and all(isinstance(i, int) and 0 <= i < len(vertices) for i in f)
            for f in faces
        ):
            return None
        organs = []
        for organ in source.get("organs", []):
            start, count = int(organ["vertex_start"]), int(organ["vertex_count"])
            if start < 0 or count < 1 or start + count > len(vertices):
                return None
            organs.append(
                dict(
                    label=str(organ.get("leaf_id", f"Leaf {len(organs) + 1}")),
                    start=start,
                    count=count,
                )
            )
        return dict(vertices=vertices, faces=faces, organs=organs)
    except (TypeError, KeyError, ValueError):
        return None


def _removal(outputs: Path):
    source = _json(outputs / RUN / "jeb_vision_v2/numerical_stability_summary.json") or {}
    edges = []
    for row in source.get("edge_summaries", []):
        if _number(row.get("baseline_mean512")) is None:
            continue
        edges.append(
            dict(
                donor=str(row.get("removed_leaf", "Unspecified")),
                receiver=str(row.get("receiver_leaf", "Unspecified")),
                value=_number(row.get("baseline_mean512")),
                low=_number(row.get("grid_min_mean512")),
                high=_number(row.get("grid_max_mean512")),
                resolved=row.get(
                    "all_configurations_resolved_positive",
                    row.get("all_eight_resolved_positive"),
                )
                is True,
            )
        )
    edges.sort(key=lambda row: -row["value"])
    if not edges:
        return [], _missing(
            "Removal outcomes unavailable",
            "Run the fixed-shape removal experiment and its two-grid numerical comparison to populate directed receiver effects.",
        )
    table = _table(
        [
            "Removed organ → fixed receiver",
            "Baseline release fraction",
            "Finite-family minimum",
            "Finite-family maximum",
            "Resolved in all configurations",
        ],
        [
            [
                f"{r['donor']} → {r['receiver']}",
                _fmt(r["value"]),
                _fmt(r["low"]),
                _fmt(r["high"]),
                "Yes" if r["resolved"] else "No / not recorded",
            ]
            for r in edges
        ],
    )
    return edges, table


def _geometry_diagnostics(outputs: Path) -> str:
    paths = sorted(outputs.glob("organ_inverse_triangles/*/summary.json"))
    if not paths:
        return ""
    source = _json(paths[0]) or {}
    scores = source.get("triangle_render_development_heldout", {})
    if not scores:
        return ""
    return (
        '<h3>Development-view mask consistency</h3><p class="small">Repeatedly inspected same-acquisition views; not independent anatomy accuracy.</p>'
        + _table(
            [
                "Mean triangle IoU",
                "Precision",
                "Recall",
                "Fitting views",
                "Development evaluation views",
            ],
            [
                [
                    _fmt(scores.get("iou")),
                    _fmt(scores.get("precision")),
                    _fmt(scores.get("recall")),
                    len(source.get("train_indices", [])),
                    len(source.get("heldout_indices", [])),
                ]
            ],
        )
    )


def _reposition(outputs: Path):
    folder = outputs / RUN / "jeb_v5/reposition_fixed_domain"
    data = []
    for row in _csv(folder / "direction_averaged_responses.csv"):
        keys = [
            "candidate_id",
            "resolution",
            "angle_degrees",
            "equal9_receiver_change_fraction",
            "equal9_whole_change_H2",
        ]
        values = {key: _number(row.get(key)) for key in keys}
        if any(value is None for value in values.values()):
            continue
        data.append(
            dict(
                candidate=int(values["candidate_id"]),
                resolution=int(values["resolution"]),
                mode=str(row.get("mode", "Unspecified")),
                angle=values["angle_degrees"],
                receiver=values["equal9_receiver_change_fraction"],
                whole=values["equal9_whole_change_H2"],
            )
        )
    if not data:
        return [], _missing(
            "Reposition outcomes unavailable",
            "Run the fixed-domain rigid-neighbour experiment to populate signed receiver and whole-shoot changes.",
        )
    resolutions = sorted({r["resolution"] for r in data})
    best = resolutions[-1]
    # Prescribed baseline configuration, retaining every angle in the finest grid.
    baseline = sorted(
        [r for r in data if r["candidate"] == 0 and r["resolution"] == best],
        key=lambda r: (r["mode"], r["angle"]),
    )
    details = _table(
        ["Movement", "Angle (degrees)", "Receiver fraction change", "Whole-shoot change / H²"],
        [
            [
                r["mode"].replace("_degrees", "").title(),
                f"{r['angle']:g}",
                _fmt(r["receiver"]),
                _fmt(r["whole"]),
            ]
            for r in baseline
        ],
    )
    summary = _json(folder / "summary.json") or {}
    ledger = (
        '<p class="small">Recorded transition closure residual: '
        + _fmt(summary.get("max_transition_closure_H2"))
        + " H². Available grids: "
        + ", ".join(map(str, resolutions))
        + " pixels per side.</p>"
    )
    return (
        data,
        '<canvas id="response-chart" aria-label="Signed fixed-receiver response across every prescribed baseline angle"></canvas>'
        + details
        + ledger,
    )


def _recovery(outputs: Path):
    rows = []
    for cohort, label in [
        ("recovery", "Designed shoots"),
        ("recovery_distichous", "Exploratory alternating-azimuth shoots"),
    ]:
        source = _json(outputs / RUN / "jeb_v5" / cohort / "summary.json") or {}
        for record in source.get("results", []):
            case = record.get("case", {})
            case_id = str(case.get("id", "Unspecified case"))
            if record.get("status") != "completed":
                rows.append([label, case_id, "Fit failed / incomplete"] + ["Unavailable"] * 5)
                continue
            metrics = record.get("metrics", {})
            initial, fitted = metrics.get("initial", {}), metrics.get("fitted", {})
            before = _number(initial.get("release_mean_abs_error"))
            after = _number(fitted.get("release_mean_abs_error"))
            change = (
                "Unavailable"
                if before is None or after is None
                else "Worsened"
                if after > before
                else "Improved"
                if after < before
                else "Unchanged"
            )
            rows.append(
                [
                    label,
                    case_id,
                    _fmt(initial.get("mean_withheld_clean_mask_iou")),
                    _fmt(fitted.get("mean_withheld_clean_mask_iou")),
                    _fmt(fitted.get("surface_symmetric_distance")),
                    _fmt(before),
                    _fmt(after),
                    change,
                ]
            )
    if not rows:
        return _missing(
            "Known-answer diagnostics unavailable",
            "Run the declared assisted synthetic recovery cases to populate image, sampled-geometry and downstream-effect errors.",
        )
    return _table(
        [
            "Cohort",
            "Case",
            "Initial withheld IoU",
            "Fitted withheld IoU",
            "Fitted nearest-vertex distance",
            "Initial release mean absolute error",
            "Fitted release mean absolute error",
            "Release endpoint",
        ],
        rows,
    )


class _ResourceCheck(HTMLParser):
    """Reject active remote/file resources; ordinary reference links are allowed."""

    def __init__(self):
        super().__init__()
        self.safe = True

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        for key in ("src", "poster", "srcset"):
            if key in attrs and not attrs[key].startswith(("data:", "#")):
                self.safe = False
        if tag == "link" and attrs.get("href", "").strip():
            self.safe = False
        if tag in ("object", "embed", "iframe", "base"):
            self.safe = False


def _atlas(outputs: Path):
    paths = sorted(outputs.glob("spatial_fusion/*/CSI_maize_RGB_NoIR_spatial_atlas.html"))
    if not paths:
        return _missing(
            "Spatial atlas unavailable",
            "The supporting observation stage has no generated self-contained atlas in this workspace.",
        )
    source = paths[-1].read_text(encoding="utf-8")
    # Remove the original template's dormant local-viewer fallback and purely
    # internal preservation note. The scientific payload is left unchanged.
    source = source.replace(
        "CSI-MAIZE · Original images and existing manuscript preserved.",
        "CSI-MAIZE · Supporting paired-camera observations.",
    )
    source = re.sub(
        r"document\.body\.innerHTML=.*?;\s*\n",
        "document.body.textContent='Atlas data unavailable.';\n",
        source,
        count=1,
    )
    # Ignore encoded image bytes: arbitrary base64 can coincidentally spell a
    # prohibited attribution without containing any readable metadata at all.
    textual = re.sub(r'data:[^"\'<>\s]*;base64,[A-Za-z0-9+/=]+', "data:embedded", source)
    forbidden = re.search(
        r"(?i)(?:[a-z]:[\\/]+(?:users|home)[\\/]+|/(?:home|users)/|\bonedrive\b|google\s*drive)",
        textual,
    )
    checker = _ResourceCheck()
    checker.feed(source)
    if (
        forbidden
        or not checker.safe
        or re.search(r"\b(?:fetch|XMLHttpRequest|WebSocket)\s*\(", textual)
    ):
        return _missing(
            "Spatial atlas unavailable",
            "The generated atlas did not pass the self-contained public-resource check; regenerate it with the public template.",
        )
    return (
        '<iframe title="Supporting RGB and NoIR spatial atlas" loading="lazy" sandbox="allow-scripts allow-downloads" srcdoc="'
        + html.escape(source, quote=True)
        + '"></iframe>'
    )


def _run_context(workspace: Path):
    """Read execution provenance without interpreting a display as a new run."""
    config = _json(workspace / "run_config.json") or {}
    manifest = _json(workspace / "run_manifest.json") or {}
    config = config if isinstance(config, dict) else {}
    manifest = manifest if isinstance(manifest, dict) else {}
    modes = {config.get("mode"), manifest.get("mode")}
    if modes.intersection({"example", "synthetic_example"}):
        mode = "synthetic_example"
    elif manifest.get("mode", config.get("mode")) == "raw":
        mode = "raw"
    else:
        mode = "stored_results_inspection"
    status = str(manifest.get("status") or "not recorded")
    return mode, status, manifest


def _disclosure(mode: str, status: str) -> str:
    if mode == "synthetic_example":
        title = "PROCEDURAL SYNTHETIC EXAMPLE"
        explanation = (
            "Generated images, leaf-only foregrounds, organ identities, cameras and complete projected paths "
            "are supplied synthetic truth. No pretrained segmentation or detection is executed by this example. "
            "These outputs are not publication observations or independent real-plant validation."
        )
    elif mode == "raw":
        title = "RAW STUDY WORKFLOW"
        explanation = (
            "This display reads the recorded workspace outputs. Generating or rebuilding the report does not "
            "execute missing analyses, verify original inputs or establish biological validation."
        )
    else:
        title = "STORED RESULTS INSPECTION"
        explanation = (
            "This is a display of available stored outputs, not a raw-data recomputation. "
            "No raw execution mode or completed end-to-end reproduction is inferred from these files."
        )
    return (
        '<aside class="boundary" role="note" style="margin:1rem;padding:1rem">'
        f"<b>{title}</b><p>{explanation}</p>"
        f"<p>Execution status at report generation: {html.escape(status)}</p></aside>"
    )


def _choose_target(workspace: Path, target: Path, document: bytes, manifest: dict, explicit: bool):
    """Preserve the identity of every completed, manifested output."""
    if manifest.get("status") != "completed":
        return target
    protected = {
        (workspace / relative).resolve(): digest
        for relative, digest in manifest.get("output_sha256", {}).items()
    }
    expected = protected.get(target.resolve())
    actual = hashlib.sha256(document).hexdigest()
    if expected is None or actual == expected:
        return target
    if explicit:
        raise ValueError(
            "Refusing to change a completed manifest output; choose a different output path"
        )
    suffix = 1
    while True:
        label = "_rebuilt" if suffix == 1 else f"_rebuilt_{suffix}"
        candidate = target.with_name(target.stem + label + target.suffix)
        if candidate.resolve() not in protected:
            if not candidate.exists() or candidate.read_bytes() == document:
                return candidate
        suffix += 1


def build_report(workspace: Path, output: Path | None = None) -> Path:
    """Build the public architecture-first report from existing workspace results.

    Missing or invalid stages are shown explicitly. This function neither fits
    models nor substitutes demonstration numbers for unavailable observations.
    """
    workspace = Path(workspace)
    mode, status, manifest = _run_context(workspace)
    outputs = workspace / "prototype/research_v2/outputs"
    target = (
        Path(output) if output is not None else workspace / "reports/CSI_maize_architecture.html"
    )
    mesh = _mesh(outputs)
    edges, removal = _removal(outputs)
    reposition_data, reposition = _reposition(outputs)
    geometry = (
        (
            '<div class="model-toolbar"><label>Visible organs <select id="organ-select"><option value="all">All organs</option></select></label><button id="reset-view" type="button">Reset view</button><span>Drag to orbit · scroll to zoom</span></div>'
            '<canvas id="mesh-view" aria-label="Interactive fitted source-assisted organ surfaces"></canvas>'
            '<p class="small" id="mesh-note"></p>'
        )
        if mesh
        else _missing(
            "Geometry unavailable",
            "No valid explicit-organ triangle mesh is available. Run the source-assisted organ fit before interpreting architectural interventions.",
        )
    )
    payload = (
        json.dumps(
            dict(mode=mode, status=status, mesh=mesh, removal=edges, reposition=reposition_data),
            allow_nan=False,
            separators=(",", ":"),
        )
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    replacements = dict(
        CSS=(WEB / "report.css").read_text(encoding="utf-8"),
        JS=(WEB / "report.js").read_text(encoding="utf-8"),
        RUN_DISCLOSURE=_disclosure(mode, status),
        REPORT_TITLE="Procedural synthetic example"
        if mode == "synthetic_example"
        else "Leaf-scale architectural experiments",
        LEAD=(
            "Inspect a procedural plant with supplied geometry inputs, hold a receiver leaf fixed, "
            "and calculate neighbour effects under designed beams."
            if mode == "synthetic_example"
            else "Inspect a source-assisted plant model, hold a receiver leaf fixed, and ask how its neighbours change projected exposure."
        ),
        GEOMETRY_CONTEXT=(
            "<p>Procedural images, leaf-only foregrounds, local organ labels, cameras and complete projected paths "
            "are provided inputs. Known-answer fitting updates the declared leaf-surface parameters.</p>"
            "<p>There is no real-image segmentation, automatic organ discovery or camera calibration in this example. "
            "Synthetic dimensions use an arbitrary parameter gauge.</p>"
            if mode == "synthetic_example"
            else "<p>Local organ identities are assisted hypotheses anchored in source views. Smooth leaf sheets and a shared stem "
            "provide an explicit surface on which visibility and neighbour effects can be calculated.</p>"
            "<p>The camera uses a fitted perspective gauge with assumed nominal rotation angles. Shape, scale, hidden lamina and "
            "organ correspondence retain those assumptions. Model dimensions are not physical measurements.</p>"
        ),
        INTERPRETATION=(
            "Example masks and identities are supplied procedural truth. This run has no real-image evaluation, anatomical "
            "annotation or learned segmentation stage."
            if mode == "synthetic_example"
            else "Whole-plant masks are automated observations. Real-image evaluation views have been reused during method development. "
            "Their consistency scores are exploratory same-acquisition diagnostics, not an untouched anatomical benchmark."
        ),
        GEOMETRY=geometry + _geometry_diagnostics(outputs),
        REMOVAL=removal,
        REPOSITION=reposition,
        RECOVERY=_recovery(outputs),
        ATLAS=_atlas(outputs),
        DATA=payload,
    )
    document = (WEB / "report.html").read_text(encoding="utf-8")
    for name, value in replacements.items():
        document = document.replace(f"__{name}__", value)
    encoded = document.encode("utf-8")
    target = _choose_target(workspace, target, encoded, manifest, output is not None)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded)
    return target
