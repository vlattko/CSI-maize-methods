"""Equal-effort camera-pair specificity challenge on frozen archive masks.

Wrong-pair pass rates are engineering null diagnostics, not anatomical accuracy.
No JPEG intensities, new labels, model training or cross-round identity are used.
"""

from common import workspace_relative

import os

for _key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_key] = "2"
from bootstrap import (
    BASE,
    ROOT,
    OUT,
    read_csv,
    write_csv,
    validate_rotation_records,
    method_source_hashes,
)
from multimodal_leaf_core import affine_align, mask_iou, sha
from channel_fusion_core import POLICY, match_instances, warp_to_rgb, interior, np, cv2
from run_channel_fusion import registration_with_empty_guard, load_prediction
from PIL import Image
from pathlib import Path
from functools import lru_cache
from threadpoolctl import threadpool_limits
import collections, contextlib, datetime, hashlib, json, sys, time, traceback

DEST = OUT / "leaf_dynamics/2026-10-24/jeb_v5/pair_controls"
FROZEN = OUT / "channel_fusion/2026-10-24"
PRED = OUT / "multimodal_leaf/2026-10-24/predictions"
VIEWS = list(range(0, 40, 4))
CONDITIONS = [
    "nominal",
    "same_acquisition_shift10",
    "same_acquisition_shift20",
    "different_genotype_same_round",
]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def relative(path):
    return workspace_relative(Path(path))


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def cross_assignments(visits):
    """Smallest cyclic label-only bijection with different genotypes per round."""
    mapping = {}
    for round_ in sorted({r["measurement"] for r in visits}):
        group = sorted(
            [r for r in visits if r["measurement"] == round_], key=lambda r: r["session"]
        )
        for shift in range(1, len(group)):
            if all(
                r["genotype"] != group[(i + shift) % len(group)]["genotype"]
                for i, r in enumerate(group)
            ):
                mapping.update(
                    {
                        r["session"]: group[(i + shift) % len(group)]["session"]
                        for i, r in enumerate(group)
                    }
                )
                break
        else:
            raise ValueError(
                "No label-only different-genotype cyclic bijection for round " + str(round_)
            )
    return mapping


def moment_registration(a, b):
    """Exact moment initialisation used inside the existing affine_align()."""
    if a.sum() < 10 or b.sum() < 10:
        return np.array([[1, 0, 0], [0, 1, 0]], np.float32), 0.0, "empty_or_tiny_foreground"

    def moments(x):
        y, z = np.where(x)
        return np.array([z.mean(), y.mean()]), np.array([max(z.std(), 1), max(y.std(), 1)])

    ca, sa = moments(a)
    cb, sb = moments(b)
    scale = np.clip(sb / sa, 0.6, 1.6)
    matrix = np.array(
        [[scale[0], 0, cb[0] - ca[0] * scale[0]], [0, scale[1], cb[1] - ca[1] * scale[1]]],
        np.float32,
    )
    moved = (
        cv2.warpAffine(
            a.astype(np.uint8), matrix, (b.shape[1], b.shape[0]), flags=cv2.INTER_NEAREST
        )
        > 0
    )
    return matrix, mask_iou(moved, b), "moment_initialisation"


def score_pair(a, b, rgb, noir, matrix, whole_iou):
    """Apply the frozen Hungarian, margin, geometry and interior-support rules."""
    warped_masks = [warp_to_rgb(m.astype(np.uint8), matrix, a.shape, True)[0] > 0 for _, m in noir]
    ious = np.array([[mask_iou(m, w) for w in warped_masks] for _, m in rgb]).reshape(
        len(rgb), len(noir)
    )
    proposals = match_instances(ious, 0.25, POLICY["assignment_margin_min"])
    # This field is exactly the original bilinear intensity interpolation support;
    # its values are independent of image intensities, so no JPEG is needed.
    _, field = warp_to_rgb(np.ones(b.shape, np.float32), matrix, a.shape)
    results = []
    used_r = set()
    used_n = set()
    for proposal in proposals:
        i, j = proposal["i"], proposal["j"]
        r, rm = rgb[i]
        n, nm = noir[j]
        rid, nid = int(r["detection_index"]), int(n["detection_index"])
        used_r.add(rid)
        used_n.add(nid)
        ri = interior(rm & a)
        ni = interior(nm & b)
        wn = warp_to_rgb(ni.astype(np.uint8), matrix, a.shape, True)[0] > 0
        common = ri & wn & field
        retained = float(common.sum() / max(1, min(ri.sum(), wn.sum())))
        reasons = []
        if whole_iou < POLICY["whole_iou_min"]:
            reasons.append("whole_mask_IoU_below_gate")
        if proposal["iou"] < POLICY["leaf_iou_min"]:
            reasons.append("leaf_IoU_below_gate")
        if not proposal["accepted"]:
            reasons.append("ambiguous_assignment")
        geometry_ok = not reasons
        if common.sum() < POLICY["min_common_interior_pixels"]:
            reasons.append("too_few_common_interior_pixels")
        if retained < POLICY["min_interior_overlap_fraction"]:
            reasons.append("low_common_interior_fraction")
        results.append(
            dict(
                rgb_detection_index=rid,
                noir_detection_index=nid,
                leaf_iou=proposal["iou"],
                assignment_row_margin=proposal["row_margin"],
                assignment_col_margin=proposal["col_margin"],
                geometry_candidate=geometry_ok,
                pixel_hypothesis=not reasons,
                common_interior_pixels=int(common.sum()),
                common_interior_fraction=retained,
                failure_reasons=reasons,
            )
        )
    return dict(
        proposals=results,
        proposal_count=len(results),
        geometry_candidates=sum(r["geometry_candidate"] for r in results),
        pixel_hypotheses=sum(r["pixel_hypothesis"] for r in results),
        unmatched_rgb_ids=[
            int(r["detection_index"]) for r, _ in rgb if int(r["detection_index"]) not in used_r
        ],
        unmatched_noir_ids=[
            int(r["detection_index"]) for r, _ in noir if int(r["detection_index"]) not in used_n
        ],
    )


@lru_cache(maxsize=120)
def load_whole(path):
    with Image.open(path) as image:
        return np.asarray(image.resize((512, 385), Image.Resampling.NEAREST)) > 0


@lru_cache(maxsize=4)
def frozen_predictions(session, camera):
    return load_prediction(session, camera)


def design(rows):
    visit_lookup = {}
    lookup = {}
    for row in rows:
        visit_lookup[row["session"]] = {
            key: row[key]
            for key in ["session", "measurement", "genotype", "replicate", "treatment"]
        }
        key = row["session"], row["camera"], int(row["view"])
        if key in lookup:
            raise ValueError("Duplicate camera source key")
        lookup[key] = row
    visits = list(visit_lookup.values())
    cross = cross_assignments(visits)
    records = []
    for session in sorted(visit_lookup):
        for view in VIEWS:
            for condition in CONDITIONS:
                destination = cross[session] if condition == CONDITIONS[3] else session
                nview = (
                    view
                    + (
                        10
                        if condition == CONDITIONS[1]
                        else 20
                        if condition == CONDITIONS[2]
                        else 0
                    )
                ) % 40
                records.append(
                    dict(
                        anchor_session=session,
                        anchor_view=view,
                        condition=condition,
                        noir_session=destination,
                        noir_view=nview,
                        source_round=visit_lookup[session]["measurement"],
                        rgb_genotype=visit_lookup[session]["genotype"],
                        noir_genotype=visit_lookup[destination]["genotype"],
                        rgb_replicate_label=visit_lookup[session]["replicate"],
                        noir_replicate_label=visit_lookup[destination]["replicate"],
                    )
                )
    return records, lookup, cross


def replay_nominal(frame_rows, proposal_rows):
    oldframes = {
        (r["session"], int(r["view"])): r
        for r in json.loads((FROZEN / "frame_registration.json").read_text(encoding="utf-8"))[
            "records"
        ]
    }
    oldtargets = collections.defaultdict(dict)
    for r in json.loads((FROZEN / "shared_targets.json").read_text(encoding="utf-8"))["records"]:
        if int(r["view"]) in VIEWS:
            oldtargets[r["session"], int(r["view"])][
                int(r["rgb_detection_index"]), int(r["noir_detection_index"])
            ] = r
    newtargets = collections.defaultdict(dict)
    for r in proposal_rows:
        if r["condition"] == "nominal" and r["method"] == "ECC_selected":
            newtargets[r["anchor_session"], r["anchor_view"]][
                r["rgb_detection_index"], r["noir_detection_index"]
            ] = r
    checks = []
    for row in frame_rows:
        if row["condition"] != "nominal" or row["method"] != "ECC_selected":
            continue
        key = row["anchor_session"], row["anchor_view"]
        old = oldframes[key]
        fields = []
        matrix_error = float(
            np.max(
                np.abs(
                    np.array(row["affine_source_to_target"])
                    - np.array(old["affine_source_to_target"])
                )
            )
        )
        iou_error = abs(row["whole_mask_iou"] - old["whole_mask_iou"])
        if matrix_error > 1e-8:
            fields.append("affine_matrix")
        if iou_error > 1e-12:
            fields.append("whole_mask_iou")
        for new, previous in [
            ("proposal_count", "proposals"),
            ("geometry_candidates", "supported_targets"),
            ("pixel_hypotheses", "pixel_hypotheses"),
        ]:
            if row[new] != old[previous]:
                fields.append(new)
        if row["registration_status"] != old["status"]:
            fields.append("registration_status")
        aa, bb = oldtargets[key], newtargets[key]
        if set(aa) != set(bb):
            fields.append("proposal_identity_set")
        for pair in set(aa) & set(bb):
            for name in [
                "geometry_candidate",
                "pixel_hypothesis",
                "common_interior_pixels",
                "failure_reasons",
            ]:
                if aa[pair][name] != bb[pair][name]:
                    fields.append("target_" + name)
            for name in [
                "leaf_iou",
                "assignment_row_margin",
                "assignment_col_margin",
                "common_interior_fraction",
            ]:
                if abs(aa[pair][name] - bb[pair][name]) > 1e-12:
                    fields.append("target_" + name)
        checks.append(
            dict(
                anchor_session=key[0],
                anchor_view=key[1],
                exact_within_tolerance=not fields,
                matrix_max_abs_error=matrix_error,
                whole_IoU_abs_error=iou_error,
                mismatched_fields=sorted(set(fields)),
            )
        )
    dump(
        DEST / "nominal_replay.json",
        dict(
            frames=len(checks),
            matching=sum(r["exact_within_tolerance"] for r in checks),
            discrepant=sum(not r["exact_within_tolerance"] for r in checks),
            records=checks,
            scope="Original frozen ECC-selected outputs replayed; differences are preserved rather than corrected by borrowing matrices.",
        ),
    )
    return checks


def summarise(frames):
    acquisitions = []
    totals = []
    for method in ["moment_only", "ECC_selected"]:
        for condition in CONDITIONS:
            group = [r for r in frames if r["method"] == method and r["condition"] == condition]
            for session in sorted({r["anchor_session"] for r in group}):
                rows = [r for r in group if r["anchor_session"] == session]
                acquisitions.append(
                    dict(
                        method=method,
                        condition=condition,
                        anchor_session=session,
                        frames=len(rows),
                        median_whole_mask_iou=float(np.median([r["whole_mask_iou"] for r in rows])),
                        frames_with_geometry_candidate=sum(
                            r["geometry_candidates"] > 0 for r in rows
                        ),
                        frames_with_pixel_hypothesis=sum(r["pixel_hypotheses"] > 0 for r in rows),
                        geometry_candidates=sum(r["geometry_candidates"] for r in rows),
                        pixel_hypotheses=sum(r["pixel_hypotheses"] for r in rows),
                        proposals=sum(r["proposal_count"] for r in rows),
                        geometry_frame_fraction=sum(r["geometry_candidates"] > 0 for r in rows)
                        / len(rows),
                        pixel_frame_fraction=sum(r["pixel_hypotheses"] > 0 for r in rows)
                        / len(rows),
                    )
                )
            acq = [r for r in acquisitions if r["method"] == method and r["condition"] == condition]
            totals.append(
                dict(
                    method=method,
                    condition=condition,
                    frames=len(group),
                    acquisitions=len(acq),
                    median_of_acquisition_median_whole_mask_iou=float(
                        np.median([r["median_whole_mask_iou"] for r in acq])
                    ),
                    frames_with_geometry_candidate=sum(r["geometry_candidates"] > 0 for r in group),
                    frames_with_pixel_hypothesis=sum(r["pixel_hypotheses"] > 0 for r in group),
                    geometry_candidates=sum(r["geometry_candidates"] for r in group),
                    pixel_hypotheses=sum(r["pixel_hypotheses"] for r in group),
                    proposals=sum(r["proposal_count"] for r in group),
                    ECC_selected_frames=sum(
                        r["registration_status"] == "affine_ECC" for r in group
                    ),
                )
            )
    write_csv(DEST / "acquisition_summary.csv", acquisitions)
    write_csv(DEST / "condition_summary.csv", totals)
    contrasts = []
    for method in ["moment_only", "ECC_selected"]:
        for condition in CONDITIONS[1:]:
            for session in sorted({r["anchor_session"] for r in frames}):
                nominal = next(
                    r
                    for r in acquisitions
                    if r["method"] == method
                    and r["condition"] == "nominal"
                    and r["anchor_session"] == session
                )
                control = next(
                    r
                    for r in acquisitions
                    if r["method"] == method
                    and r["condition"] == condition
                    and r["anchor_session"] == session
                )
                contrasts.append(
                    dict(
                        method=method,
                        condition=condition,
                        anchor_session=session,
                        control_minus_nominal_median_whole_IoU=control["median_whole_mask_iou"]
                        - nominal["median_whole_mask_iou"],
                        control_minus_nominal_geometry_frame_fraction=control[
                            "geometry_frame_fraction"
                        ]
                        - nominal["geometry_frame_fraction"],
                        control_minus_nominal_pixel_frame_fraction=control["pixel_frame_fraction"]
                        - nominal["pixel_frame_fraction"],
                    )
                )
    write_csv(DEST / "paired_acquisition_contrasts.csv", contrasts)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, field, ylabel in zip(
        axes,
        ["median_whole_mask_iou", "geometry_frame_fraction"],
        ["Whole-mask IoU: acquisition median", "Fraction of 10 anchors with a geometric candidate"],
    ):
        data = [
            [
                r[field]
                for r in acquisitions
                if r["method"] == method and r["condition"] == condition
            ]
            for condition in CONDITIONS
            for method in ["moment_only", "ECC_selected"]
        ]
        bp = ax.boxplot(
            data, positions=np.arange(8), widths=0.6, patch_artist=True, showfliers=True
        )
        for patch, color in zip(bp["boxes"], ["#a2a9a3", "#257465"] * 4):
            patch.set_facecolor(color)
        ax.set_xticks(
            np.arange(4) * 2 + 0.5, ["Nominal", "+10 views", "+20 views", "Different genotype"]
        )
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"Equal-effort registration: moment-only (grey) and ECC-selected (green)\n{len({r['anchor_session'] for r in frames})} paired acquisition summaries; no anatomical truth or biological p-values",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(DEST / "specificity_summary.png", dpi=170)
    fig.savefig(DEST / "specificity_summary.pdf")
    plt.close(fig)
    return acquisitions, totals


def main_analysis():
    started = time.time()
    cv2.setNumThreads(2)
    DEST.mkdir(parents=True, exist_ok=True)
    rows = read_csv(OUT / "sam_traits/view_traits.csv")
    acquisition_count = validate_rotation_records(rows)
    cases, lookup, cross = design(rows)
    assert (
        len(cases) == acquisition_count * len(VIEWS) * len(CONDITIONS)
        and len(cross) == acquisition_count
    )
    sources = [
        BASE / "v5_pair_controls.py",
        BASE / "v5_pair_controls_test.py",
        BASE / "multimodal_leaf_core.py",
        BASE / "channel_fusion_core.py",
        BASE / "run_channel_fusion.py",
        BASE / "bootstrap.py",
        OUT / "sam_traits/view_traits.csv",
        FROZEN / "frame_registration.json",
        FROZEN / "shared_targets.json",
        BASE / "config.json",
    ]
    sources += sorted(PRED.glob("*.json")) + sorted(PRED.glob("*.npz"))
    maskpaths = set()
    for case in cases:
        maskpaths.add(ROOT / lookup[case["anchor_session"], "rgb", case["anchor_view"]]["mask"])
        maskpaths.add(ROOT / lookup[case["noir_session"], "noir", case["noir_view"]]["mask"])
    sources += sorted(maskpaths)
    input_hashes = method_source_hashes("v5_pair_controls.py")
    input_hashes.update({relative(p): sha(p) for p in sources})
    configuration = dict(
        created_utc=utc(),
        views=VIEWS,
        conditions=CONDITIONS,
        cases=cases,
        cross_assignment=cross,
        cross_rule="Within round, smallest positive cyclic shift in sorted acquisition order having a different genotype for every source; label-only bijection.",
        registration="Exact existing affine_align and empty guard for every pair; independent moment initialisation, ECC 80 iterations/1e-5, retain only higher source-to-target whole-mask IoU.",
        methods=["moment_only", "ECC_selected"],
        policy=dict(POLICY, proposal_iou_min=0.25),
        input_sha256=input_hashes,
        cpu_threads_max=2,
        no_GPU=True,
        runtime_versions=dict(python=sys.version, numpy=np.__version__, opencv=cv2.__version__),
        meaning="Engineering specificity challenge; deliberately mismatched capture pass rates are not anatomical false-positive accuracy.",
    )
    signed = {k: v for k, v in configuration.items() if k != "created_utc"}
    configuration["signature"] = hashlib.sha256(
        json.dumps(signed, sort_keys=True).encode()
    ).hexdigest()
    configpath = DEST / "configuration_before_run.json"
    if configpath.exists():
        previous = json.loads(configpath.read_text(encoding="utf-8"))
        if previous["signature"] != configuration["signature"]:
            raise ValueError(
                "Frozen run configuration differs; preserve it and use an explicitly new experiment."
            )
    else:
        dump(configpath, configuration)
    write_csv(DEST / "pair_design.csv", cases)
    print(
        "CONFIG frozen before computation",
        configuration["signature"],
        "cases",
        len(cases),
        "source masks",
        len(maskpaths),
        flush=True,
    )
    frames = []
    proposals = []
    observations = []
    for number, case in enumerate(cases):
        session = case["anchor_session"]
        view = case["anchor_view"]
        ns = case["noir_session"]
        nv = case["noir_view"]
        a = load_whole(str(ROOT / lookup[session, "rgb", view]["mask"]))
        b = load_whole(str(ROOT / lookup[ns, "noir", nv]["mask"]))
        rgb = frozen_predictions(session, "rgb").get(view, [])
        noir = frozen_predictions(ns, "noir").get(nv, [])
        moment = moment_registration(a, b)
        selected = registration_with_empty_guard(a, b)
        for method, (matrix, whole_iou, status) in [
            ("moment_only", moment),
            ("ECC_selected", selected),
        ]:
            scored = score_pair(a, b, rgb, noir, matrix, whole_iou)
            common = dict(case, method=method)
            frame = dict(
                common,
                whole_mask_iou=whole_iou,
                registration_status=status,
                affine_source_to_target=matrix.tolist(),
                rgb_detection_ids=[int(r["detection_index"]) for r, _ in rgb],
                noir_detection_ids=[int(r["detection_index"]) for r, _ in noir],
                missing_rgb_detections=not rgb,
                missing_noir_detections=not noir,
                **{k: v for k, v in scored.items() if k != "proposals"},
            )
            frames.append(frame)
            state = {}
            for record in scored["proposals"]:
                target = f"{Path(session).name}__v{view:03}__{case['condition']}__{method}__{Path(ns).name}__nv{nv:03}__r{record['rgb_detection_index']}__n{record['noir_detection_index']}"
                proposals.append(dict(common, target_id=target, **record))
                label = (
                    "supported_candidate" if record["geometry_candidate"] else "rejected_proposal"
                )
                state["rgb", record["rgb_detection_index"]] = (target, label)
                state["noir", record["noir_detection_index"]] = (target, label)
            for camera, items in [("rgb", rgb), ("noir", noir)]:
                for record, _ in items:
                    target, label = state.get(
                        (camera, int(record["detection_index"])), ("", "unmatched")
                    )
                    observations.append(
                        dict(
                            common,
                            camera=camera,
                            detection_index=int(record["detection_index"]),
                            source_session=session if camera == "rgb" else ns,
                            source_view=view if camera == "rgb" else nv,
                            source=record["source"],
                            checkpoint_sha256=record["checkpoint_sha256"],
                            target_id=target,
                            state=label,
                        )
                    )
        if (number + 1) % 40 == 0:
            print(
                "COMPUTED",
                number + 1,
                "/",
                len(cases),
                "seconds",
                round(time.time() - started, 1),
                flush=True,
            )
    dump(DEST / "frame_records.json", dict(signature=configuration["signature"], records=frames))
    write_csv(
        DEST / "frame_records.csv",
        [
            {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()}
            for r in frames
        ],
    )
    dump(DEST / "proposal_records.json", dict(records=proposals))
    write_csv(DEST / "proposal_records.csv", proposals)
    write_csv(DEST / "observation_states.csv", observations)
    replay = replay_nominal(frames, proposals)
    acquisitions, totals = summarise(frames)
    findings = []
    for method in ["moment_only", "ECC_selected"]:
        for r in [x for x in totals if x["method"] == method]:
            findings.append(
                f"{method}, {r['condition']}: {r['frames_with_geometry_candidate']}/{len(cross) * len(VIEWS)} anchors had a geometric candidate and {r['frames_with_pixel_hypothesis']}/{len(cross) * len(VIEWS)} had common-interior support; acquisition-balanced median whole-mask IoU {r['median_of_acquisition_median_whole_mask_iou']:.4f}."
            )
    findings.append(
        f"The nominal ECC replay matched {sum(r['exact_within_tolerance'] for r in replay)}/{len(cross) * len(VIEWS)} frozen frames within declared tolerances; every discrepancy is retained."
    )
    summary = dict(
        status="completed_exploratory_specificity_challenge",
        question="Can the same registration and gates qualify deliberately mismatched capture pairs?",
        methods={
            k: configuration[k]
            for k in [
                "views",
                "conditions",
                "cross_rule",
                "registration",
                "methods",
                "policy",
                "meaning",
            ]
        },
        metrics=dict(
            cases=len(cases),
            scored_method_cases=len(frames),
            acquisitions=len(cross),
            condition_summaries=totals,
            nominal_replay_matches=sum(r["exact_within_tolerance"] for r in replay),
            nominal_replay_discrepancies=sum(not r["exact_within_tolerance"] for r in replay),
            source_masks=len(maskpaths),
            source_files=len(sources),
            runtime_seconds=time.time() - started,
            observation_states=dict(collections.Counter(r["state"] for r in observations)),
        ),
        findings=findings,
        limitations=[
            "Engineering gates are not calibrated anatomical accuracy. Negative-pair pass rates are not biological false-positive rates.",
            "Nominal pairs are unreviewed; rejecting wrong pairs does not prove their anatomical correctness.",
            "Camera-local detections share weak-teacher origins. Technical views and reused acquisitions are dependent.",
            "The cross-acquisition null uses one deterministic different-genotype bijection, not every possible mismatched pair.",
            "The frozen registration helper returns moment fallback after either an ECC error or no IoU improvement; these reasons are not separately instrumented.",
            "No radiometric, physiological, temporal-identity or hidden-geometry validation is performed.",
        ],
        input_sha256=configuration["input_sha256"],
        signature=configuration["signature"],
        artifacts=sorted(
            set(
                [relative(p) for p in DEST.iterdir() if p.is_file()]
                + [relative(DEST / "summary.json"), relative(DEST / "findings.txt")]
            )
        ),
    )
    dump(DEST / "summary.json", summary)
    (DEST / "findings.txt").write_text("\n\n".join(findings) + "\n", encoding="utf-8")
    print(json.dumps(dict(metrics=summary["metrics"], findings=findings), indent=2), flush=True)


class Tee:
    def __init__(self, terminal, handle):
        self.terminal = terminal
        self.handle = handle

    def write(self, text):
        self.terminal.write(text)
        self.handle.write(text)
        self.handle.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.handle.flush()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with (
        (DEST / "execution.log").open("a", encoding="utf-8") as handle,
        contextlib.redirect_stdout(Tee(sys.stdout, handle)),
        contextlib.redirect_stderr(Tee(sys.stderr, handle)),
        threadpool_limits(limits=2),
    ):
        print(
            "START UTC",
            utc(),
            "COMMAND",
            sys.executable,
            " ".join(sys.argv),
            "CPU <=2, no GPU",
            flush=True,
        )
        try:
            main_analysis()
            print("END UTC", utc(), "STATUS complete exploratory engineering null", flush=True)
        except Exception:
            traceback.print_exc()
            print("END UTC", utc(), "STATUS failed", flush=True)
            raise


if __name__ == "__main__":
    main()
