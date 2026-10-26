"""Independent saved-output binding/gate checks; no registration is refitted."""

from common import workspace_relative

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
from bootstrap import BASE, ROOT, OUT
from pathlib import Path
import collections, csv, datetime, hashlib, json
import numpy as np

DEST = OUT / "leaf_dynamics/2026-10-24/jeb_v5/pair_controls"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    return json.loads((DEST / name).read_text(encoding="utf-8"))


def main():
    config = load("configuration_before_run.json")
    frames = load("frame_records.json")["records"]
    proposals = load("proposal_records.json")["records"]
    replay = load("nominal_replay.json")
    read = lambda name: list(csv.DictReader((DEST / name).open(encoding="utf-8-sig", newline="")))
    observations = read("observation_states.csv")
    acquisitions = read("acquisition_summary.csv")
    key = lambda r: (r["anchor_session"], int(r["anchor_view"]), r["condition"], r["method"])
    lookup = {key(r): r for r in frames}
    checks = {}
    acquisitions_expected = len(config["cross_assignment"])
    anchors_expected = acquisitions_expected * len(config["views"])
    cases_expected = anchors_expected * len(config["conditions"])
    methods_expected = len(config["methods"])
    checks["all_input_hashes_match"] = bool(config["input_sha256"]) and all(
        digest(ROOT / p) == h for p, h in config["input_sha256"].items()
    )
    checks["complete_design_and_unique_method_records"] = (
        len(config["cases"]) == cases_expected
        and len(frames) == len(lookup) == cases_expected * methods_expected
    )
    checks["each_method_condition_has_all_anchors"] = set(
        collections.Counter((r["method"], r["condition"]) for r in frames).values()
    ) == {anchors_expected}
    with (OUT / "sam_traits/view_traits.csv").open(encoding="utf-8-sig", newline="") as handle:
        source_round = {r["session"]: r["measurement"] for r in csv.DictReader(handle)}
    checks["all_cross_pairs_different_genotype_same_round"] = all(
        r["rgb_genotype"] != r["noir_genotype"]
        and source_round[r["anchor_session"]]
        == source_round[r["noir_session"]]
        == r["source_round"]
        for r in config["cases"]
        if r["condition"] == "different_genotype_same_round"
    )
    mapping = config["cross_assignment"]
    checks["cross_assignment_bijective"] = (
        set(mapping) == set(mapping.values()) and len(mapping) == acquisitions_expected
    )
    checks["all_exact40view_index_rules"] = all(
        int(r["noir_view"])
        == (
            int(r["anchor_view"])
            + (
                {"same_acquisition_shift10": 10, "same_acquisition_shift20": 20}.get(
                    r["condition"], 0
                )
            )
        )
        % 40
        for r in frames
    )
    checks["ECC_retention_never_decreases_whole_IoU"] = all(
        r["whole_mask_iou"] + 1e-15
        >= lookup[(r["anchor_session"], r["anchor_view"], r["condition"], "moment_only")][
            "whole_mask_iou"
        ]
        for r in frames
        if r["method"] == "ECC_selected"
    )
    grouped = collections.defaultdict(list)
    for r in proposals:
        grouped[key(r)].append(r)
    bindings = True
    gates = True
    uniqueness = True
    negative_passes = []
    for k, f in lookup.items():
        pp = grouped[k]
        pairs = [(r["rgb_detection_index"], r["noir_detection_index"]) for r in pp]
        uniqueness &= len(set(i for i, j in pairs)) == len(pp) == len(set(j for i, j in pairs))
        bindings &= all(
            i in f["rgb_detection_ids"] and j in f["noir_detection_ids"] for i, j in pairs
        )
        bindings &= (
            len(pp) == f["proposal_count"]
            and sum(r["geometry_candidate"] for r in pp) == f["geometry_candidates"]
            and sum(r["pixel_hypothesis"] for r in pp) == f["pixel_hypotheses"]
        )
        bindings &= set(f["rgb_detection_ids"]) - set(i for i, j in pairs) == set(
            f["unmatched_rgb_ids"]
        )
        bindings &= set(f["noir_detection_ids"]) - set(j for i, j in pairs) == set(
            f["unmatched_noir_ids"]
        )
        for r in pp:
            geometry = (
                f["whole_mask_iou"] >= 0.65
                and r["leaf_iou"] >= 0.50
                and r["assignment_row_margin"] >= 0.10
                and r["assignment_col_margin"] >= 0.10
            )
            pixel = (
                geometry
                and r["common_interior_pixels"] >= 40
                and r["common_interior_fraction"] >= 0.35
            )
            gates &= r["geometry_candidate"] == geometry and r["pixel_hypothesis"] == pixel
            if r["condition"] != "nominal" and geometry:
                negative_passes.append(r)
    checks["all_proposals_bound_and_counts_reconstructed"] = bool(bindings)
    checks["every_proposal_one_to_one"] = bool(uniqueness)
    checks["independent_boolean_gate_replay"] = bool(gates)
    observed = collections.Counter(
        (
            r["anchor_session"],
            int(r["anchor_view"]),
            r["condition"],
            r["method"],
            r["camera"],
            int(r["detection_index"]),
        )
        for r in observations
    )
    expected = {
        (f["anchor_session"], f["anchor_view"], f["condition"], f["method"], c, i)
        for f in frames
        for c in ["rgb", "noir"]
        for i in f[c + "_detection_ids"]
    }
    checks["all_camera_observations_once_and_unmatched_retained"] = set(
        observed
    ) == expected and set(observed.values()) == {1}
    checks["paired_acquisition_summaries"] = len(acquisitions) == acquisitions_expected * len(
        config["conditions"]
    ) * methods_expected and all(int(r["frames"]) == len(config["views"]) for r in acquisitions)
    checks["all_nominal_frozen_frames_replay"] = (
        replay["frames"] == replay["matching"] == anchors_expected and replay["discrepant"] == 0
    )
    determinants = {}
    for method in ["moment_only", "ECC_selected"]:
        for condition in config["conditions"]:
            values = [
                float(np.linalg.det(np.array(r["affine_source_to_target"])[:, :2]))
                for r in frames
                if r["method"] == method and r["condition"] == condition
            ]
            determinants[method + "|" + condition] = dict(
                min=min(values), max=max(values), nonpositive=sum(x <= 0 for x in values)
            )
    result = dict(
        status="passed" if all(checks.values()) else "failed",
        verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checks=checks,
        negative_passing_proposals=negative_passes,
        transform_determinant_diagnostics=determinants,
        figure_visual_review="Not performed by this numerical verifier; inspect generated figures separately.",
        verification_code_sha256=digest(__file__),
    )
    (DEST / "verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = load("summary.json")
    summary["artifacts"] = sorted(
        set(
            summary["artifacts"]
            + [
                workspace_relative(DEST / "verification.json").replace("\\", "/"),
                workspace_relative(DEST / "artifact_manifest.json").replace("\\", "/"),
            ]
        )
    )
    (DEST / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        str(p.relative_to(DEST)).replace("\\", "/"): {
            "sha256": digest(p),
            "bytes": p.stat().st_size,
        }
        for p in DEST.iterdir()
        if p.is_file() and p.name != "artifact_manifest.json"
    }
    (DEST / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(
                status=result["status"],
                checks=checks,
                negative_proposals=len(negative_passes),
                determinants=determinants,
            ),
            indent=2,
        )
    )
    assert all(checks.values()), checks


if __name__ == "__main__":
    main()
