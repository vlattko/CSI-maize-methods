"""E12: assemble source-linked challenge cases, not invented reference labels."""

from common import workspace_relative

from bootstrap import BASE, ROOT, OUT, read_csv, write_csv
from experiment_lab import RUN, utc, sha256_file, append_event
from collections import Counter
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import time


def validate_case(case):
    if (
        case.get("is_ground_truth")
        and case.get("annotation_status") != "independent_manual_reference"
    ):
        raise ValueError("An automatic mask is not independent ground truth")
    if case.get("split") == "independent_test":
        raise ValueError("Reused development cases cannot claim an untouched test split")
    return case


def source_case(case_id, domain, path, root=ROOT):
    return {
        "case_id": case_id,
        "domain": domain,
        "source": Path(path).relative_to(root).as_posix(),
        "source_sha256": sha256_file(path),
        "annotation_status": "automatic_prediction",
        "is_ground_truth": False,
        "split": "reused_development_challenge",
    }


def preview(row, path):
    from traits import load_image

    image, _ = load_image(ROOT / row["source"])
    mask = np.array(Image.open(ROOT / row["mask"])) > 0
    if image.shape[:2] != mask.shape:
        raise ValueError("Mask coordinate mismatch")
    overlay = image.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([65, 225, 155])).astype(np.uint8)
    canvas = Image.new("RGB", (2048, 820), "#142b25")
    canvas.paste(Image.fromarray(image), (0, 51))
    canvas.paste(Image.fromarray(overlay), (1024, 51))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (14, 10),
        row["session"].split("/")[-1]
        + " / "
        + row["camera"]
        + " view "
        + str(row["view"])
        + " | automatic mask, not reference annotation",
        fill="white",
    )
    draw.text(
        (14, 29),
        "QC: "
        + str(row["qc_issues"] or "pass")
        + " | clipping and thin boundaries require independent review",
        fill="#e5c78f",
    )
    canvas.thumbnail((1536, 650))
    canvas.save(path, quality=88)


def main():
    started = utc()
    clock = time.time()
    folder = RUN / "e12_benchmark"
    folder.mkdir(parents=True, exist_ok=True)
    append_event(
        RUN / "events.jsonl",
        "E12",
        "started",
        {
            "command": "additive_benchmark.py",
            "selection": "all automatic QC failures plus one deterministic minimum-area case per genotype/round/camera stratum",
        },
    )
    rows = read_csv(OUT / "sam_traits/view_traits.csv")
    fail = [r for r in rows if r["qc_pass"] != "True"]
    strata = {}
    for r in rows:
        if r["qc_pass"] != "True":
            continue
        key = (r["genotype"], r["measurement"], r["camera"])
        if key not in strata or float(r["silhouette_area_px"]) < float(
            strata[key]["silhouette_area_px"]
        ):
            strata[key] = r
    selected = {r["source"]: r for r in [*fail, *strata.values()]}
    cases = []
    inputs = {}
    for index, row in enumerate(selected.values()):
        case = source_case(
            "segmentation_" + str(index).zfill(3), "segmentation_qc", ROOT / row["source"]
        )
        case.update(
            {
                k: row[k]
                for k in [
                    "session",
                    "genotype",
                    "measurement",
                    "camera",
                    "view",
                    "qc_pass",
                    "qc_issues",
                    "mask",
                ]
            }
        )
        case["mask_sha256"] = sha256_file(ROOT / row["mask"])
        case["selection_reason"] = (
            "QC failure"
            if row["qc_pass"] != "True"
            else "Minimum-area view in available genotype/round/camera stratum"
        )
        inputs[case["source"]] = case["source_sha256"]
        inputs[case["mask"]] = case["mask_sha256"]
        cases.append(validate_case(case))
    video_path = OUT / "leaf_instance_tracking/summary.json"
    video = json.loads(video_path.read_text(encoding="utf-8"))
    # Keep every reported pair-duplication frame, using actual saved diagnostic records.
    pair_candidates = list((OUT / "leaf_instance_tracking").glob("*.csv"))
    duplicate_rows = []
    for path in pair_candidates:
        inputs[workspace_relative(path)] = sha256_file(path)
        for r in read_csv(path):
            a = r.get("object_a", r.get("id_a"))
            b = r.get("object_b", r.get("id_b"))
            value = r.get("iou", r.get("mask_iou", 0))
            if a is not None and b is not None and float(value) > 0.5:
                duplicate_rows.append({"diagnostic_file": workspace_relative(path), **r})
    observations_path = OUT / "organ_observations/capture_session_C2_5-1/observations.json"
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    for record in observations["views"]:
        uncertain = [
            r for r in record["leaves"] if r["association_status"] != "provisionally_consistent"
        ]
        if not uncertain:
            continue
        source = next(r for r in observations["inputs"] if r["view"] == record["view"])
        case = source_case(
            "organ_association_" + str(record["view"]).zfill(2),
            "organ_association",
            ROOT / source["source"],
        )
        case.update(
            {
                "view": record["view"],
                "uncertain_leaf_ids": [r["leaf_id"] for r in uncertain],
                "uncertainty_flags": {r["leaf_id"]: r["uncertainty_flags"] for r in uncertain},
                "annotation_status": "assisted_hypothesis",
            }
        )
        cases.append(validate_case(case))
        inputs[case["source"]] = case["source_sha256"]
    ambiguity_path = RUN / "e08_ambiguity/summary.json"
    alternative = {"status": "not_yet_assembled"}
    if ambiguity_path.exists():
        ambiguity = json.loads(ambiguity_path.read_text(encoding="utf-8"))
        alternative = {
            "status": ambiguity["status"],
            "summary": workspace_relative(ambiguity_path),
            "summary_sha256": sha256_file(ambiguity_path),
            "metrics": ambiguity["metrics"],
            "artifacts": ambiguity["artifacts"],
        }
        inputs[workspace_relative(ambiguity_path)] = sha256_file(ambiguity_path)
    independent_reference_template = [
        {
            "case_id": r["case_id"],
            "source": r["source"],
            "reference_mask": "",
            "leaf_id_if_verified": "",
            "annotator": "",
            "annotation_date": "",
            "adjudication_status": "not_annotated",
            "independent_anatomical_measurement": "",
            "units": "",
        }
        for r in cases
    ]
    write_csv(folder / "reference_annotation_template.csv", independent_reference_template)
    write_csv(
        folder / "challenge_cases.csv",
        [
            {
                **r,
                "uncertain_leaf_ids": json.dumps(r.get("uncertain_leaf_ids", [])),
                "uncertainty_flags": json.dumps(r.get("uncertainty_flags", {})),
            }
            for r in cases
        ],
    )
    (folder / "challenge_cases.json").write_text(json.dumps(cases, indent=2))
    (folder / "tracking_failure.json").write_text(
        json.dumps(
            {
                "recorded_duplicate_frame_count": video["pair_frame_iou_above_0_5"],
                "diagnostic_rows": duplicate_rows,
                "loading_info": video["loading_info"],
                "object_id_convention": "SAM video prompt IDs, different from fitted leaf IDs",
                "independent_reference": False,
            },
            indent=2,
        )
    )
    (folder / "alternative_geometry_manifest.json").write_text(json.dumps(alternative, indent=2))
    preview_rows = sorted(
        selected.values(),
        key=lambda r: (r["qc_pass"] == "True", -float(r.get("dark_pixel_fraction") or 0)),
    )[:4]
    preview_rows += [
        min(strata.values(), key=lambda r: float(r["silhouette_area_px"])),
        max(strata.values(), key=lambda r: float(r["silhouette_area_px"])),
    ]
    for i, row in enumerate(preview_rows):
        preview(row, folder / f"challenge_preview_{i:02d}.jpg")
    evaluation = {
        "status": "benchmark_draft_no_independent_reference",
        "segmentation_metrics_when_annotated": [
            "IoU",
            "Dice",
            "boundary F score with declared tolerance",
            "tip recall",
            "trait error against independent measurement",
        ],
        "organ_geometry_metrics_when_annotated": [
            "held-out landmark reprojection",
            "leaf-length and width error",
            "identity consistency",
            "coverage of independently measured traits by declared feasible ranges",
        ],
        "partition_rule": "Keep views and repeated physical individuals together; use new reserved plants for untouched evaluation. Existing challenge cases were reused in development.",
        "public_external_resources": [
            {
                "name": "Pheno4D",
                "url": "https://www.ipb.uni-bonn.de/data/pheno4d/index.html",
                "status": "reference catalogued; not downloaded or evaluated in this run",
            },
            {
                "name": "MaizeField3D",
                "url": "https://huggingface.co/datasets/BGLab/MaizeField3D",
                "status": "reference catalogued; not downloaded or evaluated in this run",
            },
        ],
    }
    (folder / "evaluation_protocol.json").write_text(json.dumps(evaluation, indent=2))
    inputs.update(
        {
            workspace_relative(p): sha256_file(p)
            for p in [
                BASE / "additive_benchmark.py",
                OUT / "sam_traits/view_traits.csv",
                video_path,
                observations_path,
            ]
        }
    )
    (folder / "BENCHMARK_README.txt").write_text(
        "This is a source-linked development challenge collection, not an independently annotated accuracy benchmark. Automatic QC is not accuracy. Automatic masks and assisted leaf associations are retained as predictions. Fill the annotation template with reviewed manual masks and verified anatomy, then reserve new physical plants for independent evaluation. Native files remain outside this draft package. External geometry resources are catalogued, not evaluated.\n"
    )
    metrics = {
        "challenge_cases": len(cases),
        "segmentation_cases": len(selected),
        "automatic_qc_failure_cases": len(fail),
        "stratified_minimum_area_cases": len(strata),
        "organ_association_cases": sum(r["domain"] == "organ_association" for r in cases),
        "tracking_duplicate_frame_count": video["pair_frame_iou_above_0_5"],
        "independent_reference_annotations": 0,
        "external_datasets_evaluated": 0,
    }
    summary = {
        "id": "E12",
        "title": "Ambiguity and quality-control challenge benchmark",
        "status": "partial_requires_new_data",
        "question": "Can we release source-inspectable failure and ambiguity cases with defensible future scoring rules?",
        "started_utc": started,
        "finished_utc": utc(),
        "runtime_seconds": time.time() - clock,
        "methods": {
            "case_selection": "All automatic QC failures plus minimum-area case in every available genotype/round/camera stratum; uncertain organ paths retained",
            "reference_status": "No independent anatomical or mask reference created by this automatic assembly",
        },
        "metrics": metrics,
        "findings": [
            f"Assembled {len(cases)} source-linked challenge cases, including all {len(fail)} automatic QC failures.",
            f"Retained the genuine SAM video failure with {video['pair_frame_iou_above_0_5']} duplicated-mask frames; no successful five-leaf tracking claim.",
            "Created blank reference annotation templates and evaluation rules; no independent accuracy scores can yet be reported.",
        ],
        "limitations": [
            "Cases were selected from repeatedly used development images",
            "Automatic masks, QC scores and assisted IDs are not independent reference labels",
            "External datasets were catalogued but not downloaded or evaluated",
            "This assembly is not a Zenodo publication or completed Scientific Data descriptor",
        ],
        "next_requirements": [
            "Independent stratified mask/organ annotations and adjudication",
            "Verified physical identities, source protocols and reuse rights",
            "New reserved plants and independent anatomical measurements for an untouched test set",
        ],
        "input_sha256": inputs,
        "artifacts": [
            workspace_relative(p)
            for p in sorted(folder.iterdir())
            if p.is_file() and p.name != "summary.json"
        ],
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    append_event(
        RUN / "events.jsonl",
        "E12",
        "finished",
        {"status": summary["status"], "summary_sha256": sha256_file(folder / "summary.json")},
    )
    print(
        json.dumps(
            {"id": "E12", "metrics": metrics, "runtime_seconds": summary["runtime_seconds"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
