"""Extract exploratory visible traits from SAM masks with explicit automated QC.

Passing QC is not human-annotated segmentation validation. All views are retained
in the audit table; session features require 80 passing views to avoid silently
mixing incomplete rotations. Skeleton measures remain unvalidated proxies.
"""

from common import workspace_relative

from bootstrap import BASE, ROOT, OUT, read_csv, write_csv, model_directory
from sam_segmentation import MODEL_ID, REVISION, make_prompts, overlay
from traits import load_image, segment, measure_mask, annotate
from concurrent.futures import ThreadPoolExecutor
import argparse, json, time, hashlib
import numpy as np
import cv2
import torch
from PIL import Image


def quality_checks(image, seed, mask, score):
    area = int(mask.sum())
    inter = int((seed & mask).sum())
    coverage = inter / max(1, int(seed.sum()))
    ratio = area / max(1, int(seed.sum()))
    dark = float(np.mean(image[mask].max(axis=1) < 25)) if area else 1.0
    issues = []
    if score < 0.85:
        issues.append("low_model_score")
    if not 0.0005 < mask.mean() < 0.15:
        issues.append("implausible_area_fraction")
    if coverage < 0.85:
        issues.append("seed_not_covered")
    if not 0.6 < ratio < 2.5:
        issues.append("large_seed_disagreement")
    if dark > 0.12:
        issues.append("dark_background_leakage")
    return {
        "qc_pass": not issues,
        "qc_issues": ";".join(issues),
        "sam_score": score,
        "seed_coverage": coverage,
        "mask_seed_area_ratio": ratio,
        "dark_pixel_fraction": dark,
    }


def main():
    from transformers import Sam2Model, Sam2Processor

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    folder = OUT / "sam_traits"
    folder.mkdir(exist_ok=True)
    for name in ["masks", "annotations", "crops"]:
        (folder / name).mkdir(exist_ok=True)
    torch.set_num_threads(4)
    model, loading = Sam2Model.from_pretrained(
        model_directory("sam2.1-hiera-small"), output_loading_info=True
    )
    if any(
        loading[k] for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
    ):
        raise ValueError(str(loading))
    model = model.to("cuda").eval()
    processor = Sam2Processor.from_pretrained(model_directory("sam2.1-hiera-small"))
    sessions = [r for r in read_csv(OUT / "sessions.csv") if r["eligible"] == "True"]
    if args.limit:
        sessions = sessions[: args.limit]
    all_rows = []
    aggregates = []
    started = time.time()
    for number, record in enumerate(sessions, 1):
        tasks = [
            (cam, path)
            for cam in ["rgb", "noir"]
            for path in sorted((ROOT / record["session"] / cam).glob("*.jpg"))
        ]
        session_rows = []
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start : start + args.batch_size]
            ready = []
            with ThreadPoolExecutor(max_workers=4) as pool:
                images = list(pool.map(lambda task: load_image(task[1]), batch))
            for (camera, path), (image, native) in zip(batch, images):
                try:
                    seed = segment(image, camera)
                    box, points, labels = make_prompts(seed, image)
                    if seed.mean() > 0.15:
                        raise ValueError("Seed occupies >15% of frame")
                    ready.append((camera, path, image, native, seed, box))
                except Exception as error:
                    session_rows.append(
                        {
                            "session": record["session"],
                            "camera": camera,
                            "view": int(path.stem.rsplit("_", 1)[1]),
                            "qc_pass": False,
                            "qc_issues": "seed_error: " + str(error),
                        }
                    )
            if not ready:
                continue
            inputs = processor(
                images=[Image.fromarray(r[2]) for r in ready],
                input_boxes=[[r[5]] for r in ready],
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                output = model(**inputs, multimask_output=False)
            masks = processor.post_process_masks(output.pred_masks.cpu(), inputs["original_sizes"])
            scores = output.iou_scores.detach().cpu().numpy().reshape(-1)
            for item, tensor, score in zip(ready, masks, scores):
                camera, path, image, native, seed, box = item
                mask = tensor[0, 0].numpy().astype(bool)
                idx = int(path.stem.rsplit("_", 1)[1])
                stem = path.parent.parent.name + "__" + camera + "__" + str(idx).zfill(3)
                mask_path = folder / "masks" / (stem + ".png")
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                qc = quality_checks(image, seed, mask, float(score))
                row = {
                    **{
                        k: record[k]
                        for k in ["session", "treatment", "measurement", "genotype", "replicate"]
                    },
                    "camera": camera,
                    "view": idx,
                    "source": workspace_relative(path),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "mask": workspace_relative(mask_path),
                    "box_xyxy": json.dumps(box),
                    "processing_width": image.shape[1],
                    "processing_height": image.shape[0],
                    "native_width": native[0],
                    "native_height": native[1],
                    **qc,
                }
                if mask.sum() >= 10:
                    values, skeleton = measure_mask(image, mask, camera)
                    row.update(values)
                    if idx % 10 == 0:
                        label = (
                            stem
                            + " | "
                            + (
                                "AUTOMATED QC PASS"
                                if qc["qc_pass"]
                                else "REVIEW: " + qc["qc_issues"]
                            )
                        )
                        annotate(image, mask, skeleton, values, label).save(
                            folder / "annotations" / (stem + ".jpg"), quality=90
                        )
                    if idx % 5 == 0:
                        ys, xs = np.where(mask)
                        x0 = max(0, int(xs.min()) - 20)
                        x1 = min(image.shape[1], int(xs.max()) + 21)
                        y0 = max(0, int(ys.min()) - 20)
                        y1 = min(image.shape[0], int(ys.max()) + 21)
                        crop = image[y0:y1, x0:x1].copy()
                        crop[~mask[y0:y1, x0:x1]] = [124, 116, 104]
                        Image.fromarray(crop).save(folder / "crops" / (stem + ".png"))
                session_rows.append(row)
        all_rows.extend(session_rows)
        good = [r for r in session_rows if r["qc_pass"]]
        if len(good) == 80:
            entry = {
                k: record[k]
                for k in ["session", "treatment", "measurement", "genotype", "replicate"]
            }
            entry["valid_views"] = 80
            for camera in ["rgb", "noir"]:
                subset = [r for r in good if r["camera"] == camera]
                for key in measure_mask(
                    np.ones((4, 4, 3), np.uint8) * 50, np.ones((4, 4), bool), camera
                )[0]:
                    vals = np.array([r[key] for r in subset], float)
                    for stat, value in [
                        ("median", np.median(vals)),
                        ("max", vals.max()),
                        ("iqr", np.percentile(vals, 75) - np.percentile(vals, 25)),
                    ]:
                        entry[camera + "__" + key + "__" + stat] = float(value)
            aggregates.append(entry)
        write_csv(folder / "view_traits.csv", all_rows)
        write_csv(
            folder / "session_traits.csv",
            aggregates,
            fields=list(aggregates[0]) if aggregates else ["session"],
        )
        print(
            f"{number}/{len(sessions)} {record['session']}: {len(good)}/80 QC pass, {round(time.time() - started, 1)} seconds",
            flush=True,
        )
    summary = {
        "model": MODEL_ID,
        "revision": REVISION,
        "mode": "bounding_box_from_conservative_colour_seed",
        "loading_info": loading,
        "sessions_processed": len(sessions),
        "complete_qc_sessions": len(aggregates),
        "views": len(all_rows),
        "views_qc_pass": sum(r["qc_pass"] for r in all_rows),
        "manual_annotation_validation": False,
        "status": "exploratory_image_traits",
        "processing_max_dimension": 1024,
        "limitations": [
            "Automated QC is not segmentation accuracy",
            "Skeleton endpoints are not organ counts",
            "Pixel dimensions have no metric scale",
            "Camera colour is not measured chlorophyll concentration",
            "Frame truncation censors whole-plant extent",
        ],
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
