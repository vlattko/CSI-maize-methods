"""Compare public SAM 2.1 image masks against colour seeds, with saved prompts.

Outputs are candidates awaiting annotated validation, never automatic ground truth.
"""

from common import workspace_relative

from bootstrap import BASE, ROOT, OUT, read_csv, write_csv, model_directory
from traits import load_image, segment
import argparse, json, hashlib, time
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import torch

MODEL_ID = "facebook/sam2.1-hiera-small"
REVISION = "ee5bba1d82bb8749febdf90f45e84b687142ba03"


def make_prompts(seed, image):
    """Deterministic farthest-point coverage of a conservative seed mask."""
    ys, xs = np.where(seed)
    h, w = seed.shape
    box = [
        max(0, int(xs.min()) - 24),
        max(0, int(ys.min()) - 24),
        min(w - 1, int(xs.max()) + 24),
        min(h - 1, int(ys.max()) + 24),
    ]
    distance = ndi.distance_transform_edt(seed)
    eligible = np.argwhere(seed & (distance >= max(1.0, float(distance.max()) * 0.12)))
    first = np.unravel_index(distance.argmax(), distance.shape)
    selected = [np.asarray(first)]
    nearest = ((eligible - selected[0]) ** 2).sum(axis=1).astype(float)
    for _ in range(11):
        index = int(nearest.argmax())
        if nearest[index] < 20**2:
            break
        selected.append(eligible[index])
        nearest = np.minimum(nearest, ((eligible - selected[-1]) ** 2).sum(axis=1))
    points = [[int(p[1]), int(p[0])] for p in selected]
    labels = [1] * len(points)
    # Black-background anchors are safe negatives; no assumption that weakly green pixels are background.
    for fy, fx in [
        (0.05, 0.05),
        (0.05, 0.95),
        (0.5, 0.05),
        (0.5, 0.95),
        (0.95, 0.05),
        (0.95, 0.95),
        (0.05, 0.5),
    ]:
        y, x = int(fy * (h - 1)), int(fx * (w - 1))
        if image[y, x].max() < 40 and not seed[y, x]:
            points.append([x, y])
            labels.append(0)
    return box, points, labels


def overlay(image, mask, points=None, labels=None):
    display = image.copy()
    display[~mask] = (display[~mask] * 0.22).astype(np.uint8)
    edge = mask & ~ndi.binary_erosion(mask)
    display[edge] = [100, 255, 150]
    canvas = Image.fromarray(display)
    draw = ImageDraw.Draw(canvas)
    for (x, y), label in zip(points or [], labels or []):
        draw.ellipse(
            (x - 4, y - 4, x + 4, y + 4), fill=("#63ff90" if label else "#ff6472"), outline="white"
        )
    return canvas


def main():
    from transformers import Sam2Model, Sam2Processor

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    folder = OUT / "sam_comparison"
    folder.mkdir(exist_ok=True)
    model_path = model_directory("sam2.1-hiera-small")
    model = Sam2Model.from_pretrained(model_path).to("cuda").eval()
    processor = Sam2Processor.from_pretrained(model_path)
    records = read_csv(OUT / "sessions.csv")
    chosen = [
        r
        for r in records
        if r["session"].endswith(("capture_session_C1_1-1", "capture_session_C2_5-1"))
    ]
    tasks = [
        (record, camera, path)
        for record in chosen
        for camera in ["rgb", "noir"]
        for path in sorted((ROOT / record["session"] / camera).glob("*.jpg"))
        if int(path.stem.rsplit("_", 1)[1]) in [0, 10, 20, 30]
    ]
    if args.limit:
        tasks = tasks[: args.limit]
    rows = []
    for record, camera, path in tasks:
        started = time.time()
        image, native = load_image(path)
        seed = segment(image, camera)
        box, points, labels = make_prompts(seed, image)
        stem = path.parent.parent.name + "__" + camera + "__" + path.stem.rsplit("_", 1)[1]
        panels = [("Colour seed", overlay(image, seed))]
        results = {}
        for mode in ["box", "prompted"]:
            kwargs = {"input_boxes": [[box]]}
            if mode == "prompted":
                kwargs.update(input_points=[[points]], input_labels=[[labels]])
            inputs = processor(images=Image.fromarray(image), return_tensors="pt", **kwargs).to(
                "cuda"
            )
            with torch.inference_mode():
                outputs = model(**inputs, multimask_output=False)
            mask = (
                processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0][
                    0, 0
                ]
                .numpy()
                .astype(bool)
            )
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                folder / (stem + "__" + mode + ".png")
            )
            panels.append(
                (
                    "SAM 2.1 " + mode,
                    overlay(image, mask, points if mode == "prompted" else None, labels),
                )
            )
            intersection = int((mask & seed).sum())
            union = int((mask | seed).sum())
            results[mode] = {
                "model_quality_score": float(outputs.iou_scores.flatten()[0].cpu()),
                "area_px": int(mask.sum()),
                "seed_coverage": intersection / int(seed.sum()),
                "agreement_iou_with_seed": intersection / max(1, union),
            }
            rows.append(
                {
                    "sample": stem,
                    "camera": camera,
                    "mode": mode,
                    **results[mode],
                    "validation_status": "candidate_unvalidated",
                }
            )
        canvas = Image.new("RGB", (image.shape[1] * 3, image.shape[0] + 54), "#10231d")
        draw = ImageDraw.Draw(canvas)
        for i, (title, panel) in enumerate(panels):
            canvas.paste(panel, (i * image.shape[1], 54))
            draw.text((i * image.shape[1] + 12, 12), title, fill="white")
        draw.text(
            (12, 32), stem + " | no reference mask: agreement is not accuracy", fill="#e7b46b"
        )
        canvas.save(folder / (stem + "__comparison.jpg"), quality=92)
        manifest = {
            "image": workspace_relative(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "model": MODEL_ID,
            "revision": REVISION,
            "processing_size": [image.shape[1], image.shape[0]],
            "native_size": native,
            "box_xyxy": box,
            "points_xy": points,
            "labels": labels,
            "results": results,
            "validation_status": "candidate_unvalidated",
        }
        (folder / (stem + ".json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(stem, results, "seconds", round(time.time() - started, 2), flush=True)
    write_csv(folder / "metrics.csv", rows)


if __name__ == "__main__":
    main()
