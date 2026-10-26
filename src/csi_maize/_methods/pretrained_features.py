"""Frozen DINOv2 representations and relative monocular-depth demonstrations."""

from common import workspace_relative

from bootstrap import (
    BASE,
    ROOT,
    OUT,
    read_csv,
    write_csv,
    model_directory,
    publication_case,
    method_config,
)
from traits import load_image
import argparse, json, hashlib
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation
from scipy.stats import mannwhitneyu
import cv2


def embeddings():
    summary = json.loads((OUT / "sam_traits/summary.json").read_text(encoding="utf-8"))
    expected = method_config().get("expected_acquisitions", 41) if publication_case() else None
    if expected is not None and summary["sessions_processed"] != int(expected):
        raise ValueError("SAM extraction must finish first")
    records = read_csv(OUT / "sam_traits/session_traits.csv")
    directory = model_directory("dinov2-small")
    processor = AutoImageProcessor.from_pretrained(directory, use_fast=False)
    model = AutoModel.from_pretrained(directory).to("cuda").eval()
    torch.set_num_threads(4)
    sessions = []
    features = []
    for record in records:
        vectors = []
        for camera in ["rgb", "noir"]:
            paths = sorted(
                (OUT / "sam_traits/crops").glob(
                    record["session"].split("/")[-1] + "__" + camera + "__*.png"
                )
            )
            if len(paths) != 8:
                raise ValueError("Expected 8 distinct crops per camera")
            images = [
                ImageOps.pad(Image.open(path).convert("RGB"), (224, 224), color=(124, 116, 104))
                for path in paths
            ]
            inputs = processor(
                images=images, do_resize=False, do_center_crop=False, return_tensors="pt"
            ).to("cuda")
            with torch.inference_mode():
                tokens = model(**inputs).last_hidden_state[:, 0, :]
            normalized = torch.nn.functional.normalize(tokens, dim=1).mean(dim=0).cpu().numpy()
            vectors.append(normalized)
        sessions.append(record["session"])
        features.append(np.concatenate(vectors))
        print("DINO", len(sessions), "/", len(records), record["session"], flush=True)
    np.savez_compressed(
        OUT / "dinov2_embeddings.npz", sessions=np.array(sessions), features=np.array(features)
    )
    meta = {
        "model": "facebook/dinov2-small",
        "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        "sessions": len(sessions),
        "dimensions": 768,
        "pooling": "mean of L2-normalized CLS vectors over 8 views per camera; concatenate RGB and NoIR",
        "preprocessing": "SAM foreground on constant background, aspect-preserving padding to 224 x 224, ImageNet normalization; no central cropping",
        "limitations": [
            "Frozen generic representation, not trained to measure chlorophyll",
            "NoIR is outside normal RGB pretraining",
            "Automatic mask QC; no annotated accuracy estimate",
        ],
    }
    (OUT / "dinov2_manifest.json").write_text(json.dumps(meta, indent=2))


def depth():
    folder = OUT / "relative_depth"
    folder.mkdir(exist_ok=True)
    directory = model_directory("Depth-Anything-V2-Small-hf")
    processor = AutoImageProcessor.from_pretrained(directory, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(directory).to("cuda").eval()
    torch.set_num_threads(4)
    manifests = json.loads((BASE / "models/pretrained_manifest.json").read_text(encoding="utf-8"))
    revision = next(r["revision"] for r in manifests if r["model"].endswith("Small-hf"))
    records = [
        r
        for r in read_csv(OUT / "sessions.csv")
        if r["session"].endswith(("capture_session_C1_1-1", "capture_session_C2_5-1"))
    ]
    rows = []
    for record in records:
        for path in sorted((ROOT / record["session"] / "rgb").glob("*.jpg")):
            idx = int(path.stem.rsplit("_", 1)[1])
            if idx not in [0, 10, 20, 30]:
                continue
            image, native = load_image(path)
            h, w = image.shape[:2]
            inputs = processor(images=Image.fromarray(image), return_tensors="pt").to("cuda")
            with torch.inference_mode():
                prediction = model(**inputs).predicted_depth
                array = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
                    )[0, 0]
                    .cpu()
                    .numpy()
                )
            stem = path.parent.parent.name + "__rgb__" + str(idx).zfill(3)
            # Use the SAM comparison box mask, identical processing coordinates.
            mask = (
                np.array(
                    Image.open(
                        OUT
                        / "sam_comparison"
                        / (path.parent.parent.name + "__rgb__" + str(idx).zfill(3) + "__box.png")
                    )
                )
                > 0
            )
            if array.shape != mask.shape:
                raise ValueError("Depth/mask coordinates differ")
            np.savez_compressed(folder / (stem + ".npz"), relative_inverse_depth=array, mask=mask)
            low, high = np.percentile(array, [1, 99])
            norm = np.clip((array - low) / max(float(high - low), 1e-8), 0, 1)
            colours = cv2.cvtColor(
                cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
                cv2.COLOR_BGR2RGB,
            )
            isolated = colours.copy()
            isolated[~mask] = [15, 29, 24]
            canvas = Image.new("RGB", (w * 3, h + 60), "#10231d")
            draw = ImageDraw.Draw(canvas)
            for j, (title, view) in enumerate(
                [
                    ("Source RGB", image),
                    ("Relative inverse depth", colours),
                    ("SAM foreground / same depth map", isolated),
                ]
            ):
                canvas.paste(Image.fromarray(view), (j * w, 60))
                draw.text((j * w + 12, 12), title, fill="white")
            draw.text(
                (12, 35),
                stem
                + " | learned relative depth; no metric scale or reconstruction accuracy validation",
                fill="#e7b46b",
            )
            canvas.save(folder / (stem + ".jpg"), quality=91)
            foreground = array[mask]
            background = array[~mask][::50]
            separation = float(
                mannwhitneyu(foreground[::5], background, alternative="two-sided").statistic
                / (len(foreground[::5]) * len(background))
            )
            rows.append(
                {
                    "sample": stem,
                    "image": workspace_relative(path),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "foreground_median": float(np.median(foreground)),
                    "background_median": float(np.median(background)),
                    "plant_background_rank_separation": separation,
                    "validation_status": "relative_depth_demonstration_no_ground_truth",
                }
            )
            print("DEPTH", stem, "separation", round(separation, 4), flush=True)
    write_csv(folder / "metrics.csv", rows)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "model": "depth-anything/Depth-Anything-V2-Small-hf",
                "revision": revision,
                "images": len(rows),
                "quantity": "relative inverse depth, arbitrary scale; greater values conventionally nearer",
                "limitations": [
                    "No calibrated metric distance",
                    "Foreground/background ranking is not geometric accuracy",
                    "Independent monocular predictions do not establish consistent multiview geometry",
                    "Thin leaves and dark background are a challenging transfer domain",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["embeddings", "depth"])
    args = parser.parse_args()
    embeddings() if args.mode == "embeddings" else depth()
