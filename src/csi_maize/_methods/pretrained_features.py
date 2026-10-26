"""Frozen DINOv2 representations for the declared SAM foreground crops."""

from bootstrap import OUT, read_csv, model_directory, publication_case, method_config
import argparse
import json
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["embeddings"])
    parser.parse_args()
    embeddings()
