"""Controlled recorded-blue-channel NoIR ablation using the same weak labels."""

from multimodal_leaf_core import *
from bootstrap import preferred_device
from PIL import Image
import shutil, time, platform


def main():
    from csi_maize.ultralytics_offline import configure_ultralytics, require_checkpoint

    configure_ultralytics(BASE, require_amp=True)
    import torch, ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(4)
    source = DEST / "weak_datasets/noir"
    target = DEST / "weak_datasets/noir_blue"
    for split in ["train", "validation"]:
        for part in ["images", "labels"]:
            (target / part / split).mkdir(parents=True, exist_ok=True)
        for p in (source / "images" / split).glob("*.jpg"):
            a = np.asarray(Image.open(p).convert("RGB"))
            b = np.repeat(a[:, :, 2:3], 3, axis=2)
            Image.fromarray(b).save(target / "images" / split / p.name, quality=95)
        for p in (source / "labels" / split).glob("*.txt"):
            shutil.copy2(p, target / "labels" / split / p.name)
    (target / "data.yaml").write_text(
        f"path: {target.as_posix()}\ntrain: images/train\nval: images/validation\nnames:\n  0: leaf\n",
        encoding="utf-8",
    )
    summary = json.loads((DEST / "training_summary.json").read_text(encoding="utf-8"))
    for scale in ["n", "s"]:
        tag = f"noir_blue_{scale}_weak_e30"
        folder = DEST / "training" / tag
        if (folder / "run_manifest.json").exists():
            cached = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
            if cached.get("run") != tag:
                raise ValueError("Cached training record does not match requested recipe")
            summary["runs"] = [row for row in summary["runs"] if row.get("run") != tag]
            summary["runs"].append(cached)
            dump(DEST / "training_summary.json", summary)
            continue
        pretrained = DEST / "pretrained" / f"yolo26{scale}-seg.pt"
        started = time.time()
        checkpoint = require_checkpoint(pretrained, base=BASE, model_id=f"yolo26{scale}-seg")
        model = YOLO(str(checkpoint))
        result = model.train(
            data=str(target / "data.yaml"),
            epochs=30,
            imgsz=768,
            batch=8,
            workers=0,
            device=preferred_device(),
            project=str(DEST / "training"),
            name=tag,
            exist_ok=True,
            seed=20261024,
            deterministic=True,
            optimizer="AdamW",
            lr0=0.001,
            patience=12,
            amp=True,
            cache=False,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            degrees=0.0,
            translate=0.02,
            scale=0.1,
            fliplr=0.5,
            flipud=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.15,
            plots=True,
            save=True,
            verbose=False,
        )
        r = {
            "run": tag,
            "camera": "noir",
            "representation": "recorded_blue_channel_triplicated",
            "scale": scale,
            "status": "executed_weak_label_bootstrap",
            "metrics_meaning": "Same NoIR teacher labels and recipe as raw-channel run; teacher agreement, not independent accuracy or calibrated NIR.",
            "metrics": {k: float(v) for k, v in result.results_dict.items()},
            "epochs_requested": 30,
            "elapsed_seconds": time.time() - started,
            "pretrained_sha256": sha(pretrained),
            "best_checkpoint": str(folder / "weights/best.pt"),
            "best_sha256": sha(folder / "weights/best.pt"),
            "data_yaml_sha256": sha(target / "data.yaml"),
            "code_sha256": sha(__file__),
            "runtime": {
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
                "python": platform.python_version(),
                "GPU": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
            },
        }
        dump(folder / "run_manifest.json", r)
        summary["runs"].append(r)
        dump(DEST / "training_summary.json", summary)
        print("COMPLETED", tag, r["metrics"], flush=True)


if __name__ == "__main__":
    main()
