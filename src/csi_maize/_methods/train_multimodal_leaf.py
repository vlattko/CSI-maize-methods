"""Bounded weak-teacher YOLO26 experiments; never reports independent leaf accuracy."""

from multimodal_leaf_core import *
from bootstrap import preferred_device
from bootstrap import write_csv
import argparse, time, shutil, platform


def main():
    from csi_maize.ultralytics_offline import configure_ultralytics, require_checkpoint

    p = argparse.ArgumentParser()
    p.add_argument("--allow-weak-labels", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--scales", default="n,s")
    args = p.parse_args()
    if not args.allow_weak_labels:
        raise ValueError(
            "This dataset is unreviewed. Pass --allow-weak-labels only for the explicitly labelled bootstrap experiment, or export corrected supervised labels."
        )
    configure_ultralytics(BASE, require_amp=True)
    import ultralytics, torch
    from ultralytics import YOLO, settings

    settings.update(
        {
            k: False
            for k in [
                "sync",
                "wandb",
                "tensorboard",
                "mlflow",
                "clearml",
                "comet",
                "dvc",
                "neptune",
            ]
            if k in settings
        }
    )
    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    prep = json.loads((DEST / "preparation_summary.json").read_text(encoding="utf-8"))
    weights = DEST / "pretrained"
    weights.mkdir(exist_ok=True)
    runs = []
    for scale in args.scales.split(","):
        name = f"yolo26{scale}-seg.pt"
        target = weights / name
        if not target.exists():
            existing = BASE / name
            if existing.exists():
                shutil.copy2(existing, target)
            else:
                raise FileNotFoundError(
                    "Missing declared pretrained checkpoint "
                    + name
                    + "; run the models stage first"
                )
        for camera in ["rgb", "noir"]:
            tag = f"{camera}_{scale}_weak_e{args.epochs}"
            folder = DEST / "training" / tag
            if (folder / "run_manifest.json").exists():
                runs.append(json.loads((folder / "run_manifest.json").read_text(encoding="utf-8")))
                continue
            started = time.time()
            checkpoint = require_checkpoint(target, base=BASE, model_id=name[:-3])
            model = YOLO(str(checkpoint))
            result = model.train(
                data=str(DEST / "weak_datasets" / camera / "data.yaml"),
                epochs=args.epochs,
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
                hsv_h=0.0 if camera == "noir" else 0.01,
                hsv_s=0.0 if camera == "noir" else 0.2,
                hsv_v=0.15,
                plots=True,
                save=True,
                verbose=False,
            )
            metric = {k: float(v) for k, v in result.results_dict.items()}
            record = {
                "run": tag,
                "camera": camera,
                "scale": scale,
                "status": "executed_weak_label_bootstrap",
                "label_origin": "Frozen RGB model ownership; NoIR uses whole-mask-gated affine transfer. No human leaf truth.",
                "validation_scope": "Different angles of the same two development acquisitions. Not plant-independent evaluation.",
                "metrics_meaning": "Agreement with weak teacher annotations; not biological leaf accuracy.",
                "metrics": metric,
                "epochs_requested": args.epochs,
                "elapsed_seconds": time.time() - started,
                "pretrained_sha256": sha(target),
                "best_checkpoint": str(folder / "weights/best.pt"),
                "best_sha256": sha(folder / "weights/best.pt"),
                "data_yaml_sha256": sha(DEST / "weak_datasets" / camera / "data.yaml"),
                "preparation_summary_sha256": sha(DEST / "preparation_summary.json"),
                "code_sha256": sha(__file__),
                "runtime": {
                    "ultralytics": ultralytics.__version__,
                    "torch": torch.__version__,
                    "python": platform.python_version(),
                    "GPU": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
                },
            }
            dump(folder / "run_manifest.json", record)
            runs.append(record)
            dump(
                DEST / "training_summary.json",
                {
                    "runs": runs,
                    "inference": "Weak teacher agreement only; corrected independent plant tracks required.",
                },
            )
            print("COMPLETED", tag, json.dumps(metric), flush=True)
    print("All requested bootstrap models completed", len(runs))


if __name__ == "__main__":
    main()
