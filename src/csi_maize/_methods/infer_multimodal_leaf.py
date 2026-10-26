"""Unlabelled archive survey, prediction-only tracks and cross-camera diagnostics."""

from multimodal_leaf_core import *
from bootstrap import preferred_device
from bootstrap import read_csv, write_csv
from circular_leaf_tracker import CircularTracker, harmonic_predict
from PIL import Image, ImageDraw
import time, collections

COLORS = np.array(
    [
        [0, 180, 240],
        [245, 170, 0],
        [0, 205, 150],
        [220, 115, 190],
        [130, 140, 250],
        [245, 220, 110],
        [240, 120, 90],
        [120, 220, 245],
    ],
    np.uint8,
)


def main():
    from csi_maize.ultralytics_offline import configure_ultralytics, require_checkpoint

    configure_ultralytics(BASE, require_amp=False)
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(4)
    started = time.time()
    allrows = read_csv(OUT / "sam_traits/view_traits.csv")
    training = json.loads((DEST / "training_summary.json").read_text(encoding="utf-8"))["runs"]
    selected = {
        cam: max(
            [r for r in training if r["camera"] == cam],
            key=lambda r: r["metrics"]["metrics/mAP50-95(M)"],
        )
        for cam in ["rgb", "noir"]
    }
    rows = []
    tracksummary = []
    allpred = {}
    gifs = DEST / "animations"
    gifs.mkdir(exist_ok=True)
    predfolder = DEST / "predictions"
    predfolder.mkdir(exist_ok=True)
    for cam in ["rgb", "noir"]:
        chosen = selected[cam]
        checkpoint = require_checkpoint(
            chosen["best_checkpoint"], base=BASE, expected_sha256=chosen["best_sha256"]
        )
        model = YOLO(str(checkpoint))
        sessions = sorted({r["session"] for r in allrows if r["camera"] == cam})
        for session in sessions:
            rr = sorted(
                [r for r in allrows if r["session"] == session and r["camera"] == cam],
                key=lambda r: int(r["view"]),
            )
            assert len(rr) == 40
            raw = [
                np.asarray(
                    Image.open(ROOT / r["source"])
                    .convert("RGB")
                    .resize((512, 385), Image.Resampling.LANCZOS)
                )
                for r in rr
            ]
            inputs = [
                np.repeat(a[:, :, 2:3], 3, axis=2)
                if chosen.get("representation") == "recorded_blue_channel_triplicated"
                else a
                for a in raw
            ]
            pred = model.predict(
                [a[:, :, ::-1].copy() for a in inputs],
                imgsz=768,
                batch=8,
                device=preferred_device(),
                conf=0.25,
                iou=0.7,
                max_det=12,
                retina_masks=True,
                verbose=False,
            )
            tracker = {mode: CircularTracker(mode) for mode in ["overlap", "circular"]}
            masks = []
            metadata = []
            animation = []
            sessionpred = []
            for view, (r, result, a) in enumerate(zip(rr, pred, raw)):
                obs = []
                if result.masks is not None:
                    for mm, conf in zip(
                        result.masks.data.cpu().numpy() > 0.5, result.boxes.conf.cpu().numpy()
                    ):
                        assert mm.shape == (385, 512)
                        yy, xx = np.where(mm)
                        if len(xx) < 30:
                            continue
                        centered = np.column_stack([xx - xx.mean(), yy - yy.mean()])
                        cov = centered.T @ centered / max(1, len(xx) - 1)
                        vals, vec = np.linalg.eigh(cov)
                        axis = vec[:, -1]
                        orientation = float(np.degrees(np.arctan2(axis[1], axis[0])))
                        obs.append(
                            {
                                "mask": mm,
                                "centroid": [float(xx.mean()), float(yy.mean())],
                                "area": len(xx),
                                "confidence": float(conf),
                                "visible_extent_y_px": int(yy.max() - yy.min() + 1),
                                "visible_extent_x_px": int(xx.max() - xx.min() + 1),
                                "PCA_axis_degrees": orientation,
                            }
                        )
                ids = {mode: t.step(view, obs) for mode, t in tracker.items()}
                whole = (
                    np.asarray(
                        Image.open(ROOT / r["mask"]).resize((512, 385), Image.Resampling.NEAREST)
                    )
                    > 0
                )
                for j, o in enumerate(obs):
                    mask = o["mask"]
                    sample = a[mask]
                    record = {
                        "session": session,
                        "camera": cam,
                        "view": view,
                        "detection_index": j,
                        "source": r["source"],
                        "source_sha256_recorded": r["source_sha256"],
                        "checkpoint_sha256": chosen["best_sha256"],
                        "scientific_split": next(
                            x["scientific_split"] for x in readpairs if x["session"] == session
                        ),
                        "overlap_track_id": ids["overlap"][j],
                        "circular_track_id": ids["circular"][j],
                        "confidence": o["confidence"],
                        "projected_area_px": o["area"],
                        "centroid_x": o["centroid"][0],
                        "centroid_y": o["centroid"][1],
                        "visible_extent_y_px": o["visible_extent_y_px"],
                        "visible_extent_x_px": o["visible_extent_x_px"],
                        "PCA_axis_degrees": o["PCA_axis_degrees"],
                        "outside_SAM_foreground_fraction": float(
                            np.count_nonzero(mask & ~whole) / o["area"]
                        ),
                        "source_R_median": float(np.median(sample[:, 0])),
                        "source_G_median": float(np.median(sample[:, 1])),
                        "source_B_median": float(np.median(sample[:, 2])),
                        "units": "processing pixels and recorded camera DN, not metric size or reflectance",
                    }
                    masks.append(mask.reshape(-1))
                    metadata.append(record)
                    rows.append(record)
                # Store per-camera tracks independently, without teacher/geometry ownership IDs.
                sessionpred.append((obs, ids["circular"]))
                if Path(session).name in ["capture_session_C1_5-1", "capture_session_C2_5-1"]:
                    panel = a.copy()
                    panel[~whole] = 0
                    for o, k in zip(obs, ids["circular"]):
                        mm = o["mask"] & whole
                        panel[mm] = (
                            0.55 * panel[mm] + 0.45 * COLORS[(k - 1) % len(COLORS)]
                        ).astype(np.uint8)
                    im = Image.fromarray(panel)
                    draw = ImageDraw.Draw(im)
                    draw.text(
                        (8, 8),
                        f"{cam.upper()} view {view:02} | prediction IDs, unverified",
                        fill="white",
                    )
                    for o, k in zip(obs, ids["circular"]):
                        draw.text(
                            tuple(o["centroid"]),
                            str(k),
                            fill="white",
                            stroke_width=1,
                            stroke_fill="black",
                        )
                    animation.append(im)
            for mode, t in tracker.items():
                lengths = [len(x["views"]) for x in t.tracks.values()]
                fiterr = []
                for tr in t.tracks.values():
                    if len(tr["views"]) >= 6:
                        residual = [
                            np.linalg.norm(harmonic_predict(tr["views"], tr["centroids"], v) - xy)
                            for v, xy in zip(tr["views"], tr["centroids"])
                        ]
                        fiterr.extend(residual)
                tracksummary.append(
                    {
                        "session": session,
                        "camera": cam,
                        "mode": mode,
                        "created_tracks": len(lengths),
                        "observations": sum(lengths),
                        "median_observations_per_track": float(np.median(lengths))
                        if lengths
                        else 0.0,
                        "tracks_at_least_8_views": sum(n >= 8 for n in lengths),
                        "median_harmonic_residual_px": float(np.median(fiterr)) if fiterr else None,
                        "interpretation": "Prediction fragmentation and fitted nominal-motion residual, not identity accuracy or botanical leaf count.",
                    }
                )
            key = Path(session).name + "__" + cam
            np.savez_compressed(
                predfolder / (key + ".npz"),
                packed_masks=np.packbits(np.array(masks, np.uint8), axis=1)
                if masks
                else np.empty((0, 24640), np.uint8),
                mask_shape=np.array([385, 512]),
            )
            dump(
                predfolder / (key + ".json"),
                {
                    "records": metadata,
                    "checkpoint": chosen["run"],
                    "representation": chosen.get("representation", "recorded_RGB_channels"),
                    "teacher_ids_used_by_tracker": False,
                },
            )
            if animation:
                animation[0].save(
                    gifs / (key + ".gif"),
                    save_all=True,
                    append_images=animation[1:],
                    duration=180,
                    loop=0,
                )
                allpred[session, cam] = sessionpred
            print("INFERRED", cam, Path(session).name, "detections", len(metadata), flush=True)
    cross = []
    registration = json.loads((DEST / "registration.json").read_text(encoding="utf-8"))["records"]
    for r in registration:
        session = next(s for s, c in allpred if Path(s).name == r["session"])
        v = r["view"]
        rgb, rgbids = allpred[session, "rgb"][v]
        nir, nirids = allpred[session, "noir"][v]
        matrix = np.array(r["affine_source_to_target"], np.float32)
        warped = [
            cv2.warpAffine(o["mask"].astype(np.uint8), matrix, (512, 385), flags=cv2.INTER_NEAREST)
            > 0
            for o in rgb
        ]
        cost = np.array([[1 - mask_iou(a, b["mask"]) for b in nir] for a in warped]).reshape(
            len(warped), len(nir)
        )
        matches = assign_tracks(cost, 0.75)
        cross.append(
            {
                "session": session,
                "view": v,
                "rgb_detections": len(rgb),
                "noir_detections": len(nir),
                "one_to_one_mask_matches_iou_at_least_025": len(matches),
                "unmatched_RGB_candidates": len(rgb) - len(matches),
                "unmatched_NoIR_candidates": len(nir) - len(matches),
                "mean_matched_mask_iou": float(np.mean([1 - cost[i, j] for i, j in matches]))
                if matches
                else None,
                "affine_whole_mask_iou": r["whole_mask_iou"],
                "matched_local_track_pairs": ";".join(
                    f"{rgbids[i]}:{nirids[j]}" for i, j in matches
                ),
                "interpretation": "Same-index affine correspondence diagnostic; unmatched candidates are not proven additional leaves. Shared weak teachers make this non-independent.",
            }
        )
    write_csv(DEST / "leaf_predictions.csv", rows)
    write_csv(DEST / "tracking_diagnostics.csv", tracksummary)
    write_csv(DEST / "cross_camera_diagnostics.csv", cross)
    summary = {
        "status": "executed_unlabelled_archive_survey",
        "images": len(allrows),
        "sequences": len(tracksummary) // 2,
        "detected_leaf_observations": len(rows),
        "selected_models": selected,
        "selection_rule": "Highest within-acquisition weak-teacher mask AP50-95 per camera, before archive inference. Not independent plant selection.",
        "tracker_modes": ["overlap", "circular"],
        "prediction_confidence": 0.25,
        "cross_camera_diagnostic_frames": len(cross),
        "code_sha256": sha(__file__),
        "tracker_code_sha256": sha(BASE / "circular_leaf_tracker.py"),
        "elapsed_seconds": time.time() - started,
        "inference": "No independent leaf identity labels. Counts and consistency are diagnostic; no biological performance or longitudinal tracking accuracy is claimed.",
    }
    dump(DEST / "inference_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "selected_models"}, indent=2))


if __name__ == "__main__":
    readpairs = read_csv(DEST / "paired_frames.csv")
    main()
