"""Audit paired cameras and export explicitly weak, correctable leaf proposals."""

from common import workspace_relative

from multimodal_leaf_core import *
from bootstrap import read_csv, write_csv, validate_rotation_records
from PIL import Image, ImageDraw
from leaf_dynamics_geometry import assembled, frontmost_ownership
from additive_geometry import projection
import time, collections

COLORS = np.array(
    [[0, 180, 240], [245, 170, 0], [0, 205, 150], [220, 115, 190], [130, 140, 250]], np.uint8
)


def image(path):
    return np.asarray(Image.open(path).convert("RGB").resize((512, 385), Image.Resampling.LANCZOS))


def mask(path):
    return np.asarray(Image.open(path).resize((512, 385), Image.Resampling.NEAREST)) > 0


def features(rows):
    a = np.array(
        [
            [
                float(r[k])
                for k in ["silhouette_width_px", "silhouette_area_px", "visible_height_px"]
            ]
            for r in sorted(rows, key=lambda r: int(r["view"]))
        ]
    )
    return (a - a.mean(0)) / np.maximum(a.std(0), 1)


def main():
    started = time.time()
    DEST.mkdir(parents=True, exist_ok=True)
    assets = DEST / "assets"
    assets.mkdir(exist_ok=True)
    source = OUT / "sam_traits/view_traits.csv"
    rows = read_csv(source)
    validate_rotation_records(rows)
    pairs, missing = pair_records(rows)
    if missing:
        raise ValueError("Unpaired camera records: " + str(missing[:5]))
    groups = {r["session"]: plant_group(r) for r in rows}
    splits = {g: split_group(g) for g in set(groups.values())}
    teachers = {
        "capture_session_C1_5-1": OUT / "leaf_dynamics/2026-10-24/geometry/earlier_organ_model.npz",
        "capture_session_C2_5-1": OUT
        / "organ_inverse_triangles/capture_session_C2_5-1/organ_model.npz",
    }
    # Declare all label-source plant groups development/training before modelling.
    for r in rows:
        if Path(r["session"]).name in teachers:
            splits[groups[r["session"]]] = "train"
    pairrows = []
    phase = []
    pool = []
    hashes = {
        workspace_relative(source): sha(source),
        workspace_relative(BASE.joinpath("multimodal_leaf_core.py")): sha(
            BASE / "multimodal_leaf_core.py"
        ),
    }
    for session in sorted(groups):
        rr = [r for r in rows if r["session"] == session]
        a = features([r for r in rr if r["camera"] == "rgb"])
        b = features([r for r in rr if r["camera"] == "noir"])
        scores = np.array([np.mean((a - np.roll(b, -shift, axis=0)) ** 2) for shift in range(40)])
        order = np.argsort(scores)
        phase.append(
            dict(
                session=session,
                group=groups[session],
                scientific_split=splits[groups[session]],
                best_shift=int(order[0]),
                second_shift=int(order[1]),
                cost_best=float(scores[order[0]]),
                cost_second=float(scores[order[1]]),
                zero_shift_cost=float(scores[0]),
                interpretation="Shape-descriptor phase proposal; bilateral aliases and camera effects remain, not verified synchronisation.",
            )
        )
    for p in pairs:
        split = splits[groups[p["session"]]]
        pairrows.append(
            {
                "session": p["session"],
                "view": p["view"],
                "group": groups[p["session"]],
                "scientific_split": split,
                "nominal_turn_degrees": p["view"] * 9,
                "rgb_source": p["rgb"]["source"],
                "noir_source": p["noir"]["source"],
                "rgb_source_sha256": p["rgb"]["source_sha256"],
                "noir_source_sha256": p["noir"]["source_sha256"],
                "pair_status": "same acquisition and filename index; physical synchronisation unverified",
            }
        )
        if p["view"] in [0, 13, 26]:
            for camera in ["rgb", "noir"]:
                r = p[camera]
                name = f"{Path(r['session']).name}__{camera}__{p['view']:03}"
                raw = image(ROOT / r["source"])
                whole = mask(ROOT / r["mask"])
                seg = raw.copy()
                seg[~whole] = 0
                Image.fromarray(raw).save(assets / (name + "_raw.jpg"), quality=95)
                Image.fromarray(seg).save(assets / (name + "_segmented.png"))
                pool.append(
                    dict(
                        image_id=name,
                        session=r["session"],
                        camera=camera,
                        view=p["view"],
                        group=groups[r["session"]],
                        scientific_split=split,
                        source=r["source"],
                        asset="assets/" + name + "_raw.jpg",
                        review_status="unlabelled",
                        objects=[],
                        width=512,
                        height=385,
                    )
                )
    # All-view evaluation tasks from two held-out plant groups, selected without masks/labels.
    chosen = []
    for session in sorted(groups):
        if splits[groups[session]] == "test" and groups[session] not in [groups[s] for s in chosen]:
            chosen.append(session)
        if len(chosen) == 2:
            break
    evaluation = [r for r in pairrows if r["session"] in chosen]
    write_csv(DEST / "paired_frames.csv", pairrows)
    write_csv(DEST / "phase_audit.csv", phase)
    write_csv(DEST / "heldout_sequence_tasks.csv", evaluation)
    dump(
        DEST / "annotation_pool.json",
        {
            "coordinate_frame": [512, 385],
            "records": pool,
            "annotation_rule": "Visible leaf instances, one class leaf. IDs local within acquisition. Mark uncertainty/occlusion; never invent hidden lamina. Human review requires completeness and reviewer.",
            "scientific_split_unit": "Conservative treatment/genotype/replicate group, both cameras and repeated rounds kept together.",
        },
    )
    registration = []
    proposals = []
    exported = []
    for session, modelpath in teachers.items():
        saved = np.load(modelpath)
        v, f, o = assembled(saved["vertices"], saved["stem"])
        fo = o[f[:, 0]]
        camera = json.loads(str(saved["camera"]))
        n = len(saved["vertices"])
        hashes[workspace_relative(modelpath)] = sha(modelpath)
        selected = [p for p in pairs if Path(p["session"]).name == session]
        for p in selected:
            view = p["view"]
            rgbmask = mask(ROOT / p["rgb"]["mask"])
            nirmask = mask(ROOT / p["noir"]["mask"])
            matrix, agreement, status = affine_align(rgbmask, nirmask)
            registration.append(
                dict(
                    session=session,
                    view=view,
                    source_camera="rgb",
                    target_camera="noir",
                    affine_source_to_target=matrix.tolist(),
                    whole_mask_iou=agreement,
                    status=status,
                    transfer_eligible=agreement >= 0.65,
                    interpretation="Image-based affine hypothesis, not calibrated correspondence.",
                )
            )
            xy, depth = projection(v, np.full(len(v), view * 2 * np.pi / 40), camera)
            xy = (xy + 0.5) * [0.5, 385 / 769] - 0.5
            owner = frontmost_ownership(xy, depth, f, fo, (385, 512))
            for cam, whole in [("rgb", rgbmask), ("noir", nirmask)]:
                row = p[cam]
                src = ROOT / row["source"]
                assert sha(src) == row["source_sha256"]
                hashes[row["source"]] = sha(src)
                hashes[row["mask"]] = sha(ROOT / row["mask"])
                raw = image(src)
                name = f"{session}__{cam}__{view:03}"
                labels = np.zeros((385, 512), np.uint8)
                objects = []
                fail = []
                for i in range(n):
                    base = (owner == i).astype(np.uint8)
                    if cam == "noir":
                        base = cv2.warpAffine(base, matrix, (512, 385), flags=cv2.INTER_NEAREST)
                    instance = (base > 0) & whole
                    if instance.sum() < 30:
                        continue
                    try:
                        poly = export_polygon(instance)
                    except ValueError as err:
                        fail.append(f"L{i + 1}: " + str(err))
                        continue
                    # Preserve original mask in master. The single dominant polygon is training-only.
                    labels[instance] = i + 1
                    objects.append(
                        {
                            "id": f"L{i + 1}",
                            "class": "leaf",
                            "polygon": poly.tolist(),
                            "pixels": int(instance.sum()),
                            "origin": "frozen RGB organ model intersected with camera SAM foreground"
                            + (" after affine transfer" if cam == "noir" else ""),
                        }
                    )
                # Stem excluded, missing tiny/fragmented organs retained as flags, never asserted absent.
                labels_path = assets / (name + "_instances.png")
                Image.fromarray(labels).save(labels_path)
                Image.fromarray(raw).save(assets / (name + "_raw.jpg"), quality=95)
                over = raw.copy()
                fg = labels > 0
                over[fg] = (0.35 * raw[fg] + 0.65 * COLORS[(labels[fg] - 1) % 5]).astype(np.uint8)
                Image.fromarray(over).save(assets / (name + "_proposal.jpg"), quality=95)
                seg = raw.copy()
                seg[~whole] = 0
                Image.fromarray(seg).save(assets / (name + "_segmented.png"))
                record = {
                    "image_id": name,
                    "session": p["session"],
                    "camera": cam,
                    "view": view,
                    "group": groups[p["session"]],
                    "scientific_split": splits[groups[p["session"]]],
                    "source": row["source"],
                    "source_sha256": hashes[row["source"]],
                    "asset": "assets/" + name + "_raw.jpg",
                    "segmented_asset": "assets/" + name + "_segmented.png",
                    "proposal_asset": "assets/" + name + "_proposal.jpg",
                    "labelmask_asset": "assets/" + name + "_instances.png",
                    "width": 512,
                    "height": 385,
                    "objects": objects,
                    "review_status": "unreviewed_model_proposal",
                    "reviewer": None,
                    "complete": False,
                    "annotation_flags": fail,
                    "affine_iou": agreement if cam == "noir" else None,
                    "training_eligible_weak": bool(
                        len(objects) >= 2 and not fail and (cam == "rgb" or agreement >= 0.65)
                    ),
                    "angle_pilot_split": "validation" if view % 5 == 0 else "train",
                }
                proposals.append(record)
                if record["training_eligible_weak"]:
                    split = record["angle_pilot_split"]
                    folder = DEST / "weak_datasets" / cam
                    for kind in ["images", "labels"]:
                        (folder / kind / split).mkdir(parents=True, exist_ok=True)
                    # Same source processing frame for teachers and students; no synthetic foreground fill in training.
                    Image.fromarray(raw).save(
                        folder / "images" / split / (name + ".jpg"), quality=95
                    )
                    lines = [
                        "0 " + " ".join(f"{x:.7f}" for point in ob["polygon"] for x in point)
                        for ob in objects
                    ]
                    (folder / "labels" / split / (name + ".txt")).write_text(
                        "\n".join(lines) + "\n", encoding="utf-8"
                    )
                    exported.append(
                        dict(
                            image_id=name,
                            camera=cam,
                            split=split,
                            instances=len(objects),
                            group=record["group"],
                        )
                    )
            if view % 10 == 0:
                print(session, view, "affine IoU", round(agreement, 3), flush=True)
        for cam in ["rgb", "noir"]:
            folder = DEST / "weak_datasets" / cam
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "data.yaml").write_text(
                "path: "
                + folder.as_posix()
                + "\ntrain: images/train\nval: images/validation\nnames:\n  0: leaf\n",
                encoding="utf-8",
            )
    dump(
        DEST / "proposals.json",
        {"label_status": "weak model-derived proposals; not ground truth", "records": proposals},
    )
    dump(DEST / "registration.json", {"records": registration, "gate_iou": 0.65})
    write_csv(DEST / "weak_export_manifest.csv", exported)
    counts = {
        cam: {
            s: sum(r["camera"] == cam and r["split"] == s for r in exported)
            for s in ["train", "validation"]
        }
        for cam in ["rgb", "noir"]
    }
    summary = {
        "status": "prepared_weak_label_pilot_and_review_pool",
        "paired_frames": len(pairs),
        "acquisitions": len(groups),
        "candidate_plant_groups": len(splits),
        "scientific_group_splits": dict(collections.Counter(splits.values())),
        "annotation_pool_images": len(pool),
        "heldout_full_acquisitions": chosen,
        "heldout_pairs": len(evaluation),
        "teacher_acquisitions": list(teachers),
        "proposed_images": len(proposals),
        "weak_dataset_images": counts,
        "registration_median_iou": float(np.median([r["whole_mask_iou"] for r in registration])),
        "registration_gate_pass": sum(r["transfer_eligible"] for r in registration),
        "phase_zero_best": sum(r["best_shift"] == 0 for r in phase),
        "input_sha256": hashes,
        "code_sha256": sha(__file__),
        "elapsed_seconds": time.time() - started,
        "inference": "Fine-tuning development uses angle splits of model-derived labels, not independent plants. Scientific group splits are reserved for corrected data. No leaf accuracy, calibration or physical time identity is established.",
    }
    dump(DEST / "preparation_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
