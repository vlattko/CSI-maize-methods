"""Controlled, explicitly uncalibrated turntable silhouette tomography.

One camera model and rotation axis explain every view. Outside-frame rays are
censored; foreground disagreements are counted separately. A nested view split
separates camera/tolerance selection from the final withheld-view diagnostic.
This yields a silhouette envelope, not uniquely recovered leaf tissue.
"""

from bootstrap import BASE, ROOT, OUT, read_csv, write_csv
from pathlib import Path
import argparse, json, hashlib, time, itertools
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

cv2.setNumThreads(2)


def project_points(points, angle, camera):
    """World X, upward Y, Z; camera projection in original processing pixels."""
    p = np.asarray(points)
    c, s = np.cos(angle), np.sin(angle)
    xx = c * p[:, 0] + s * p[:, 2]
    zz = -s * p[:, 0] + c * p[:, 2]
    e = np.radians(camera["elevation_deg"])
    up = np.cos(e) * p[:, 1] + np.sin(e) * zz
    depth = np.cos(e) * zz - np.sin(e) * p[:, 1]
    distance = camera.get("distance", 0.0)
    denom = distance + depth if distance else np.ones_like(depth)
    factor = distance / np.maximum(denom, 1e-8) if distance else np.ones_like(depth)
    roll = np.radians(camera.get("roll_deg", 0.0))
    sx = xx * factor
    sy = -up * factor
    return (
        camera["axis_x"] + np.cos(roll) * sx - np.sin(roll) * sy,
        camera["base_y"] + np.sin(roll) * sx + np.cos(roll) * sy,
        denom > 0,
    )


def carve(points, masks, angles, camera, tolerance=0, device="cpu"):
    """Return background contradictions and observable-view counts per voxel."""
    if device == "cpu":
        wrong = np.zeros(len(points), np.uint8)
        observed = np.zeros(len(points), np.uint8)
        for mask, angle in zip(masks, angles):
            m = (
                cv2.dilate(mask.astype(np.uint8), np.ones((2 * tolerance + 1,) * 2, np.uint8))
                if tolerance
                else mask
            )
            u, v, front = project_points(points, angle, camera)
            u = np.rint(u).astype(int)
            v = np.rint(v).astype(int)
            valid = front & (u >= 0) & (v >= 0) & (u < m.shape[1]) & (v < m.shape[0])
            observed += valid
            wrong[valid] += (m[v[valid], u[valid]] == 0).astype(np.uint8)
        return wrong, observed
    import torch

    p = torch.as_tensor(points, device=device)
    wrong = torch.zeros(len(p), dtype=torch.uint8, device=device)
    observed = wrong.clone()
    e = np.radians(camera["elevation_deg"])
    distance = camera.get("distance", 0.0)
    for mask, angle in zip(masks, angles):
        m = (
            cv2.dilate(mask.astype(np.uint8), np.ones((2 * tolerance + 1,) * 2, np.uint8))
            if tolerance
            else mask
        )
        mt = torch.as_tensor(m, device=device)
        c, s = np.cos(angle), np.sin(angle)
        xx = c * p[:, 0] + s * p[:, 2]
        zz = -s * p[:, 0] + c * p[:, 2]
        up = np.cos(e) * p[:, 1] + np.sin(e) * zz
        depth = np.cos(e) * zz - np.sin(e) * p[:, 1]
        if distance:
            denominator = distance + depth
            factor = distance / denominator.clamp_min(1e-8)
            front = denominator > 0
        else:
            factor = 1.0
            front = torch.ones(len(p), dtype=torch.bool, device=device)
        roll = np.radians(camera.get("roll_deg", 0.0))
        sx = xx * factor
        sy = -up * factor
        u = torch.round(camera["axis_x"] + np.cos(roll) * sx - np.sin(roll) * sy).long()
        v = torch.round(camera["base_y"] + np.sin(roll) * sx + np.cos(roll) * sy).long()
        valid = front & (u >= 0) & (v >= 0) & (u < m.shape[1]) & (v < m.shape[0])
        hits = mt[v.clamp(0, m.shape[0] - 1), u.clamp(0, m.shape[1] - 1)] > 0
        observed += valid.to(torch.uint8)
        wrong += (valid & ~hits).to(torch.uint8)
    return wrong.cpu().numpy(), observed.cpu().numpy()


def render_points(points, angle, camera, shape, radius=1):
    result = np.zeros(shape, np.uint8)
    if len(points):
        u, v, front = project_points(points, angle, camera)
        u = np.rint(u).astype(int)
        v = np.rint(v).astype(int)
        valid = front & (u >= 0) & (v >= 0) & (u < shape[1]) & (v < shape[0])
        result[v[valid], u[valid]] = 1
        if radius:
            result = cv2.dilate(result, np.ones((2 * radius + 1,) * 2, np.uint8))
    return result > 0


def mask_scores(predicted, observed):
    intersection = int((predicted & observed).sum())
    union = int((predicted | observed).sum())
    return {
        "iou": intersection / max(union, 1),
        "precision": intersection / max(int(predicted.sum()), 1),
        "recall": intersection / max(int(observed.sum()), 1),
        "observed_pixels": int(observed.sum()),
        "predicted_pixels": int(predicted.sum()),
    }


def load_session(name, camera="rgb"):
    rows = [
        r
        for r in read_csv(OUT / "sam_traits/view_traits.csv")
        if Path(r["session"]).name == name and r["camera"] == camera
    ]
    rows.sort(key=lambda r: int(r["view"]))
    if len(rows) != 40 or [int(r["view"]) for r in rows] != list(range(40)):
        raise ValueError("Expected 40 indexed views")
    masks = [np.array(Image.open(ROOT / r["mask"])) > 0 for r in rows]
    if len({m.shape for m in masks}) != 1:
        raise ValueError("Masks must share one processing frame")
    return rows, masks


def train_geometry(masks, indices, angles):
    """Estimate one shared horizontal axis, never per-test-frame offsets."""
    h, w = masks[0].shape
    anchors = []
    bounds = []
    for i in indices:
        y, x = np.where(masks[i])
        bounds.append((float(x.min()), float(x.max()), float(y.min())))
        # Basal band is a proxy for a rotating point; residual is saved, not corrected.
        yy, xx = np.where(masks[i][h - 25 : h - 5])
        anchors.append(float(np.median(xx)) if len(xx) else np.nan)
    anchors = np.array(anchors)
    ok = np.isfinite(anchors)
    design = np.stack(
        [np.ones(len(indices)), np.cos(angles[indices]), np.sin(angles[indices])], axis=1
    )
    if ok.sum() < 4:
        raise ValueError("Insufficient basal stem evidence")
    coef = np.linalg.lstsq(design[ok], anchors[ok], rcond=None)[0]
    axis = float(coef[0])
    residual = anchors[ok] - design[ok] @ coef
    radius = max(max(abs(lo - axis), abs(hi - axis)) for lo, hi, _ in bounds) * 1.15 + 10
    top = min(y for _, _, y in bounds)
    height = h - 1 - top
    return (
        {"axis_x": axis, "base_y": float(h - 1), "elevation_deg": 0.0, "distance": 0.0},
        radius,
        (-radius * 0.4, height + radius * 0.5 + 10),
        {
            "fit_indices": list(map(int, indices)),
            "basal_band_y": [h - 25, h - 5],
            "basal_sinusoid_coefficients": coef.tolist(),
            "basal_residual_rmse_processing_px": float(np.sqrt(np.mean(residual**2))),
            "axis_limit": "Stem-band centroid is a train-only proxy; not an independently measured turntable axis",
        },
    )


def grid_axes(radius, ybounds, resolution):
    step = 2 * radius / (resolution - 1)
    y = np.arange(ybounds[0], ybounds[1] + step * 0.5, step, dtype=np.float32)
    x = np.linspace(-radius, radius, resolution, dtype=np.float32)
    z = x.copy()
    return x, y, z, float(step)


def grid_points(radius, ybounds, resolution):
    x, y, z, step = grid_axes(radius, ybounds, resolution)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    return points, (len(x), len(y), len(z)), [float(x[0]), float(y[0]), float(z[0])], float(step)


def envelope(points, masks, indices, angles, camera, tolerance, allowed, device):
    wrong, observable = carve(
        points, [masks[i] for i in indices], angles[indices], camera, tolerance, device
    )
    # Never allow a wholly censored voxel to count as reconstructed geometry.
    selected = (wrong <= allowed) & (observable >= int(np.ceil(0.75 * len(indices))))
    return selected, wrong, observable


def score_views(points, masks, indices, angles, camera, radius):
    return [
        {
            "view": int(i),
            **mask_scores(
                render_points(points, angles[i], camera, masks[i].shape, radius), masks[i]
            ),
        }
        for i in indices
    ]


def mesh_export(folder, selected, shape, origin, spacing, stem="envelope"):
    volume = selected.reshape(shape)
    # Padded domain caps are computational, not evidence of closed leaf tissue.
    vertices, faces, _, _ = marching_cubes(
        np.pad(volume.astype(np.uint8), 1), 0.5, spacing=(spacing,) * 3
    )
    vertices += np.array(origin) - spacing
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(folder / (stem + ".ply"))
    mesh.export(folder / (stem + ".obj"))
    return vertices, faces


def browser_mesh(folder, vertices, faces, rows, masks, angles, camera, name, view_indices=None):
    colours = colour_points(vertices, rows, masks, angles, camera, view_indices)
    data = {
        "vertices": vertices.round(3).tolist(),
        "faces": faces.tolist(),
        "colors": colours.tolist(),
        "units": "processing-pixel geometry, unknown physical scale",
        "representation": "censored silhouette envelope",
    }
    (folder / name).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def colour_points(points, rows, masks, angles, camera, view_indices=None):
    """Median silhouette-compatible projected colours; no visibility/reflectance claim."""
    samples = []
    view_indices = list(range(0, 40, 5)) if view_indices is None else list(view_indices)[::4]
    for i in view_indices:
        im = (
            Image.open(ROOT / rows[i]["source"])
            .convert("RGB")
            .resize((masks[i].shape[1], masks[i].shape[0]), Image.Resampling.LANCZOS)
        )
        image = np.array(im)
        u, v, front = project_points(points, angles[i], camera)
        u = np.rint(u).astype(int)
        v = np.rint(v).astype(int)
        valid = front & (u >= 0) & (v >= 0) & (u < image.shape[1]) & (v < image.shape[0])
        col = np.full((len(points), 3), np.nan, np.float32)
        ix = np.flatnonzero(valid)
        ix = ix[masks[i][v[ix], u[ix]]]
        col[ix] = image[v[ix], u[ix]]
        samples.append(col)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col = np.nanmedian(samples, axis=0)
    return np.nan_to_num(col, nan=100).clip(0, 255).astype(np.uint8)


def make_overlays(folder, rows, masks, points, indices, angles, camera, splat):
    paths = []
    for i in indices:
        source = np.array(
            Image.open(ROOT / rows[i]["source"])
            .convert("RGB")
            .resize((masks[i].shape[1], masks[i].shape[0]), Image.Resampling.LANCZOS)
        )
        pred = render_points(points, angles[i], camera, masks[i].shape, splat)
        truth = masks[i]
        overlay = source.copy()
        overlay[pred & truth] = (
            0.4 * overlay[pred & truth] + 0.6 * np.array([64, 210, 150])
        ).astype(np.uint8)
        overlay[pred & ~truth] = [242, 138, 55]
        overlay[truth & ~pred] = [212, 67, 140]
        panel = Image.new("RGB", (source.shape[1] * 2, source.shape[0] + 48), "#142b20")
        panel.paste(Image.fromarray(source), (0, 48))
        panel.paste(Image.fromarray(overlay), (source.shape[1], 48))
        draw = ImageDraw.Draw(panel)
        score = mask_scores(pred, truth)
        draw.text((16, 14), f"WITHHELD VIEW {i:02d} | original RGB", fill="white")
        draw.text(
            (source.shape[1] + 16, 14),
            f"IoU {score['iou']:.3f} | mint match / orange excess / pink missed",
            fill="white",
        )
        path = folder / f"heldout_{i:02d}.jpg"
        panel.save(path, quality=92)
        paths.append(path.name)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="capture_session_C2_5-1")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--name", default="turntable_tomography_v2")
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    start = time.time()
    rows, masks = load_session(args.session)
    angles = np.arange(40) * 2 * np.pi / 40
    outer = np.array([i for i in range(40) if i % 4 == 2])
    train = np.array([i for i in range(40) if i % 4 != 2])
    inner_val = train[::5]
    inner_train = np.array([i for i in train if i not in inner_val])
    camera, radius, ybounds, axis_audit = train_geometry(masks, inner_train, angles)
    coarse, coarse_shape, _, step = grid_points(radius, ybounds, 96)
    splat = max(1, int(np.ceil(step * 0.6)))
    candidates = []
    # This grid is fixed before the outer holdout is scored.
    for elevation, roll, tolerance, allowed in itertools.product(
        [-12.0, -6.0, 0.0, 6.0, 12.0], [-4.0, 0.0, 4.0], [5, 12], [1, 3]
    ):
        distance = 0.0
        cam = {**camera, "elevation_deg": elevation, "distance": distance, "roll_deg": roll}
        selected, _, _ = envelope(
            coarse, masks, inner_train, angles, cam, tolerance, allowed, args.device
        )
        scores = score_views(coarse[selected], masks, inner_val, angles, cam, splat)
        record = {
            "elevation_deg": elevation,
            "roll_deg": roll,
            "distance": distance,
            "tolerance_processing_px": tolerance,
            "allowed_contradictions": allowed,
            "inner_mean_iou": float(np.mean([s["iou"] for s in scores])),
            "coarse_voxels": int(selected.sum()),
        }
        candidates.append(record)
    best = max(candidates, key=lambda r: r["inner_mean_iou"])
    if args.refine:
        center = dict(best)
        nominal = angles.copy()
        # Low-dimensional periodic response: no per-view angle corrections.
        # Repeated outer-view development is disclosed, never confirmatory.
        for elevation, roll, a, b in itertools.product(
            np.arange(center["elevation_deg"] - 4, center["elevation_deg"] + 4.1, 2),
            np.arange(center["roll_deg"] - 2, center["roll_deg"] + 2.1, 1),
            [-0.06, 0.0, 0.06],
            [-0.06, 0.0, 0.06],
        ):
            test_angles = nominal + a * np.sin(nominal) + b * (np.cos(nominal) - 1)
            assert np.all(np.diff(np.r_[test_angles, 2 * np.pi]) > 0)
            cam = {
                **camera,
                "elevation_deg": float(elevation),
                "roll_deg": float(roll),
                "distance": 0.0,
            }
            selected, _, _ = envelope(
                coarse,
                masks,
                inner_train,
                test_angles,
                cam,
                center["tolerance_processing_px"],
                center["allowed_contradictions"],
                args.device,
            )
            scores = score_views(coarse[selected], masks, inner_val, test_angles, cam, splat)
            candidates.append(
                {
                    "elevation_deg": float(elevation),
                    "roll_deg": float(roll),
                    "distance": 0.0,
                    "warp_sin_rad": a,
                    "warp_cos_rad": b,
                    "tolerance_processing_px": center["tolerance_processing_px"],
                    "allowed_contradictions": center["allowed_contradictions"],
                    "inner_mean_iou": float(np.mean([s["iou"] for s in scores])),
                    "coarse_voxels": int(selected.sum()),
                }
            )
        best = max(candidates, key=lambda r: r["inner_mean_iou"])
        angles = (
            nominal
            + best.get("warp_sin_rad", 0.0) * np.sin(nominal)
            + best.get("warp_cos_rad", 0.0) * (np.cos(nominal) - 1)
        )
    print("Selected on INNER views only:", best, flush=True)
    # Freeze family above; this is a second exploratory development run after v1.
    camera, radius, ybounds, axis_audit = train_geometry(masks, train, angles)
    camera.update({k: best[k] for k in ["elevation_deg", "roll_deg", "distance"]})
    points, shape, origin, spacing = grid_points(radius, ybounds, args.resolution)
    splat = max(1, int(np.ceil(spacing * 0.6)))
    chosen, wrong, observable = envelope(
        points,
        masks,
        train,
        angles,
        camera,
        best["tolerance_processing_px"],
        best["allowed_contradictions"],
        args.device,
    )
    withheld_scores = score_views(points[chosen], masks, outer, angles, camera, splat)
    baseline_camera = {**camera, "elevation_deg": 0.0, "roll_deg": 0.0, "distance": 0.0}
    baseline, _, _ = envelope(points, masks, train, angles, baseline_camera, 2, 0, args.device)
    baseline_scores = score_views(points[baseline], masks, outer, angles, baseline_camera, splat)
    if not args.name.replace("_", "").isalnum():
        raise ValueError("Simple output name required")
    folder = OUT / args.name / args.session
    folder.mkdir(parents=True, exist_ok=True)
    # This strict comparator changes tilt AND tolerance/allowed contradictions;
    # matched_ablations.csv supplies the isolated camera/angle comparisons.
    write_csv(folder / "inner_candidates.csv", candidates)
    write_csv(
        folder / "heldout_scores.csv",
        [{"model": "inner_selected", **s} for s in withheld_scores]
        + [{"model": "zero_tilt_strict_selected_angle_schedule", **s} for s in baseline_scores],
    )
    ax, ay, az, _ = grid_axes(radius, ybounds, args.resolution)
    np.savez_compressed(
        folder / "heldout_envelope.npz",
        occupancy=chosen.reshape(shape),
        camera=json.dumps(camera),
        origin=origin,
        spacing=spacing,
        axis_x_coordinates=ax,
        axis_y_coordinates=ay,
        axis_z_coordinates=az,
        train_indices=train,
        heldout_indices=outer,
        contradictions=wrong.reshape(shape),
        observable=observable.reshape(shape),
    )
    overlays = make_overlays(folder, rows, masks, points[chosen], outer, angles, camera, splat)
    held_vertices, held_faces = mesh_export(
        folder, chosen, shape, origin, spacing, "heldout_envelope"
    )
    browser_mesh(
        folder,
        held_vertices,
        held_faces,
        rows,
        masks,
        angles,
        camera,
        "browser_mesh_heldout.json",
        train,
    )
    # Separate all-view refit for display, on the exact frozen camera/domain.
    all_indices = np.arange(40)
    final, wrong_all, obs_all = envelope(
        points,
        masks,
        all_indices,
        angles,
        camera,
        best["tolerance_processing_px"],
        best["allowed_contradictions"],
        args.device,
    )
    if not final.any():
        raise ValueError("No surviving geometry; preserve diagnostics rather than fabricate mesh")
    np.savez_compressed(
        folder / "allview_envelope.npz",
        occupancy=final.reshape(shape),
        camera=json.dumps(camera),
        origin=origin,
        spacing=spacing,
        axis_x_coordinates=ax,
        axis_y_coordinates=ay,
        axis_z_coordinates=az,
        contradictions=wrong_all.reshape(shape),
        observable=obs_all.reshape(shape),
    )
    vertices, faces = mesh_export(folder, final, shape, origin, spacing)
    browser_mesh(folder, vertices, faces, rows, masks, angles, camera, "browser_mesh.json")
    boundary = final.reshape(shape) & ~ndi.binary_erosion(final.reshape(shape))
    surface = points[boundary.ravel()]
    stride = max(1, int(np.ceil(len(surface) / 18000)))
    surface = surface[::stride]
    colours = colour_points(surface, rows, masks, angles, camera)
    payload = np.concatenate([surface, colours], axis=1).round(3).tolist()
    (folder / "surface_points.json").write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    np.save(folder / "angles_rad.npy", angles)
    summary = {
        "session": args.session,
        "method": "shared-camera censored silhouette tomography",
        "camera": camera,
        "angle_model": "theta_i=2*pi*i/40 + a*sin(theta_nominal) + b*(cos(theta_nominal)-1); a/b from selected_inner_config, default zero; not measured motor angles",
        "development_status": "Exploratory model family refined after earlier development results; heldout masks excluded from each fit, not an untouched confirmatory benchmark",
        "source_camera": "rgb",
        "train_indices": train.tolist(),
        "heldout_indices": outer.tolist(),
        "inner_train_indices": inner_train.tolist(),
        "inner_validation_indices": inner_val.tolist(),
        "selected_inner_config": best,
        "axis_fit": axis_audit,
        "processing_shape": list(masks[0].shape),
        "grid_shape": list(shape),
        "grid_origin": origin,
        "voxel_spacing_processing_px": spacing,
        "render_splat_radius_processing_px": splat,
        "domain_radius_processing_px": radius,
        "allview_voxels": int(final.sum()),
        "mesh_vertices": len(vertices),
        "mesh_faces": len(faces),
        "display_surface_points": len(surface),
        "heldout_mean_iou": float(np.mean([s["iou"] for s in withheld_scores])),
        "heldout_mean_precision": float(np.mean([s["precision"] for s in withheld_scores])),
        "heldout_mean_recall": float(np.mean([s["recall"] for s in withheld_scores])),
        "strict_orthographic_heldout_mean_iou": float(np.mean([s["iou"] for s in baseline_scores])),
        "allview_training_scores": score_views(
            points[final], masks, all_indices, angles, camera, splat
        ),
        "heldout_overlays": overlays,
        "runtime_seconds": time.time() - start,
        "source_hashes": [
            {
                "source": r["source"],
                "sha256": r["source_sha256"],
                "mask": r["mask"],
                "mask_sha256": hashlib.sha256((ROOT / r["mask"]).read_bytes()).hexdigest(),
            }
            for r in rows
        ],
        "limitations": [
            "Uncalibrated shared camera selected on inner silhouette prediction; not metrology",
            "Nominal one-turn angle hypothesis and handedness ambiguity",
            "Single RGB ring yields a silhouette envelope, not unique thin-leaf surfaces",
            "Mask labels are automatic SAM outputs, not manual ground truth",
            "Heldout views are from the same acquisition, not a new plant or independent physical reference",
            "Finite voxel grid, tolerance and splatting affect apparent thickness and surface area",
            "Outside-frame rays censored; finite domain caps are computational, never measured leaf boundaries",
            "Possible plant nonrigidity and perspective/axis mismatch remain",
        ],
        "status": "executed_heldout_view_consistency_experiment",
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ["source_hashes", "allview_training_scores", "limitations"]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
