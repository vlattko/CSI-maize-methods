"""Known-answer assisted inverse recovery, separate from archived end-to-end validation.

Known camera/angles/straight stem axis and four identity-labelled 2D midrib
paths (including occluded portions) are supplied. Truth controls are never
passed into initialization or fitting. No biological accuracy is implied.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
from bootstrap import BASE, OUT, write_csv
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from organ_inverse import project, scene, triangle_silhouette
from anatomical_ribbons import ribbon_faces
from pathlib import Path
import argparse, datetime, hashlib, json, time, traceback

cv2.setNumThreads(2)
torch.set_num_threads(2)
DEST = OUT / "leaf_dynamics/2026-10-24/jeb_v5/recovery"
SCALE = 100.0


def design():
    return {
        "seed": 20261024,
        "known_camera": True,
        "known_straight_stem_axis": True,
        "camera": {
            "axis_x": 80.0,
            "base_y": 119.0,
            "principal_x": 80.0,
            "principal_y": 64.0,
            "distance": 270.0,
            "elevation_deg": -4.0,
            "roll_deg": 3.0,
        },
        "shape": [128, 160],
        "views": 16,
        "train_views": [i for i in range(16) if i % 4 != 2],
        "test_views": [i for i in range(16) if i % 4 == 2],
        "assisted_path_views": [0, 4, 8, 12],
        "assisted_inputs": "Known IDs, four complete projected 2D midrib polylines including hidden portions; no point-to-point 3D correspondence supplied. Known camera, angles and straight stem axis.",
        "initialization": "2D observations only; no truth controls",
        "initial_width_controls": 3.0,
        "initial_twist": 0.0,
        "iterations": 350,
        "optimizer": "Adam control/attachment lr0.002, width/twist lr0.012, fixed final iterate",
        "fit_surface": [32, 5],
        "evaluation_surface": [64, 9],
        "fit_renderer": "organ_inverse.triangle_silhouette sigma0.3 radius5; leaf-only silhouette",
        "observation_renderer": "hard triangles OpenCV rounded integer vertices at128x160; leaf-only, stem omitted in all conditions",
        "perturbation": "assisted path iid Gaussian sigma0.65px; alternating one-pixel dilation/erosion; fixed 8x16 rectangular mask occlusion on every third fitting view",
        "evaluation": "Clean truth geometry and four untouched synthetic evaluation masks. Same withheld angles in all cases. No model/config selection.",
        "limitations": [
            "Not the archived full optimizer or end-to-end pipeline validation",
            "Known camera and angles; no organ discovery, correspondence or camera estimation",
            "Complete synthetic midrib paths including hidden portions are privileged assisted input",
            "Truth stem axis is known and straight; no stem error assessed",
            "Only four designed shoots and one perturbation per truth; no population inference",
            "Finite cubic fit family; out-of-family wavy truth deliberately retained",
        ],
        "cases": [
            {
                "id": f"{n}leaf_{family}_{noise}",
                "leaves": n,
                "truth_family": family,
                "condition": noise,
                "truth_seed": 0,
                "observation_seed": 20261024 + 100 * n + (family == "wavy") * 10,
            }
            for n in [3, 5]
            for family in ["cubic", "wavy"]
            for noise in ["clean", "perturbed"]
        ],
    }


def project_np(points, angles, camera):
    with torch.no_grad():
        return project(
            torch.tensor(points, dtype=torch.float64),
            torch.tensor(angles, dtype=torch.float64),
            camera,
        ).numpy()


def make_truth(n, family, seed=0):
    attachments = np.linspace(13.0, 66.0, n)
    cps = []
    widths = []
    twists = []
    for i, y in enumerate(attachments):
        a = i * 2.39996323 + 0.22 * (n - 3)
        radial = np.array([np.cos(a), 0.0, np.sin(a)])
        delta = radial * (43 + 4 * np.sin(i + 0.4)) + np.array([0.0, 28.0 - 4 * i, 0.0])
        base = np.array([0.0, y, 0.0])
        cps.append(
            [base, base + 0.26 * delta + [0, 18, 0], base + 0.72 * delta + [0, 15, 0], base + delta]
        )
        widths.append(np.array([4.0, 6.0, 3.0]) * (0.83 + 0.09 * i))
        twists.append([0.12 * i, -0.35 + 0.1 * i, 0.25 - 0.05 * i])
    cp = torch.tensor(np.array(cps) / SCALE, dtype=torch.float64)
    w = torch.tensor(np.array(widths) / SCALE, dtype=torch.float64)
    tw = torch.tensor(np.array(twists), dtype=torch.float64)
    from anatomical_ribbons import cubic_ribbons

    v, mid, faces = cubic_ribbons(cp, w, tw, 64, 9)
    v = v.numpy() * SCALE
    mid = mid.numpy() * SCALE
    if family == "wavy":
        t = np.linspace(0, 1, 64)
        u = np.linspace(-1, 1, 9)
        for i in range(n):
            a = i * 2.39996323 + 0.22 * (n - 3)
            transverse = np.array([-np.sin(a), 0.0, np.cos(a)])
            warp = 3.0 * np.sin(4 * np.pi * t) * np.sin(np.pi * t)
            change = warp[:, None] * transverse
            mid[i] += change
            v[i] += change[:, None, :]
            ripple = (
                1.4 * np.sin(6 * np.pi * t)[:, None] * (u * u)[None, :] * np.sin(np.pi * t)[:, None]
            )
            v[i, :, :, 1] += ripple
    return {
        "vertices": v,
        "midribs": mid,
        "faces": faces,
        "generating_control_points": np.array(cps),
        "width_controls": np.array(widths),
        "twists": np.array(twists),
    }


def render_masks(vertices, angles, camera, shape):
    n, s, c, _ = vertices.shape
    faces = ribbon_faces(s, c)
    xy = project_np(vertices.reshape(-1, 3), angles, camera).reshape(len(angles), n, s * c, 2)
    masks = np.zeros((len(angles), *shape), np.uint8)
    for k in range(len(angles)):
        for leaf in range(n):
            for f in faces:
                cv2.fillConvexPoly(masks[k], np.rint(xy[k, leaf, f]).astype(np.int32), 1)
    return masks


def resample_path(p, n=32):
    dist = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1).cumsum()]
    return np.stack(
        [np.interp(np.linspace(0, dist[-1], n), dist, p[:, j]) for j in range(2)], axis=-1
    )


def observations_from_truth(generated, case, d):
    angles = np.arange(16) * 2 * np.pi / 16
    pv = d["assisted_path_views"]
    mids = generated["midribs"]
    paths = project_np(mids.reshape(-1, 3), angles[pv], d["camera"]).reshape(4, len(mids), 64, 2)
    paths = np.array([[resample_path(p) for p in frame] for frame in paths])
    clean = render_masks(generated["vertices"], angles, d["camera"], d["shape"])
    observed = clean[d["train_views"]].copy()
    if case["condition"] == "perturbed":
        rng = np.random.default_rng(case["observation_seed"])
        paths += rng.normal(0, 0.65, paths.shape)
        for k, view in enumerate(d["train_views"]):
            observed[k] = (
                cv2.dilate(observed[k], np.ones((3, 3), np.uint8))
                if k % 2 == 0
                else cv2.erode(observed[k], np.ones((3, 3), np.uint8))
            )
            if k % 3 == 0:
                observed[k, 50:66, 72:80] = 0
    return {
        "path_views": np.array(pv),
        "paths": paths,
        "train_views": np.array(d["train_views"]),
        "masks": observed,
        "angles": angles,
    }, clean


def tiny_observations(camera):
    angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    mid = np.linspace([0, 20, 0], [20, 60, 12], 32)[None]
    paths = project_np(mid.reshape(-1, 3), angles, camera).reshape(4, 1, 32, 2)
    return {"paths": paths, "angles": angles, "path_views": np.arange(4)}


def initialize_from_observations(observations, camera):
    paths = observations["paths"]
    angles = observations["angles"][observations["path_views"]]
    control = []
    attached = []
    for i in range(paths.shape[1]):
        tipxy = paths[:, i, -1]
        startxy = paths[:, i, 0]
        tip = least_squares(
            lambda xyz: (project_np(xyz[None], angles, camera)[:, 0] - tipxy).ravel(),
            [0.0, 60.0, 0.0],
            max_nfev=100,
        ).x
        y = least_squares(
            lambda yy: (
                project_np(np.array([[0.0, float(yy[0]), 0.0]]), angles, camera)[:, 0] - startxy
            ).ravel(),
            [25.0],
            max_nfev=50,
        ).x[0]
        base = np.array([0.0, y, 0.0])
        delta = tip - base
        control.append([base + 0.28 * delta + [0, 8, 0], base + 0.68 * delta + [0, 6, 0], tip])
        attached.append(y)
    return {
        "control": np.array(control),
        "attachment": np.array(attached),
        "widths": np.full((len(control), 3), 3.0),
        "twists": np.zeros((len(control), 3)),
    }


def surface_from_parameters(parameters, samples=64, cross=9):
    args = [
        torch.tensor(parameters[k], dtype=torch.float64)
        for k in ["control", "attachment", "widths", "twists"]
    ]
    control, attachment, widths, twists = args
    with torch.no_grad():
        v, mid, cp, stem, faces = scene(
            control / SCALE,
            attachment / SCALE,
            torch.zeros(4, dtype=torch.float64),
            widths / SCALE,
            twists,
            SCALE,
            samples,
            cross,
        )
    return v.numpy(), mid.numpy(), faces


def fit_case(observations, camera, initial, iterations, device):
    cp = torch.nn.Parameter(
        torch.tensor(initial["control"] / SCALE, dtype=torch.float32, device=device)
    )
    at = torch.nn.Parameter(
        torch.tensor(initial["attachment"] / SCALE, dtype=torch.float32, device=device)
    )
    wr = torch.nn.Parameter(
        torch.tensor(
            np.log(np.expm1(initial["widths"] / SCALE)), dtype=torch.float32, device=device
        )
    )
    tw = torch.nn.Parameter(torch.tensor(initial["twists"], dtype=torch.float32, device=device))
    opt = torch.optim.Adam([{"params": [cp, at], "lr": 0.002}, {"params": [wr, tw], "lr": 0.012}])
    angles = torch.tensor(
        observations["angles"][observations["train_views"]], dtype=torch.float32, device=device
    )
    pathangles = torch.tensor(
        observations["angles"][observations["path_views"]], dtype=torch.float32, device=device
    )
    target = torch.tensor(observations["masks"], dtype=torch.float32, device=device)
    paths = torch.tensor(observations["paths"], dtype=torch.float32, device=device)
    n = paths.shape[1]
    leaf_faces = ribbon_faces(32, 5)
    faces = np.concatenate([leaf_faces + i * 32 * 5 for i in range(n)])
    history = []
    area = target.sum((1, 2)).clamp_min(1)
    for step in range(iterations):
        opt.zero_grad(set_to_none=True)
        v, mid, _, _, _ = scene(
            cp, at, torch.zeros(4, device=device), F.softplus(wr), tw, SCALE, 32, 5
        )
        batch = (torch.arange(4, device=device) + 4 * step) % len(angles)
        xy = project(v.reshape(-1, 3), angles[batch], camera)
        prob = triangle_silhouette(xy, faces, target.shape[1:], sigma=0.3, radius=5).clamp(
            1e-5, 1 - 1e-5
        )
        image = (
            (
                -(target[batch] * prob.log()).sum((1, 2))
                - 0.8 * ((1 - target[batch]) * torch.log1p(-prob)).sum((1, 2))
            )
            .div(area[batch])
            .mean()
        )
        projected = project(mid.reshape(-1, 3), pathangles, camera).reshape(
            len(pathangles), n, 32, 2
        )
        distances = torch.cdist(projected.reshape(-1, 32, 2), paths.reshape(-1, 32, 2)).square()
        path = (distances.min(-1).values.mean() + distances.min(-2).values.mean()) / 9
        endpoints = ((projected[:, :, [0, -1]] - paths[:, :, [0, -1]]).square().mean()) / 9
        regular = tw.square().mean() * 0.001
        loss = image + path + 0.3 * endpoints + regular
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite fitting loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([cp, at, wr, tw], 20)
        opt.step()
        with torch.no_grad():
            at.clamp_(0.01, 1.1)
            wr.clamp_(-6, -1.7)
            tw.clamp_(-2, 2)
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach()),
                "image": float(image.detach()),
                "path": float(path.detach()),
                "endpoints": float(endpoints.detach()),
            }
        )
    return {
        "control": cp.detach().cpu().numpy() * SCALE,
        "attachment": at.detach().cpu().numpy() * SCALE,
        "widths": F.softplus(wr).detach().cpu().numpy() * SCALE,
        "twists": tw.detach().cpu().numpy(),
    }, history


def geometry_metrics(pred, pmid, ref, rmid, faces):
    def areas(v):
        tri = v.reshape(len(v), -1, 3)[:, faces]
        return 0.5 * np.linalg.norm(
            np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0]), axis=-1
        ).sum(-1)

    def length(x):
        return np.linalg.norm(np.diff(x, axis=1), axis=-1).sum(-1)

    def width(x):
        return np.linalg.norm(x[:, :, -1] - x[:, :, 0], axis=-1).max(-1)

    pa, ra = areas(pred), areas(ref)
    pl, rl = length(pmid), length(rmid)
    pw, rw = width(pred), width(ref)
    distances = []
    for a, b in zip(pred, ref):
        a = a.reshape(-1, 3)
        b = b.reshape(-1, 3)
        distances.append((cKDTree(a).query(b)[0].mean() + cKDTree(b).query(a)[0].mean()) / 2)
    return {
        "surface_symmetric_distance": float(np.mean(distances)),
        "surface_symmetric_distance_per_leaf": list(map(float, distances)),
        "mean_area_relative_abs_error": float(np.mean(abs(pa - ra) / ra)),
        "mean_midrib_length_relative_abs_error": float(np.mean(abs(pl - rl) / rl)),
        "mean_width_relative_abs_error": float(np.mean(abs(pw - rw) / rw)),
        "mean_tip_distance": float(np.linalg.norm(pmid[:, -1] - rmid[:, -1], axis=-1).mean()),
        "mean_midrib_symmetric_distance": float(
            np.mean(
                [
                    (cKDTree(a).query(b)[0].mean() + cKDTree(b).query(a)[0].mean()) / 2
                    for a, b in zip(pmid, rmid)
                ]
            )
        ),
        "area_per_leaf": pa.tolist(),
        "truth_area_per_leaf": ra.tolist(),
        "midrib_length_per_leaf": pl.tolist(),
        "truth_midrib_length_per_leaf": rl.tolist(),
        "width_per_leaf": pw.tolist(),
        "truth_width_per_leaf": rw.tolist(),
    }


def simple_test_surface():
    mid = np.linspace([0, 0, 0], [0, 10, 0], 8)[None]
    v = mid[:, :, None, :] + np.array([-1, 0, 1])[None, None, :, None] * np.array([1, 0, 0])
    return v, mid, ribbon_faces(8, 3)


def mask_metrics(pred, ref):
    i = (pred.astype(bool) & ref.astype(bool)).sum((1, 2))
    u = (pred.astype(bool) | ref.astype(bool)).sum((1, 2))
    return i / np.maximum(u, 1)


def release_matrix(vertices):
    """Nine designed beams, hard barycentric ownership, common fixed square gauge."""
    n, s, c, _ = vertices.shape
    faces = ribbon_faces(s, c)
    effects = []
    directions = [np.array([0.0, 1.0, 0.0])] + [
        np.array([np.cos(a), 1.0, np.sin(a)]) / np.sqrt(2) for a in np.arange(8) * np.pi / 4
    ]
    for direction in directions:
        right = np.cross(direction, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, direction)
        xy = np.stack([vertices @ right, vertices @ up], axis=-1) * 1.28 + 128
        depth = vertices @ direction
        grids = np.full((n, 256, 256), -np.inf)
        for k in range(n):
            uv = xy[k].reshape(-1, 2)
            z = depth[k].ravel()
            for f in faces:
                tri = uv[f]
                lo = np.maximum(np.floor(tri.min(0)).astype(int), 0)
                hi = np.minimum(np.ceil(tri.max(0)).astype(int), 255)
                if np.any(lo > hi):
                    continue
                a, b, c0 = tri
                ab = b - a
                ac = c0 - a
                det = ab[0] * ac[1] - ab[1] * ac[0]
                if abs(det) < 1e-8:
                    continue
                yy, xx = np.mgrid[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
                q = np.stack([xx, yy], axis=-1) - a
                v = (q[..., 0] * (c0 - a)[1] - q[..., 1] * (c0 - a)[0]) / det
                w = ((b - a)[0] * q[..., 1] - (b - a)[1] * q[..., 0]) / det
                u = 1 - v - w
                valid = (u >= 0) & (v >= 0) & (w >= 0)
                zz = u * z[f[0]] + v * z[f[1]] + w * z[f[2]]
                patch = grids[k, lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
                np.maximum(patch, np.where(valid, zz, -np.inf), out=patch)
        visible = np.isfinite(grids)
        full = grids.argmax(0)
        fullvalid = visible.any(0)
        mat = np.zeros((n, n))
        for j in range(n):
            deleted = grids.copy()
            deleted[j] = -np.inf
            owner = deleted.argmax(0)
            valid = np.isfinite(deleted).any(0)
            for i in range(n):
                if i != j:
                    mat[j, i] = np.sum(valid & (owner == i) & ~(fullvalid & (full == i))) / max(
                        1, visible[i].sum()
                    )
        effects.append(mat)
    return np.mean(effects, axis=0), np.array(effects)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run():
    d = design()
    DEST.mkdir(parents=True, exist_ok=True)
    pre = DEST / "predeclared_design.json"
    if pre.exists() and json.loads(pre.read_text()) != d:
        raise RuntimeError("predeclared design differs; do not silently overwrite")
    pre.write_text(json.dumps(d, indent=2))
    started = utc()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    print("BEGIN", started, "device", device, "cases", len(d["cases"]), flush=True)
    with threadpool_limits(limits=2):
        for case in d["cases"]:
            folder = DEST / case["id"]
            folder.mkdir(exist_ok=True)
            t = time.time()
            print("CASE", case["id"], flush=True)
            try:
                generated = make_truth(case["leaves"], case["truth_family"], case["truth_seed"])
                obs, clean = observations_from_truth(generated, case, d)
                initial = initialize_from_observations(obs, d["camera"])
                np.savez_compressed(
                    folder / "synthetic_inputs.npz",
                    **{"truth_" + k: v for k, v in generated.items()},
                    **{"observed_" + k: v for k, v in obs.items()},
                    clean_masks=clean,
                )
                np.savez_compressed(folder / "initial_parameters.npz", **initial)
                fitted, history = fit_case(obs, d["camera"], initial, d["iterations"], device)
                np.savez_compressed(folder / "fitted_parameters.npz", **fitted)
                write_csv(folder / "loss.csv", history)
                allgeo = {}
                arrays = {}
                releases = {}
                truth_release, truth_beams = release_matrix(generated["vertices"])
                for label, pars in [("initial", initial), ("fitted", fitted)]:
                    v, mid, faces = surface_from_parameters(pars)
                    arrays[label + "_vertices"] = v
                    arrays[label + "_midribs"] = mid
                    metrics = geometry_metrics(
                        v, mid, generated["vertices"], generated["midribs"], faces
                    )
                    masks = render_masks(v, obs["angles"], d["camera"], d["shape"])
                    ious = mask_metrics(masks, clean)
                    release, beams = release_matrix(v)
                    releases[label] = release
                    metrics.update(
                        {
                            "mean_train_clean_mask_iou": float(ious[d["train_views"]].mean()),
                            "mean_withheld_clean_mask_iou": float(ious[d["test_views"]].mean()),
                            "per_view_clean_mask_iou": ious.tolist(),
                            "release_mean_abs_error": float(
                                abs(release - truth_release)[
                                    ~np.eye(case["leaves"], dtype=bool)
                                ].mean()
                            ),
                            "release_max_abs_error": float(abs(release - truth_release).max()),
                        }
                    )
                    arrays[label + "_masks"] = masks
                    arrays[label + "_beam_release"] = beams
                    allgeo[label] = metrics
                arrays["truth_release"] = truth_release
                arrays["truth_beam_release"] = truth_beams
                np.savez_compressed(folder / "recovery_arrays.npz", **arrays)
                for j in d["test_views"]:
                    cv2.imwrite(
                        str(folder / f"withheld_{j:02d}_truth_initial_fitted.png"),
                        np.concatenate(
                            [clean[j], arrays["initial_masks"][j], arrays["fitted_masks"][j]],
                            axis=1,
                        )
                        * 255,
                    )
                row = {
                    "case": case,
                    "status": "completed",
                    "metrics": allgeo,
                    "seconds": time.time() - t,
                    "units": "Known synthetic world gauge with parameter scale100, not actual shoot height or physical dimensions; relative errors dimensionless",
                    "start_or_end_note": "All specified fits retained; fixed last iterate, no tuning by evaluation.",
                }
                (folder / "summary.json").write_text(json.dumps(row, indent=2))
                results.append(row)
                print(
                    "RESULT",
                    case["id"],
                    {
                        k: {
                            m: round(vv[m], 4)
                            for m in [
                                "surface_symmetric_distance",
                                "mean_area_relative_abs_error",
                                "mean_withheld_clean_mask_iou",
                                "release_mean_abs_error",
                            ]
                        }
                        for k, vv in allgeo.items()
                    },
                    "seconds",
                    round(time.time() - t, 1),
                    flush=True,
                )
            except Exception as e:
                row = {
                    "case": case,
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                results.append(row)
                (folder / "summary.json").write_text(json.dumps(row, indent=2))
                print(row["traceback"], flush=True)
    inputs = {
        str(p.relative_to(BASE)).replace("\\", "/"): sha(p)
        for p in [
            Path(__file__),
            BASE / "test_v5_recovery_benchmark.py",
            BASE / "organ_inverse.py",
            BASE / "anatomical_ribbons.py",
            BASE / "bootstrap.py",
        ]
    }
    summary = {
        "title": "Known-answer assisted inverse recovery",
        "status": "completed"
        if all(x["status"] == "completed" for x in results)
        else "completed_with_failures",
        "started_utc": started,
        "ended_utc": utc(),
        "design": d,
        "input_sha256": inputs,
        "device": device,
        "torch": torch.__version__,
        "results": results,
        "claim": "Recovery of a declared cubic surface fitting configuration under known cameras and privileged supplied 2D organ paths. Not independent real-plant anatomy or validation of the archived full pipeline.",
    }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2))
    print("END", summary["ended_utc"], summary["status"], flush=True)


if __name__ == "__main__":
    run()
