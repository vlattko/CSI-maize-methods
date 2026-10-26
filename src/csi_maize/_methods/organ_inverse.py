"""Explicit-organ inverse graphics; assisted identities are preserved as inputs.

This fits smooth leaf sheets and a common stem directly to training images.
No voxel or opacity-field surface is reused as the leaf representation.
"""

from bootstrap import BASE, ROOT, OUT, read_csv, write_csv, preferred_device
from anatomical_ribbons import cubic_ribbons
import argparse, json, hashlib, time
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F

cv2.setNumThreads(2)


def project(points, angles, camera):
    """One shared camera; points in processing-pixel world gauge, Y upward."""
    c = angles.cos()[:, None]
    s = angles.sin()[:, None]
    xx = c * points[None, :, 0] + s * points[None, :, 2]
    zz = -s * points[None, :, 0] + c * points[None, :, 2]
    e = np.radians(camera["elevation_deg"])
    r = np.radians(camera["roll_deg"])
    cr, sr = np.cos(r), np.sin(r)
    up = np.cos(e) * points[None, :, 1] + np.sin(e) * zz
    distance = camera.get("distance", 0.0)
    if distance:
        cx = camera["principal_x"]
        cy = camera["principal_y"]
        dx = camera["axis_x"] - cx
        dy = camera["base_y"] - cy
        tx = cr * dx + sr * dy
        ty = sr * dx - cr * dy
        xx = xx + tx
        up = up + ty
        depth = distance + np.cos(e) * zz - np.sin(e) * points[None, :, 1]
        return torch.stack(
            [
                cx + distance * (cr * xx + sr * up) / depth.clamp_min(1),
                cy + distance * (sr * xx - cr * up) / depth.clamp_min(1),
            ],
            dim=-1,
        )
    return torch.stack(
        [camera["axis_x"] + cr * xx + sr * up, camera["base_y"] + sr * xx - cr * up], dim=-1
    )


def soft_silhouette(xy, weights, shape):
    """Differentiable position splats; this is a surface quadrature renderer."""
    b, n, _ = xy.shape
    h, w = shape
    x = xy[..., 0]
    y = xy[..., 1]
    ix = x.floor().long()
    iy = y.floor().long()
    fx = x - ix
    fy = y - iy
    result = xy.new_zeros(b * h * w)
    batch = torch.arange(b, device=xy.device)[:, None] * h * w
    for dx, dy, weight in [
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ]:
        xx = ix + dx
        yy = iy + dy
        valid = (xx >= 0) & (yy >= 0) & (xx < w) & (yy < h)
        index = batch + yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1)
        result = result.scatter_add(0, index.ravel(), (weight * weights * valid).ravel())
    # A fixed image footprint accommodates finite surface quadrature spacing.
    kernel = xy.new_tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])[None, None] / 16
    mass = F.conv2d(result.reshape(b, 1, h, w), kernel, padding=1)[:, 0]
    return -torch.expm1(-mass)


def triangle_silhouette(vertices, faces, shape, sigma=0.15, radius=5):
    """Differentiable triangle coverage on local pixel tiles, no opacity field.

    The local tile is centred on each projected triangle's bounding box. Every
    in-frame integer pixel centre in a nondegenerate triangle's bounding box
    must fit in that tile; otherwise raise rather than truncate its interior.
    The sigmoid outside the triangle remains a finite-tile approximation.
    Degenerate tip triangles are explicitly omitted.
    """
    b, n, _ = vertices.shape
    h, w = shape
    f = torch.as_tensor(faces, device=vertices.device, dtype=torch.long)
    tri = vertices[:, f]
    lo = tri.amin(2)
    hi = tri.amax(2)
    center = ((lo + hi) / 2).floor()
    offset = torch.arange(-radius, radius + 1, device=vertices.device)
    oy, ox = torch.meshgrid(offset, offset, indexing="ij")
    pixels = center[:, :, None, :] + torch.stack([ox.ravel(), oy.ravel()], dim=-1)[None, None]
    p = pixels + 0.0
    a = tri[:, :, 0, None]
    q = tri[:, :, 1, None]
    c = tri[:, :, 2, None]

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    area = cross(q - a, c - a)
    sign = torch.sign(area)
    # Check the exact integer bbox used by hard coverage, rather than requiring
    # fractional/offscreen vertices to lie inside the tile. This single detached
    # reduction synchronizes CUDA once, without adding a gradient dependency.
    with torch.no_grad():
        pixel_lo = lo.ceil().clamp_min(0)
        pixel_hi = torch.minimum(hi.floor(), vertices.new_tensor([w - 1, h - 1]))
        in_frame = (pixel_lo <= pixel_hi).all(-1)
        truncated = ((pixel_lo < center - radius) | (pixel_hi > center + radius)).any(-1)
        if bool((in_frame & truncated & (area[..., 0].abs() > 1e-4)).any()):
            raise ValueError(
                "Projected triangle exceeds its local raster tile; increase tessellation or radius"
            )
    distances = []
    for first, second in [(a, q), (q, c), (c, a)]:
        edge = second - first
        distances.append(
            cross(edge, p - first) * sign / torch.linalg.vector_norm(edge, dim=-1).clamp_min(1e-5)
        )
    inside = torch.stack(distances, dim=-1).amin(-1)
    coverage = torch.sigmoid(inside / sigma) * (area.abs() > 1e-4)
    ix = pixels[..., 0].long()
    iy = pixels[..., 1].long()
    valid = (ix >= 0) & (iy >= 0) & (ix < w) & (iy < h)
    index = (
        torch.arange(b, device=vertices.device)[:, None, None] * h * w
        + iy.clamp(0, h - 1) * w
        + ix.clamp(0, w - 1)
    )
    mass = vertices.new_zeros(b * h * w).scatter_add(
        0, index.ravel(), (-torch.log1p(-coverage.clamp_max(1 - 1e-5)) * valid).ravel()
    )
    return -torch.expm1(-mass.reshape(b, h, w))


def read_observations():
    path = OUT / "organ_observations/capture_session_C2_5-1/observations.json"
    data = json.loads(path.read_text())
    ids = [r["leaf_id"] for r in data["leaves"]]
    lookup = {k: i for i, k in enumerate(ids)}
    observations = []
    for view in data["views"]:
        for leaf in view["leaves"]:
            if leaf["association_status"] == "provisionally_consistent" or view["view"] in [0, 20]:
                observations.append(
                    {**leaf, "view": view["view"], "index": lookup[leaf["leaf_id"]]}
                )
    return data, ids, observations


def camera_and_tips(data, observations):
    independent = OUT / "organ_observations" / data["session"] / "camera_tip_fit.json"
    if independent.exists():
        result = json.loads(independent.read_text())
        best = next(r for r in result["results"] if r["model"] == "perspective_image_center")
        source = OUT / "organ_observations" / data["session"] / "observations.json"
        assert (
            result["source_observations_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        )
        camera = {
            **data["camera"],
            "principal_x": 512.0,
            "principal_y": 384.5,
            "elevation_deg": best["camera"]["elevation_deg"],
            "distance": best["camera"]["distance"],
        }
        return (
            camera,
            np.array([r["tip_xyz"] for r in best["leaves"]]),
            {
                "independent_mentor_camera_comparison": result,
                "camera": camera,
                "principal_point": "Assumed image center, not measured",
                "world_scale": "f=D gauge, not physical units",
            },
        )
    # Fallback also permits standalone execution before the independent mentor fit.
    camera = {**data["camera"], "principal_x": 512.0, "principal_y": 384.5}
    tips = []
    for leaf in data["leaves"]:
        coef = np.array(leaf["trajectory_coefficients_derolled_xy"])
        tips.append([coef[1, 0], coef[0, 1], coef[2, 0]])
    tips = np.array(tips)
    angles = np.array([r["view"] * 2 * np.pi / 40 for r in observations])
    ids = np.array([r["index"] for r in observations])
    truth = np.array([r["tip_xy"] for r in observations])

    def prediction(parameter):
        e, d = parameter[:2]
        points = parameter[2:].reshape(-1, 3)[ids]
        c = np.cos(angles)
        s = np.sin(angles)
        xx = c * points[:, 0] + s * points[:, 2]
        zz = -s * points[:, 0] + c * points[:, 2]
        er = np.radians(e)
        r = np.radians(camera["roll_deg"])
        cr, sr = np.cos(r), np.sin(r)
        up = np.cos(er) * points[:, 1] + np.sin(er) * zz
        depth = d + np.cos(er) * zz - np.sin(er) * points[:, 1]
        tx = cr * (camera["axis_x"] - 512) + sr * (768 - 384.5)
        ty = sr * (camera["axis_x"] - 512) - cr * (768 - 384.5)
        xx += tx
        up += ty
        return np.stack(
            [
                512 + d * (cr * xx + sr * up) / np.maximum(depth, 20),
                384.5 + d * (sr * xx - cr * up) / np.maximum(depth, 20),
            ],
            axis=1,
        )

    fits = []
    for distance, elevation in [(650.0, 0.0), (1100.0, 5.0), (1800.0, 10.0)]:
        p = np.r_[elevation, distance, tips.ravel()]
        fit = least_squares(
            lambda p: (prediction(p) - truth).ravel(),
            p,
            bounds=(
                np.r_[-25.0, 400.0, np.tile([-1500.0, 0.0, -1500.0], len(tips))],
                np.r_[25.0, 3000.0, np.tile([1500.0, 1600.0, 1500.0], len(tips))],
            ),
            loss="soft_l1",
            f_scale=5,
            max_nfev=400,
        )
        fits.append(fit)
    fit = min(fits, key=lambda f: f.cost)
    camera.update(elevation_deg=float(fit.x[0]), distance=float(fit.x[1]))
    tips = fit.x[2:].reshape(-1, 3)
    residual = np.linalg.norm(prediction(fit.x) - truth, axis=1)
    audit = {
        "tip_observations": len(observations),
        "median_tip_error_px": float(np.median(residual)),
        "rmse_tip_error_px": float(np.sqrt(np.mean(residual**2))),
        "camera": camera,
        "principal_point": "Assumed image center; not independently measured",
        "world_scale": "f=D gauge; processing pixels, not physical units",
        "initializations": [
            {"cost": float(f.cost), "elevation_deg": float(f.x[0]), "distance": float(f.x[1])}
            for f in fits
        ],
    }
    return camera, tips, audit


def sample_polyline(path, n):
    p = np.array(path)
    d = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1).cumsum()]
    return np.stack([np.interp(np.linspace(0, d[-1], n), d, p[:, j]) for j in range(2)], axis=1)


def scene(control, attachment, stem_shape, widths, twists, scale, samples=72, cross=11):
    # All C0 positions are constrained to the same continuous stem.
    y = attachment
    x = stem_shape[0] * y + stem_shape[1] * y * y
    z = stem_shape[2] * y + stem_shape[3] * y * y
    base = torch.stack([x, y, z], dim=-1)
    cp = torch.cat([base[:, None], control], dim=1)
    vertices, midribs, faces = cubic_ribbons(cp, widths, twists, samples, cross)
    sy = torch.linspace(0, 1, 100, device=control.device) * attachment.max() * 1.03
    sx = stem_shape[0] * sy + stem_shape[1] * sy * sy
    sz = stem_shape[2] * sy + stem_shape[3] * sy * sy
    angle = torch.linspace(0, 2 * torch.pi, 13, device=control.device)[:-1]
    radius = 0.0055 * (1 - 0.45 * sy / sy.max().clamp_min(0.01))
    stem = torch.stack(
        [
            sx[:, None] + radius[:, None] * angle.cos(),
            sy[:, None].expand(-1, 12),
            sz[:, None] + radius[:, None] * angle.sin(),
        ],
        dim=-1,
    )
    return vertices * scale, midribs * scale, cp * scale, stem * scale, faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=900)
    parser.add_argument("--name", default="organ_inverse")
    parser.add_argument("--triangles", action="store_true")
    parser.add_argument("--warm-start", action="store_true")
    args = parser.parse_args()
    start = time.time()
    torch.manual_seed(201)
    torch.set_num_threads(4)
    device = preferred_device()
    scale = 768.0
    data, ids, observations = read_observations()
    camera, tips, audit = camera_and_tips(data, observations)
    folder = OUT / args.name / data["session"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "camera_fit.json").write_text(json.dumps(audit, indent=2))
    print("Perspective camera", camera, flush=True)
    train = np.array(data["train_indices"])
    heldout = np.array(data["excluded_indices"])
    angles = torch.tensor(train * 2 * np.pi / 40, dtype=torch.float32, device=device)
    rows = [
        r
        for r in read_csv(OUT / "sam_traits/view_traits.csv")
        if r["session"].endswith(data["session"]) and r["camera"] == "rgb"
    ]
    rows.sort(key=lambda r: int(r["view"]))
    shape = (192, 256)
    masks = np.array(
        [
            cv2.resize(
                np.array(Image.open(ROOT / rows[i]["mask"])),
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
            for i in train
        ]
    )
    target = torch.tensor(masks, dtype=torch.float32, device=device)
    nfg = target.sum((1, 2)).clamp_min(1)
    signed = np.array(
        [distance_transform_edt(~m) - distance_transform_edt(m) for m in masks], np.float32
    )
    sdf = torch.tensor(signed[:, None], device=device)
    # Invert seed-node projection onto the stem-axis plane; image height is not 3D Y.
    attached = []
    cr = np.cos(np.radians(camera["roll_deg"]))
    sr = np.sin(np.radians(camera["roll_deg"]))
    ce = np.cos(np.radians(camera["elevation_deg"]))
    se = np.sin(np.radians(camera["elevation_deg"]))
    d = camera["distance"]
    ty = sr * (camera["axis_x"] - 512) - cr * (768 - 384.5)
    for leaf_id in ids:
        ys = []
        for record in observations:
            if record["leaf_id"] == leaf_id and record["view"] in [0, 20]:
                u, v = record["attachment_xy"]
                up = sr * (u - 512) - cr * (v - 384.5)
                ys.append(d * (up - ty) / (d * ce + up * se))
        attached.append(np.mean(ys) / scale)
    attached = np.array(attached)
    attachment = torch.nn.Parameter(torch.tensor(attached, dtype=torch.float32, device=device))
    stem_shape = torch.nn.Parameter(torch.zeros(4, device=device))
    initial = []
    for tip, height in zip(tips, attached):
        p0 = np.array([0, height * scale, 0])
        delta = tip - p0
        initial.append(
            [
                p0 + delta * 0.28 + np.array([0, min(60, np.linalg.norm(delta) * 0.15), 0]),
                p0 + delta * 0.68 + np.array([0, min(50, np.linalg.norm(delta) * 0.1), 0]),
                tip,
            ]
        )
    control = torch.nn.Parameter(
        torch.tensor(np.array(initial) / scale, dtype=torch.float32, device=device)
    )
    w0 = (
        np.array([[14, 21, 10], [12, 18, 10], [18, 26, 17], [15, 26, 17], [5, 8, 5]], np.float32)
        / scale
    )
    width_raw = torch.nn.Parameter(torch.tensor(np.log(np.expm1(w0)), device=device))
    twists = torch.nn.Parameter(torch.zeros((len(ids), 3), device=device))
    if args.warm_start:
        saved = np.load(OUT / "organ_inverse" / data["session"] / "organ_model.npz")
        previous = saved["control_points"] / scale
        y = previous[:, 0, 1]
        basis = np.stack([y, y * y], axis=1)
        xz = np.r_[
            np.linalg.lstsq(basis, previous[:, 0, 0], rcond=None)[0],
            np.linalg.lstsq(basis, previous[:, 0, 2], rcond=None)[0],
        ]
        with torch.no_grad():
            attachment.copy_(torch.tensor(y, device=device))
            stem_shape.copy_(torch.tensor(xz, device=device))
            control.copy_(torch.tensor(previous[:, 1:], device=device))
            width_raw.copy_(
                torch.tensor(np.log(np.expm1(saved["width_controls"] / scale)), device=device)
            )
            twists.copy_(torch.tensor(saved["twist_controls"], device=device))
    optimizer = torch.optim.Adam([control, attachment, stem_shape, width_raw, twists], lr=0.003)
    view_lookup = {int(v): i for i, v in enumerate(train)}
    obs_leaf = torch.tensor([r["index"] for r in observations], device=device)
    obs_view = torch.tensor([view_lookup[r["view"]] for r in observations], device=device)
    obs_tip = torch.tensor(
        np.array([r["tip_xy"] for r in observations]), dtype=torch.float32, device=device
    )
    obs_path = torch.tensor(
        np.array([sample_polyline(r["polyline_xy"], 32) for r in observations]),
        dtype=torch.float32,
        device=device,
    )
    history = []
    original_attached = attachment.detach().clone()
    source_tips = torch.tensor(tips, dtype=torch.float32, device=device)
    for step in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        widths = F.softplus(width_raw)
        leaves, mid, cp, stem, faces = scene(control, attachment, stem_shape, widths, twists, scale)
        b = (torch.arange(6, device=device) + step * 6) % len(train)
        theta = angles[b]
        leaf_xy = project(leaves.reshape(-1, 3), theta, camera).reshape(len(b), len(ids), 72, 11, 2)
        leaf_small = leaf_xy * leaf_xy.new_tensor([shape[1] / 1024, shape[0] / 769])
        # Local projected quadrature area, including foreshortening.
        ds = leaf_small[:, :, 2:, 1:-1] - leaf_small[:, :, :-2, 1:-1]
        dc = leaf_small[:, :, 1:-1, 2:] - leaf_small[:, :, 1:-1, :-2]
        area = (ds[..., 0] * dc[..., 1] - ds[..., 1] * dc[..., 0]).abs() / 4
        area = F.pad(area, (1, 1, 1, 1), mode="replicate").reshape(len(b), -1).clamp(0.015, 6) * 3
        stem_xy = project(stem.reshape(-1, 3), theta, camera) * leaf_xy.new_tensor(
            [shape[1] / 1024, shape[0] / 769]
        )
        xy = torch.cat([leaf_small.reshape(len(b), -1, 2), stem_xy], dim=1)
        weight = torch.cat([area, torch.ones(stem_xy.shape[:2], device=device) * 0.3], dim=1)
        if args.triangles:
            sampled = leaves[:, ::2][:, :, [0, 2, 5, 8, 10]]
            ss, cc = sampled.shape[1:3]
            from anatomical_ribbons import ribbon_faces

            lf = ribbon_faces(ss, cc)
            all_faces = np.concatenate([lf + i * ss * cc for i in range(len(ids))])
            screen = project(sampled.reshape(-1, 3), theta, camera) * xy.new_tensor(
                [shape[1] / 1024, shape[0] / 769]
            )
            leaf_probability = triangle_silhouette(screen, all_faces, shape)
            stem_probability = soft_silhouette(
                stem_xy, torch.ones(stem_xy.shape[:2], device=device) * 0.5, shape
            )
            probability = (1 - (1 - leaf_probability) * (1 - stem_probability)).clamp(
                1e-5, 1 - 1e-5
            )
        else:
            probability = soft_silhouette(xy, weight, shape).clamp(1e-5, 1 - 1e-5)
        positive = -(target[b] * probability.log()).sum((1, 2)) / nfg[b]
        negative = -((1 - target[b]) * torch.log1p(-probability)).sum((1, 2)) / nfg[b]
        image_loss = (positive + 0.8 * negative).mean()
        # Identity-bound projected midribs keep organs from exchanging targets.
        all_mid_xy = project(mid.reshape(-1, 3), angles, camera).reshape(
            len(train), len(ids), 72, 2
        )
        predicted = all_mid_xy[obs_view, obs_leaf]
        dist = torch.cdist(predicted / scale, obs_path / scale)
        path_loss = (dist.min(-1).values.mean() + dist.min(-2).values.mean()) * 12
        tip_loss = F.smooth_l1_loss(predicted[:, -1] / 12, obs_tip / 12, beta=1) * 0.6
        # Surface SDF gives useful coordinate gradients even far outside masks.
        grid = xy.clone()
        grid[..., 0] = grid[..., 0] * 2 / (shape[1] - 1) - 1
        grid[..., 1] = grid[..., 1] * 2 / (shape[0] - 1) - 1
        distances = F.grid_sample(sdf[b], grid[:, None], align_corners=True, padding_mode="border")[
            :, 0, 0
        ]
        visible = (grid.abs() <= 1).all(-1)
        outside = (F.relu(distances - 1.0) * visible).sum() / visible.sum().clamp_min(1) / 20
        curvature = (
            (
                (cp[:, 0] - 2 * cp[:, 1] + cp[:, 2]).square().sum()
                + (cp[:, 1] - 2 * cp[:, 2] + cp[:, 3]).square().sum()
            )
            / scale**2
            * 0.03
        )
        prior = (
            (attachment - original_attached).square().mean() * 8
            + stem_shape.square().mean() * 0.1
            + twists.square().mean() * 0.002
        )
        loss = image_loss + path_loss + tip_loss + outside + curvature + prior
        loss.backward()
        torch.nn.utils.clip_grad_norm_([control, attachment, stem_shape, width_raw, twists], 10)
        optimizer.step()
        with torch.no_grad():
            attachment.clamp_(0.02, 1.1)
            width_raw.clamp_(min=-8, max=-1.8)
            twists.clamp_(-2, 2)
        history.append(
            {
                "iteration": step + 1,
                "loss": float(loss.detach()),
                "image": float(image_loss.detach()),
                "path": float(path_loss.detach()),
                "tip": float(tip_loss.detach()),
                "outside": float(outside.detach()),
            }
        )
        if step % 100 == 0:
            print(history[-1], flush=True)
    with torch.no_grad():
        leaves, mid, cp, stem, faces = scene(
            control, attachment, stem_shape, F.softplus(width_raw), twists, scale, 96, 13
        )
    np.savez_compressed(
        folder / "organ_model.npz",
        control_points=cp.cpu().numpy(),
        width_controls=F.softplus(width_raw).detach().cpu().numpy() * scale,
        twist_controls=twists.detach().cpu().numpy(),
        vertices=leaves.cpu().numpy(),
        midribs=mid.cpu().numpy(),
        stem=stem.cpu().numpy(),
        stem_shape=stem_shape.detach().cpu().numpy(),
        camera=json.dumps(camera),
        train_indices=train,
        heldout_indices=heldout,
    )
    write_csv(folder / "training_loss.csv", history)
    export_model(
        folder,
        leaves.cpu().numpy(),
        mid.cpu().numpy(),
        cp.cpu().numpy(),
        stem.cpu().numpy(),
        faces,
        camera,
        rows,
        ids,
        data,
        train,
        heldout,
        audit,
    )
    summary_path = folder / "summary.json"
    saved_summary = json.loads(summary_path.read_text())
    saved_summary["fit_run"] = {
        "iterations": args.iterations,
        "renderer": "differentiable local triangle coverage"
        if args.triangles
        else "differentiable surface quadrature splats",
        "training_resolution": [256, 192],
        "warm_start": args.warm_start,
        "seed": 201,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU",
    }
    summary_path.write_text(json.dumps(saved_summary, indent=2))
    print("Completed explicit organs", folder, "seconds", time.time() - start, flush=True)


def export_model(
    folder, leaves, mid, cp, stem, faces, camera, rows, ids, data, train, heldout, audit
):
    from anatomical_ribbons import ribbon_faces
    from turntable_tomography import mask_scores
    import trimesh

    colors = np.array(
        [
            [244, 188, 71],
            [67, 208, 188],
            [240, 118, 138],
            [138, 170, 249],
            [201, 137, 234],
            [111, 156, 92],
        ]
    )
    vertices = np.concatenate([leaves.reshape(-1, 3), stem.reshape(-1, 3)])
    mesh_faces = np.concatenate([faces + i * np.prod(leaves.shape[1:3]) for i in range(len(ids))])
    # Wrapped cylindrical stem faces.
    stem_faces = []
    offset = leaves.reshape(-1, 3).shape[0]
    for j in range(stem.shape[0] - 1):
        for k in range(stem.shape[1]):
            a = offset + j * 12 + k
            b = offset + (j + 1) * 12 + k
            c = offset + (j + 1) * 12 + (k + 1) % 12
            d = offset + j * 12 + (k + 1) % 12
            stem_faces.extend([[a, b, c], [a, c, d]])
    mesh_faces = np.concatenate([mesh_faces, np.array(stem_faces)])
    tri = vertices[mesh_faces]
    nondegenerate = (
        np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) > 1e-6
    )
    mesh_faces = mesh_faces[nondegenerate]
    vcolors = np.concatenate(
        [np.repeat(colors[i, None], np.prod(leaves.shape[1:3]), axis=0) for i in range(len(ids))]
        + [np.repeat(colors[-1, None], stem.reshape(-1, 3).shape[0], axis=0)]
    )
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=mesh_faces, vertex_colors=vcolors, process=False
    )
    mesh.export(folder / "organ_plant.ply")
    mesh.export(folder / "organ_plant.obj")
    # Compatibility is not calibrated confidence, and does not establish organ identity.
    hits = np.zeros(len(vertices), int)
    seen = np.zeros(len(vertices), int)
    for view in train:
        xy = project(
            torch.tensor(vertices, dtype=torch.float32),
            torch.tensor([view * 2 * np.pi / 40]),
            camera,
        )[0].numpy()
        u, v = np.rint(xy).astype(int).T
        valid = (u >= 0) & (u < 1024) & (v >= 0) & (v < 769)
        truth = np.array(Image.open(ROOT / rows[view]["mask"])) > 0
        seen += valid
        ix = np.flatnonzero(valid)
        hits[ix] += truth[v[ix], u[ix]]
    compatibility = hits / np.maximum(seen, 1)
    payload = {
        "vertices": vertices.round(4).tolist(),
        "faces": mesh_faces.tolist(),
        "colors": vcolors.tolist(),
        "organs": [],
        "camera": camera,
        "foreground_compatibility": compatibility.round(4).tolist(),
        "support_note": "Fraction of in-frame TRAIN projections compatible with the whole-plant SAM mask; not per-leaf truth or confidence",
        "units": "processing-pixel gauge; no metric calibration",
        "representation": "five assisted-ID smooth leaf hypotheses with shared stem",
    }
    traits = []
    for i, leaf_id in enumerate(ids):
        length = float(np.linalg.norm(np.diff(mid[i], axis=0), axis=1).sum())
        width = float(np.linalg.norm(leaves[i, :, -1] - leaves[i, :, 0], axis=1).max())
        direction = mid[i, 3] - mid[i, 0]
        nearest = int(np.argmin(abs(stem[:, :, 1].mean(axis=1) - cp[i, 0, 1])))
        before = max(0, nearest - 1)
        after = min(len(stem) - 1, nearest + 1)
        stem_tangent = stem[after].mean(axis=0) - stem[before].mean(axis=0)
        leaf_tangent = cp[i, 1] - cp[i, 0]
        angle = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(stem_tangent, leaf_tangent)
                        / (np.linalg.norm(stem_tangent) * np.linalg.norm(leaf_tangent)),
                        -1,
                        1,
                    )
                )
            )
        )
        start = int(i * np.prod(leaves.shape[1:3]))
        end = int((i + 1) * np.prod(leaves.shape[1:3]))
        seed_junctions = []
        for view_record in data["views"]:
            if view_record["view"] not in [0, 20]:
                continue
            observed = next((r for r in view_record["leaves"] if r["leaf_id"] == leaf_id), None)
            if observed is None:
                continue
            prediction = project(
                torch.tensor(cp[i, :1], dtype=torch.float32),
                torch.tensor([view_record["view"] * 2 * np.pi / 40]),
                camera,
            )[0, 0].numpy()
            seed_junctions.append(
                {
                    "view": view_record["view"],
                    "observed_2d_junction_xy": observed["attachment_xy"],
                    "model_attachment_projected_xy": prediction.tolist(),
                    "residual_px": float(
                        np.linalg.norm(prediction - np.array(observed["attachment_xy"]))
                    ),
                    "junction_status": observed["attachment_status"],
                    "botanical_attachment_measured": False,
                }
            )
        trait = {
            "leaf_id": leaf_id,
            "model_midrib_length_px": length,
            "model_max_width_px": width,
            "model_attachment_y_px": float(cp[i, 0, 1]),
            "model_insertion_angle_deg": angle,
            "seed_junction_mean_residual_px": float(
                np.mean([r["residual_px"] for r in seed_junctions])
            ),
            "train_foreground_compatibility_mean": float(compatibility[start:end].mean()),
            "status": "prior-conditioned fitted organ; insertion and length are not independently measured traits",
        }
        traits.append(trait)
        payload["organs"].append(
            {
                **trait,
                "seed_junction_diagnostics": seed_junctions,
                "junction_note": "Observed 2D skeleton junctions can be caused by occlusion, not botanical insertion. Large residual flags a disagreement; small residual does not validate insertion.",
                "vertex_start": int(i * np.prod(leaves.shape[1:3])),
                "vertex_count": int(np.prod(leaves.shape[1:3])),
                "midrib": mid[i].round(3).tolist(),
            }
        )
    write_csv(folder / "organ_traits.csv", traits)
    (folder / "browser_mesh.json").write_text(json.dumps(payload, separators=(",", ":")))
    # Actual triangle rasterization for evaluation, unlike the training splat approximation.
    all_scores = []
    for view in range(40):
        xy = project(
            torch.tensor(vertices, dtype=torch.float32),
            torch.tensor([view * 2 * np.pi / 40]),
            camera,
        )[0].numpy()
        mask = np.zeros((769, 1024), np.uint8)
        for triangle in mesh_faces:
            cv2.fillConvexPoly(mask, np.rint(xy[triangle]).astype(np.int32), 1)
        truth = np.array(Image.open(ROOT / rows[view]["mask"])) > 0
        score = {
            "view": view,
            "split": "train" if view in train else "development_heldout",
            **mask_scores(mask > 0, truth),
        }
        all_scores.append(score)
        im = np.array(
            Image.open(ROOT / rows[view]["source"])
            .convert("RGB")
            .resize((1024, 769), Image.Resampling.LANCZOS)
        )
        overlay = im.copy()
        for i in range(len(ids)):
            line = project(
                torch.tensor(mid[i], dtype=torch.float32),
                torch.tensor([view * 2 * np.pi / 40]),
                camera,
            )[0].numpy()
            cv2.polylines(
                overlay, [np.rint(line).astype(np.int32)], False, tuple(map(int, colors[i])), 2
            )
            p = tuple(np.rint(line[-1]).astype(int))
            cv2.putText(
                overlay, ids[i], p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(map(int, colors[i])), 1
            )
        compare = im.copy()
        compare[(mask > 0) & truth] = (
            0.4 * compare[(mask > 0) & truth] + 0.6 * np.array([64, 210, 150])
        ).astype(np.uint8)
        compare[(mask > 0) & ~truth] = [242, 138, 55]
        compare[(mask == 0) & truth] = [212, 67, 140]
        panel = Image.new("RGB", (2048, 811), "#142b20")
        panel.paste(Image.fromarray(overlay), (0, 42))
        panel.paste(Image.fromarray(compare), (1024, 42))
        ImageDraw.Draw(panel).text(
            (12, 12),
            f"VIEW {view:02d} {score['split']} | FIVE ASSISTED ORGAN HYPOTHESES | triangle IoU {score['iou']:.3f}",
            fill="white",
        )
        panel.save(folder / f"view_{view:02d}.jpg", quality=91)
    write_csv(folder / "view_scores.csv", all_scores)
    summary = {
        "session": data["session"],
        "method": "assisted explicit-organ perspective inverse graphics",
        "leaf_hypotheses": len(ids),
        "camera": camera,
        "camera_fit": audit,
        "train_indices": train.tolist(),
        "heldout_indices": heldout.tolist(),
        "triangle_render_development_heldout": {
            k: float(np.mean([r[k] for r in all_scores if r["split"] == "development_heldout"]))
            for k in ["iou", "precision", "recall"]
        },
        "manual_initialization": data["initialization"],
        "mesh_vertices": len(vertices),
        "mesh_faces": len(mesh_faces),
        "limitations": [
            "Five IDs initialized in two source views; not fully automated leaf discovery",
            "Projected skeleton midlines are not measured veins",
            "Perspective intrinsics/axis assumptions are fitted, not independently calibrated",
            "Continuous leaf sheets fill missing evidence through a prior; inferred hidden portions are not newly observed",
            "Masks and evaluation views reused during development; no independent dimensional validation",
            "Model-derived length/width/angles are conditional processing-scale quantities",
        ],
        "observations_sha256": hashlib.sha256(
            (OUT / "organ_observations" / data["session"] / "observations.json").read_bytes()
        ).hexdigest(),
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ["camera_fit", "manual_initialization"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
