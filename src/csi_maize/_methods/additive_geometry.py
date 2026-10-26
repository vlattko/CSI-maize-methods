"""Execute four bounded additive geometry experiments on saved source evidence.

CPU-only, <=3 native/PyTorch threads. No fitting updates existing model files.
All inferred quantities retain the camera/identity/scale assumptions of inputs.
"""

from common import workspace_relative

import os

for _name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "3"
from bootstrap import ROOT, BASE, OUT, read_csv, write_csv
from pathlib import Path
import argparse, datetime, hashlib, itertools, json, sys, time
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
import torch
from anatomical_ribbons import cubic_ribbons
from organ_camera import project as camera_project

cv2.setNumThreads(3)
torch.set_num_threads(3)
DEST = OUT / "additive_experiments/2026-10-24"
MODEL = OUT / "organ_inverse_triangles/capture_session_C2_5-1"
OBS = OUT / "organ_observations/capture_session_C2_5-1/observations.json"
CAMFIT = OBS.parent / "camera_tip_fit.json"
FOLDERS = {"E05": "e05_mechanics", "E07": "e07_light", "E08": "e08_ambiguity", "E09": "e09_views"}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return workspace_relative(Path(path))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def event(folder, stage, **data):
    with (folder / "execution.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(utc=utc(), stage=stage, **data), allow_nan=False) + "\n")


def inputs(paths):
    paths = list(paths) + [
        BASE / "additive_geometry.py",
        BASE / "test_additive_geometry.py",
        BASE / "organ_camera.py",
        BASE / "anatomical_ribbons.py",
    ]
    return {relative(p): sha(p) for p in dict.fromkeys(paths)}


def finish(
    folder,
    identifier,
    title,
    status,
    question,
    methods,
    metrics,
    findings,
    limitations,
    next_requirements,
    source_paths,
):
    result = dict(
        id=identifier,
        title=title,
        status=status,
        question=question,
        methods=methods,
        metrics=metrics,
        findings=findings,
        limitations=limitations,
        next_requirements=next_requirements,
        input_sha256=inputs(source_paths),
        artifacts=[],
    )
    event(folder, "completed", id=identifier, status=status, metrics=metrics)
    result["artifacts"] = [relative(p) for p in sorted(folder.glob("*")) if p.is_file()] + [
        relative(folder / "summary.json")
    ]
    result["artifacts"] = list(dict.fromkeys(result["artifacts"]))
    dump(folder / "summary.json", result)
    print(
        json.dumps(dict(id=identifier, status=status, metrics=metrics), allow_nan=False), flush=True
    )
    return result


def load_geometry():
    saved = np.load(MODEL / "organ_model.npz")
    mesh = json.loads((MODEL / "browser_mesh.json").read_text())
    camera = json.loads(str(saved["camera"]))
    vertices = np.concatenate([saved["vertices"].reshape(-1, 3), saved["stem"].reshape(-1, 3)])
    faces = np.array(mesh["faces"], int)
    organs = np.full(len(vertices), 5, int)
    for i, o in enumerate(mesh["organs"]):
        organs[o["vertex_start"] : o["vertex_start"] + o["vertex_count"]] = i
    return saved, mesh, camera, vertices, faces, organs


def directional_raster(vertices, faces, face_organs, direction, resolution=384):
    """Opaque, double-sided orthographic projected interception, source at +n.

    Pixel-center barycentric depth interpolation; the nearest surface toward
    the light wins. Areas retain squared input units. Footprints are per-organ
    independent projected unions, not lamina surface areas.
    """
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, int)
    face_organs = np.asarray(face_organs, int)
    n = np.asarray(direction, float)
    n /= np.linalg.norm(n)
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    xy = np.column_stack([vertices @ u, vertices @ v])
    depth = vertices @ n
    span = float(np.max(np.ptp(xy, axis=0)))
    step = max(span / (resolution - 4), 1e-12)
    origin = (xy.min(0) + xy.max(0)) / 2 - step * resolution / 2
    screen = (xy - origin) / step - 0.5
    z = np.full((resolution, resolution), -np.inf)
    owner = np.full(z.shape, -1, int)
    count = int(face_organs.max()) + 1
    footprints = np.zeros((count, resolution, resolution), bool)
    for face, organ in zip(faces, face_organs):
        triangle = screen[face]
        lo = np.maximum(np.ceil(triangle.min(0)).astype(int), 0)
        hi = np.minimum(np.floor(triangle.max(0)).astype(int), resolution - 1)
        if np.any(hi < lo):
            continue
        a, b, c = triangle
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12:
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        wa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / den
        wb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / den
        wc = 1 - wa - wb
        inside = (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9)
        if not inside.any():
            continue
        iz = wa * depth[face[0]] + wb * depth[face[1]] + wc * depth[face[2]]
        ys = yy[inside]
        xs = xx[inside]
        newz = iz[inside]
        footprints[organ, ys, xs] = True
        # Use the same transitive order as layered compositing: strict maximum
        # source-facing depth, then smaller organ ID on an exact depth tie.
        closer = (newz > z[ys, xs]) | ((newz == z[ys, xs]) & (organ < owner[ys, xs]))
        z[ys[closer], xs[closer]] = newz[closer]
        owner[ys[closer], xs[closer]] = organ
    projected = np.array([m.sum() * step**2 for m in footprints])
    intercepted = np.array([(owner == i).sum() * step**2 for i in range(count)])
    return dict(
        intercepted_area=float((owner >= 0).sum() * step**2),
        organ_intercepted_area=intercepted.tolist(),
        organ_unoccluded_projected_area=projected.tolist(),
        owner=owner,
        pixel_area=step**2,
    )


def projection(points, angles, camera):
    return camera_project(
        points,
        angles,
        camera["elevation_deg"],
        camera["distance"],
        camera["roll_deg"],
        (camera["principal_x"], camera["principal_y"]),
        (camera["axis_x"], camera["base_y"]),
    )


def triangulate_tip(theta, xy, camera):
    theta = np.asarray(theta, float)
    xy = np.asarray(xy, float)
    if len(theta) < 2:
        return dict(status="insufficient_views", point=None)
    cr, sr = np.cos(np.radians(camera["roll_deg"])), np.sin(np.radians(camera["roll_deg"]))
    ce, se = (
        np.cos(np.radians(camera["elevation_deg"])),
        np.sin(np.radians(camera["elevation_deg"])),
    )
    d = camera["distance"]
    matrix = []
    right = []
    for a, p in zip(theta, xy):
        c, s = np.cos(a), np.sin(a)
        xx = np.array([c, 0, s])
        up = np.array([-se * s, ce, se * c])
        depth = np.array([-ce * s, -se, ce * c])
        un = (p[0] - camera["principal_x"]) / d
        vn = (p[1] - camera["principal_y"]) / d
        matrix.extend([cr * xx + sr * up - un * depth, sr * xx - cr * up - vn * depth])
        right.extend(
            [
                un * d - (camera["axis_x"] - camera["principal_x"]),
                vn * d - (camera["base_y"] - camera["principal_y"]),
            ]
        )
    matrix = np.array(matrix)
    initial, _, rank, _ = np.linalg.lstsq(matrix, np.array(right), rcond=None)
    if rank < 3:
        return dict(status="rank_deficient", point=None)
    fit = least_squares(
        lambda p: (projection(np.tile(p, (len(theta), 1)), theta, camera)[0] - xy).ravel(),
        initial,
        loss="soft_l1",
        f_scale=5,
        max_nfev=150,
    )
    return dict(
        status="fitted", point=fit.x.tolist(), condition_number=float(np.linalg.cond(matrix))
    )


def select_views(pool, k, strategy, visibility=None):
    pool = sorted(map(int, pool))
    k = min(k, len(pool))
    if strategy == "uniform":
        selected = []
        for target in np.linspace(0, 40, k, endpoint=False):
            candidate = min(
                [i for i in pool if i not in selected],
                key=lambda i: (min(abs(i - target), 40 - abs(i - target)), i),
            )
            selected.append(candidate)
        return sorted(selected)
    if strategy != "preview_balanced":
        raise ValueError(strategy)
    n = len(visibility[pool[0]])
    information = np.tile(np.eye(3)[None] * 0.05, (n, 1, 1))
    selected = []
    while len(selected) < k:
        choices = []
        for i in pool:
            if i in selected:
                continue
            a = i * 2 * np.pi / 40
            v = np.array([1, np.cos(a), np.sin(a)])
            addition = np.outer(v, v)
            updated = information + np.asarray(visibility[i])[:, None, None] * addition
            gain = float(
                np.linalg.slogdet(updated)[1].sum() - np.linalg.slogdet(information)[1].sum()
            )
            choices.append((gain, -i, i, updated))
        best = max(choices, key=lambda x: (x[0], x[1]))
        selected.append(best[2])
        information = best[3]
    return sorted(selected)


def perturb_controls(
    cp, widths, twists, stem_shape, width_scale, twist_delta, attachment_delta, bow_delta
):
    cp = np.array(cp, float).copy()
    widths = np.array(widths, float) * width_scale
    twists = np.array(twists, float) + twist_delta
    old = cp[:, 0].copy()
    height = old[:, 1] + attachment_delta
    y = height / 768.0
    new = np.column_stack(
        [
            (stem_shape[0] * y + stem_shape[1] * y * y) * 768,
            height,
            (stem_shape[2] * y + stem_shape[3] * y * y) * 768,
        ]
    )
    delta = new - old
    cp[:, 0] = new
    cp[:, 1] += delta * 0.65
    cp[:, 2] += delta * 0.25
    cp[:, 1:3, 1] += bow_delta
    return cp, widths, twists


def accept_training(score, baseline, tolerance):
    return bool(score >= baseline - tolerance - 1e-12)


def view_common_residuals(records):
    """No residual exists for a view with no observed tips; never invent one."""
    return {
        v: np.median([[r["dx_px"], r["dy_px"]] for r in records if r["view"] == v], axis=0)
        for v in sorted({r["view"] for r in records})
    }


def e05(folder):
    saved, mesh, camera, vertices, faces, organs = load_geometry()
    data = json.loads(OBS.read_text())
    ids = [l["leaf_id"] for l in data["leaves"]]
    tip_fit = json.loads(CAMFIT.read_text())
    camera_points = {r["leaf_id"]: r["tip_xyz"] for r in tip_fit["results"][1]["leaves"]}
    records = []
    for view in data["views"]:
        for leaf in view["leaves"]:
            i = ids.index(leaf["leaf_id"])
            theta = view["nominal_angle_rad"]
            truth = np.array(leaf["tip_xy"])
            pred, _ = projection(saved["control_points"][i, 3][None], np.array([theta]), camera)
            rigid, _ = projection(
                np.array(camera_points[leaf["leaf_id"]])[None], np.array([theta]), camera
            )
            residual = truth - pred[0]
            records.append(
                dict(
                    view=view["view"],
                    leaf_id=leaf["leaf_id"],
                    angle_rad=theta,
                    dx_px=float(residual[0]),
                    dy_px=float(residual[1]),
                    norm_px=float(np.linalg.norm(residual)),
                    conditional_camera_tip_residual_px=float(np.linalg.norm(truth - rigid[0])),
                    association_status=leaf["association_status"],
                    uncertainty_flags=";".join(leaf["uncertainty_flags"]),
                )
            )
    view_common = view_common_residuals(records)
    for r in records:
        common = view_common[r["view"]]
        r["within_view_common_removed_norm_px"] = float(
            np.linalg.norm(np.array([r["dx_px"], r["dy_px"]]) - common)
        )
    write_csv(folder / "organ_view_residuals.csv", records)
    per_leaf = []
    for lid in ids:
        rows = [r for r in records if r["leaf_id"] == lid]
        values = np.array([r["norm_px"] for r in rows])
        xy = np.array([[r["dx_px"], r["dy_px"]] for r in rows])
        theta = np.array([r["angle_rad"] for r in rows])
        a = np.column_stack([np.ones(len(rows)), np.cos(theta), np.sin(theta)])
        fit = a @ np.linalg.lstsq(a, xy, rcond=None)[0]
        per_leaf.append(
            dict(
                leaf_id=lid,
                n=len(rows),
                median_px=float(np.median(values)),
                rmse_px=float(np.sqrt(np.mean(values**2))),
                max_px=float(values.max()),
                fitted_first_harmonic_residual_rms_px=float(np.sqrt(np.mean(fit**2))),
            )
        )
    write_csv(folder / "per_organ_diagnostics.csv", per_leaf)
    protocol = dict(
        status="required_not_acquired",
        record=[
            "stationary high-frame-rate bursts before and after known gentle move",
            "rigid fiducials fixed to pot and lab; actual angle and exposure timestamps",
            "matched clockwise/counterclockwise repeats; repeated zero and full-turn closure",
            "several dwell times, fixed exposure/focus/lights, no-move repeats",
            "leaf length/width and mass/boundary-condition evidence before interpreting stiffness",
        ],
        falsification=[
            "Motion shared by markers is camera/table motion, not leaf flexibility.",
            "Nonrepeatable or association-dependent residuals do not identify compliance.",
            "Damping/frequency needs temporal samples; archived one-image-per-stop residuals cannot identify stiffness.",
        ],
    )
    dump(folder / "required_mechanics_protocol.json", protocol)
    source_paths = [OBS, CAMFIT, MODEL / "organ_model.npz"]
    for source in data["inputs"]:
        source_paths.extend([ROOT / source["source"], ROOT / source["mask"]])
    norms = np.array([r["norm_px"] for r in records])
    return finish(
        folder,
        "E05",
        "Rigid organ residuals and a falsifiable mechanics protocol",
        "partial_requires_new_data",
        "Which organ/view residuals merit a future dynamics experiment?",
        dict(
            observations=86,
            camera="fixed supplied perspective camera",
            angle_model="nominal uniform one turn",
            common_motion_diagnostic="Within-view median residual; diagnostic only, not independently measured camera motion",
        ),
        dict(
            observations=len(records),
            organs=len(ids),
            views_with_observed_tips=len(view_common),
            views_without_observed_tips=sorted(set(data["train_indices"]) - set(view_common)),
            median_residual_px=float(np.median(norms)),
            rmse_residual_px=float(np.sqrt(np.mean(norms**2))),
            max_residual_px=float(norms.max()),
            per_organ=per_leaf,
        ),
        [
            "Archive residuals are quantified per organ and view; competing explanations remain camera, association, mask and deformation errors.",
            "A within-view common-residual subtraction is descriptive and does not establish a mechanical signal.",
        ],
        [
            "No stationary video, known excitation or independently measured rigidity in these stills.",
            "No stiffness, damping, turgor or causal stress response measured.",
        ],
        protocol["record"],
        source_paths,
    )


def e07(folder):
    saved, mesh, camera, vertices, faces, organs = load_geometry()
    height = float(np.ptp(vertices[:, 1]))
    normalized = vertices / height
    face_organs = organs[faces[:, 0]]
    directions = [("zenith", np.array([0.0, 1.0, 0.0]))]
    for elevation in [25, 60]:
        for azimuth in range(0, 360, 45):
            e, a = np.radians([elevation, azimuth])
            directions.append(
                (
                    f"e{elevation}_a{azimuth}",
                    np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)]),
                )
            )
    records = []
    totals = []
    panels = []
    for name, direction in directions:
        result = directional_raster(normalized, faces, face_organs, direction, resolution=384)
        for i in range(6):
            unoccluded = result["organ_unoccluded_projected_area"][i]
            hit = result["organ_intercepted_area"][i]
            records.append(
                dict(
                    direction=name,
                    organ="stem" if i == 5 else f"leaf_{i + 1}",
                    intercepted_area_H2=hit,
                    unoccluded_projected_footprint_H2=unoccluded,
                    surviving_interorgan_fraction=hit / unoccluded if unoccluded > 0 else None,
                    fraction_of_total_interception=hit / result["intercepted_area"],
                )
            )
        totals.append(
            dict(
                direction=name,
                source_direction_xyz=direction.tolist(),
                total_intercepted_area_H2=result["intercepted_area"],
            )
        )
        palette = np.array(
            [
                [245, 190, 70],
                [65, 210, 190],
                [240, 115, 140],
                [140, 170, 250],
                [205, 135, 235],
                [110, 155, 90],
            ],
            np.uint8,
        )
        owner = result["owner"]
        image = np.full((*owner.shape, 3), [20, 40, 30], np.uint8)
        valid = owner >= 0
        image[valid] = palette[owner[valid]]
        image = Image.fromarray(image).resize((240, 240))
        panels.append((name, image))
        event(
            folder, "light_direction", name=name, total_interception_H2=result["intercepted_area"]
        )
    write_csv(folder / "organ_interception.csv", records)
    dump(folder / "direction_totals.json", totals)
    montage = Image.new("RGB", (5 * 240, 4 * 270), "#14281e")
    draw = ImageDraw.Draw(montage)
    for j, (name, image) in enumerate(panels):
        x = (j % 5) * 240
        y = (j // 5) * 270
        montage.paste(image, (x, y + 25))
        draw.text((x + 8, y + 7), name, fill="white")
    montage.save(folder / "directional_interception.png")
    areas = [r["total_intercepted_area_H2"] for r in totals]
    return finish(
        folder,
        "E07",
        "Conditional directional light interception and self-shading",
        "completed_exploratory",
        "How does the fitted seedling distribute interception of a directional beam among its organs?",
        dict(
            resolution=384,
            directions=len(directions),
            source_at_positive_direction=True,
            normalization="Coordinates divided by full model vertical extent H; area in H squared",
            radiative_model="uniform collimated unit irradiance, opaque double-sided leaf sheets and opaque stem, nearest surface wins",
        ),
        dict(
            directions=len(directions),
            normalizing_model_height_px_gauge=height,
            min_intercepted_area_H2=min(areas),
            max_intercepted_area_H2=max(areas),
            max_to_min_ratio=max(areas) / min(areas),
        ),
        [
            "Different source directions change both total projected interception and its per-organ distribution.",
            "Projected union footprints and frontmost per-organ interception are retained separately, including occlusion losses.",
        ],
        [
            "One prior-conditioned seedling, not a calibrated canopy.",
            "Opaque and double-sided assumptions omit reflection, transmission, scattering and spectral leaf optics.",
            "Interception is not absorbed PAR, photosynthesis, biomass or a measured physiological response.",
            "Finite raster and inferred surfaces affect occlusion.",
        ],
        [
            "Calibrated light source/irradiance and measured leaf reflectance/transmittance for absorption estimates.",
            "Independent geometry and repeated seedlings for architectural conclusions.",
        ],
        [MODEL / "organ_model.npz", MODEL / "browser_mesh.json"],
    )


def render_mask(vertices, faces, angle, camera, shape=(385, 512)):
    xy, depth = projection(vertices, np.full(len(vertices), angle), camera)
    if not (depth > 0).all():
        raise ValueError("Alternative intersects camera plane")
    xy = (xy + 0.5) * np.array([shape[1] / 1024, shape[0] / 769]) - 0.5
    xy = np.rint(xy).astype(np.int32)
    result = np.zeros(shape, np.uint8)
    for f in faces:
        cv2.fillConvexPoly(result, xy[f], 1)
    return result > 0


def metrics(predicted, truth):
    intersection = int((predicted & truth).sum())
    return dict(
        iou=intersection / max(1, int((predicted | truth).sum())),
        precision=intersection / max(1, int(predicted.sum())),
        recall=intersection / max(1, int(truth.sum())),
    )


def mean_metric(records):
    return {k: float(np.mean([r[k] for r in records])) for k in ["iou", "precision", "recall"]}


def model_traits(cp, widths, twists, stem_shape):
    with torch.no_grad():
        leaves, mid, _ = cubic_ribbons(
            torch.tensor(cp, dtype=torch.float32),
            torch.tensor(widths, dtype=torch.float32),
            torch.tensor(twists, dtype=torch.float32),
            96,
            13,
        )
    leaves = leaves.numpy()
    mid = mid.numpy()
    out = []
    for i in range(5):
        y = cp[i, 0, 1] / 768
        st = np.array(
            [stem_shape[0] + 2 * stem_shape[1] * y, 1, stem_shape[2] + 2 * stem_shape[3] * y]
        )
        lt = cp[i, 1] - cp[i, 0]
        angle = np.degrees(
            np.arccos(np.clip(st @ lt / np.linalg.norm(st) / np.linalg.norm(lt), -1, 1))
        )
        out.append(
            dict(
                leaf_id=f"leaf_{i + 1}",
                midrib_length_px_gauge=float(np.linalg.norm(np.diff(mid[i], axis=0), axis=1).sum()),
                max_width_px_gauge=float(
                    np.linalg.norm(leaves[i, :, -1] - leaves[i, :, 0], axis=1).max()
                ),
                attachment_Y_px_gauge=float(cp[i, 0, 1]),
                basal_model_angle_deg=float(angle),
            )
        )
    return leaves, out


def e08(folder):
    saved, mesh, camera, vertices, faces, organs = load_geometry()
    data = json.loads(OBS.read_text())
    train = data["train_indices"]
    test = data["excluded_indices"]
    rows = [
        r
        for r in read_csv(OUT / "sam_traits/view_traits.csv")
        if r["session"].endswith(data["session"]) and r["camera"] == "rgb"
    ]
    rows.sort(key=lambda r: int(r["view"]))
    shape = (385, 512)
    train_masks = {
        i: cv2.resize(
            np.asarray(Image.open(ROOT / rows[i]["mask"])),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        > 0
        for i in train
    }
    # Fixed before evaluating alternatives. Baseline, eight one-factor perturbations, sixteen joint corners.
    configs = [(1.0, 0.0, 0.0, 0.0)]
    deltas = [(0.85, 1.15), (-0.18, 0.18), (-6.0, 6.0), (-8.0, 8.0)]
    for j, values in enumerate(deltas):
        for value in values:
            c = [1.0, 0.0, 0.0, 0.0]
            c[j] = value
            configs.append(tuple(c))
    configs += list(itertools.product(*deltas))
    assert len(configs) == 25
    dump(
        folder / "frozen_grid.json",
        dict(
            utc=utc(),
            configs=configs,
            acceptance_absolute_train_iou_loss=0.02,
            render_shape=list(shape),
            selection="Train masks only; all accepted alternatives subsequently assessed on development-heldout without selection",
        ),
    )
    candidates = []
    all_traits = []
    geometries = []
    parameter_sets = []
    base = None
    for j, config in enumerate(configs):
        cp, w, t = perturb_controls(
            saved["control_points"],
            saved["width_controls"],
            saved["twist_controls"],
            saved["stem_shape"],
            *config,
        )
        leaves, traits = model_traits(cp, w, t, saved["stem_shape"])
        v = np.r_[leaves.reshape(-1, 3), saved["stem"].reshape(-1, 3)]
        scores = [
            metrics(render_mask(v, faces, i * 2 * np.pi / 40, camera, shape), train_masks[i])
            for i in train
        ]
        score = mean_metric(scores)
        if base is None:
            base = score["iou"]
        accepted = accept_training(score["iou"], base, 0.02)
        record = dict(
            candidate_id=j,
            width_scale=config[0],
            twist_delta_rad=config[1],
            attachment_delta_px=config[2],
            bow_delta_px=config[3],
            train_iou=score["iou"],
            train_precision=score["precision"],
            train_recall=score["recall"],
            accepted=accepted,
        )
        candidates.append(record)
        all_traits.extend(dict(candidate_id=j, accepted=accepted, **r) for r in traits)
        if accepted:
            geometries.append((j, v))
            parameter_sets.append(
                dict(
                    candidate_id=j,
                    control_points=cp.tolist(),
                    width_controls=w.tolist(),
                    twist_controls=t.tolist(),
                )
            )
        event(folder, "candidate", **record)
        print(
            "E08 candidate", j, "trainIoU", round(score["iou"], 4), "accepted", accepted, flush=True
        )
    dump(folder / "accepted_parameters.json", parameter_sets)
    # Selection frozen above; only now access held-out masks.
    test_masks = {
        i: cv2.resize(
            np.asarray(Image.open(ROOT / rows[i]["mask"])),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        > 0
        for i in test
    }
    byid = {r["candidate_id"]: r for r in candidates}
    for j, v in geometries:
        score = mean_metric(
            [
                metrics(render_mask(v, faces, i * 2 * np.pi / 40, camera, shape), test_masks[i])
                for i in test
            ]
        )
        byid[j].update(
            development_heldout_iou=score["iou"],
            development_heldout_precision=score["precision"],
            development_heldout_recall=score["recall"],
        )
    np.savez_compressed(
        folder / "accepted_continuous_meshes.npz",
        candidate_ids=np.array([j for j, _ in geometries]),
        vertices=np.array([v for _, v in geometries]),
        faces=faces,
    )
    # Save one nonbaseline accepted display mesh using grid order, never held-out score.
    if len(geometries) > 1:
        j, v = geometries[1]
        dump(
            folder / "alternative_mesh.json",
            dict(
                candidate_id=j,
                vertices=v.round(4).tolist(),
                faces=faces.tolist(),
                organs=mesh["organs"],
                units="processing-pixel gauge",
                note="Organ numeric traits in this mesh metadata are omitted; see candidate_traits.csv.",
            ),
        )
        alternative = json.loads((folder / "alternative_mesh.json").read_text())
        alternative["organs"] = [
            {k: o[k] for k in ["leaf_id", "vertex_start", "vertex_count"]} for o in mesh["organs"]
        ]
        dump(folder / "alternative_mesh.json", alternative)
    write_csv(folder / "candidate_scores.csv", candidates)
    write_csv(folder / "candidate_traits.csv", all_traits)
    ranges = []
    for i in range(5):
        rows_ = [r for r in all_traits if r["accepted"] and r["leaf_id"] == f"leaf_{i + 1}"]
        ranges.append(
            dict(
                leaf_id=f"leaf_{i + 1}",
                **{
                    k: dict(min=min(r[k] for r in rows_), max=max(r[k] for r in rows_))
                    for k in [
                        "midrib_length_px_gauge",
                        "max_width_px_gauge",
                        "attachment_Y_px_gauge",
                        "basal_model_angle_deg",
                    ]
                },
            )
        )
    dump(folder / "conditional_trait_ranges.json", ranges)
    accepted_scores = [r["development_heldout_iou"] for r in candidates if r["accepted"]]
    sources = [MODEL / "organ_model.npz", MODEL / "browser_mesh.json", OBS] + [
        ROOT / r["mask"] for r in rows
    ]
    return finish(
        folder,
        "E08",
        "Alternative continuous organs under a training-defined tolerance",
        "completed_exploratory",
        "Which model traits vary across this finite grid of image-compatible continuous ribbon geometries?",
        dict(
            candidates=25,
            selection="absolute training-mean IoU loss <= .02 from baseline",
            render_shape=list(shape),
            grid="baseline + eight one-factor changes + sixteen joint corners; no alternative reoptimization",
            assessment="All accepted candidates scored on development-heldout after acceptance; no selection by these scores",
        ),
        dict(
            candidates=25,
            accepted=len(geometries),
            baseline_train_iou=base,
            accepted_development_heldout_iou_min=min(accepted_scores),
            accepted_development_heldout_iou_max=max(accepted_scores),
            conditional_trait_ranges=ranges,
        ),
        [
            f"{len(geometries)} of 25 predefined continuous geometry hypotheses satisfy the training tolerance.",
            "Accepted alternatives change width, twist, stem attachment and/or midrib controls; real parameters and meshes are saved.",
        ],
        [
            "Finite deterministic sensitivity grid, not a confidence interval, posterior or complete feasible set.",
            "Supplied camera, leaf identities and baseline reflect earlier development on this acquisition.",
            "Trait ranges are model gauge quantities; insertion is not an independently observed anatomical attachment.",
        ],
        [
            "Independent leaf boundaries/lengths, calibration and elevated views to challenge alternative models."
        ],
        sources,
    )


def e09(folder):
    data = json.loads(OBS.read_text())
    camera = json.loads(str(np.load(MODEL / "organ_model.npz")["camera"]))
    ids = [l["leaf_id"] for l in data["leaves"]]
    views = {v["view"]: v for v in data["views"]}
    evaluation = [3, 8, 13, 23, 28, 33]
    pool = sorted(set(views) - set(evaluation))
    visibility = {
        i: np.array([any(l["leaf_id"] == lid for l in views[i]["leaves"]) for lid in ids])
        for i in pool
    }
    dump(
        folder / "frozen_design.json",
        dict(
            evaluation_views=evaluation,
            coordinate_annotation_pool=pool,
            budgets=[4, 8, 12, 20, 30],
            strategies=["uniform", "preview_balanced"],
            camera_prior="Supplied from all30 original training views, including the six diagnostic evaluation views; outside annotation budget",
            identity_prior="Five IDs initialized in views0/20 and globally associated across30; supplied, not rediscovered within budget",
            preview_cost="preview_balanced sees organ-presence metadata on all24 eligible pool views, not tip coordinates or evaluation visibility",
            budget30="Full30 refit has in-sample diagnostics only; no unseen evaluation left",
        ),
    )
    all_rows = []
    fits = []
    for strategy in ["uniform", "preview_balanced"]:
        for budget in [4, 8, 12, 20, 30]:
            selected = (
                sorted(views) if budget == 30 else select_views(pool, budget, strategy, visibility)
            )
            target_views = sorted(views) if budget == 30 else evaluation
            errors = []
            missing = 0
            fitted_count = 0
            for lid in ids:
                train_observations = [
                    (i, l) for i in selected for l in views[i]["leaves"] if l["leaf_id"] == lid
                ]
                fit = triangulate_tip(
                    np.array([i * 2 * np.pi / 40 for i, _ in train_observations]),
                    np.array([l["tip_xy"] for _, l in train_observations]),
                    camera,
                )
                fits.append(
                    dict(
                        strategy=strategy,
                        budget=budget,
                        leaf_id=lid,
                        coordinate_views=[i for i, _ in train_observations],
                        **fit,
                    )
                )
                targets = [
                    (i, l) for i in target_views for l in views[i]["leaves"] if l["leaf_id"] == lid
                ]
                if fit["point"] is None:
                    missing += len(targets)
                    continue
                fitted_count += 1
                for i, l in targets:
                    predicted, _ = projection(
                        np.array(fit["point"])[None], np.array([i * 2 * np.pi / 40]), camera
                    )
                    error = float(np.linalg.norm(predicted[0] - l["tip_xy"]))
                    errors.append(error)
            all_rows.append(
                dict(
                    strategy=strategy,
                    coordinate_view_budget=budget,
                    selected_views=";".join(map(str, selected)),
                    fitted_organs=fitted_count,
                    preview_view_cost=24 if strategy == "preview_balanced" and budget < 30 else 0,
                    supplied_camera_calibration_view_cost=30,
                    evaluation_scope="in_sample_full30"
                    if budget == 30
                    else "six_excluded_coordinate_views_conditional_on_supplied_camera_and_IDs",
                    evaluated_tip_observations=len(errors),
                    unpredictable_tip_observations=missing,
                    median_error_px=float(np.median(errors)) if errors else None,
                    rmse_error_px=float(np.sqrt(np.mean(np.array(errors) ** 2)))
                    if errors
                    else None,
                )
            )
            event(folder, "budget_result", **all_rows[-1])
    write_csv(folder / "view_budget_results.csv", all_rows)
    dump(folder / "fitted_tip_points.json", fits)
    preview20 = next(
        r
        for r in all_rows
        if r["strategy"] == "preview_balanced" and r["coordinate_view_budget"] == 20
    )
    uniform20 = next(
        r for r in all_rows if r["strategy"] == "uniform" and r["coordinate_view_budget"] == 20
    )
    return finish(
        folder,
        "E09",
        "Conditional view and annotation budgets for explicit tip recovery",
        "completed_exploratory",
        "How many annotated view coordinates suffice for tip recovery when camera and organ identities are supplied?",
        dict(
            budgets=[4, 8, 12, 20, 30],
            strategies=[
                "uniform angle spacing",
                "preview organ-presence balanced log-determinant design",
            ],
            evaluation_views=evaluation,
            pool_views=pool,
            triangulation="Known-camera linear initialization plus robust three-coordinate point fit",
            calibration_cost="Shared camera informed by all30 views, supplied outside budget; this is not an end-to-end capture budget",
            preview_cost="24 pool previews for presence-based selection; no evaluation views/coordinates supplied to selector",
            full30="All30 refit evaluated in sample, explicitly not an unseen-view result",
        ),
        dict(
            results=all_rows,
            uniform20_evaluation_median_px=uniform20["median_error_px"],
            preview20_evaluation_median_px=preview20["median_error_px"],
        ),
        [
            "Feasible retrospective coordinate-annotation budgets are executed for 4/8/12/20 fitting views on a 24-view pool.",
            "The30-view fit is retained as an in-sample descriptive reference, not a falsely held-out result.",
            "Presence-based selection has an explicit preview cost and preserves missing/unpredictable organs.",
            "Preview-balanced selection does not uniformly improve residuals over uniform spacing across the tested budgets.",
        ],
        [
            "Camera fit uses all30 source views and globally assisted identities are supplied: conditional evaluation only.",
            "Six excluded coordinate views are from the same repeatedly developed acquisition.",
            "Only tip triangulation is evaluated; no claim of full leaf reconstruction from four views.",
        ],
        [
            "Lock an independently calibrated camera and prospectively acquire selected views plus new evaluation views."
        ],
        [OBS, CAMFIT, MODEL / "organ_model.npz"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["all", "E05", "E07", "E08", "E09"], default="all")
    args = parser.parse_args()
    with threadpool_limits(limits=3):
        for identifier, run in [("E05", e05), ("E07", e07), ("E08", e08), ("E09", e09)]:
            if args.experiment not in ["all", identifier]:
                continue
            folder = DEST / FOLDERS[identifier]
            folder.mkdir(parents=True, exist_ok=True)
            event(
                folder,
                "started",
                id=identifier,
                command=[sys.executable, *sys.argv],
                cpu_threads=3,
                gpu_used=False,
                python=sys.version,
                script_sha256=sha(__file__),
                ruling="Use existing checkout and isolated output folders per assigned scope; no worktree or dependency changes.",
            )
            start = time.time()
            try:
                run(folder)
                event(folder, "runtime", seconds=time.time() - start)
            except Exception as error:
                event(folder, "failed", error=repr(error))
                raise


if __name__ == "__main__":
    main()
