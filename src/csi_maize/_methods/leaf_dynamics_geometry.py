"""Scoped, source-assisted leaf dynamics pilot. No physical longitudinal claim.

All shape fitting and automatic associations use indices i%4!=2. Evaluation
opens development-heldout masks only after parameters are saved. The shared
perspective projection is a conditional f=D gauge, not metric calibration.
"""

from common import workspace_relative

import os

for _key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_key] = "3"
from bootstrap import ROOT, BASE, OUT, read_csv, write_csv, preferred_device
from pathlib import Path
import argparse, datetime, hashlib, itertools, json, sys, time
import numpy as np
from PIL import Image, ImageDraw
import cv2
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F
from organ_observations import extract_paths, deroll, design, raw_cost, robust_coefficients
from organ_inverse import project, scene, soft_silhouette, triangle_silhouette, sample_polyline
from anatomical_ribbons import ribbon_faces
from additive_geometry import directional_raster, projection, render_mask, metrics, load_geometry

DEST = OUT / "leaf_dynamics/2026-10-24/geometry"
EARLY = "capture_session_C1_5-1"
LATE = "capture_session_C2_5-1"
COLORS = [(255, 190, 70), (60, 220, 200), (210, 140, 255), (140, 170, 255), (255, 110, 130)]
# Source-inspected visible distal tips, processing coordinates. These are
# assisted hypotheses, not botanical rank labels or measured insertions.
SEEDS = {
    0: {"early_1": [549, 631], "early_2": [398, 398], "early_3": [482, 322]},
    20: {"early_1": [441, 617], "early_2": [629, 405], "early_3": [518, 346]},
}


def train_indices():
    return [i for i in range(40) if i % 4 != 2]


def require_training(indices):
    if any(i not in train_indices() for i in indices):
        raise ValueError("Development-heldout index in fitting data")


def rank_matchings(n_early, n_late):
    return list(itertools.combinations(range(n_late), n_early))


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def rel(path):
    return workspace_relative(Path(path))


def accepted_shape_family(candidate_ids):
    """Describe the executed accepted family without assuming historical IDs."""
    ids = [int(value) for value in candidate_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Accepted geometry family must contain unique candidate IDs")
    return dict(accepted_shape_alternatives=len(ids), accepted_shape_candidate_ids=ids)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def log(stage, **values):
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / "execution.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(utc=utc(), stage=stage, **values)) + "\n")


def light_counterfactual(vertices, faces, face_organs, direction, removed, resolution=256):
    """Fixed geometry and fixed raster extent; remove just one organ's triangles."""
    before = directional_raster(vertices, faces, face_organs, direction, resolution)
    keep = np.asarray(face_organs) != removed
    after = directional_raster(
        vertices, np.asarray(faces)[keep], np.asarray(face_organs)[keep], direction, resolution
    )
    n = int(np.max(face_organs)) + 1
    first = np.array(before["organ_intercepted_area"])
    second = np.pad(after["organ_intercepted_area"], (0, n - len(after["organ_intercepted_area"])))
    return dict(before=first.tolist(), after=second.tolist(), gain=(second - first).tolist())


def frontmost_ownership(xy, depth, faces, face_organs, shape):
    """Camera z-buffer with perspective-correct reciprocal-depth interpolation."""
    h, w = shape
    z = np.full(shape, np.inf)
    owner = np.full(shape, -1, int)
    for face, organ in zip(faces, face_organs):
        if np.any(depth[face] <= 0):
            continue
        tri = xy[face]
        lo = np.maximum(np.ceil(tri.min(0)).astype(int), 0)
        hi = np.minimum(np.floor(tri.max(0)).astype(int), [w - 1, h - 1])
        if np.any(hi < lo):
            continue
        a, b, c = tri
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-10:
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        wa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / den
        wb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / den
        wc = 1 - wa - wb
        inside = (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9)
        if not inside.any():
            continue
        value = 1 / (wa / depth[face[0]] + wb / depth[face[1]] + wc / depth[face[2]])
        ys = yy[inside]
        xs = xx[inside]
        zz = value[inside]
        near = zz < z[ys, xs]
        z[ys[near], xs[near]] = zz[near]
        owner[ys[near], xs[near]] = organ
    return owner


def source_appearance(early_fit):
    """Appearance of source RGB pixels conditioned on frontmost fitted organ ID."""
    records = []
    source_hashes = {}
    summaries = []
    for session in [EARLY, LATE]:
        if session == EARLY:
            saved = np.load(DEST / "earlier_organ_model.npz")
            vertices, faces, organs = assembled(saved["vertices"], saved["stem"])
            camera = json.loads(str(saved["camera"]))
            n = 3
        else:
            saved, _, camera, vertices, faces, organs = load_geometry()
            n = 5
        rows = rows_for(session)
        face_organs = organs[faces[:, 0]]
        for view in train_indices():
            theta = np.full(len(vertices), view * 2 * np.pi / 40)
            xy, depth = projection(vertices, theta, camera)
            xy = (xy + 0.5) * np.array([512 / 1024, 385 / 769]) - 0.5
            owner = frontmost_ownership(xy, depth, faces, face_organs, (385, 512))
            source = ROOT / rows[view]["source"]
            maskpath = ROOT / rows[view]["mask"]
            source_hashes[rel(source)] = sha(source)
            source_hashes[rel(maskpath)] = sha(maskpath)
            rgb = np.array(
                Image.open(source).convert("RGB").resize((512, 385), Image.Resampling.LANCZOS)
            )
            mask = (
                cv2.resize(
                    np.array(Image.open(maskpath)), (512, 385), interpolation=cv2.INTER_NEAREST
                )
                > 0
            )
            source_interior = (
                cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
            )
            for i in range(n):
                raw = owner == i
                interior = (
                    cv2.erode(raw.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
                )
                selected = interior & source_interior
                nonblack = rgb.max(2) > 12
                nonsaturated = rgb.max(2) < 250
                valid = selected & nonblack & nonsaturated
                values = rgb[valid].astype(float)
                count = len(values)
                row = dict(
                    session=session,
                    view=view,
                    leaf_id=("early_" if session == EARLY else "leaf_") + str(i + 1),
                    rendered_visible_pixels=int(raw.sum()),
                    eroded_pixels=int(interior.sum()),
                    mask_compatible_pixels=int(selected.sum()),
                    usable_pixels=count,
                    rejected_black=int((selected & ~nonblack).sum()),
                    rejected_saturated=int((selected & ~nonsaturated).sum()),
                    qc="pass_50_pixels" if count >= 50 else "insufficient_visible_interior",
                    cross_visit_gate="conditional_candidate_only"
                    if early_fit["credibility_flag"]
                    else "earlier_geometry_failed_no_cross_visit_comparison",
                )
                for key in [
                    "r_median",
                    "g_median",
                    "b_median",
                    "green_chromatic_median",
                    "excess_green_chromatic_median",
                    "green_chromatic_iqr",
                ]:
                    row[key] = None
                if count:
                    total = values.sum(1)
                    chromatic = values / total[:, None]
                    med = np.median(values, axis=0)
                    row.update(
                        r_median=float(med[0]),
                        g_median=float(med[1]),
                        b_median=float(med[2]),
                        green_chromatic_median=float(np.median(chromatic[:, 1])),
                        excess_green_chromatic_median=float(
                            np.median(2 * chromatic[:, 1] - chromatic[:, 0] - chromatic[:, 2])
                        ),
                        green_chromatic_iqr=float(
                            np.diff(np.percentile(chromatic[:, 1], [25, 75]))[0]
                        ),
                    )
                records.append(row)
            if view in [0, 20]:
                overlay = rgb.copy()
                for i in range(n):
                    overlay[owner == i] = (
                        0.6 * overlay[owner == i] + 0.4 * np.array(COLORS[i])
                    ).astype(np.uint8)
                im = Image.fromarray(overlay)
                ImageDraw.Draw(im).text(
                    (7, 7),
                    f"{session} view{view} frontmost IDs; eroded source pixels measured",
                    fill="white",
                )
                im.save(DEST / f"appearance_ownership_{session}_{view:02}.png")
        for i in range(n):
            leaf = ("early_" if session == EARLY else "leaf_") + str(i + 1)
            good = [
                r
                for r in records
                if r["session"] == session and r["leaf_id"] == leaf and r["qc"] == "pass_50_pixels"
            ]
            values = [r["green_chromatic_median"] for r in good]
            summaries.append(
                dict(
                    session=session,
                    leaf_id=leaf,
                    qualified_train_views=len(good),
                    total_usable_pixels=sum(r["usable_pixels"] for r in good),
                    median_of_view_green_chromatic_medians=float(np.median(values))
                    if values
                    else None,
                    view_green_chromatic_min=min(values) if values else None,
                    view_green_chromatic_max=max(values) if values else None,
                    note="sRGB appearance under unknown exposure/white balance, model-conditioned ownership; not chlorophyll or physiology",
                )
            )
        print("source appearance complete", session, flush=True)
    write_csv(DEST / "leaf_source_appearance_by_view.csv", records)
    write_csv(DEST / "leaf_source_appearance_summary.csv", summaries)
    dump(
        DEST / "source_appearance_protocol.json",
        dict(
            resolution=[512, 385],
            views=train_indices(),
            frontmost="minimum positive camera depth, reciprocal-depth interpolation",
            sampling="two 3x3 erosions of organ ownership; one erosion of whole-plant SAM; intersect; reject maxRGB<=12 or>=250",
            qualified_view_minimum_pixels=50,
            colour="median camera sRGB and chromatic coordinates; no radiometric calibration, no cross-date physiology",
            hashes=source_hashes,
        ),
    )
    return dict(
        summaries=summaries,
        input_sha256=source_hashes,
        view_leaf_rows=len(records),
        qualified_rows=sum(r["qc"] == "pass_50_pixels" for r in records),
    )


def light_figure():
    """Dependency-free labelled scientific figure from already saved numerical tables."""
    from PIL import ImageFont

    light = read_csv(DEST / "current_leaf_directional_light.csv")
    counter = read_csv(DEST / "fixed_shape_neighbour_removal.csv")
    sensitivity = read_csv(DEST / "current_leaf_light_sensitivity.csv")
    image = Image.new("RGB", (1350, 640), "white")
    draw = ImageDraw.Draw(image)
    # Pillow's bundled font is available on Windows, macOS and Linux.
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    draw.text(
        (30, 15),
        "Conditional leaf visibility and fixed-shape neighbour effects",
        font=font,
        fill="black",
    )
    draw.text(
        (30, 43),
        "Opaque double-sided geometry, normalized areas; not measured irradiance or photosynthesis",
        font=small,
        fill="black",
    )
    x0, y0 = 75, 110
    pw, ph = 570, 390
    for g in range(6):
        yy = y0 + ph - ph * g / 5
        draw.line((x0, yy, x0 + pw, yy), fill="#dddddd")
        draw.text((35, yy - 8), f"{g / 5:.1f}", font=small, fill="black")
    directions = list(dict.fromkeys(r["direction"] for r in light))
    ranges = []
    for i in range(5):
        leaf = f"leaf_{i + 1}"
        points = []
        for k, direction in enumerate(directions):
            value = float(
                next(r for r in light if r["leaf_id"] == leaf and r["direction"] == direction)[
                    "visible_projected_fraction"
                ]
            )
            points.append((x0 + pw * k / 8, y0 + ph * (1 - value)))
            options = [
                float(r["visible_projected_fraction"])
                for r in sensitivity
                if r["leaf_id"] == leaf and r["direction"] == direction
            ]
            ranges.append(
                dict(
                    leaf_id=leaf,
                    direction=direction,
                    base=value,
                    grid_min=min(options),
                    grid_max=max(options),
                    grid_span=max(options) - min(options),
                )
            )
        draw.line(points, fill=COLORS[i], width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=COLORS[i])
        draw.text((80 + i * 110, 550), f"Leaf {i + 1}", font=font, fill=COLORS[i])
    for k, label in enumerate(["Z", "0", "45", "90", "135", "180", "225", "270", "315"]):
        draw.text((x0 + pw * k / 8 - 10, 510), label, font=small, fill="black")
    draw.text(
        (130, 588), "Z=zenith; others elevation45deg / azimuth(deg)", font=small, fill="black"
    )
    matrix = np.zeros((5, 5))
    for r in counter:
        a = int(r["receiver_leaf"][-1]) - 1
        b = int(r["removed_leaf"][-1]) - 1
        matrix[a, b] = max(matrix[a, b], float(r["gain_over_height_squared"]))
    maximum = matrix.max()
    draw.text((740, 88), "Maximum receiver gain / baseline H^2", font=font, fill="black")
    draw.text((850, 120), "Removed leaf", font=small, fill="black")
    for i in range(5):
        draw.text((700, 185 + i * 70), f"Receiver {i + 1}", font=small, fill="black")
        draw.text((835 + i * 90, 145), str(i + 1), font=small, fill="black")
        for j in range(5):
            value = matrix[i, j]
            f = value / maximum if maximum else 0
            colour = (int(245 - 180 * f), int(245 - 100 * f), int(250 - 60 * f))
            x = 800 + j * 90
            y = 170 + i * 70
            draw.rectangle((x, y, x + 87, y + 67), fill=colour)
            draw.text((x + 8, y + 24), f"{value:.4f}", font=small, fill="black")
    draw.text(
        (730, 550), "All other shapes held fixed; removed leaf excluded.", font=small, fill="black"
    )
    image.save(DEST / "leaf_light_and_neighbour_effects.png")
    write_csv(DEST / "current_leaf_light_per_direction_grid_ranges.csv", ranges)


def cross_visit_descriptors(early_fit):
    """Candidate differences remain tabulated for every admissible map, unselected."""
    saved = np.load(DEST / "earlier_organ_model.npz")
    vertices, faces, organs = assembled(saved["vertices"], saved["stem"])
    height = np.ptp(vertices[:, 1])
    vertices = vertices / height
    directions = [("zenith", np.array([0.0, 1.0, 0.0]))]
    for az in range(0, 360, 45):
        a = np.radians(az)
        directions.append(
            (
                f"e45_a{az:03}",
                np.array([np.cos(a) / np.sqrt(2), 1 / np.sqrt(2), np.sin(a) / np.sqrt(2)]),
            )
        )
    early_light = []
    for name, direction in directions:
        light = directional_raster(vertices, faces, organs[faces[:, 0]], direction, 256)
        for i in range(3):
            intercepted = light["organ_intercepted_area"][i]
            footprint = light["organ_unoccluded_projected_area"][i]
            early_light.append(
                dict(
                    leaf_id=f"early_{i + 1}",
                    direction=name,
                    intercepted_over_own_height_squared=intercepted,
                    visible_projected_fraction=intercepted / footprint if footprint else 0.0,
                )
            )
    write_csv(DEST / "earlier_leaf_directional_light.csv", early_light)
    late_light = read_csv(DEST / "current_leaf_directional_light.csv")
    appearance = read_csv(DEST / "leaf_source_appearance_summary.csv")
    matches = read_csv(DEST / "rank_order_candidate_matching.csv")
    augmented = []
    for match in matches:
        e = next(r for r in appearance if r["leaf_id"] == match["earlier_leaf"])
        l = next(r for r in appearance if r["leaf_id"] == match["later_leaf"])
        ev = np.mean(
            [
                float(r["visible_projected_fraction"])
                for r in early_light
                if r["leaf_id"] == match["earlier_leaf"]
            ]
        )
        lv = np.mean(
            [
                float(r["visible_projected_fraction"])
                for r in late_light
                if r["leaf_id"] == match["later_leaf"]
            ]
        )
        qualified = (
            early_fit["credibility_flag"]
            and int(e["qualified_train_views"]) >= 8
            and int(l["qualified_train_views"]) >= 8
        )
        augmented.append(
            {
                **match,
                "earlier_appearance_qualified_views": e["qualified_train_views"],
                "later_appearance_qualified_views": l["qualified_train_views"],
                "camera_green_chromatic_difference": float(
                    l["median_of_view_green_chromatic_medians"]
                )
                - float(e["median_of_view_green_chromatic_medians"])
                if qualified
                else None,
                "colour_comparison_qc": "descriptive_exposure_and_identity_unconfirmed"
                if qualified
                else "insufficient_model_or_visible_views",
                "mean_directional_projected_visibility_difference": float(lv - ev),
                "light_difference_qc": "conditional_composition_comparison_within_own_footprint_not_photosynthetic_change",
            }
        )
    write_csv(DEST / "candidate_leaf_differences_geometry_appearance_light.csv", augmented)
    summary = dict(
        hypothesis_count=10,
        maps_selected=0,
        uniform_enumeration_entropy_bits=float(np.log2(10)),
        entropy_note="Combinatorial count under hypothetical equal weights, not an inferred biological posterior",
        earlier_mean_directional_visibility={
            f"early_{i + 1}": float(
                np.mean(
                    [
                        r["visible_projected_fraction"]
                        for r in early_light
                        if r["leaf_id"] == f"early_{i + 1}"
                    ]
                )
            )
            for i in range(3)
        },
        conditional_pair_rows=len(augmented),
        colour_eligible_pair_rows=sum(
            r["colour_comparison_qc"].startswith("descriptive") for r in augmented
        ),
        no_physical_growth="Each model has its own scale/gauge. All differences are conditional shape/appearance/composition descriptors, no rates or measured growth.",
        missing_states="Later-only visible hypotheses may be born, previously occluded/folded or differently segmented. Physical plant correspondence and botanical ranks unconfirmed.",
    )
    dump(DEST / "candidate_matching_summary.json", summary)
    return summary


def rows_for(session):
    return {
        int(r["view"]): r
        for r in read_csv(OUT / "sam_traits/view_traits.csv")
        if Path(r["session"]).name == session and r["camera"] == "rgb"
    }


def early_observations(rows):
    camera = dict(axis_x=481.0, base_y=768.0, roll_deg=3.0)
    angles = np.arange(40) * 2 * np.pi / 40
    require_training(SEEDS)
    candidates = {}
    inputs = {}
    lookup = {}
    for i in train_indices():
        row = rows[i]
        for key in ["source", "mask"]:
            inputs[row[key]] = sha(ROOT / row[key])
        candidates[i], _ = extract_paths(np.array(Image.open(ROOT / row["mask"])) > 0)
    for i, anchors in SEEDS.items():
        lookup[i] = {
            leaf: int(
                np.argmin(
                    [
                        np.linalg.norm(np.array(c["observed_terminal_xy"]) - xy)
                        for c in candidates[i]
                    ]
                )
            )
            for leaf, xy in anchors.items()
        }
        if len(set(lookup[i].values())) != 3:
            raise ValueError("Seeds fail to resolve three distinct visible paths")
    ids = sorted(SEEDS[0])
    models = {}
    initial = []
    for leaf in ids:
        pairs = [(i, candidates[i][lookup[i][leaf]]) for i in SEEDS]
        reference = np.mean([c["root_geodesic_px"] for _, c in pairs])
        best = None
        for i, options in candidates.items():
            if i in SEEDS or abs(np.sin(angles[i])) < 0.35:
                continue
            for c in options:
                if c["tip_censored"]:
                    continue
                p = pairs + [(i, c)]
                coef = np.linalg.solve(
                    design([angles[k] for k, _ in p]),
                    [deroll(q["observed_terminal_xy"], camera) for _, q in p],
                )
                pred = design(angles) @ coef
                costs = [
                    min(
                        [
                            raw_cost(q, pred[j], reference, camera)
                            for q in choices
                            if not q["tip_censored"]
                        ]
                        or [2.3]
                    )
                    for j, choices in candidates.items()
                ]
                value = np.minimum(costs, 2.0).sum() - 3 * np.sum(np.array(costs) < 2.0)
                if best is None or value < best[0]:
                    best = (value, coef, i, c["candidate_id"])
        models[leaf] = dict(coefficient=best[1], reference_root_length=reference)
        initial.append(dict(leaf_id=leaf, third_view=best[2], third_candidate=best[3]))
    for iteration in range(7):
        assignments = {}
        for i, choices in candidates.items():
            if i in SEEDS:
                assignments[i] = {leaf: (index, 0.0) for leaf, index in lookup[i].items()}
                continue
            eligible = [j for j, c in enumerate(choices) if not c["tip_censored"]]
            cost = np.full((3, len(eligible) + 3), 2.3)
            for n, leaf in enumerate(ids):
                expected = (design([angles[i]]) @ models[leaf]["coefficient"])[0]
                for k, j in enumerate(eligible):
                    cost[n, k] = raw_cost(
                        choices[j], expected, models[leaf]["reference_root_length"], camera
                    )
            left, right = linear_sum_assignment(cost)
            assignments[i] = {
                ids[n]: (eligible[k], float(cost[n, k]))
                for n, k in zip(left, right)
                if k < len(eligible) and cost[n, k] < 2.3
            }
        for leaf in ids:
            pairs = [(i, candidates[i][a[leaf][0]]) for i, a in assignments.items() if leaf in a]
            models[leaf]["coefficient"] = robust_coefficients(
                [angles[i] for i, _ in pairs],
                [deroll(c["observed_terminal_xy"], camera) for _, c in pairs],
            )
    views = []
    observations = []
    for i in train_indices():
        leaves = []
        for leaf, (index, cost) in assignments[i].items():
            c = candidates[i][index]
            flags = [
                "provisional_cross_view_identity",
                "silhouette_junction_not_anatomical_attachment",
            ]
            if cost > 1.3:
                flags += ["association_weak"]
            record = dict(
                leaf_id=leaf,
                index=ids.index(leaf),
                view=i,
                tip_xy=c["observed_terminal_xy"],
                attachment_xy=c["attachment_candidate_xy"],
                polyline_xy=c["polyline_xy"],
                association_cost=cost,
                uncertainty_flags=flags,
                initialization="source_assisted"
                if i in SEEDS
                else "algorithmic_harmonic_hungarian",
            )
            leaves.append(record)
            if cost <= 1.3 or i in SEEDS:
                observations.append(record)
        views.append(
            dict(
                view=i,
                leaves=leaves,
                unassigned_candidates=[
                    c
                    for j, c in enumerate(candidates[i])
                    if j not in [v[0] for v in assignments[i].values()]
                ],
            )
        )
        if i in [0, 5, 10, 15, 20, 25, 30, 35, 39]:
            image = Image.open(ROOT / rows[i]["source"]).convert("RGB").resize((1024, 769))
            draw = ImageDraw.Draw(image)
            for leaf in leaves:
                color = COLORS[ids.index(leaf["leaf_id"])]
                draw.line([tuple(p) for p in leaf["polyline_xy"]], fill=color, width=3)
                x, y = leaf["tip_xy"]
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)
                draw.text((x + 7, y), leaf["leaf_id"], fill=color)
            image.save(DEST / f"early_association_{i:02}.jpg", quality=92)
    result = dict(
        session=EARLY,
        ids=ids,
        train_indices=train_indices(),
        excluded_indices=[i for i in range(40) if i % 4 == 2],
        initialization=dict(
            type="source_assisted_three_visible_leaf_hypotheses",
            seeds_processing_pixels=SEEDS,
            algorithmic_initializers=initial,
            note="Seed points snapped to automatically extracted skeleton terminals; upright terminal may be medial-axis biased. Counts are visible hypotheses, not total botanical leaves.",
        ),
        views=views,
        fit_observations=observations,
        input_sha256=inputs,
    )
    dump(DEST / "earlier_observations.json", result)
    log(
        "early_associations",
        counts={leaf: sum(o["leaf_id"] == leaf for o in observations) for leaf in ids},
        fit_observations=len(observations),
    )
    return result


def fit_early_camera(data):
    obs = data["fit_observations"]
    theta = np.array([o["view"] * 2 * np.pi / 40 for o in obs])
    index = np.array([o["index"] for o in obs])
    truth = np.array([o["tip_xy"] for o in obs])
    ids = data["ids"]
    initial = []
    for leaf in ids:
        subset = [o for o in obs if o["leaf_id"] == leaf]
        coef = np.linalg.lstsq(
            design([o["view"] * 2 * np.pi / 40 for o in subset]),
            [deroll(o["tip_xy"], dict(axis_x=481, base_y=768, roll_deg=3)) for o in subset],
            rcond=None,
        )[0]
        initial.append([coef[1, 0], coef[0, 1], coef[2, 0]])
    camera = dict(axis_x=481.0, base_y=768.0, roll_deg=3.0, principal_x=512.0, principal_y=384.5)

    def predicted(p):
        return projection(
            p[2:].reshape(-1, 3)[index], theta, {**camera, "elevation_deg": p[0], "distance": p[1]}
        )[0]

    fits = []
    for elevation, distance in [(0, 650), (5, 1100), (-5, 1800)]:
        fits.append(
            least_squares(
                lambda p: (predicted(p) - truth).ravel(),
                np.r_[elevation, distance, np.array(initial).ravel()],
                bounds=(
                    np.r_[-25, 400, np.tile([-1000, 0, -1000], 3)],
                    np.r_[25, 3000, np.tile([1000, 1400, 1000], 3)],
                ),
                loss="soft_l1",
                f_scale=5,
                max_nfev=600,
            )
        )
    fit = min(fits, key=lambda f: f.cost)
    camera.update(elevation_deg=float(fit.x[0]), distance=float(fit.x[1]))
    errors = np.linalg.norm(predicted(fit.x) - truth, axis=1)
    audit = dict(
        camera=camera,
        tip_observations=len(obs),
        median_tip_error_px=float(np.median(errors)),
        rmse_tip_error_px=float(np.sqrt(np.mean(errors**2))),
        starts=[
            dict(cost=float(f.cost), elevation=float(f.x[0]), distance=float(f.x[1])) for f in fits
        ],
        note="Camera fitted only from provisional TRAIN tips; assumed principal point, roll3deg, axis481, one uniform turn, f=D gauge. Not transferable metric calibration.",
    )
    dump(DEST / "earlier_camera_fit.json", audit)
    return camera, fit.x[2:].reshape(3, 3), audit


def assembled(leaves, stem):
    vertices = np.r_[leaves.reshape(-1, 3), stem.reshape(-1, 3)]
    n, s, c, _ = leaves.shape
    lf = ribbon_faces(s, c)
    faces = np.concatenate([lf + i * s * c for i in range(n)])
    offset = n * s * c
    sf = []
    for j in range(len(stem) - 1):
        for k in range(stem.shape[1]):
            a = offset + j * 12 + k
            b = a + 12
            cc = offset + (j + 1) * 12 + (k + 1) % 12
            d = offset + j * 12 + (k + 1) % 12
            sf.extend([[a, b, cc], [a, cc, d]])
    faces = np.r_[faces, np.array(sf)]
    tri = vertices[faces]
    faces = faces[
        np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) > 1e-6
    ]
    organs = np.r_[np.repeat(np.arange(n), s * c), np.full(len(stem) * 12, n)]
    return vertices, faces, organs


def fit_early(data, rows, iterations):
    torch.manual_seed(2410)
    torch.set_num_threads(3)
    device = preferred_device()
    scale = 768.0
    ids = data["ids"]
    obs = data["fit_observations"]
    camera, tips, audit = fit_early_camera(data)
    train = np.array(train_indices())
    angles = torch.tensor(train * 2 * np.pi / 40, dtype=torch.float32, device=device)
    shape = (192, 256)
    masks = np.array(
        [
            cv2.resize(
                np.array(Image.open(ROOT / rows[i]["mask"])),
                (256, 192),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
            for i in train
        ]
    )
    target = torch.tensor(masks, dtype=torch.float32, device=device)
    nfg = target.sum((1, 2)).clamp_min(1)
    sdf = torch.tensor(
        np.array(
            [distance_transform_edt(~m) - distance_transform_edt(m) for m in masks], np.float32
        )[:, None],
        device=device,
    )
    # A 2D silhouette junction initializes a latent root, never measures insertion.
    heights = []
    r = np.radians(3)
    e = np.radians(camera["elevation_deg"])
    d = camera["distance"]
    ty = np.sin(r) * (481 - 512) - np.cos(r) * (768 - 384.5)
    for leaf in ids:
        pp = [o["attachment_xy"] for o in obs if o["leaf_id"] == leaf and o["view"] in SEEDS]
        up = np.array([np.sin(r) * (p[0] - 512) - np.cos(r) * (p[1] - 384.5) for p in pp])
        heights.append(np.mean(d * (up - ty) / (d * np.cos(e) + up * np.sin(e))) / scale)
    initial = []
    for tip, height in zip(tips, heights):
        p0 = np.array([0, height * scale, 0])
        delta = tip - p0
        initial.append([p0 + 0.28 * delta + [0, 30, 0], p0 + 0.68 * delta + [0, 25, 0], tip])
    control = torch.nn.Parameter(
        torch.tensor(np.array(initial) / scale, dtype=torch.float32, device=device)
    )
    attachment = torch.nn.Parameter(torch.tensor(heights, dtype=torch.float32, device=device))
    stem_shape = torch.nn.Parameter(torch.zeros(4, device=device))
    width_raw = torch.nn.Parameter(
        torch.tensor(
            np.log(np.expm1(np.array([[12, 17, 8], [15, 24, 12], [7, 12, 6]]) / scale)),
            dtype=torch.float32,
            device=device,
        )
    )
    twists = torch.nn.Parameter(torch.zeros((3, 3), device=device))
    parameters = [control, attachment, stem_shape, width_raw, twists]
    optimizer = torch.optim.Adam(parameters, lr=0.003)
    original = attachment.detach().clone()
    lookup = {int(i): n for n, i in enumerate(train)}
    oi = torch.tensor([o["index"] for o in obs], device=device)
    ov = torch.tensor([lookup[o["view"]] for o in obs], device=device)
    ot = torch.tensor(np.array([o["tip_xy"] for o in obs]), dtype=torch.float32, device=device)
    op = torch.tensor(
        np.array([sample_polyline(o["polyline_xy"], 32) for o in obs]),
        dtype=torch.float32,
        device=device,
    )
    history = []
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        leaves, mid, cp, stem, faces = scene(
            control, attachment, stem_shape, F.softplus(width_raw), twists, scale
        )
        b = (torch.arange(6, device=device) + step * 6) % 30
        theta = angles[b]
        leafxy = project(leaves.reshape(-1, 3), theta, camera).reshape(6, 3, 72, 11, 2)
        small = leafxy * leafxy.new_tensor([256 / 1024, 192 / 769])
        stemxy = project(stem.reshape(-1, 3), theta, camera) * small.new_tensor(
            [256 / 1024, 192 / 769]
        )
        xy = torch.cat([small.reshape(6, -1, 2), stemxy], 1)
        if step >= iterations * 2 // 3:
            sampled = leaves[:, ::2][:, :, [0, 2, 5, 8, 10]]
            ss, cc = sampled.shape[1:3]
            lf = ribbon_faces(ss, cc)
            ff = np.concatenate([lf + i * ss * cc for i in range(3)])
            screen = project(sampled.reshape(-1, 3), theta, camera) * xy.new_tensor(
                [256 / 1024, 192 / 769]
            )
            lp = triangle_silhouette(screen, ff, shape)
            sp = soft_silhouette(stemxy, torch.ones(stemxy.shape[:2], device=device) * 0.5, shape)
            prob = (1 - (1 - lp) * (1 - sp)).clamp(1e-5, 1 - 1e-5)
        else:
            ds = small[:, :, 2:, 1:-1] - small[:, :, :-2, 1:-1]
            dc = small[:, :, 1:-1, 2:] - small[:, :, 1:-1, :-2]
            area = (ds[..., 0] * dc[..., 1] - ds[..., 1] * dc[..., 0]).abs() / 4
            area = F.pad(area, (1, 1, 1, 1), mode="replicate").reshape(6, -1).clamp(0.015, 6) * 3
            weight = torch.cat([area, torch.ones(stemxy.shape[:2], device=device) * 0.3], 1)
            prob = soft_silhouette(xy, weight, shape).clamp(1e-5, 1 - 1e-5)
        image_loss = (
            -(target[b] * prob.log()).sum((1, 2)) / nfg[b]
            - 0.8 * ((1 - target[b]) * torch.log1p(-prob)).sum((1, 2)) / nfg[b]
        ).mean()
        mxy = project(mid.reshape(-1, 3), angles, camera).reshape(30, 3, 72, 2)
        pred = mxy[ov, oi]
        dist = torch.cdist(pred / scale, op / scale)
        path = (dist.min(-1).values.mean() + dist.min(-2).values.mean()) * 12
        tip = F.smooth_l1_loss(pred[:, -1] / 12, ot / 12) * 0.6
        grid = xy.clone()
        grid[..., 0] = grid[..., 0] * 2 / 255 - 1
        grid[..., 1] = grid[..., 1] * 2 / 191 - 1
        distances = F.grid_sample(sdf[b], grid[:, None], align_corners=True, padding_mode="border")[
            :, 0, 0
        ]
        visible = (grid.abs() <= 1).all(-1)
        outside = (F.relu(distances - 1) * visible).sum() / visible.sum().clamp_min(1) / 20
        curve = (
            (
                (cp[:, 0] - 2 * cp[:, 1] + cp[:, 2]).square().sum()
                + (cp[:, 1] - 2 * cp[:, 2] + cp[:, 3]).square().sum()
            )
            / scale**2
            * 0.03
        )
        prior = (
            (attachment - original).square().mean() * 8
            + stem_shape.square().mean() * 0.1
            + twists.square().mean() * 0.002
        )
        loss = image_loss + path + tip + outside + curve + prior
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10)
        optimizer.step()
        with torch.no_grad():
            attachment.clamp_(0.02, 1.1)
            width_raw.clamp_(-8, -1.8)
            twists.clamp_(-2, 2)
        history.append(
            dict(
                iteration=step + 1,
                loss=float(loss.detach()),
                image=float(image_loss.detach()),
                path=float(path.detach()),
                tip=float(tip.detach()),
                renderer="triangles" if step >= iterations * 2 // 3 else "splats",
            )
        )
        if step % 100 == 0:
            print("earlier fit", history[-1], flush=True)
    with torch.no_grad():
        leaves, mid, cp, stem, _ = scene(
            control, attachment, stem_shape, F.softplus(width_raw), twists, scale, 96, 13
        )
    leaves, mid, cp, stem = [x.cpu().numpy() for x in [leaves, mid, cp, stem]]
    vertices, faces, organs = assembled(leaves, stem)
    np.savez_compressed(
        DEST / "earlier_organ_model.npz",
        control_points=cp,
        width_controls=F.softplus(width_raw).detach().cpu().numpy() * scale,
        twist_controls=twists.detach().cpu().numpy(),
        vertices=leaves,
        midribs=mid,
        stem=stem,
        stem_shape=stem_shape.detach().cpu().numpy(),
        camera=json.dumps(camera),
        train_indices=train,
        heldout_indices=np.array([i for i in range(40) if i % 4 == 2]),
    )
    dump(
        DEST / "earlier_mesh.json",
        dict(
            vertices=vertices.tolist(),
            faces=faces.tolist(),
            vertex_organs=organs.tolist(),
            leaf_ids=ids,
            camera=camera,
            units="conditional f=D processing-pixel gauge",
            status="three source-assisted hypotheses, no anatomical validation",
        ),
    )
    write_csv(DEST / "earlier_training_loss.csv", history)
    log(
        "earlier_parameters_frozen",
        model_sha256=sha(DEST / "earlier_organ_model.npz"),
        iterations=iterations,
        camera=camera,
    )
    # FIRST evaluation access to heldout masks is below the frozen save.
    scores = []
    for i in range(40):
        truth = (
            cv2.resize(
                np.array(Image.open(ROOT / rows[i]["mask"])),
                (512, 385),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        mask = render_mask(vertices, faces, i * 2 * np.pi / 40, camera)
        score = dict(
            view=i, split="train" if i in train else "development_heldout", **metrics(mask, truth)
        )
        scores.append(score)
        if i in [0, 2, 10, 20, 22, 30]:
            image = np.array(
                Image.open(ROOT / rows[i]["source"]).convert("RGB").resize((1024, 769))
            )
            for k in range(3):
                line = project(torch.tensor(mid[k]), torch.tensor([i * 2 * np.pi / 40]), camera)[
                    0
                ].numpy()
                cv2.polylines(image, [np.rint(line).astype(np.int32)], False, COLORS[k], 2)
            im = Image.fromarray(image)
            ImageDraw.Draw(im).text(
                (10, 10),
                f"{i:02} {score['split']} three source-assisted hypotheses IoU512 {score['iou']:.3f}",
                fill="white",
            )
            im.save(DEST / f"earlier_fit_{i:02}.jpg", quality=92)
    write_csv(DEST / "earlier_view_scores.csv", scores)
    result = dict(
        camera_fit=audit,
        iterations=iterations,
        renderer_schedule="2/3 surface quadrature, 1/3 differentiable triangles",
        train_mean_iou=float(np.mean([s["iou"] for s in scores if s["split"] == "train"])),
        development_heldout_mean_iou=float(
            np.mean([s["iou"] for s in scores if s["split"] != "train"])
        ),
        development_heldout_mean_recall=float(
            np.mean([s["recall"] for s in scores if s["split"] != "train"])
        ),
        criterion="predeclared descriptive credibility flag: heldout IoU>=0.55 and tip median<=8px; not anatomy validation",
    )
    result["credibility_flag"] = bool(
        result["development_heldout_mean_iou"] >= 0.55 and audit["median_tip_error_px"] <= 8
    )
    dump(DEST / "earlier_fit_summary.json", result)
    return result


def leaf_traits(saved, session):
    leaves = saved["vertices"]
    mid = saved["midribs"]
    cp = saved["control_points"]
    height = float(np.ptp(np.r_[leaves.reshape(-1, 3), saved["stem"].reshape(-1, 3)][:, 1]))
    records = []
    for i in range(len(leaves)):
        chord = mid[i, -1] - mid[i, 0]
        length = float(np.linalg.norm(np.diff(mid[i], axis=0), axis=1).sum())
        width = float(np.linalg.norm(leaves[i, :, -1] - leaves[i, :, 0], axis=1).max())
        records.append(
            dict(
                session=session,
                leaf_id=("early_" if session == EARLY else "leaf_") + str(i + 1),
                visible_model_order=i + 1,
                length_over_own_model_height=length / height,
                width_over_own_model_height=width / height,
                root_height_over_own_model_height=float(cp[i, 0, 1] / height),
                tip_root_elevation_deg=float(
                    np.degrees(np.arctan2(chord[1], np.hypot(chord[0], chord[2])))
                ),
                length_width_ratio=length / width,
                own_model_height_gauge=height,
            )
        )
    return records


def lighting_and_matching():
    saved, mesh, camera, vertices, faces, organs = load_geometry()
    height = float(np.ptp(vertices[:, 1]))
    face_organs = organs[faces[:, 0]]
    vertices = vertices / height
    directions = [("zenith", np.array([0.0, 1.0, 0.0]))]
    for az in range(0, 360, 45):
        a = np.radians(az)
        e = np.radians(45)
        directions.append(
            (f"e45_a{az:03}", np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)]))
        )
    light = []
    counter = []
    for name, direction in directions:
        result = directional_raster(vertices, faces, face_organs, direction, 256)
        for i in range(5):
            intercepted = result["organ_intercepted_area"][i]
            footprint = result["organ_unoccluded_projected_area"][i]
            light.append(
                dict(
                    candidate_id="base",
                    direction=name,
                    leaf_id=f"leaf_{i + 1}",
                    intercepted_over_height_squared=intercepted,
                    unoccluded_projected_over_height_squared=footprint,
                    visible_projected_fraction=intercepted / footprint if footprint else 0.0,
                    whole_plant_share=intercepted / result["intercepted_area"],
                )
            )
        for removed in range(5):
            cf = light_counterfactual(vertices, faces, face_organs, direction, removed, 256)
            for i in range(5):
                if i == removed:
                    continue
                counter.append(
                    dict(
                        direction=name,
                        removed_leaf=f"leaf_{removed + 1}",
                        receiver_leaf=f"leaf_{i + 1}",
                        baseline_intercepted=cf["before"][i],
                        counterfactual_intercepted=cf["after"][i],
                        gain_over_height_squared=cf["gain"][i],
                    )
                )
        print("lighting base/counterfactual", name, flush=True)
    accepted = OUT / "additive_experiments/2026-10-24/e08_ambiguity/accepted_continuous_meshes.npz"
    alternatives = np.load(accepted)
    family = accepted_shape_family(alternatives["candidate_ids"])
    sensitivity = []
    for name, alternative in zip(alternatives["candidate_ids"], alternatives["vertices"]):
        name = str(name)
        for direction_name, direction in directions:
            # ONE fixed baseline denominator across alternatives, not renormalizing differences away.
            result = directional_raster(
                alternative / height, alternatives["faces"], face_organs, direction, 256
            )
            for i in range(5):
                a = result["organ_intercepted_area"][i]
                p = result["organ_unoccluded_projected_area"][i]
                sensitivity.append(
                    dict(
                        candidate_id=name,
                        direction=direction_name,
                        leaf_id=f"leaf_{i + 1}",
                        intercepted_over_baseline_height_squared=a,
                        visible_projected_fraction=a / p if p else 0.0,
                    )
                )
        print("lighting accepted alternative", name, flush=True)
    write_csv(DEST / "current_leaf_directional_light.csv", light)
    write_csv(DEST / "fixed_shape_neighbour_removal.csv", counter)
    write_csv(DEST / "current_leaf_light_sensitivity.csv", sensitivity)
    ranges = []
    for i in range(5):
        leaf = f"leaf_{i + 1}"
        base = [r for r in light if r["leaf_id"] == leaf]
        sens = [r for r in sensitivity if r["leaf_id"] == leaf]
        ranges.append(
            dict(
                leaf_id=leaf,
                baseline_directional_visibility_min=min(
                    r["visible_projected_fraction"] for r in base
                ),
                baseline_directional_visibility_max=max(
                    r["visible_projected_fraction"] for r in base
                ),
                accepted_grid_visibility_min=min(r["visible_projected_fraction"] for r in sens),
                accepted_grid_visibility_max=max(r["visible_projected_fraction"] for r in sens),
                mean_interception=float(
                    np.mean([r["intercepted_over_height_squared"] for r in base])
                ),
                max_neighbour_removal_gain=max(
                    r["gain_over_height_squared"] for r in counter if r["receiver_leaf"] == leaf
                ),
            )
        )
    dump(
        DEST / "current_leaf_light_ranges.json",
        dict(
            direction_count=len(directions),
            accepted_grid_count=family["accepted_shape_alternatives"],
            accepted_grid_candidate_ids=family["accepted_shape_candidate_ids"],
            ranges=ranges,
            interpretation="Grid and directional ranges, not confidence intervals. Opaque double-sided surfaces; no scattering/absorption, no actual sun trajectory or photosynthesis. Each receiver remains fixed after neighbour deletion.",
        ),
    )
    early = np.load(DEST / "earlier_organ_model.npz")
    early_traits = leaf_traits(early, EARLY)
    late_traits = leaf_traits(saved, LATE)
    write_csv(DEST / "conditional_leaf_traits.csv", early_traits + late_traits)
    # Enumerate ALL injective order-preserving hypotheses. No winner is selected.
    matches = []
    for hypothesis, mapping in enumerate(rank_matchings(3, 5)):
        for i, j in enumerate(mapping):
            e, l = early_traits[i], late_traits[j]
            matches.append(
                dict(
                    hypothesis=f"order_{hypothesis:02}",
                    earlier_leaf=e["leaf_id"],
                    later_leaf=l["leaf_id"],
                    state="candidate_correspondence_unconfirmed",
                    normalized_length_difference=l["length_over_own_model_height"]
                    - e["length_over_own_model_height"],
                    normalized_width_difference=l["width_over_own_model_height"]
                    - e["width_over_own_model_height"],
                    tip_root_elevation_difference_deg=l["tip_root_elevation_deg"]
                    - e["tip_root_elevation_deg"],
                    newly_visible_later_ids=";".join(
                        f"leaf_{k + 1}" for k in range(5) if k not in mapping
                    ),
                    interpretation="shape composition difference, not growth; unseen organs may already exist; physical identity/rank unconfirmed",
                )
            )
    write_csv(DEST / "rank_order_candidate_matching.csv", matches)
    states = [
        dict(
            earlier_leaf=r["leaf_id"],
            state="ambiguous_order_candidate",
            candidate_later_ids=";".join(
                sorted({m["later_leaf"] for m in matches if m["earlier_leaf"] == r["leaf_id"]})
            ),
        )
        for r in early_traits
    ]
    states.extend(
        dict(
            earlier_leaf="",
            state="later_only_visible_hypothesis_birth_or_earlier_occlusion_not_distinguished",
            candidate_later_ids=f"leaf_{i + 1}",
        )
        for i in range(5)
    )
    write_csv(DEST / "candidate_identity_states.csv", states)
    light_figure()
    return dict(
        per_leaf=ranges,
        directions=9,
        **family,
        counterfactual_receiver_cases=len(counter),
        order_preserving_hypotheses=10,
        maximum_fixed_shape_neighbour_gain=max(r["gain_over_height_squared"] for r in counter),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=750)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse frozen fit and completed numerical lighting tables after ancillary plotting failure",
    )
    args = p.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    log(
        "started",
        command=sys.argv,
        parameters=dict(
            seed=2410,
            train_rule="view%4!=2",
            early_leaf_count=3,
            roll_deg=3,
            iterations=args.iterations,
            light_resolution=256,
            directions="zenith plus elevation45 azimuth0:45:315",
        ),
        test_red="2026-10-24: unittest failed before implementation with missing module; command retained in tests_red.txt",
    )
    rows = rows_for(EARLY)
    if args.resume:
        log(
            "resume_after_ancillary_failure",
            failure="Initial run completed fit and numerical lighting tables then failed on missing optional matplotlib; no reconstruction rerun or holdout-driven parameter changes",
            frozen_model_sha256=sha(DEST / "earlier_organ_model.npz"),
        )
        data = json.loads((DEST / "earlier_observations.json").read_text(encoding="utf-8"))
        fit = json.loads((DEST / "earlier_fit_summary.json").read_text(encoding="utf-8"))
        ranges = json.loads((DEST / "current_leaf_light_ranges.json").read_text(encoding="utf-8"))
        accepted = (
            OUT / "additive_experiments/2026-10-24/e08_ambiguity/accepted_continuous_meshes.npz"
        )
        with np.load(accepted) as saved_family:
            family = accepted_shape_family(saved_family["candidate_ids"])
        if (
            ranges["accepted_grid_count"] != family["accepted_shape_alternatives"]
            or ranges.get("accepted_grid_candidate_ids", family["accepted_shape_candidate_ids"])
            != family["accepted_shape_candidate_ids"]
        ):
            raise ValueError("Saved light ranges do not match the current accepted geometry family")
        counter = read_csv(DEST / "fixed_shape_neighbour_removal.csv")
        light = dict(
            per_leaf=ranges["ranges"],
            directions=9,
            **family,
            counterfactual_receiver_cases=len(counter),
            order_preserving_hypotheses=10,
            maximum_fixed_shape_neighbour_gain=max(
                float(r["gain_over_height_squared"]) for r in counter
            ),
        )
        light_figure()
    else:
        data = early_observations(rows)
        fit = fit_early(data, rows, args.iterations)
        light = lighting_and_matching()
    appearance = source_appearance(fit)
    matching = cross_visit_descriptors(fit)
    sources = [
        BASE / "leaf_dynamics_geometry.py",
        BASE / "test_leaf_dynamics_geometry.py",
        BASE / "organ_inverse.py",
        BASE / "organ_observations.py",
        BASE / "anatomical_ribbons.py",
        BASE / "additive_geometry.py",
        BASE / "organ_camera.py",
        OUT / f"organ_inverse_triangles/{LATE}/organ_model.npz",
        OUT / f"organ_inverse_triangles/{LATE}/browser_mesh.json",
        OUT / "additive_experiments/2026-10-24/e08_ambiguity/accepted_continuous_meshes.npz",
        OUT / "additive_experiments/2026-10-24/e08_ambiguity/candidate_scores.csv",
    ]
    hashes = {rel(path): sha(path) for path in sources}
    hashes.update(data["input_sha256"])
    hashes.update(appearance["input_sha256"])
    for i in range(40):
        hashes[rows[i]["mask"]] = sha(ROOT / rows[i]["mask"])
    result = dict(
        scope="Source-assisted earlier 3-leaf pilot plus current 5-leaf light/uncertainty and rank-constrained correspondence hypotheses",
        status="completed_exploratory"
        if fit["credibility_flag"]
        else "partial_earlier_model_failed_credibility_flag",
        methods=dict(
            earlier_fit=fit,
            lighting=f"Opaque double-sided directional rasterization; fixed-shape neighbour-removal; {light['accepted_shape_alternatives']} TRAIN-compatible grid alternatives inherited without reselection (candidate IDs {light['accepted_shape_candidate_ids']})",
            matching="Exhaustive 10 injective order-preserving maps; no selected physical correspondence",
        ),
        metrics=dict(earlier=fit, current_light=light),
        findings=[
            "Earlier source views0/20 show three distinct visible blade hypotheses, versus five later fitted hypotheses. This is not a measured birth count.",
            "All earlier fitting/association uses 30 training masks, two source-assisted training seeds, no development-heldout selection.",
            "Current per-leaf access depends jointly on own pose, light direction and neighbouring geometry; neighbour-removal isolates a conditional shading interaction.",
            "Ten rank-order candidate maps remain; source labels alone do not establish physical plant or leaf identity.",
        ],
        limitations=[
            "Same-plant longitudinal identity and ages unconfirmed; no rates/day or physical growth claims.",
            "Model IDs are not measured botanical ranks; order constraint itself is a conditional assumption and could be wrong.",
            "Fixed roll/principal point, uniform one-turn angles and f=D gauge are assumptions; no metric calibration or stationary video.",
            "Earlier counts include visible blades only; absent leaves can be occluded/folded or unsegmented.",
            "Earlier 2D junctions are overlap/sheath silhouettes, not measured insertions; inferred hidden surfaces remain priors.",
            "Camera fit residuals concern provisional tip observations, not independent surface truth.",
            "Normalized lengths use each model own height and measure composition; cross-date differences cannot measure growth.",
            "Lighting is normalized geometrical interception, not measured irradiance/absorptance/photosynthesis. Grid ranges are not confidence intervals.",
            "No colour physiology inferred: RGB exposure/white balance and leaf-side effects are uncalibrated.",
        ],
        input_sha256=hashes,
        artifacts=[],
    )
    result["metrics"]["source_appearance"] = {
        k: v for k, v in appearance.items() if k != "input_sha256"
    }
    result["metrics"]["candidate_matching"] = matching
    result["methods"]["source_appearance"] = (
        "Positive-depth frontmost camera triangle ownership with perspective-correct reciprocal-depth interpolation; eroded organ and SAM interiors; TRAIN views only; robust per-view uncalibrated camera RGB/chromatic coordinates"
    )
    result["limitations"].append(
        "RGB appearance sampling uses fitted frontmost ownership, which can misassign overlapping laminae. Pixel counts and view dependence are retained; no chlorophyll inference or selected cross-visit identity."
    )
    log(
        "completed",
        status=result["status"],
        earlier_heldout_iou=fit["development_heldout_mean_iou"],
        appearance_qualified_rows=appearance["qualified_rows"],
    )
    result["artifacts"] = [rel(path) for path in sorted(DEST.glob("*")) if path.is_file()] + [
        rel(DEST / "summary.json")
    ]
    dump(DEST / "summary.json", result)
    print(json.dumps(dict(status=result["status"], earlier=fit, light=light), indent=2), flush=True)


if __name__ == "__main__":
    main()
