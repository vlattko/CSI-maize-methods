"""Assisted source-anchored leaf identities from TRAIN RGB/SAM observations.

Only indices i % 4 != 2 are opened. Two source-inspected training views anchor
five visible organs. Remaining associations are algorithmic, conditional and
auditable; skeleton merge junctions are not assumed botanical attachments.
"""

from bootstrap import ROOT, OUT, read_csv
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize
from traits import prune_skeleton

SESSION = "capture_session_C2_5-1"
# Source-inspected identities, anchored to automatically extracted terminals.
# Coordinates are processing pixels in unchanged 1024 x 769 image coordinates.
SEEDS = {
    0: {
        "leaf_1": [554, 649],
        "leaf_2": [349, 394],
        "leaf_3": [898, 119],
        "leaf_4": [313, 54],
        "leaf_5": [509, 299],
    },
    20: {
        "leaf_1": [429, 618],
        "leaf_2": [672, 409],
        "leaf_3": [234, 181],
        "leaf_4": [742, 42],
        "leaf_5": [501, 308],
    },
}
DESCRIPTIONS = {
    "leaf_1": "Lowest visible small blade with brown tip in view 0.",
    "leaf_2": "Intermediate left-pointing blade in view 0.",
    "leaf_3": "Long right-pointing blade in view 0.",
    "leaf_4": "Long left-pointing upper blade in view 0.",
    "leaf_5": "Emerging central upright blade in view 0; folded shape possible.",
}
COLORS = {
    "leaf_1": (255, 190, 70),
    "leaf_2": (60, 220, 200),
    "leaf_3": (255, 110, 130),
    "leaf_4": (140, 170, 255),
    "leaf_5": (210, 140, 255),
}


def deroll(xy, camera):
    xy = np.asarray(xy, float)
    dx = xy[..., 0] - camera["axis_x"]
    dy = xy[..., 1] - camera["base_y"]
    r = np.radians(camera["roll_deg"])
    return np.stack([np.cos(r) * dx + np.sin(r) * dy, np.sin(r) * dx - np.cos(r) * dy], axis=-1)


def reroll(uv, camera):
    uv = np.asarray(uv, float)
    r = np.radians(camera["roll_deg"])
    return np.stack(
        [
            camera["axis_x"] + np.cos(r) * uv[..., 0] + np.sin(r) * uv[..., 1],
            camera["base_y"] + np.sin(r) * uv[..., 0] - np.cos(r) * uv[..., 1],
        ],
        axis=-1,
    )


def sample_path(path, spacing=5):
    path = np.asarray(path, float)
    if len(path) < 2:
        return path
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    at = np.arange(0.0, cumulative[-1], spacing)
    at = np.r_[at, cumulative[-1]]
    return np.column_stack([np.interp(at, cumulative, path[:, j]) for j in range(2)])


def extract_paths(mask, min_leaf_length=40):
    sk = prune_skeleton(skeletonize(mask), 20)
    ys, xs = np.where(sk)
    if not len(xs):
        return [], None
    lookup = np.full(mask.shape, -1, int)
    lookup[ys, xs] = np.arange(len(xs))
    edges = []
    weights = []
    for dy, dx in [(0, 1), (1, -1), (1, 0), (1, 1)]:
        ny = ys + dy
        nx = xs + dx
        ok = (ny >= 0) & (ny < mask.shape[0]) & (nx >= 0) & (nx < mask.shape[1])
        src = np.flatnonzero(ok)
        dst = lookup[ny[ok], nx[ok]]
        hit = dst >= 0
        edges.extend(zip(src[hit], dst[hit]))
        weights.extend([np.hypot(dx, dy)] * int(hit.sum()))
    if not edges:
        return [], None
    edges = np.asarray(edges)
    graph = csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(len(xs), len(xs)))
    graph = graph + graph.T
    degree = np.diff(graph.indptr)
    # Root is a basal source-mask point, not a measured turntable center.
    root = int(np.argmax(ys - np.abs(xs - 482) * 0.02))
    distance, predecessor = dijkstra(graph, indices=root, return_predecessors=True)
    candidates = []
    for terminal in np.flatnonzero(degree == 1):
        if terminal == root or not np.isfinite(distance[terminal]):
            continue
        path = [int(terminal)]
        at = int(terminal)
        while at != root and predecessor[at] >= 0:
            at = int(predecessor[at])
            path.append(at)
        junction = next((q for q in path[1:] if degree[q] >= 3), root)
        branch_length = float(distance[terminal] - distance[junction])
        if branch_length < min_leaf_length:
            continue
        j = path.index(junction)
        visible = np.array([[xs[q], ys[q]] for q in path[: j + 1][::-1]], float)
        root_path = np.array([[xs[q], ys[q]] for q in path[::-1]], float)
        tip = [int(xs[terminal]), int(ys[terminal])]
        censored = (
            tip[0] < 8 or tip[0] >= mask.shape[1] - 8 or tip[1] < 10 or tip[1] >= mask.shape[0] - 8
        )
        candidates.append(
            dict(
                candidate_id=len(candidates),
                observed_terminal_xy=tip,
                tip_censored=bool(censored),
                attachment_candidate_xy=[int(xs[junction]), int(ys[junction])],
                polyline_xy=sample_path(visible).round(3).tolist(),
                root_path_xy=sample_path(root_path).round(3).tolist(),
                visible_branch_length_px=branch_length,
                root_geodesic_px=float(distance[terminal]),
            )
        )
    return candidates, [int(xs[root]), int(ys[root])]


def design(theta):
    theta = np.asarray(theta)
    return np.column_stack([np.ones(theta.size), np.cos(theta), np.sin(theta)])


def robust_coefficients(theta, values):
    a = design(theta)
    values = np.asarray(values, float)
    coefficient = np.linalg.lstsq(a, values, rcond=None)[0]
    for _ in range(8):
        residual = np.linalg.norm(values - a @ coefficient, axis=1)
        scale = max(8.0, 1.4826 * np.median(abs(residual - np.median(residual))))
        weight = np.minimum(1.0, 1.5 * scale / np.maximum(residual, 1e-9))
        coefficient = np.linalg.lstsq(
            a * weight[:, None] ** 0.5, values * weight[:, None] ** 0.5, rcond=None
        )[0]
    return coefficient


def raw_cost(candidate, expected_uv, expected_root_length, camera):
    observed = deroll(candidate["observed_terminal_xy"], camera)
    # Broad errors permit silhouette medial-axis bias, but do not make a fit exact.
    delta = (observed - expected_uv) / np.array([25.0, 22.0])
    length_penalty = max(0.0, abs(candidate["root_geodesic_px"] - expected_root_length) - 180) / 100
    return float(np.linalg.norm(delta) + length_penalty)


def initial_models(candidates, seed_lookup, angles, camera):
    models = {}
    diagnostics = {}
    for leaf in sorted(DESCRIPTIONS):
        seed_pairs = [(i, candidates[i][seed_lookup[i][leaf]]) for i in sorted(SEEDS)]
        reference_length = float(np.mean([c["root_geodesic_px"] for _, c in seed_pairs]))
        best = None
        for i, options in candidates.items():
            if i in SEEDS or abs(np.sin(angles[i])) < 0.35:
                continue
            for candidate in options:
                if candidate["tip_censored"]:
                    continue
                pairs = seed_pairs + [(i, candidate)]
                coefficient = np.linalg.solve(
                    design([angles[k] for k, _ in pairs]),
                    np.array([deroll(c["observed_terminal_xy"], camera) for _, c in pairs]),
                )
                expected = design(angles) @ coefficient
                costs = []
                for j, choices in candidates.items():
                    eligible = [c for c in choices if not c["tip_censored"]]
                    if eligible:
                        costs.append(
                            min(
                                raw_cost(c, expected[j], reference_length, camera) for c in eligible
                            )
                        )
                costs = np.array(costs)
                inlier = costs < 2.0
                score = float(np.minimum(costs, 2.0).sum() - 3 * inlier.sum())
                if best is None or score < best[0]:
                    best = (score, coefficient, i, candidate["candidate_id"], int(inlier.sum()))
        if best is None:
            raise ValueError("No supported trajectory hypothesis for " + leaf)
        models[leaf] = dict(coefficient=best[1], reference_root_length=reference_length)
        diagnostics[leaf] = dict(
            initial_third_view=best[2], initial_candidate_id=best[3], initial_inlier_views=best[4]
        )
    return models, diagnostics


def associate(candidates, models, seed_lookup, angles, camera, gate=2.3):
    leaves = sorted(models)
    assignments = {}
    for i, choices in candidates.items():
        if i in seed_lookup:
            assignments[i] = {
                leaf: (index, 0.0, "assisted_source_initialization")
                for leaf, index in seed_lookup[i].items()
            }
            continue
        eligible = [j for j, c in enumerate(choices) if not c["tip_censored"]]
        cost = np.full((len(leaves), len(eligible) + len(leaves)), gate)
        for n, leaf in enumerate(leaves):
            expected = design([angles[i]]) @ models[leaf]["coefficient"]
            for k, j in enumerate(eligible):
                cost[n, k] = raw_cost(
                    choices[j], expected[0], models[leaf]["reference_root_length"], camera
                )
        left, right = linear_sum_assignment(cost)
        assignments[i] = {
            leaves[n]: (eligible[k], float(cost[n, k]), "algorithmic_global_trajectory")
            for n, k in zip(left, right)
            if k < len(eligible) and cost[n, k] < gate
        }
    return assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="organ_observations")
    args = parser.parse_args()
    if args.name != "organ_observations":
        raise ValueError("Output scope is fixed to outputs/organ_observations")
    rows = [
        r
        for r in read_csv(OUT / "sam_traits/view_traits.csv")
        if Path(r["session"]).name == SESSION and r["camera"] == "rgb" and int(r["view"]) % 4 != 2
    ]
    rows.sort(key=lambda r: int(r["view"]))
    indices = [int(r["view"]) for r in rows]
    assert indices == [i for i in range(40) if i % 4 != 2]
    camera = dict(axis_x=481.0, base_y=768.0, roll_deg=3.0, elevation_deg=0.0, distance=0.0)
    angles = np.arange(40) * 2 * np.pi / 40
    candidates = {}
    roots = {}
    inputs = []
    row_map = {}
    for row in rows:
        i = int(row["view"])
        row_map[i] = row
        mask_path = ROOT / row["mask"]
        source = ROOT / row["source"]
        mask = np.asarray(Image.open(mask_path)) > 0
        candidates[i], roots[i] = extract_paths(mask)
        inputs.append(
            dict(
                view=i,
                mask=row["mask"],
                mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                source=row["source"],
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
        )
    seed_lookup = {}
    for i, anchors in SEEDS.items():
        seed_lookup[i] = {}
        for leaf, xy in anchors.items():
            distances = [
                np.linalg.norm(np.array(c["observed_terminal_xy"]) - xy) for c in candidates[i]
            ]
            j = int(np.argmin(distances))
            assert distances[j] <= 3, (i, leaf, distances)
            seed_lookup[i][leaf] = j
        assert len(set(seed_lookup[i].values())) == 5
    models, initial = initial_models(candidates, seed_lookup, angles, camera)
    for _ in range(6):
        assignments = associate(candidates, models, seed_lookup, angles, camera)
        for leaf in models:
            pairs = [
                (i, candidates[i][found[leaf][0]])
                for i, found in assignments.items()
                if leaf in found
            ]
            if len(pairs) >= 5:
                models[leaf]["coefficient"] = robust_coefficients(
                    [angles[i] for i, _ in pairs],
                    [deroll(c["observed_terminal_xy"], camera) for _, c in pairs],
                )
    assignments = associate(candidates, models, seed_lookup, angles, camera)
    folder = OUT / "organ_observations" / SESSION
    folder.mkdir(parents=True, exist_ok=True)
    views = []
    leaf_summary = []
    for leaf in sorted(models):
        seed_attachments = [
            deroll(candidates[i][seed_lookup[i][leaf]]["attachment_candidate_xy"], camera)
            for i in SEEDS
        ]
        attachment_height = float(np.mean(np.array(seed_attachments)[:, 1]))
        models[leaf]["attachment_height"] = attachment_height
        found = [(i, obs[leaf]) for i, obs in assignments.items() if leaf in obs]
        residual = []
        for i, (j, cost, method) in found:
            predicted = (design([angles[i]]) @ models[leaf]["coefficient"])[0]
            residual.append(
                np.linalg.norm(deroll(candidates[i][j]["observed_terminal_xy"], camera) - predicted)
            )
        leaf_summary.append(
            dict(
                leaf_id=leaf,
                source_description=DESCRIPTIONS[leaf],
                identity_status="source_anchored_assisted_hypothesis_not_an_independent_botanical_label",
                initialization_views=sorted(SEEDS),
                n_observed_uncensored_tips=len(found),
                trajectory_coefficients_derolled_xy=models[leaf]["coefficient"].tolist(),
                coefficient_order=["constant", "cos(theta)", "sin(theta)"],
                tip_trajectory_rmse_px=float(np.sqrt(np.mean(np.array(residual) ** 2))),
                tip_trajectory_median_error_px=float(np.median(residual)),
                provisional_attachment_height_up_px=attachment_height,
                attachment_source="Two source-inspected seed-view skeleton junctions; not measured anatomy",
                **initial[leaf],
            )
        )
    for i in indices:
        observations = []
        for leaf, (j, cost, method) in assignments[i].items():
            candidate = candidates[i][j]
            expected = (design([angles[i]]) @ models[leaf]["coefficient"])[0]
            residual = float(
                np.linalg.norm(deroll(candidate["observed_terminal_xy"], camera) - expected)
            )
            attachment_height = deroll(candidate["attachment_candidate_xy"], camera)[1]
            attachment_ambiguous = abs(attachment_height - models[leaf]["attachment_height"]) > 20
            other = []
            for other_leaf, model in models.items():
                if other_leaf != leaf:
                    other.append(
                        raw_cost(
                            candidate,
                            (design([angles[i]]) @ model["coefficient"])[0],
                            model["reference_root_length"],
                            camera,
                        )
                    )
            margin = float(min(other) - cost) if other else None
            flags = []
            if cost > 1.3 or residual > 30:
                flags.append("large_trajectory_residual")
            if margin is not None and margin < 0.4:
                flags.append("competing_leaf_identity")
            if attachment_ambiguous:
                flags.append("junction_not_consistent_with_seed_attachment")
            observations.append(
                dict(
                    leaf_id=leaf,
                    candidate_id=j,
                    tip_xy=candidate["observed_terminal_xy"],
                    attachment_xy=candidate["attachment_candidate_xy"],
                    attachment_status="possible_occlusion_or_merge_junction"
                    if attachment_ambiguous
                    else "source_consistent_skeleton_junction",
                    botanical_attachment_measured=False,
                    polyline_xy=candidate["polyline_xy"],
                    root_path_xy=candidate["root_path_xy"],
                    root_geodesic_px=candidate["root_geodesic_px"],
                    visible_branch_length_px=candidate["visible_branch_length_px"],
                    tip_censored=False,
                    association_method=method,
                    association_cost=cost,
                    competing_identity_margin=margin,
                    trajectory_residual_px=residual,
                    uncertainty_flags=flags,
                    association_status="uncertain" if flags else "provisionally_consistent",
                )
            )
        missing = [leaf for leaf in models if leaf not in assignments[i]]
        views.append(
            dict(
                view=i,
                nominal_angle_rad=float(angles[i]),
                root_xy=roots[i],
                leaves=observations,
                unobserved_leaf_ids=missing,
                unassigned_candidates=[
                    c
                    for c in candidates[i]
                    if c["candidate_id"] not in [x[0] for x in assignments[i].values()]
                ],
                source=row_map[i]["source"],
                mask=row_map[i]["mask"],
            )
        )
        image = (
            Image.open(ROOT / row_map[i]["source"])
            .convert("RGB")
            .resize((1024, 769), Image.Resampling.LANCZOS)
        )
        draw = ImageDraw.Draw(image)
        for candidate in candidates[i]:
            tip = candidate["observed_terminal_xy"]
            draw.ellipse((tip[0] - 3, tip[1] - 3, tip[0] + 3, tip[1] + 3), outline="white")
        for observation in observations:
            color = COLORS[observation["leaf_id"]]
            poly = [tuple(p) for p in observation["polyline_xy"]]
            draw.line(poly, fill=color, width=3)
            x, y = observation["tip_xy"]
            draw.text(
                (x + 5, y + 3),
                observation["leaf_id"] + (" ?" if observation["uncertainty_flags"] else ""),
                fill=color,
            )
            ax, ay = observation["attachment_xy"]
            draw.ellipse((ax - 4, ay - 4, ax + 4, ay + 4), outline=color, width=2)
        draw.rectangle((0, 0, 1024, 27), fill="#142b20")
        draw.text(
            (10, 8),
            f"TRAIN VIEW {i:02d} | assisted 0/20 seed IDs; trajectory matches, ? uncertain; unassigned={len(candidates[i]) - len(observations)}",
            fill="white",
        )
        image.save(folder / f"view_{i:02d}_organs.jpg", quality=92)
    result = dict(
        session=SESSION,
        coordinate_system="Original 1024x769 processing RGB frame, x right/y down",
        camera=camera,
        angle_assumption="uniform one turn theta_i=2*pi*i/40; not measured",
        train_indices=indices,
        excluded_indices=[i for i in range(40) if i % 4 == 2],
        heldout_masks_opened=False,
        initialization=dict(
            method="Source-inspected leaf identity labels in training views 0 and 20; nearest extracted endpoint within 3px",
            manually_declared_coordinates=SEEDS,
            reason="Both source views have five long rooted terminal paths, associated by visible attachment level and organ appearance.",
        ),
        leaves=leaf_summary,
        views=views,
        inputs=inputs,
        limitations=[
            "Leaf IDs are source-anchored hypotheses; uncertain associations remain flagged.",
            "A rooted silhouette skeleton is a projected medial axis, not a vein measurement.",
            "Skeleton junctions may be occlusions/merges; no botanical attachment truth asserted.",
            "No unseen tip or organ positions synthesized; unobserved IDs have no invented polyline.",
            "Border terminals are censored candidates and excluded from uncensored tip fitting.",
            "Sinusoidal tips use uniform-angle and small-motion assumptions, not calibrated trajectories.",
            "Independent per-leaf constant/harmonic terms can absorb camera/model mismatch; not a unique 3D plant.",
        ],
    )
    (folder / "observations.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(
                folder=str(folder),
                train_views=len(indices),
                leaves=leaf_summary,
                observations=sum(len(v["leaves"]) for v in views),
                uncertain=sum(bool(l["uncertainty_flags"]) for v in views for l in v["leaves"]),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
