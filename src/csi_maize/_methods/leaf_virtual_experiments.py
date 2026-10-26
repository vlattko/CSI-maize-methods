"""Leaf-resolved model interventions with explicit overlapping-screen bookkeeping.

Frozen geometry only. Light scenarios are not measured irradiance, physiological
responses or independent replicates. Stem and receiver remain in every removal.
"""

from pathlib import Path
import hashlib, json, itertools, datetime, time
import numpy as np
from bootstrap import ROOT, BASE, OUT, read_csv, write_csv

DEST = OUT / "leaf_dynamics/2026-10-24/jeb_vision_v2"
EXPERIMENT = DEST / "experiments"
GEO = OUT / "leaf_dynamics/2026-10-24/geometry"
GRID = OUT / "additive_experiments/2026-10-24/e08_ambiguity"
EPS = 1e-10  # Near-tie reporting threshold only; ownership uses a total order.


def validate_candidate_ids(candidate_ids, geometry_count):
    """Retain the actual accepted family, with its baseline in position zero.

    Acceptance is decided upstream from fitting evidence. Downstream virtual
    experiments must not replace it with a historical list of accepted IDs.
    """
    values = np.asarray(candidate_ids)
    if (
        values.ndim != 1
        or not len(values)
        or len(values) != geometry_count
        or not np.issubdtype(values.dtype, np.integer)
    ):
        raise ValueError("Accepted candidate IDs must match a nonempty geometry family")
    ids = values.tolist()
    if ids[0] != 0 or len(set(ids)) != len(ids) or any(i < 0 for i in ids):
        raise ValueError("Accepted family needs unique nonnegative IDs and baseline zero first")
    return ids


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, v):
    Path(p).write_text(
        json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n"
    )


def compose_owners(depths, active):
    """Largest depth toward the source wins; exact ties use the smaller ID.

    Strict depth ordering is transitive across all organ subsets. Treating
    pairwise near ties as equal would make blocker bookkeeping inconsistent.
    """
    best = np.full(depths.shape[1:], -np.inf)
    owner = np.full(best.shape, -1, int)
    for i in sorted(active):
        take = depths[i] > best
        owner[take] = i
        best[take] = depths[i][take]
    return owner


def screening_bookkeeping(depths, receiver, leaf_ids, stem_id):
    footprint = np.isfinite(depths[receiver])
    neighbours = [i for i in leaf_ids if i != receiver]
    stem_free = footprint & (compose_owners(depths, [receiver, stem_id]) == receiver)
    bits = np.zeros(footprint.shape, np.uint8)
    for b, j in enumerate(neighbours):
        blocking = footprint & (compose_owners(depths, [receiver, j]) != receiver)
        bits[blocking] |= 1 << b
    multiplicity = np.zeros(bits.shape, int)
    for b in range(len(neighbours)):
        multiplicity += (bits >> b) & 1
    full = int(np.count_nonzero(stem_free & (bits == 0)))
    counts = [
        int(np.count_nonzero(stem_free & (multiplicity == k))) for k in range(len(neighbours) + 1)
    ]
    allbits = (1 << len(neighbours)) - 1
    coalitions = []
    for removed in range(allbits + 1):
        received = int(np.count_nonzero(stem_free & ((bits & (allbits ^ removed)) == 0)))
        coalitions.append(
            {
                "removal_bits": removed,
                "removed_ids": [j for b, j in enumerate(neighbours) if removed >> b & 1],
                "received_pixels": received,
                "gain_pixels": received - full,
            }
        )
    singles = {j: coalitions[1 << b]["gain_pixels"] for b, j in enumerate(neighbours)}
    result = {
        "receiver": receiver,
        "neighbours": neighbours,
        "footprint_pixels": int(footprint.sum()),
        "full_pixels": full,
        "stem_blocked_pixels": int(np.count_nonzero(footprint & ~stem_free)),
        "all_neighbours_removed_pixels": int(stem_free.sum()),
        "exclusive_pixels": sum(singles.values()),
        "redundant_pixels": sum(counts[2:]),
        "multiplicity_pixels": counts,
        "single_removal_pixels": singles,
        "coalitions": coalitions,
        "blocker_bits": bits,
        "stem_free_mask": stem_free,
    }
    assert counts[0] == full and counts[1] == sum(singles.values())
    assert (
        result["footprint_pixels"]
        == full
        + result["stem_blocked_pixels"]
        + result["exclusive_pixels"]
        + result["redundant_pixels"]
    )
    return result


def layered_raster(vertices, faces, face_organs, direction, resolution=256):
    """Independent per-organ depth buffers on the original full-geometry grid."""
    vertices = np.asarray(vertices, float)
    n = np.asarray(direction, float)
    n = n / np.linalg.norm(n)
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    xy = np.column_stack([vertices @ u, vertices @ v])
    depth = vertices @ n
    step = max(float(np.max(np.ptp(xy, axis=0))) / (resolution - 4), 1e-12)
    origin = (xy.min(0) + xy.max(0)) / 2 - step * resolution / 2
    screen = (xy - origin) / step - 0.5
    layers = np.full((int(face_organs.max()) + 1, resolution, resolution), -np.inf)
    for face, organ in zip(faces, face_organs):
        tri = screen[face]
        lo = np.maximum(np.ceil(tri.min(0)).astype(int), 0)
        hi = np.minimum(np.floor(tri.max(0)).astype(int), resolution - 1)
        if np.any(hi < lo):
            continue
        a, b, c = tri
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
        zz = (wa * depth[face[0]] + wb * depth[face[1]] + wc * depth[face[2]])[inside]
        ys = yy[inside]
        xs = xx[inside]
        take = zz > layers[organ, ys, xs]
        layers[organ, ys[take], xs[take]] = zz[take]
    return layers, step**2


def run(resolution=256):
    from additive_geometry import load_geometry, directional_raster

    global EXPERIMENT
    EXPERIMENT = DEST / ("experiments" if resolution == 256 else f"experiments_{resolution}")
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    inputs = [
        Path(__file__),
        BASE / "additive_geometry.py",
        BASE / "test_leaf_virtual_experiments.py",
        GRID / "accepted_continuous_meshes.npz",
        GRID / "candidate_scores.csv",
        GRID / "frozen_grid.json",
        OUT / "organ_inverse_triangles/capture_session_C2_5-1/organ_model.npz",
        OUT / "organ_inverse_triangles/capture_session_C2_5-1/browser_mesh.json",
        GEO / "fixed_shape_neighbour_removal.csv",
        GEO / "current_leaf_directional_light.csv",
        GEO / "leaf_source_appearance_summary.csv",
    ]
    before_hash = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    saved, mesh, camera, raw, faces, organs = load_geometry()
    H = float(np.ptp(raw[:, 1]))
    fo = organs[faces[:, 0]]
    assert all(np.all(organs[f] == fo[i]) for i, f in enumerate(faces))
    grid = np.load(inputs[3])
    assert np.array_equal(grid["faces"], faces)
    difference = float(np.max(np.abs(grid["vertices"][0] - raw)))
    assert difference < 0.001
    ids = validate_candidate_ids(grid["candidate_ids"], len(grid["vertices"]))
    # The saved export is the baseline used by the original counterfactual CSV.
    # Candidate0's regenerated mesh differs by <=0.001 gauge units through rounding.
    geometries = [raw] + [x for x in grid["vertices"][1:]]
    directions = [("zenith", np.array([0.0, 1.0, 0.0]))] + [
        (
            f"e45_a{a:03}",
            np.array(
                [
                    np.cos(np.radians(a)) / np.sqrt(2),
                    1 / np.sqrt(2),
                    np.sin(np.radians(a)) / np.sqrt(2),
                ]
            ),
        )
        for a in range(0, 360, 45)
    ]
    edges = []
    states = []
    coalitions = []
    checks = []
    illustrations = {}
    max_original_error = 0.0
    original = read_csv(GEO / "fixed_shape_neighbour_removal.csv")
    for ci, (cid, verts) in enumerate(zip(ids, geometries)):
        vertices = verts / H
        for di, (name, direction) in enumerate(directions):
            depths, pixel_area = layered_raster(vertices, faces, fo, direction, resolution)
            owner = compose_owners(depths, range(6))
            if cid == 0:
                direct = directional_raster(vertices, faces, fo, direction, resolution)
                assert np.array_equal(owner, direct["owner"]), ("full owner mismatch", name)
                checks.append(
                    {
                        "candidate_id": cid,
                        "direction": name,
                        "removed": [],
                        "owner_pixels_different": 0,
                    }
                )
            for receiver in range(5):
                r = screening_bookkeeping(depths, receiver, list(range(5)), 5)
                den = r["footprint_pixels"]
                assert r["full_pixels"] == int(np.count_nonzero(owner == receiver)), (
                    "bookkeeping mismatch",
                    cid,
                    name,
                    receiver,
                )
                common = {
                    "candidate_id": cid,
                    "direction": name,
                    "receiver_leaf": f"leaf_{receiver + 1}",
                    "pixel_area_H2": pixel_area,
                    "isolated_pixels": den,
                }
                row = {
                    **common,
                    "received_fraction": r["full_pixels"] / den,
                    "stem_blocked_fraction": r["stem_blocked_pixels"] / den,
                    "exclusive_fraction": r["exclusive_pixels"] / den,
                    "redundant_fraction": r["redundant_pixels"] / den,
                    "all_neighbours_removed_fraction": r["all_neighbours_removed_pixels"] / den,
                }
                tied = np.zeros(depths.shape[1:], bool)
                for j in range(6):
                    if j == receiver:
                        continue
                    finite = np.isfinite(depths[receiver]) & np.isfinite(depths[j])
                    close = np.zeros_like(tied)
                    close[finite] = abs(depths[j][finite] - depths[receiver][finite]) <= EPS
                    tied |= close
                row["near_tie_fraction"] = int(tied.sum()) / den
                row.update(
                    {
                        f"multiplicity_{k}_fraction": n / den
                        for k, n in enumerate(r["multiplicity_pixels"])
                    }
                )
                states.append(row)
                for removed, gain in r["single_removal_pixels"].items():
                    edge = {
                        **common,
                        "removed_leaf": f"leaf_{removed + 1}",
                        "received_area_H2": r["full_pixels"] * pixel_area,
                        "gain_area_H2": gain * pixel_area,
                        "rescue_fraction": gain / den,
                    }
                    edges.append(edge)
                    if cid == 0 and resolution == 256:
                        old = next(
                            x
                            for x in original
                            if x["direction"] == name
                            and x["receiver_leaf"] == edge["receiver_leaf"]
                            and x["removed_leaf"] == edge["removed_leaf"]
                        )
                        err = abs(edge["gain_area_H2"] - float(old["gain_over_height_squared"]))
                        max_original_error = max(max_original_error, err)
                        assert err < 1e-12, ("single-removal replay", name, receiver, removed, err)
                for c in r["coalitions"]:
                    coalitions.append(
                        {
                            **common,
                            "removal_bits": c["removal_bits"],
                            "removed_leaves": ";".join(f"leaf_{i + 1}" for i in c["removed_ids"]),
                            "received_fraction": c["received_pixels"] / den,
                            "gain_fraction": c["gain_pixels"] / den,
                        }
                    )
                    if cid == 0 and di == 0 and receiver == 0:
                        keep = ~np.isin(fo, c["removed_ids"])
                        direct = directional_raster(
                            vertices, faces[keep], fo[keep], direction, resolution
                        )
                        predicted = compose_owners(
                            depths, [i for i in range(6) if i not in c["removed_ids"]]
                        )
                        assert np.array_equal(predicted, direct["owner"]), (
                            "all-subset direct check",
                            resolution,
                            c["removal_bits"],
                        )
                        assert int(np.count_nonzero(predicted == receiver)) == c["received_pixels"]
                        checks.append(
                            {
                                "candidate_id": cid,
                                "direction": name,
                                "receiver": 1,
                                "removed": [j + 1 for j in c["removed_ids"]],
                                "owner_pixels_different": 0,
                            }
                        )
                if cid == 0:
                    illustrations[f"{name}_L{receiver + 1}_bits"] = r["blocker_bits"]
                    illustrations[f"{name}_L{receiver + 1}_stem_free"] = r["stem_free_mask"]
                    illustrations[f"{name}_L{receiver + 1}_footprint"] = np.isfinite(
                        depths[receiver]
                    )
            # Predefined multi-removal checks, one per accepted configuration.
            if di == 0:
                removed = [2, 3]
                keep = ~np.isin(fo, removed)
                direct = directional_raster(vertices, faces[keep], fo[keep], direction, resolution)
                predicted = compose_owners(depths, [i for i in range(6) if i not in removed])
                assert np.array_equal(predicted, direct["owner"]), ("multi-removal mismatch", cid)
                checks.append(
                    {
                        "candidate_id": cid,
                        "direction": name,
                        "removed": [3, 4],
                        "owner_pixels_different": 0,
                    }
                )
        print(
            json.dumps(
                {
                    "resolution": resolution,
                    "candidate": cid,
                    "completed": ci + 1,
                    "of": len(ids),
                    "elapsed_seconds": round(time.time() - start, 1),
                }
            ),
            flush=True,
        )
    write_csv(EXPERIMENT / "directed_neighbour_effects.csv", edges)
    write_csv(EXPERIMENT / "screening_decomposition.csv", states)
    write_csv(EXPERIMENT / "all_neighbour_subset_outcomes.csv", coalitions)
    np.savez_compressed(EXPERIMENT / "baseline_screening_maps.npz", **illustrations)
    edge_summary = []
    for removed, receiver in itertools.permutations(range(5), 2):
        a = f"leaf_{removed + 1}"
        b = f"leaf_{receiver + 1}"
        means = []
        ranks = []
        for cid in ids:
            vals = [
                x["rescue_fraction"]
                for x in edges
                if x["candidate_id"] == cid and x["removed_leaf"] == a and x["receiver_leaf"] == b
            ]
            value = float(np.mean(vals))
            means.append(value)
            allmeans = [
                float(
                    np.mean(
                        [
                            x["rescue_fraction"]
                            for x in edges
                            if x["candidate_id"] == cid
                            and x["removed_leaf"] == aa
                            and x["receiver_leaf"] == bb
                        ]
                    )
                )
                for aa, bb in itertools.permutations([f"leaf_{i + 1}" for i in range(5)], 2)
            ]
            ranks.append(1 + sum(v > value + 1e-12 for v in allmeans))
        edge_summary.append(
            {
                "removed_leaf": a,
                "receiver_leaf": b,
                "baseline_mean_rescue_fraction": means[0],
                "grid_min_mean_rescue_fraction": min(means),
                "grid_max_mean_rescue_fraction": max(means),
                "positive_configurations": sum(v > 0 for v in means),
                "rank_min": min(ranks),
                "rank_max": max(ranks),
            }
        )
    edge_summary.sort(key=lambda x: -x["baseline_mean_rescue_fraction"])
    write_csv(EXPERIMENT / "edge_stability.csv", edge_summary)
    byleaf = []
    for receiver in range(5):
        rows = [
            s
            for s in states
            if s["candidate_id"] == 0 and s["receiver_leaf"] == f"leaf_{receiver + 1}"
        ]
        byleaf.append(
            {
                "receiver_leaf": f"leaf_{receiver + 1}",
                **{
                    k: float(np.mean([s[k] for s in rows]))
                    for k in [
                        "received_fraction",
                        "stem_blocked_fraction",
                        "exclusive_fraction",
                        "redundant_fraction",
                        "all_neighbours_removed_fraction",
                    ]
                },
            }
        )
    redundant = max(
        [s for s in states if s["candidate_id"] == 0], key=lambda s: s["redundant_fraction"]
    )
    after_hash = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    assert before_hash == after_hash
    result = {
        "status": "completed_computational_demonstration",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "shape_ids": ids,
        "configuration_count": len(ids),
        "resolution": resolution,
        "baseline_height_gauge": H,
        "baseline_export_vs_grid0_max_difference_gauge": difference,
        "baseline_policy": "Original saved export for exact baseline replay; every other accepted configuration retains its frozen grid vertices. All use original baseline H and each direction full-geometry raster bounds.",
        "directions": [d[0] for d in directions],
        "direction_weighting": "Equal scenario weights; not daily illumination or a probability distribution.",
        "counts": {
            "directed_edges": len(edges),
            "receiver_direction_shape_states": len(states),
            "subset_outcomes": len(coalitions),
            "direct_raster_checks": len(checks),
        },
        "baseline_single_removal_max_replay_error_H2": max_original_error,
        "direct_owner_checks": checks,
        "baseline_mean_decomposition_by_leaf": byleaf,
        "strongest_edges": edge_summary[:5],
        "baseline_max_redundant_case": redundant,
        "all_decompositions_exact_in_pixel_counts": True,
        "input_sha256": before_hash,
        "inference": "Geometric model interventions. No measured radiation, physiology, botanical rank or temporal identity is inferred.",
    }
    dump(EXPERIMENT / "summary.json", result)
    print(
        json.dumps(
            {
                "resolution": resolution,
                "completed": True,
                "counts": result["counts"],
                "max_baseline_replay_error_H2": max_original_error,
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def convergence():
    low = DEST / "experiments"
    high = DEST / "experiments_512"
    lo = read_csv(low / "directed_neighbour_effects.csv")
    hi = read_csv(high / "directed_neighbour_effects.csv")
    key = lambda r: (r["candidate_id"], r["direction"], r["removed_leaf"], r["receiver_leaf"])
    lookup = {key(r): r for r in hi}
    rows = []
    for r in lo:
        s = lookup[key(r)]
        a = float(r["rescue_fraction"])
        b = float(s["rescue_fraction"])
        tol = max(1 / int(r["isolated_pixels"]), 1 / int(s["isolated_pixels"]), abs(a - b))
        rows.append(
            {
                "candidate_id": int(r["candidate_id"]),
                "direction": r["direction"],
                "removed_leaf": r["removed_leaf"],
                "receiver_leaf": r["receiver_leaf"],
                "rescue_256": a,
                "rescue_512": b,
                "numerical_tolerance": tol,
                "resolved_positive": min(a, b) > tol,
            }
        )
    write_csv(DEST / "edge_resolution_checks.csv", rows)
    ids = sorted(set(r["candidate_id"] for r in rows))
    edges = []
    for a, b in itertools.permutations([f"leaf_{i + 1}" for i in range(5)], 2):
        options = [r for r in rows if r["removed_leaf"] == a and r["receiver_leaf"] == b]
        means = []
        for cid in ids:
            rr = [r for r in options if r["candidate_id"] == cid]
            x = float(np.mean([r["rescue_256"] for r in rr]))
            y = float(np.mean([r["rescue_512"] for r in rr]))
            tol = float(np.mean([r["numerical_tolerance"] for r in rr]))
            means.append(
                {
                    "candidate_id": cid,
                    "mean256": x,
                    "mean512": y,
                    "tolerance": tol,
                    "lower": min(x, y) - tol,
                    "upper": max(x, y) + tol,
                }
            )
        edges.append(
            {
                "removed_leaf": a,
                "receiver_leaf": b,
                "baseline_mean256": means[0]["mean256"],
                "baseline_mean512": means[0]["mean512"],
                "grid_min_mean256": min(r["mean256"] for r in means),
                "grid_max_mean256": max(r["mean256"] for r in means),
                "grid_min_mean512": min(r["mean512"] for r in means),
                "grid_max_mean512": max(r["mean512"] for r in means),
                "numerically_resolved_configurations": sum(r["lower"] > 0 for r in means),
                "all_configurations_resolved_positive": all(r["lower"] > 0 for r in means),
                "candidates": means,
            }
        )
    edges.sort(key=lambda r: -r["baseline_mean256"])
    for edge in edges:
        ranksets = []
        for m in edge["candidates"]:
            others = [
                next(c for c in e["candidates"] if c["candidate_id"] == m["candidate_id"])
                for e in edges
                if e is not edge
            ]
            ranksets.append(
                {
                    "candidate_id": m["candidate_id"],
                    "strictly_larger_edges": sum(o["lower"] > m["upper"] for o in others),
                    "possibly_larger_edges": sum(o["upper"] >= m["lower"] for o in others),
                }
            )
        edge["rank_bounds_across_grid"] = [
            1 + min(r["strictly_larger_edges"] for r in ranksets),
            1 + max(r["possibly_larger_edges"] for r in ranksets),
        ]
    decomposition = []
    lows = read_csv(low / "screening_decomposition.csv")
    highs = read_csv(high / "screening_decomposition.csv")
    kk = lambda r: (r["candidate_id"], r["direction"], r["receiver_leaf"])
    hmap = {kk(r): r for r in highs}
    for r in lows:
        s = hmap[kk(r)]
        a = float(r["redundant_fraction"])
        b = float(s["redundant_fraction"])
        tol = max(1 / int(r["isolated_pixels"]), 1 / int(s["isolated_pixels"]), abs(a - b))
        decomposition.append(
            {
                "candidate_id": int(r["candidate_id"]),
                "direction": r["direction"],
                "receiver_leaf": r["receiver_leaf"],
                "redundant256": a,
                "redundant512": b,
                "numerical_tolerance": tol,
                "resolved_redundant": min(a, b) > tol,
            }
        )
    write_csv(DEST / "shared_screening_resolution_checks.csv", decomposition)
    result = {
        "numerical_rule": "Per edge/state, max of one receiver-footprint pixel on either grid and absolute256/512 fraction difference. Pooled directional tolerances are arithmetic means; not statistical confidence bounds.",
        "edge_summaries": edges,
        "configuration_count": len(ids),
        "candidate_ids": ids,
        "resolved_edges_all_configurations": sum(
            e["all_configurations_resolved_positive"] for e in edges
        ),
        "shared_screening_resolved_cases": sum(r["resolved_redundant"] for r in decomposition),
        "baseline_shared_screening_resolved_cases": sum(
            r["resolved_redundant"] and r["candidate_id"] == 0 for r in decomposition
        ),
        "max_redundant256": max(r["redundant256"] for r in decomposition),
        "max_redundant512": max(r["redundant512"] for r in decomposition),
        "maximum_near_tie_fraction": max(float(r["near_tie_fraction"]) for r in lows + highs),
        "input_sha256": {
            p.relative_to(ROOT).as_posix(): sha(p)
            for p in [
                Path(__file__),
                low / "summary.json",
                high / "summary.json",
                low / "directed_neighbour_effects.csv",
                high / "directed_neighbour_effects.csv",
                low / "screening_decomposition.csv",
                high / "screening_decomposition.csv",
            ]
        },
    }
    dump(DEST / "numerical_stability_summary.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["edge_summaries", "input_sha256"]},
            indent=2,
        )
    )


def main():
    run(256)
    run(512)
    convergence()


if __name__ == "__main__":
    main()
