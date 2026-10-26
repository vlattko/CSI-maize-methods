"""Additional question-led analyses for expanded v3; frozen v2 remains intact."""

from common import workspace_relative

from pathlib import Path
import json, hashlib, itertools, time, datetime
import numpy as np
from bootstrap import ROOT, BASE, OUT, read_csv, write_csv
from leaf_virtual_experiments import layered_raster, compose_owners

DEST = OUT / "leaf_dynamics/2026-10-24/jeb_expanded_v3"
V2 = DEST.parent / "jeb_vision_v2"
GEO = DEST.parent / "geometry"
GRID = OUT / "additive_experiments/2026-10-24/e08_ambiguity"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dump(p, v):
    Path(p).write_text(
        json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n"
    )


def direction_weights(q):
    if not 0 <= q <= 1:
        raise ValueError("Zenith scenario weight must lie in[0,1]")
    return np.array([q] + [(1 - q) / 8] * 8)


def beam_budget(before, after, removed, pixel_area=1.0):
    removed = set(removed)
    retained = [i for i in range(6) if i not in removed]
    old = np.array([np.count_nonzero(before == i) for i in range(6)]) * pixel_area
    new = np.array([np.count_nonzero(after == i) for i in range(6)]) * pixel_area
    assert all(new[i] >= old[i] - 1e-12 for i in retained)
    oldtotal = float(old.sum())
    newtotal = float(new.sum())
    loss = sum(old[i] for i in removed)
    leafgain = sum(new[i] - old[i] for i in retained if i < 5)
    stemgain = new[5] - old[5]
    escape = oldtotal - newtotal
    assert escape >= -1e-12
    return {
        "before_total": oldtotal,
        "after_total": newtotal,
        "removed_interception": float(loss),
        "retained_leaf_gain": float(leafgain),
        "stem_gain": float(stemgain),
        "escape_gain": float(escape),
        "budget_residual": float(loss - leafgain - stemgain - escape),
    }


def angular_sequence(n):
    order = [0]
    while len(order) < n:
        candidates = [i for i in range(n) if i not in order]
        chosen = max(
            candidates, key=lambda i: (min(min(abs(i - j), n - abs(i - j)) for j in order), -i)
        )
        order.append(chosen)
    return order


def select_coverage(support):
    n = support.shape[0]
    covered = np.zeros(support.shape[1:], bool)
    order = []
    curve = []
    while len(order) < n:
        chosen = max(
            [i for i in range(n) if i not in order],
            key=lambda i: (
                np.count_nonzero(support[i] & ~covered),
                int(np.min(np.count_nonzero(covered | support[i], axis=1))),
                -i,
            ),
        )
        covered |= support[chosen]
        order.append(chosen)
        curve.append(float(covered.mean()))
    return order, curve


def weights_and_metrics():
    low = read_csv(V2 / "experiments/directed_neighbour_effects.csv")
    high = read_csv(V2 / "experiments_512/directed_neighbour_effects.csv")
    keys = lambda r: (r["candidate_id"], r["direction"], r["removed_leaf"], r["receiver_leaf"])
    lo = {keys(r): r for r in low}
    ids = sorted({int(r["candidate_id"]) for r in high})
    directions = ["zenith"] + [f"e45_a{a:03}" for a in range(0, 360, 45)]
    pairs = list(itertools.permutations(range(5), 2))
    values = {}
    tolerances = {}
    records = []
    sweep = []
    curves = []
    for cid in ids:
        for metric, col in [
            ("receiver_fraction", "rescue_fraction"),
            ("common_height_area", "gain_area_H2"),
        ]:
            a = np.zeros((9, 20))
            b = np.zeros_like(a)
            tol = np.zeros_like(a)
            for d, direction in enumerate(directions):
                for e, (j, i) in enumerate(pairs):
                    r = next(
                        x
                        for x in high
                        if x["candidate_id"] == str(cid)
                        and x["direction"] == direction
                        and x["removed_leaf"] == f"leaf_{j + 1}"
                        and x["receiver_leaf"] == f"leaf_{i + 1}"
                    )
                    s = lo[keys(r)]
                    a[d, e] = float(r[col])
                    b[d, e] = float(s[col])
                    pxhi = (
                        1 / int(r["isolated_pixels"])
                        if metric == "receiver_fraction"
                        else float(r["pixel_area_H2"])
                    )
                    pxlo = (
                        1 / int(s["isolated_pixels"])
                        if metric == "receiver_fraction"
                        else float(s["pixel_area_H2"])
                    )
                    tol[d, e] = max(pxhi, pxlo, abs(a[d, e] - b[d, e]))
            values[cid, metric] = a
            tolerances[cid, metric] = tol
            for scheme, q in [("equal_nine", 1 / 9), ("oblique_ring", 0), ("zenith_only", 1)]:
                w = direction_weights(q)
                means = w @ a
                meanslo = w @ b
                delta = w @ tol
                for e, (j, i) in enumerate(pairs):
                    records.append(
                        {
                            "candidate_id": cid,
                            "metric": metric,
                            "scheme": scheme,
                            "zenith_weight": q,
                            "removed_leaf": f"leaf_{j + 1}",
                            "receiver_leaf": f"leaf_{i + 1}",
                            "effect512": means[e],
                            "effect256": meanslo[e],
                            "numerical_indicator": delta[e],
                            "nominal_rank": 1 + int(np.count_nonzero(means > means[e] + 1e-12)),
                        }
                    )
            for q in np.linspace(0, 1, 101):
                w = direction_weights(q)
                v = w @ a
                vl = w @ b
                dt = w @ tol
                lower = np.minimum(v, vl) - dt
                upper = np.maximum(v, vl) + dt
                winner = int(np.argmax(v))
                j, i = pairs[winner]
                contenders = np.where(upper >= max(lower))[0]
                sweep.append(
                    {
                        "candidate_id": cid,
                        "metric": metric,
                        "zenith_weight": q,
                        "nominal_removed_leaf": f"leaf_{j + 1}",
                        "nominal_receiver_leaf": f"leaf_{i + 1}",
                        "nominal_effect": v[winner],
                        "numerically_separated": len(contenders) == 1,
                        "contender_count": len(contenders),
                    }
                )
                if cid == 0:
                    for e, (j, i) in enumerate(pairs):
                        curves.append(
                            {
                                "metric": metric,
                                "zenith_weight": q,
                                "removed_leaf": f"leaf_{j + 1}",
                                "receiver_leaf": f"leaf_{i + 1}",
                                "effect": v[e],
                            }
                        )
    write_csv(DEST / "metric_weight_comparison.csv", records)
    write_csv(DEST / "direction_weight_sweep.csv", sweep)
    write_csv(DEST / "baseline_weight_curves.csv", curves)
    baseline = []
    for metric in ["receiver_fraction", "common_height_area"]:
        for scheme in ["equal_nine", "oblique_ring", "zenith_only"]:
            rows = [
                r
                for r in records
                if r["candidate_id"] == 0 and r["metric"] == metric and r["scheme"] == scheme
            ]
            rows.sort(key=lambda r: -r["effect512"])
            baseline.append({"metric": metric, "scheme": scheme, "leaders": rows[:3]})
    return {
        "baseline_rankings": baseline,
        "sweep_rows": len(sweep),
        "primary_direction_convention": "Designed weights, not measured radiation or a solar distribution.",
    }


def budgets():
    from additive_geometry import load_geometry, directional_raster

    saved, mesh, camera, raw, faces, organs = load_geometry()
    H = float(np.ptp(raw[:, 1]))
    fo = organs[faces[:, 0]]
    grid = np.load(GRID / "accepted_continuous_meshes.npz")
    ids = grid["candidate_ids"].tolist()
    geoms = [raw] + list(grid["vertices"][1:])
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
    records = []
    maxreplay = 0.0
    checks = 0
    start = time.time()
    for resolution in [256, 512]:
        previous = read_csv(
            V2
            / ("experiments" if resolution == 256 else "experiments_512")
            / "directed_neighbour_effects.csv"
        )
        old = {
            (
                int(r["candidate_id"]),
                r["direction"],
                int(r["removed_leaf"][-1]) - 1,
                int(r["receiver_leaf"][-1]) - 1,
            ): r
            for r in previous
        }
        for cid, vertices0 in zip(ids, geoms):
            vertices = vertices0 / H
            for di, (name, direction) in enumerate(directions):
                layers, pa = layered_raster(vertices, faces, fo, direction, resolution)
                before = compose_owners(layers, range(6))
                if di == 0:
                    direct = directional_raster(vertices, faces, fo, direction, resolution)
                    assert np.array_equal(before, direct["owner"])
                    checks += 1
                for bits in range(32):
                    removed = [i for i in range(5) if bits >> i & 1]
                    active = [i for i in range(6) if i not in removed]
                    after = compose_owners(layers, active)
                    r = beam_budget(before, after, removed, pa)
                    assert abs(r["budget_residual"]) < 1e-12
                    row = {
                        "resolution": resolution,
                        "candidate_id": cid,
                        "direction": name,
                        "removed_bits": bits,
                        "removed_leaves": ";".join(f"leaf_{i + 1}" for i in removed),
                        "removal_count": len(removed),
                        **r,
                    }
                    for i in range(5):
                        row[f"leaf_{i + 1}_area_H2"] = float(np.count_nonzero(after == i) * pa)
                        if len(removed) == 1 and i not in removed:
                            gain = float(
                                (np.count_nonzero(after == i) - np.count_nonzero(before == i)) * pa
                            )
                            err = abs(gain - float(old[cid, name, removed[0], i]["gain_area_H2"]))
                            maxreplay = max(maxreplay, err)
                            assert err < 1e-12
                    row["stem_area_H2"] = float(np.count_nonzero(after == 5) * pa)
                    records.append(row)
                    if bits == 12 and di == 0:
                        direct = directional_raster(
                            vertices,
                            faces[~np.isin(fo, removed)],
                            fo[~np.isin(fo, removed)],
                            direction,
                            resolution,
                        )
                        assert np.array_equal(after, direct["owner"])
                        checks += 1
            print(
                json.dumps(
                    {
                        "budget_grid": resolution,
                        "candidate": cid,
                        "elapsed_seconds": round(time.time() - start, 1),
                    }
                ),
                flush=True,
            )
    write_csv(DEST / "whole_shoot_subset_budgets.csv", records)
    # Receiver-wise trade-offs are computed from each complete removal state.
    outcomes = []
    for resolution in [256, 512]:
        source = read_csv(
            V2
            / ("experiments" if resolution == 256 else "experiments_512")
            / "screening_decomposition.csv"
        )
        f = {
            (int(r["candidate_id"]), r["direction"], int(r["receiver_leaf"][-1]) - 1): int(
                r["isolated_pixels"]
            )
            * float(r["pixel_area_H2"])
            for r in source
        }
        baseline = {
            (r["candidate_id"], r["direction"]): r
            for r in records
            if r["resolution"] == resolution and r["removed_bits"] == 0
        }
        for r in [r for r in records if r["resolution"] == resolution]:
            b = baseline[r["candidate_id"], r["direction"]]
            for i in range(5):
                if r["removed_bits"] >> i & 1:
                    continue
                outcomes.append(
                    {
                        "resolution": resolution,
                        "candidate_id": r["candidate_id"],
                        "direction": r["direction"],
                        "receiver_leaf": f"leaf_{i + 1}",
                        "removed_bits": r["removed_bits"],
                        "removal_count": r["removal_count"],
                        "receiver_release_fraction": (
                            r[f"leaf_{i + 1}_area_H2"] - b[f"leaf_{i + 1}_area_H2"]
                        )
                        / f[r["candidate_id"], r["direction"], i],
                        "whole_shoot_loss_fraction": r["escape_gain"] / r["before_total"],
                    }
                )
    write_csv(DEST / "receiver_subset_tradeoffs.csv", outcomes)
    return {
        "budget_rows": len(records),
        "receiver_subset_rows": len(outcomes),
        "direct_owner_checks": checks,
        "single_removal_replay_error_H2": maxreplay,
        "max_abs_budget_residual_H2": max(abs(r["budget_residual"]) for r in records),
    }


def camera_face_ownership(xy, depth, faces, organs, shape=(385, 512)):
    z = np.full(shape, np.inf)
    owner = np.full(shape, -1, int)
    face_id = np.full(shape, -1, int)
    for k, (face, organ) in enumerate(zip(faces, organs)):
        if np.any(depth[face] <= 0):
            continue
        tri = xy[face]
        lo = np.maximum(np.ceil(tri.min(0)).astype(int), 0)
        hi = np.minimum(np.floor(tri.max(0)).astype(int), [shape[1] - 1, shape[0] - 1])
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
        face_id[ys[near], xs[near]] = k
    return owner, face_id


def source_support(hashes):
    from leaf_dynamics_geometry import assembled, frontmost_ownership
    from additive_geometry import projection
    from PIL import Image
    import cv2

    models = [
        ("capture_session_C1_5-1", GEO / "earlier_organ_model.npz"),
        (
            "capture_session_C2_5-1",
            OUT / "organ_inverse_triangles/capture_session_C2_5-1/organ_model.npz",
        ),
    ]
    allviews = read_csv(OUT / "sam_traits/view_traits.csv")
    old = read_csv(GEO / "leaf_source_appearance_by_view.csv")
    records = []
    curves = []
    summaries = []
    assets = DEST / "assets"
    assets.mkdir(exist_ok=True)
    for session, path in models:
        hashes[workspace_relative(path)] = sha(path)
        saved = np.load(path)
        vertices, faces, organs = assembled(saved["vertices"], saved["stem"])
        camera = json.loads(str(saved["camera"]))
        n = len(saved["vertices"])
        assert saved["vertices"].shape[1:] == (96, 13, 3)
        fo = organs[faces[:, 0]]
        facebin = np.full(len(faces), -1, int)
        for i in range(n):
            fs = fo == i
            t = np.mean((faces[fs] - i * 96 * 13) // 13, axis=1) / 95
            facebin[fs] = np.minimum((t * 10).astype(int), 9)
        support = np.zeros((40, n, 10), bool)
        rows = {
            int(r["view"]): r
            for r in allviews
            if Path(r["session"]).name == session and r["camera"] == "rgb"
        }
        for view in range(40):
            p = ROOT / rows[view]["source"]
            mp = ROOT / rows[view]["mask"]
            hashes[workspace_relative(p)] = sha(p)
            hashes[workspace_relative(mp)] = sha(mp)
            rgb = np.asarray(
                Image.open(p).convert("RGB").resize((512, 385), Image.Resampling.LANCZOS)
            )
            mask = (
                cv2.resize(np.asarray(Image.open(mp)), (512, 385), interpolation=cv2.INTER_NEAREST)
                > 0
            )
            interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
            xy, depth = projection(vertices, np.full(len(vertices), view * 2 * np.pi / 40), camera)
            xy = (xy + 0.5) * [0.5, 385 / 769] - 0.5
            owner, faceids = camera_face_ownership(xy, depth, faces, fo)
            if view in [0, 10, 20, 30]:
                assert np.array_equal(owner, frontmost_ownership(xy, depth, faces, fo, (385, 512)))
            for i in range(n):
                valid = (
                    (
                        cv2.erode(
                            (owner == i).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2
                        )
                        > 0
                    )
                    & interior
                    & (rgb.max(2) > 12)
                    & (rgb.max(2) < 250)
                )
                count = int(valid.sum())
                name = ("early_" if n == 3 else "leaf_") + str(i + 1)
                qualified = count >= 50
                green = float(np.median(rgb[valid, 1] / rgb[valid].sum(1))) if count else None
                bins = (
                    np.bincount(facebin[faceids[valid]], minlength=10)
                    if count
                    else np.zeros(10, int)
                )
                support[view, i] = (bins >= 5) & qualified
                record = {
                    "session": session,
                    "view": view,
                    "nominal_angle_degrees": view * 9,
                    "split": "development_index" if view % 4 == 2 else "fitting_index",
                    "leaf_id": name,
                    "usable_pixels": count,
                    "qualified": qualified,
                    "green_chromatic_median": green,
                }
                record.update({f"bin_{b}_pixels": int(c) for b, c in enumerate(bins)})
                records.append(record)
                if view % 4 != 2:
                    prior = next(
                        r
                        for r in old
                        if r["session"] == session
                        and int(r["view"]) == view
                        and r["leaf_id"] == name
                    )
                    assert count == int(prior["usable_pixels"]), (
                        session,
                        view,
                        name,
                        count,
                        prior["usable_pixels"],
                    )
                    if count:
                        assert abs(green - float(prior["green_chromatic_median"])) < 1e-12
            if view in [5, 15, 25, 35]:
                # Pure black panels preserve PIL-resized foreground values.
                mm = np.asarray(Image.open(mp).resize((512, 385), Image.Resampling.NEAREST)) > 0
                seg = rgb.copy()
                seg[~mm] = 0
                Image.fromarray(seg).save(assets / f"{session}_view{view:02}_segmented.png")
            if view % 10 == 0:
                print("source_support", session, view, flush=True)
        greedy, gc = select_coverage(support)
        angular = angular_sequence(40)
        covered = np.zeros((n, 10), bool)
        for k in range(40):
            covered |= support[angular[k]]
            curves.append(
                {
                    "session": session,
                    "views": k + 1,
                    "angular_fraction": float(covered.mean()),
                    "posthoc_greedy_fraction": gc[k],
                    "angular_added_view": angular[k],
                    "greedy_added_view": greedy[k],
                }
            )
        np.savez_compressed(
            DEST / f"{session}_bin_support.npz", support=support, greedy=greedy, angular=angular
        )
        summaries.append(
            {
                "session": session,
                "leaves": n,
                "possible_parameter_bins": n * 10,
                "supported_bins_all40": int(np.any(support, axis=0).sum()),
                "qualified_views_by_leaf": [
                    sum(
                        r["qualified"]
                        for r in records
                        if r["session"] == session
                        and r["leaf_id"] == ("early_" if n == 3 else "leaf_") + str(i + 1)
                    )
                    for i in range(n)
                ],
                "fitting_view_records_reproduced": n * 30,
                "posthoc_design": "Greedy selection uses all observed masks after fitting; not a prospective acquisition benchmark.",
            }
        )
    write_csv(DEST / "source_support_all40.csv", records)
    write_csv(DEST / "view_complementarity.csv", curves)
    return {
        "records": len(records),
        "fitting_records_exactly_reproduced": 240,
        "additional_development_index_records": 80,
        "sessions": summaries,
        "bin_definition": "Ten uniform longitudinal model-parameter bins; at least5 usable pixels in a bin and50 per organ-view. Not physical area coverage.",
    }


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    inputs = [
        Path(__file__),
        BASE / "leaf_virtual_experiments.py",
        BASE / "test_leaf_expanded_analyses.py",
        BASE / "leaf_dynamics_geometry.py",
        BASE / "additive_geometry.py",
        GRID / "accepted_continuous_meshes.npz",
        GEO / "leaf_source_appearance_by_view.csv",
        OUT / "sam_traits/view_traits.csv",
        OUT / "organ_inverse_triangles/capture_session_C2_5-1/browser_mesh.json",
    ]
    inputs += [
        V2 / f"{folder}/{file}"
        for folder in ["experiments", "experiments_512"]
        for file in ["directed_neighbour_effects.csv", "screening_decomposition.csv"]
    ]
    hashes = {workspace_relative(p): sha(p) for p in inputs}
    weighting = weights_and_metrics()
    budget = budgets()
    source = source_support(hashes)
    assert all(sha(ROOT / rel) == h for rel, h in hashes.items())
    result = {
        "status": "completed_exploratory_model_and_source_analysis",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "weighting": weighting,
        "budget": budget,
        "source_support": source,
        "input_sha256": hashes,
        "inference": "Designed geometry and posthoc source-support analyses. No calibrated physiological or longitudinal effects.",
    }
    dump(DEST / "expanded_analysis_summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
