"""Reversible rigid-neighbour experiments on frozen source-assisted surfaces."""

from bootstrap import BASE, ROOT, OUT, read_csv, write_csv
from pathlib import Path
import numpy as np
import json, hashlib, datetime, time, argparse
from leaf_virtual_experiments import layered_raster, compose_owners, validate_candidate_ids

DEST = OUT / "leaf_dynamics/2026-10-24/jeb_v5/reposition_fixed_domain"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False), encoding="utf-8")


def event(**v):
    with (DEST / "execution.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **v))
            + "\n"
        )


def rigid_reposition(vertices, donor_mask, pivot, axis, degrees):
    v = np.asarray(vertices, float)
    mask = np.asarray(donor_mask, bool)
    p = np.asarray(pivot, float)
    a = np.asarray(axis, float)
    if (
        v.ndim != 2
        or v.shape[1] != 3
        or mask.shape != (len(v),)
        or p.shape != (3,)
        or a.shape != (3,)
    ):
        raise ValueError("Invalid geometry, mask, pivot or axis shape")
    if (
        not all(np.isfinite(x).all() for x in [v, p, a, np.asarray(degrees)])
        or np.linalg.norm(a) < 1e-12
    ):
        raise ValueError("Finite geometry and a nonzero rotation axis required")
    out = v.copy()
    if degrees == 0:
        return out
    a = a / np.linalg.norm(a)
    q = v[mask] - p
    t = np.radians(degrees)
    out[mask] = (
        p + q * np.cos(t) + np.cross(a, q) * np.sin(t) + (q @ a)[:, None] * a * (1 - np.cos(t))
    )
    return out


def basis(direction):
    n = np.asarray(direction, float)
    n = n / np.linalg.norm(n)
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def domain_anchors(geometries, direction):
    u, v = basis(direction)
    xy = np.concatenate([np.column_stack([g @ u, g @ v]) for g in geometries])
    lo = xy.min(0)
    hi = xy.max(0)
    # Strictly enclosing anchors dominate extrema even after inverse projection.
    # The 1e-9 H margin exceeds float64 round-off and is held fixed for all poses.
    pad = max(float(np.max(hi - lo)), 1.0) * 1e-9
    lo -= pad
    hi += pad
    return np.array([u * x + v * y for x in [lo[0], hi[0]] for y in [lo[1], hi[1]]])


def raster_fixed(vertices, faces, face_organs, direction, resolution, anchors, norgans=6):
    # Unreferenced anchors enforce a common grid. They do not paint pixels or
    # enter nearest-surface ownership; all actual faces retain original indices.
    layers, pa = layered_raster(
        np.vstack([vertices, anchors]), faces, face_organs, direction, resolution
    )
    if len(layers) < norgans:
        layers = np.concatenate(
            [layers, np.full((norgans - len(layers), resolution, resolution), -np.inf)]
        )
    return layers, pa


def transition_budget(before, after, norgans=6, pixel_area=1.0):
    b = np.asarray(before, int)
    a = np.asarray(after, int)
    if b.shape != a.shape or not all(((x >= -1) & (x < norgans)).all() for x in [b, a]):
        raise ValueError("Ownership arrays must share shape and valid IDs")
    matrix = np.bincount(
        (np.where(b < 0, norgans, b) * (norgans + 1) + np.where(a < 0, norgans, a)).ravel(),
        minlength=(norgans + 1) ** 2,
    ).reshape(norgans + 1, norgans + 1)
    old = matrix.sum(1) * pixel_area
    new = matrix.sum(0) * pixel_area
    changes = new - old
    return dict(
        transition_pixels=matrix.tolist(),
        organ_before=old[:-1].tolist(),
        organ_after=new[:-1].tolist(),
        organ_area_changes=changes[:-1].tolist(),
        whole_shoot_change=float(changes[:-1].sum()),
        escape_change=float(changes[-1]),
        closure=float(changes.sum()),
    )


def area(vertices, faces):
    tri = vertices[faces]
    return float(
        np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2
    )


def reposition_state_counts(candidate_ids, directions, resolutions, modes):
    """Count configured states; zero rotation is shared between motion modes."""
    multiplier = len(candidate_ids) * len(directions) * len(resolutions)
    angles = [angle for values in modes.values() for angle in values]
    unique_poses = sum(angle != 0 for angle in angles) + int(0 in angles)
    return multiplier * len(angles), multiplier * unique_poses


def run():
    from additive_geometry import load_geometry

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "cache").mkdir(exist_ok=True)
    saved, mesh, camera, raw, faces, organs = load_geometry()
    height = float(np.ptp(raw[:, 1]))
    fo = organs[faces[:, 0]]
    donor = 3
    receiver = 1
    gridpath = OUT / "additive_experiments/2026-10-24/e08_ambiguity/accepted_continuous_meshes.npz"
    grid = np.load(gridpath)
    ids = validate_candidate_ids(grid["candidate_ids"], len(grid["vertices"]))
    geometries = [raw.astype(float)] + [g.astype(float) for g in grid["vertices"][1:]]
    assert np.array_equal(grid["faces"], faces)
    inputs = [
        Path(__file__),
        BASE / "leaf_virtual_experiments.py",
        BASE / "additive_geometry.py",
        BASE / "test_v5_reposition.py",
        gridpath,
        OUT / "organ_inverse_triangles/capture_session_C2_5-1/organ_model.npz",
        OUT / "organ_inverse_triangles/capture_session_C2_5-1/browser_mesh.json",
    ]
    config = dict(
        donor="local_L4",
        receivers="L1,L2,L3,L5 and stem retained",
        modes={
            "yaw_degrees": list(range(-60, 61, 10)),
            "elevation_degrees": list(range(-30, 31, 5)),
        },
        resolutions=[256, 512],
        candidate_ids=ids,
        configuration_count=len(ids),
        direction_names=["zenith"] + [f"e45_a{a:03}" for a in range(0, 360, 45)],
        input_sha256={str(p): sha(p) for p in inputs},
        fixed_domain="Per configuration and beam, union bounds of every configured pose; anchors are unreferenced vertices, expanded by max(span,1)*1e-9 H to strictly enclose all poses after floating-point reprojection.",
        conditioning="Original saved baseline H throughout; prescribed local organ labels, frozen finite family and nominal geometry. Rigid pivot: first midrib vertex of L4. No collision/contact, elastic, radiometric or physiological model.",
        preregistration="Angles, donors, receivers, finite family and resolutions fixed before outcome evaluation; no beneficial angle is selected for inference.",
    )
    expected, unique_states = reposition_state_counts(
        ids, config["direction_names"], config["resolutions"], config["modes"]
    )
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["signature"] = signature
    if (DEST / "configuration.json").exists():
        assert json.loads((DEST / "configuration.json").read_text(encoding="utf-8")) == config, (
            "Configuration/source changed; use a new versioned output"
        )
    else:
        dump(DEST / "configuration.json", config)
    event(stage="started", signature=signature)
    started = time.time()
    rows = []
    illustrations = {}
    maxarea = 0.0
    maxinverse = 0.0
    maxclosure = 0.0
    directions = [np.array([0.0, 1.0, 0.0])] + [
        np.array(
            [np.cos(np.radians(a)) / np.sqrt(2), 1 / np.sqrt(2), np.sin(np.radians(a)) / np.sqrt(2)]
        )
        for a in range(0, 360, 45)
    ]
    mask = organs == donor
    donor_faces = faces[fo == donor]
    donor_fo = fo[fo == donor]
    local_faces = donor_faces - mesh["organs"][donor]["vertex_start"]
    ss = saved["vertices"].shape[1:3]
    stemstart = saved["vertices"].size // 3
    for cid, vertices0 in zip(ids, geometries):
        vertices = vertices0 / height
        donorgrid = vertices[mask].reshape(*ss, 3)
        pivot = donorgrid[0].mean(0)
        tangent = donorgrid[-1].mean(0) - pivot
        h = tangent * np.array([1.0, 0.0, 1.0])
        lift_axis = np.cross(h, [0.0, 1.0, 0.0])
        lift_axis /= np.linalg.norm(lift_axis)
        transformed = {}
        axes = {"yaw_degrees": np.array([0.0, 1.0, 0.0]), "elevation_degrees": lift_axis}
        original_area = area(vertices, donor_faces)
        for mode, angles in config["modes"].items():
            for angle in angles:
                v = rigid_reposition(vertices, mask, pivot, axes[mode], angle)
                assert np.array_equal(v[~mask], vertices[~mask])
                ae = abs(area(v, donor_faces) - original_area)
                maxarea = max(maxarea, ae)
                restored = rigid_reposition(v, mask, pivot, axes[mode], -angle)
                ie = float(np.max(abs(restored - vertices)))
                maxinverse = max(maxinverse, ie)
                assert ae < 1e-12 and ie < 1e-12
                transformed[mode, angle] = v
        for di, (name, direction) in enumerate(zip(config["direction_names"], directions)):
            anchors = domain_anchors(list(transformed.values()), direction)
            for resolution in config["resolutions"]:
                cache = DEST / "cache" / f"c{cid}_{name}_{resolution}.json"
                if cache.exists():
                    saved_cache = json.loads(cache.read_text(encoding="utf-8"))
                    assert saved_cache["signature"] == signature
                    rows.extend(saved_cache["rows"])
                    continue
                base, pa = raster_fixed(vertices, faces, fo, direction, resolution, anchors)
                before = compose_owners(base, range(6))
                footprint = np.isfinite(base[receiver])
                den = int(footprint.sum())
                assert den > 0
                packet = []
                for mode, angles in config["modes"].items():
                    for angle in angles:
                        if angle == 0:
                            after = before
                            layers = base
                        else:
                            only, qa = raster_fixed(
                                transformed[mode, angle],
                                donor_faces,
                                donor_fo,
                                direction,
                                resolution,
                                anchors,
                            )
                            assert qa == pa
                            layers = base.copy()
                            layers[donor] = only[donor]
                            after = compose_owners(layers, range(6))
                        assert np.array_equal(layers[receiver], base[receiver])
                        budget = transition_budget(before, after, 6, pa)
                        maxclosure = max(maxclosure, abs(budget["closure"]))
                        if angle == 0:
                            assert np.array_equal(before, after)
                        row = dict(
                            candidate_id=cid,
                            resolution=resolution,
                            direction=name,
                            mode=mode,
                            angle_degrees=angle,
                            isolated_receiver_pixels=den,
                            pixel_area_H2=pa,
                            receiver_before_fraction=int((before == receiver).sum()) / den,
                            receiver_after_fraction=int((after == receiver).sum()) / den,
                            receiver_change_fraction=int(
                                np.count_nonzero(after == receiver)
                                - np.count_nonzero(before == receiver)
                            )
                            / den,
                            whole_before_H2=float(np.count_nonzero(before >= 0) * pa),
                            whole_after_H2=float(np.count_nonzero(after >= 0) * pa),
                            whole_change_H2=budget["whole_shoot_change"],
                            transmission_change_H2=budget["escape_change"],
                            closure_H2=budget["closure"],
                            receiver_unchanged=True,
                            transition_pixels=budget["transition_pixels"],
                            organ_before=budget["organ_before"],
                            organ_after=budget["organ_after"],
                        )
                        packet.append(row)
                        if (
                            cid == 0
                            and name == "zenith"
                            and resolution == 512
                            and mode == "elevation_degrees"
                            and angle in [-30, 0, 30]
                        ):
                            illustrations[f"owner_elevation_{angle}"] = after
                            illustrations[f"receiver_elevation_{angle}"] = after == receiver
                            illustrations["fixed_receiver_footprint"] = footprint
                dump(cache, dict(signature=signature, rows=packet))
                rows.extend(packet)
                if illustrations:
                    np.savez_compressed(DEST / "baseline_intervention_maps.npz", **illustrations)
                event(
                    stage="direction_completed",
                    candidate=cid,
                    direction=name,
                    resolution=resolution,
                    rows=len(packet),
                    elapsed_seconds=round(time.time() - started, 1),
                )
                print(
                    f"candidate {cid} {name} {resolution}: {len(rows)}/{expected} rows", flush=True
                )
    if illustrations:
        np.savez_compressed(DEST / "baseline_intervention_maps.npz", **illustrations)
    assert (
        len(rows) == expected
        and len(
            {
                (r["candidate_id"], r["resolution"], r["direction"], r["mode"], r["angle_degrees"])
                for r in rows
            }
        )
        == expected
    )
    dump(DEST / "all_reposition_states.json", dict(records=rows))
    fields = [k for k in rows[0] if k not in ["transition_pixels", "organ_before", "organ_after"]]
    write_csv(DEST / "reposition_states.csv", [{k: r[k] for k in fields} for r in rows])
    summaries = []
    for cid in ids:
        for resolution in config["resolutions"]:
            for mode, angles in config["modes"].items():
                for angle in angles:
                    rr = [
                        r
                        for r in rows
                        if r["candidate_id"] == cid
                        and r["resolution"] == resolution
                        and r["mode"] == mode
                        and r["angle_degrees"] == angle
                    ]
                    assert len(rr) == 9
                    summaries.append(
                        dict(
                            candidate_id=cid,
                            resolution=resolution,
                            mode=mode,
                            angle_degrees=angle,
                            equal9_receiver_change_fraction=float(
                                np.mean([r["receiver_change_fraction"] for r in rr])
                            ),
                            equal9_whole_change_H2=float(
                                np.mean([r["whole_change_H2"] for r in rr])
                            ),
                            zenith_receiver_change_fraction=next(
                                r["receiver_change_fraction"]
                                for r in rr
                                if r["direction"] == "zenith"
                            ),
                        )
                    )
    write_csv(DEST / "direction_averaged_responses.csv", summaries)
    convergence = []
    for r in [r for r in summaries if r["resolution"] == 512]:
        low = next(
            s
            for s in summaries
            if all(s[k] == r[k] for k in ["candidate_id", "mode", "angle_degrees"])
            and s["resolution"] == 256
        )
        convergence.append(
            dict(
                candidate_id=r["candidate_id"],
                mode=r["mode"],
                angle_degrees=r["angle_degrees"],
                receiver_abs_difference=abs(
                    r["equal9_receiver_change_fraction"] - low["equal9_receiver_change_fraction"]
                ),
                whole_abs_difference_H2=abs(
                    r["equal9_whole_change_H2"] - low["equal9_whole_change_H2"]
                ),
            )
        )
    write_csv(DEST / "two_grid_differences.csv", convergence)
    for p, h in config["input_sha256"].items():
        assert sha(p) == h
    output = dict(
        status="completed",
        state_rows=len(rows),
        unique_shoot_beam_resolution_states=unique_states,
        configuration_count=len(ids),
        candidate_ids=ids,
        averaged_rows=len(summaries),
        all_receivers_unchanged=True,
        max_donor_area_error_H2=maxarea,
        max_inverse_rotation_error_H=maxinverse,
        max_transition_closure_H2=max(abs(r["closure_H2"]) for r in rows),
        maximum_equal9_receiver_two_grid_difference=max(
            r["receiver_abs_difference"] for r in convergence
        ),
        elapsed_seconds=time.time() - started,
        signature=signature,
        baseline_512=[r for r in summaries if r["candidate_id"] == 0 and r["resolution"] == 512],
        interpretation="Reversible model interventions under prescribed geometry and light. No collision/contact mechanics, physical leaf manipulation, anatomy accuracy or photosynthesis is established.",
    )
    dump(DEST / "summary.json", output)
    event(stage="completed", state_rows=len(rows))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    run()
