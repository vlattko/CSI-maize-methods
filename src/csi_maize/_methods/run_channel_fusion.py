"""Audit all frozen RGB/NoIR filename pairs; never infer calibrated spectra."""

from multimodal_leaf_core import (
    BASE,
    ROOT,
    OUT,
    DEST as OLD,
    sha,
    dump,
    affine_align,
    mask_iou,
    assign_tracks,
)
from bootstrap import read_csv, write_csv, validate_rotation_records, method_source_hashes
from common import safe_source
from channel_fusion_core import *
from PIL import Image, ExifTags
from pathlib import Path
import json, time, collections, hashlib

DEST = OUT / "channel_fusion/2026-10-24"
OFFSETS = [(0, 0), (-2, 0), (2, 0), (0, -2), (0, 2), (-4, 0), (4, 0), (0, -4), (0, 4)]


def input_fingerprint(rows):
    """Bind numerical cache reuse to actual data and every local helper source."""
    from common import workspace_relative

    inputs = method_source_hashes("run_channel_fusion.py")
    paths = [
        OUT / "sam_traits/view_traits.csv",
        OLD / "paired_frames.csv",
        OLD / "phase_audit.csv",
        BASE / "config.json",
    ]
    paths += sorted((OLD / "predictions").glob("*.json")) + sorted(
        (OLD / "predictions").glob("*.npz")
    )
    if (ROOT / "input_bindings.json").is_file():
        paths.append(ROOT / "input_bindings.json")
    for row in rows:
        source = safe_source(row["source"])
        actual = sha(source)
        if actual != row["source_sha256"]:
            raise ValueError("Source image changed: " + row["source"])
        paths.extend([source, safe_source(row["mask"])])
    for path in sorted(set(paths)):
        inputs[workspace_relative(path)] = sha(path)
    configuration = dict(
        input_sha256=inputs,
        policy=dict(POLICY, proposal_iou_min=0.25),
        spatial_offsets_RGB_grid_px=OFFSETS,
        channels=CHANNELS,
    )
    signature = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return inputs, signature


def cache_payload_hash(result):
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in result.items() if k != "payload_sha256"},
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def read_session_cache(path, signature, session):
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("signature") != signature:
        raise ValueError("Source/config changed; use a fresh output directory")
    if result.get("payload_sha256") != cache_payload_hash(result):
        raise ValueError("Session cache payload changed: " + str(path))
    frames = result.get("frames", [])
    if (
        len(frames) != 40
        or {f["view"] for f in frames} != set(range(40))
        or any(f["session"] != session for f in frames)
    ):
        raise ValueError("Invalid cached rotation: " + session)
    return result


def load_prediction(session, cam):
    key = Path(session).name + "__" + cam
    records = json.loads((OLD / "predictions" / (key + ".json")).read_text(encoding="utf-8"))[
        "records"
    ]
    with np.load(OLD / "predictions" / (key + ".npz")) as z:
        shape = tuple(z["mask_shape"])
        masks = (
            np.unpackbits(z["packed_masks"], axis=1, count=int(np.prod(shape))).reshape(-1, *shape)
            > 0
        )
    byview = collections.defaultdict(list)
    for r, m in zip(records, masks):
        byview[int(r["view"])].append((r, m))
    assert len(records) == len(masks)
    return byview


def raw(r):
    path = ROOT / r["source"]
    assert sha(path) == r["source_sha256"]
    with Image.open(path) as im:
        ex = im.getexif()
        tags = {
            ExifTags.TAGS.get(k, str(k)): str(v)
            for k, v in ex.items()
            if ExifTags.TAGS.get(k, str(k))
            in [
                "ExposureTime",
                "ISOSpeedRatings",
                "PhotographicSensitivity",
                "WhiteBalance",
                "ExposureBiasValue",
            ]
        }
        # Common EXIF exposure data are in the nested Exif IFD rather than the top level.
        try:
            tags.update(
                {
                    ExifTags.TAGS.get(k, str(k)): str(v)
                    for k, v in ex.get_ifd(34665).items()
                    if ExifTags.TAGS.get(k, str(k))
                    in [
                        "ExposureTime",
                        "ISOSpeedRatings",
                        "PhotographicSensitivity",
                        "WhiteBalance",
                        "ExposureBiasValue",
                    ]
                }
            )
        except (KeyError, TypeError, ValueError):
            pass
        image = np.asarray(im.convert("RGB").resize((512, 385), Image.Resampling.LANCZOS))
    mask = np.asarray(Image.open(ROOT / r["mask"]).resize((512, 385), Image.Resampling.NEAREST)) > 0
    return image, mask, tags


def stats(a, mask):
    sample = a[mask].astype(float)
    n = len(sample)
    return dict(
        pixels=n,
        median=[float(x) for x in np.median(sample, axis=0)] if n else [None] * 3,
        dark_fraction=[float(x) for x in np.mean(sample <= 1, axis=0)] if n else [None] * 3,
        clipped_fraction=[float(x) for x in np.mean(sample >= 254, axis=0)] if n else [None] * 3,
    )


def camera_row(record, a, mask, whole):
    support = interior(mask & whole)
    s = a[support].astype(float)
    st = stats(a, support)
    row = {
        k: record[k]
        for k in [
            "session",
            "camera",
            "view",
            "detection_index",
            "scientific_split",
            "confidence",
            "source",
            "checkpoint_sha256",
            "circular_track_id",
        ]
    }
    row.update(interior_pixels=int(support.sum()), source_sha256=record["source_sha256_recorded"])
    for j, c in enumerate("RGB"):
        row[c + "_DN_median"] = st["median"][j]
        row[c + "_dark_fraction"] = st["dark_fraction"][j]
        row[c + "_clipped_fraction"] = st["clipped_fraction"][j]
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        nd = normalized_difference(s[:, i], s[:, j])
        good = np.isfinite(nd)
        row["ND_" + "RGB"[i] + "_" + "RGB"[j]] = (
            float(np.median(nd[good])) if good.sum() >= 40 else None
        )
        row["n_" + "RGB"[i] + "_" + "RGB"[j]] = int(good.sum())
    valid = np.all((s > 1) & (s < 254), axis=1)
    if valid.sum() >= 40:
        chrom = s[valid] / s[valid].sum(axis=1, keepdims=True)
        row["GCC_display_DN"] = float(np.median(chrom[:, 1]))
        row["ExG_normalized_display_DN"] = float(np.median(3 * chrom[:, 1] - 1))
    else:
        row["GCC_display_DN"] = None
        row["ExG_normalized_display_DN"] = None
    return row, support


def registration_with_empty_guard(a, b):
    if a.sum() < 10 or b.sum() < 10:
        return np.array([[1, 0, 0], [0, 1, 0]], np.float32), 0.0, "empty_or_tiny_foreground"
    return affine_align(a, b)


def main():
    started = time.time()
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "sessions").mkdir(exist_ok=True)
    source = OUT / "sam_traits/view_traits.csv"
    rows = read_csv(source)
    lookup = {(r["session"], r["camera"], int(r["view"])): r for r in rows}
    pairs = read_csv(OLD / "paired_frames.csv")
    acquisitions = validate_rotation_records(rows)
    if len(lookup) != len(rows) or len(pairs) != len(rows) // 2:
        raise ValueError("Duplicate or missing nominal pairs")
    policy = POLICY.copy()
    policy["proposal_iou_min"] = 0.25
    inputhash, signature = input_fingerprint(rows)
    previous = DEST / "run_configuration.json"
    if (
        previous.exists()
        and json.loads(previous.read_text(encoding="utf-8")).get("signature") != signature
    ):
        raise ValueError("Source/config changed; use a fresh output directory")
    dump(
        DEST / "run_configuration.json",
        dict(
            policy=policy,
            spatial_offsets_RGB_grid_px=OFFSETS,
            channels=CHANNELS,
            feature_formula="ND_a_b=(a-b)/(a+b), recorded processed JPEG DN",
            input_sha256=inputhash,
            signature=signature,
            scope="Exploratory correspondence hypotheses, not calibrated multispectral measurements. Thresholds are engineering gates, not estimated anatomical accuracy.",
        ),
    )
    dump(
        DEST / "dependency_binding.json",
        dict(
            bindings={path: dict(sha256=digest) for path, digest in inputhash.items()},
            signature=signature,
            scope="Actual source/mask/prediction/phase/configuration hashes and complete local import closure; no delivery-package dependency.",
        ),
    )
    allframes = []
    alltargets = []
    allcamera = []
    allsensitivity = []
    allgain = []
    allobservations = []
    phase = {r["session"]: r for r in read_csv(OLD / "phase_audit.csv")}
    for session in sorted({r["session"] for r in pairs}):
        name = Path(session).name
        cache = DEST / "sessions" / (name + ".json")
        if cache.exists():
            result = read_session_cache(cache, signature, session)
        else:
            pred = {c: load_prediction(session, c) for c in ["rgb", "noir"]}
            frames = []
            targets = []
            camera = []
            sensitivity = []
            gains = []
            observations = []
            for view in range(40):
                rr = {c: lookup[session, c, view] for c in ["rgb", "noir"]}
                loaded = {c: raw(rr[c]) for c in ["rgb", "noir"]}
                a, am, ae = loaded["rgb"]
                b, bm, be = loaded["noir"]
                matrix, whole_iou, status = registration_with_empty_guard(am, bm)
                rgb = pred["rgb"][view]
                noir = pred["noir"][view]
                ints = {}
                for c, items in [("rgb", rgb), ("noir", noir)]:
                    for r, m in items:
                        cr, ii = camera_row(r, loaded[c][0], m, loaded[c][1])
                        camera.append(cr)
                        ints[c, int(r["detection_index"])] = ii
                # Geometry remains in RGB coordinates for both assignment and index sampling.
                warped_b, field = warp_to_rgb(b.astype(np.float32), matrix, am.shape)
                warped_masks = [
                    warp_to_rgb(m.astype(np.uint8), matrix, am.shape, True)[0] > 0 for _, m in noir
                ]
                ious = np.array([[mask_iou(m, wm) for wm in warped_masks] for _, m in rgb]).reshape(
                    len(rgb), len(noir)
                )
                proposals = match_instances(ious, 0.25, POLICY["assignment_margin_min"])
                accepted = {}
                candidate_by_obs = {}
                shifted = {}
                for dx, dy in OFFSETS:
                    inv = cv2.invertAffineTransform(matrix)
                    inv[:, 2] += [dx, dy]
                    shiftmatrix = cv2.invertAffineTransform(inv)
                    wb, fov = warp_to_rgb(b.astype(np.float32), shiftmatrix, am.shape)
                    wm = [
                        warp_to_rgb(m.astype(np.uint8), shiftmatrix, am.shape, True)[0] > 0
                        for _, m in noir
                    ]
                    tiou = np.array([[mask_iou(m, nn) for nn in wm] for _, m in rgb]).reshape(
                        len(rgb), len(noir)
                    )
                    tm = match_instances(tiou, 0.25, POLICY["assignment_margin_min"])
                    passed = {
                        (x["i"], x["j"])
                        for x in tm
                        if x["accepted"]
                        and x["iou"] >= POLICY["leaf_iou_min"]
                        and whole_iou >= POLICY["whole_iou_min"]
                    }
                    shifted[dx, dy] = (wb, fov, wm, passed, shiftmatrix)
                for p in proposals:
                    i, j = p["i"], p["j"]
                    r, rm = rgb[i]
                    n, nm = noir[j]
                    ri = ints["rgb", int(r["detection_index"])]
                    ni = ints["noir", int(n["detection_index"])]
                    target = f"{name}__v{view:03}__r{r['detection_index']}__n{n['detection_index']}"
                    wn = warp_to_rgb(ni.astype(np.uint8), matrix, am.shape, True)[0] > 0
                    common = ri & wn & field
                    retained = float(common.sum() / max(1, min(ri.sum(), wn.sum())))
                    reasons = []
                    if whole_iou < POLICY["whole_iou_min"]:
                        reasons.append("whole_mask_IoU_below_gate")
                    if p["iou"] < POLICY["leaf_iou_min"]:
                        reasons.append("leaf_IoU_below_gate")
                    if not p["accepted"]:
                        reasons.append("ambiguous_assignment")
                    geometry_ok = not reasons
                    if common.sum() < POLICY["min_common_interior_pixels"]:
                        reasons.append("too_few_common_interior_pixels")
                    if retained < POLICY["min_interior_overlap_fraction"]:
                        reasons.append("low_common_interior_fraction")
                    pixel_ok = not reasons
                    record = dict(
                        target_id=target,
                        session=session,
                        view=view,
                        rgb_detection_index=int(r["detection_index"]),
                        noir_detection_index=int(n["detection_index"]),
                        rgb_source=r["source"],
                        noir_source=n["source"],
                        rgb_source_sha256=r["source_sha256_recorded"],
                        noir_source_sha256=n["source_sha256_recorded"],
                        rgb_candidate_track=r["circular_track_id"],
                        noir_candidate_track=n["circular_track_id"],
                        scientific_split=r["scientific_split"],
                        whole_mask_iou=whole_iou,
                        leaf_iou=p["iou"],
                        assignment_row_margin=p["row_margin"],
                        assignment_col_margin=p["col_margin"],
                        geometry_candidate=geometry_ok,
                        pixel_hypothesis=pixel_ok,
                        failure_reasons=reasons,
                        common_interior_pixels=int(common.sum()),
                        common_interior_fraction=retained,
                        registration_source_to_target=matrix.tolist(),
                        nominal_phase_best_shift=int(phase[session]["best_shift"]),
                        independent_identity_review=False,
                    )
                    if geometry_ok:
                        accepted[i, j] = target
                        record["organ_summary"] = organ_features(
                            a[ri].astype(float), b[ni].astype(float)
                        )
                    if pixel_ok:
                        samples = np.column_stack([a[common], warped_b[common]])
                        record["common_support_indices"] = pair_features(samples)
                        record["all_six_valid_common_pixels"] = int(
                            np.all((samples > 1) & (samples < 254), axis=1).sum()
                        )
                        supports = {}
                        for dx, dy in OFFSETS:
                            wb, fov, wm, passed, shiftmatrix = shifted[dx, dy]
                            # Erode the shifted instance and intersect whole-plant-clean RGB interior.
                            wnshift = (
                                warp_to_rgb(ni.astype(np.uint8), shiftmatrix, am.shape, True)[0] > 0
                            )
                            cc = ri & wnshift & fov
                            supports[dx, dy] = cc
                            features = pair_features(np.column_stack([a[cc], wb[cc]]))
                            for key, ft in features.items():
                                sensitivity.append(
                                    dict(
                                        target_id=target,
                                        session=session,
                                        dx=dx,
                                        dy=dy,
                                        assignment_retained=(i, j) in passed,
                                        common_pixels=int(cc.sum()),
                                        index=key,
                                        n=ft["n"],
                                        pixel_median=ft["pixel_median"],
                                    )
                                )
                        core2 = np.logical_and.reduce(
                            [s for (dx, dy), s in supports.items() if abs(dx) + abs(dy) <= 2]
                        )
                        core4 = np.logical_and.reduce(list(supports.values()))
                        record["persistent_core_2px_pixels"] = int(core2.sum())
                        record["persistent_core_4px_pixels"] = int(core4.sum())
                        record["persistent_core_2px_indices"] = pair_features(
                            np.column_stack([a[core2], warped_b[core2]])
                        )
                        for dx, dy in OFFSETS:
                            if abs(dx) + abs(dy) > 2:
                                continue
                            wb, fov, wm, passed, shiftmatrix = shifted[dx, dy]
                            features = pair_features(np.column_stack([a[core2], wb[core2]]))
                            for key, ft in features.items():
                                sensitivity.append(
                                    dict(
                                        target_id=target,
                                        session=session,
                                        dx=dx,
                                        dy=dy,
                                        assignment_retained=(i, j) in passed,
                                        common_pixels=int(core2.sum()),
                                        index=key,
                                        n=ft["n"],
                                        pixel_median=ft["pixel_median"],
                                        support_mode="persistent_2px_core",
                                    )
                                )
                        # DN scaling is a counterfactual ratio stress test, not exposure correction.
                        for gain in POLICY["relative_noir_DN_scales"]:
                            for ii, jj in PAIRS:
                                if ii >= 3 or jj < 3:
                                    continue
                                aa, bb = samples[:, ii], samples[:, jj]
                                valid = np.isfinite(normalized_difference(aa, bb))
                                vv = normalized_difference(aa[valid], gain * bb[valid], guard=False)
                                gains.append(
                                    dict(
                                        target_id=target,
                                        session=session,
                                        index=CHANNELS[ii] + "__" + CHANNELS[jj],
                                        relative_noir_DN_scale=gain,
                                        n=len(vv),
                                        pixel_median=float(np.median(vv))
                                        if len(vv) >= 40
                                        else None,
                                    )
                                )
                    targets.append(record)
                    candidate_by_obs["rgb", int(r["detection_index"])] = (
                        target,
                        "supported_candidate" if geometry_ok else "rejected_proposal",
                    )
                    candidate_by_obs["noir", int(n["detection_index"])] = (
                        target,
                        "supported_candidate" if geometry_ok else "rejected_proposal",
                    )
                for c, items in [("rgb", rgb), ("noir", noir)]:
                    for r, m in items:
                        key = c, int(r["detection_index"])
                        tid, state = candidate_by_obs.get(key, ("", "unmatched"))
                        observations.append(
                            dict(
                                session=session,
                                view=view,
                                camera=c,
                                detection_index=key[1],
                                target_id=tid,
                                state=state,
                                source=r["source"],
                            )
                        )
                controlview = (view + 20) % 40
                controlmask = (
                    np.asarray(
                        Image.open(ROOT / lookup[session, "noir", controlview]["mask"]).resize(
                            (512, 385), Image.Resampling.NEAREST
                        )
                    )
                    > 0
                )
                control_warp = (
                    warp_to_rgb(controlmask.astype(np.uint8), matrix, am.shape, True)[0] > 0
                )
                same_warp = warp_to_rgb(bm.astype(np.uint8), matrix, am.shape, True)[0] > 0
                frames.append(
                    dict(
                        session=session,
                        view=view,
                        scientific_split=next(
                            x["scientific_split"] for x in pairs if x["session"] == session
                        ),
                        rgb_source=rr["rgb"]["source"],
                        noir_source=rr["noir"]["source"],
                        rgb_mask_sha256=sha(ROOT / rr["rgb"]["mask"]),
                        noir_mask_sha256=sha(ROOT / rr["noir"]["mask"]),
                        affine_source_to_target=matrix.tolist(),
                        whole_mask_iou=whole_iou,
                        whole_mask_iou_rgb_grid=mask_iou(am, same_warp),
                        status=status,
                        rgb_detections=len(rgb),
                        noir_detections=len(noir),
                        proposals=len(proposals),
                        supported_targets=len(accepted),
                        pixel_hypotheses=sum(
                            t["pixel_hypothesis"] for t in targets if t["view"] == view
                        ),
                        opposite_view_fixed_transform_iou=mask_iou(am, control_warp),
                        opposite_view_control="Different NoIR angle with same transform, no refitting; diagnostic only, not anatomical ground truth.",
                        rgb_EXIF=ae,
                        noir_EXIF=be,
                        phase_best_shift=int(phase[session]["best_shift"]),
                        rgb_channel_qc=stats(a, interior(am)),
                        noir_channel_qc=stats(b, interior(bm)),
                    )
                )
            result = dict(
                signature=signature,
                frames=frames,
                targets=targets,
                camera=camera,
                sensitivity=sensitivity,
                gains=gains,
                observations=observations,
            )
            result["payload_sha256"] = cache_payload_hash(result)
            dump(cache, result)
        allframes.extend(result["frames"])
        alltargets.extend(result["targets"])
        allcamera.extend(result["camera"])
        allsensitivity.extend(result["sensitivity"])
        allgain.extend(result["gains"])
        allobservations.extend(result["observations"])
        print(
            "FUSED",
            name,
            "targets",
            sum(t["geometry_candidate"] for t in result["targets"]),
            "pixel hypotheses",
            sum(t["pixel_hypothesis"] for t in result["targets"]),
            flush=True,
        )
    dump(DEST / "frame_registration.json", dict(records=allframes))
    dump(DEST / "shared_targets.json", dict(records=alltargets))
    write_csv(DEST / "camera_observations.csv", allcamera)
    write_csv(DEST / "observation_states.csv", allobservations)
    write_csv(DEST / "registration_perturbations.csv", allsensitivity)
    write_csv(DEST / "relative_DN_scaling.csv", allgain)
    ft = []
    org = []
    for t in alltargets:
        for key, f in t.get("common_support_indices", {}).items():
            ft.append(
                dict(target_id=t["target_id"], session=t["session"], view=t["view"], index=key, **f)
            )
        for key, value in (
            t.get("organ_summary", {}).get("nd_of_separate_visible_medians", {}).items()
        ):
            org.append(
                dict(
                    target_id=t["target_id"],
                    session=t["session"],
                    view=t["view"],
                    index=key,
                    ND_of_separate_visible_medians=value,
                )
            )
    write_csv(DEST / "common_support_indices.csv", ft)
    write_csv(DEST / "organ_summary_indices.csv", org)
    counts = collections.Counter(x["state"] for x in allobservations)
    summary = dict(
        status="executed_exploratory_DN_fusion",
        images=len(rows),
        pairs=len(allframes),
        acquisitions=len({x["session"] for x in allframes}),
        observations=len(allcamera),
        proposal_targets=len(alltargets),
        geometry_candidates=sum(t["geometry_candidate"] for t in alltargets),
        pixel_hypotheses=sum(t["pixel_hypothesis"] for t in alltargets),
        observation_states=dict(counts),
        whole_mask_iou_median=float(np.median([x["whole_mask_iou"] for x in allframes])),
        nominal_same_index_phase_best_acquisitions=sum(
            int(p["best_shift"]) == 0 for p in phase.values()
        ),
        all_six_valid_pixel_targets=sum(
            t.get("all_six_valid_common_pixels", 0) >= 40 for t in alltargets
        ),
        EXIF_exposure_available_images=sum(
            "ExposureTime" in x[c + "_EXIF"] for x in allframes for c in ["rgb", "noir"]
        ),
        elapsed_seconds=time.time() - started,
        limitations=[
            "Geometry and pixel support are engineering hypotheses, not verified anatomy or corresponding tissue.",
            "Shared weak teachers make cross-camera agreement non-independent.",
            "No calibration, camera spectral responses, camera synchronization or physical cross-date leaf identities established.",
            "All indices use processed JPEG DN. NoIR channels are not independent NIR bands.",
            "Forty angular views of one acquisition are repeated technical observations, not biological replicates.",
        ],
    )
    dump(DEST / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
