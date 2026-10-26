"""Independent robust camera/observed-tip fit; no held-out image/mask access.

Uses the source-assisted identities in organ_observations/observations.json.
Perspective principal point is image center, not the basal reference projection.
All 86 extracted tips remain in the diagnostics; association weights are frozen
before camera fitting. This estimates a conditional camera, not calibrated truth.
"""

from bootstrap import OUT
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.optimize import least_squares


def project(
    points,
    theta,
    elevation_deg,
    distance,
    roll_deg=3.0,
    center=(512.0, 384.5),
    basal=(481.0, 768.0),
):
    p = np.asarray(points, float)
    theta = np.asarray(theta, float)
    cr, sr = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    ce, se = np.cos(np.radians(elevation_deg)), np.sin(np.radians(elevation_deg))
    c, s = np.cos(theta), np.sin(theta)
    xx = c * p[:, 0] + s * p[:, 2]
    zz = -s * p[:, 0] + c * p[:, 2]
    tx = cr * (basal[0] - center[0]) + sr * (basal[1] - center[1])
    ty = sr * (basal[0] - center[0]) - cr * (basal[1] - center[1])
    camera_x = xx + tx
    camera_up = ce * p[:, 1] + se * zz + ty
    depth = distance + ce * zz - se * p[:, 1] if distance else np.ones_like(xx)
    scale = distance / np.maximum(depth, 1.0) if distance else np.ones_like(xx)
    uv = np.column_stack(
        [
            center[0] + scale * (cr * camera_x + sr * camera_up),
            center[1] + scale * (sr * camera_x - cr * camera_up),
        ]
    )
    return uv, depth


def main():
    folder = OUT / "organ_observations/capture_session_C2_5-1"
    path = folder / "observations.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    leaves = source["leaves"]
    ids = {l["leaf_id"]: i for i, l in enumerate(leaves)}
    observations = []
    for view in source["views"]:
        assert view["view"] % 4 != 2
        for leaf in view["leaves"]:
            flags = leaf["uncertainty_flags"]
            # Junction ambiguity affects attachments, not necessarily the tip ID.
            weight = (
                1.0
                if leaf["association_method"] == "assisted_source_initialization"
                else max(0.2, 1 / (1 + leaf["association_cost"] ** 2))
            )
            if "competing_leaf_identity" in flags:
                weight *= 0.25
            observations.append(
                dict(
                    view=view["view"],
                    leaf_id=leaf["leaf_id"],
                    tip_xy=leaf["tip_xy"],
                    theta=view["nominal_angle_rad"],
                    weight=float(weight),
                    association_method=leaf["association_method"],
                    uncertainty_flags=flags,
                )
            )
    index = np.array([ids[o["leaf_id"]] for o in observations])
    theta = np.array([o["theta"] for o in observations])
    target = np.array([o["tip_xy"] for o in observations])
    weight = np.sqrt([o["weight"] for o in observations])
    initial_xyz = np.array(
        [
            [
                l["trajectory_coefficients_derolled_xy"][1][0],
                l["trajectory_coefficients_derolled_xy"][0][1],
                l["trajectory_coefficients_derolled_xy"][2][0],
            ]
            for l in leaves
        ]
    )
    results = []
    for perspective in [False, True]:

        def residual(parameters):
            elevation = parameters[0]
            distance = np.exp(parameters[1]) if perspective else 0.0
            xyz = parameters[2:].reshape(-1, 3)
            predicted, depth = project(xyz[index], theta, elevation, distance)
            reprojection = ((predicted - target) * weight[:, None]).ravel()
            barrier = np.maximum(30 - depth, 0.0) if perspective else np.zeros(len(depth))
            return np.r_[reprojection, barrier]

        fits = []
        starts = (
            [(e, d) for e in [-15.0, 0.0, 15.0, 24.0] for d in [650.0, 1200.0, 2200.0]]
            if perspective
            else [(e, 1200.0) for e in [-20.0, 0.0, 20.0]]
        )
        for elevation, distance in starts:
            initial = np.r_[elevation, np.log(distance), initial_xyz.ravel()]
            fit = least_squares(
                residual,
                initial,
                bounds=(
                    np.r_[-25.0, np.log(400.0), np.tile([-1800.0, 0.0, -1800.0], len(leaves))],
                    np.r_[25.0, np.log(2400.0), np.tile([1800.0, 1800.0, 1800.0], len(leaves))],
                ),
                loss="soft_l1",
                f_scale=8.0,
                max_nfev=600,
            )
            fits.append(fit)
        fit = min(fits, key=lambda x: x.cost)
        elevation = float(fit.x[0])
        distance = float(np.exp(fit.x[1])) if perspective else 0.0
        xyz = fit.x[2:].reshape(-1, 3)
        predicted, depth = project(xyz[index], theta, elevation, distance)
        errors = np.linalg.norm(predicted - target, axis=1)
        records = [
            dict(
                **o,
                predicted_xy=predicted[i].tolist(),
                error_px=float(errors[i]),
                depth=float(depth[i]),
            )
            for i, o in enumerate(observations)
        ]
        result = dict(
            model="perspective_image_center" if perspective else "orthographic",
            camera=dict(
                elevation_deg=elevation,
                distance=distance,
                focal_length_px=distance if perspective else None,
                roll_deg=3.0,
                principal_point_xy=[512.0, 384.5],
                basal_reference_projection_xy=[481.0, 768.0],
                coordinate_gauge="focal length in pixels equals camera distance in chosen arbitrary world scale",
            ),
            robust_objective=float(fit.cost),
            loss="soft_l1 with f_scale8px, association-weighted residuals; positive-depth barrier",
            median_error_px=float(np.median(errors)),
            rmse_px=float(np.sqrt(np.mean(errors**2))),
            n_observations=len(observations),
            n_positive_depth=int((depth > 0).sum()),
            n_residuals_below_5px=int((errors < 5).sum()),
            n_residuals_below_15px=int((errors < 15).sum()),
            optimizer_success=bool(fit.success),
            optimizer_message=str(fit.message),
            start_count=len(starts),
            start_objectives=[float(f.cost) for f in fits],
            leaves=[
                dict(
                    leaf_id=l["leaf_id"],
                    tip_xyz=xyz[j].tolist(),
                    observation_count=int((index == j).sum()),
                    median_error_px=float(np.median(errors[index == j])),
                    rmse_px=float(np.sqrt(np.mean(errors[index == j] ** 2))),
                )
                for j, l in enumerate(leaves)
            ],
            residuals=records,
        )
        results.append(result)
    payload = dict(
        session=source["session"],
        source_observations_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_file=str(path.relative_to(OUT)),
        train_indices=source["train_indices"],
        heldout_images_or_masks_accessed=False,
        selection="Compare identical observed tips and frozen association weights; camera family selected on training reprojection only.",
        association_weight_rule="seed1; algorithmic max(.2,1/(1+association_cost^2)); competing-ID flag multiplies .25; attachment flags do not alter tip weights",
        results=results,
        limitations=[
            "Source-assisted leaf IDs are hypotheses, not independent ground truth.",
            "Angle schedule is the uniform one-turn assumption; no motor telemetry recovered.",
            "Camera fit explains observed tip points, not whole leaf geometry or physical focal calibration.",
            "Pose/scale/shape and motion confounding remains; no held-out predictive score here.",
            "Per-leaf observations share source masks and association fitting; errors are not independent samples.",
        ],
    )
    (folder / "camera_tip_fit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in [
                        "model",
                        "camera",
                        "robust_objective",
                        "median_error_px",
                        "rmse_px",
                        "n_residuals_below_5px",
                        "n_residuals_below_15px",
                        "n_positive_depth",
                        "optimizer_success",
                    ]
                }
                for r in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
