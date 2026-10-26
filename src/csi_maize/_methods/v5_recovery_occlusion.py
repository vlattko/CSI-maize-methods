"""Post-inspection exploratory distichous stress cohort; frozen inverse fitter.

Designed after the generic golden-angle cohort showed weak overlap. This adds
truth geometry cases without tuning any camera, initialization or fitting rule.
"""

import v5_recovery_benchmark as core
from bootstrap import BASE, OUT, write_csv
import numpy as np
import torch, cv2, json, time, traceback
from pathlib import Path
from threadpoolctl import threadpool_limits

DEST = OUT / "leaf_dynamics/2026-10-24/jeb_v5/recovery_distichous"


def stress_design():
    d = core.design()
    d["cases"] = [
        {
            "id": f"5leaf_distichous_{family}_{condition}",
            "leaves": 5,
            "truth_family": family,
            "condition": condition,
            "truth_seed": 0,
            "observation_seed": 20261524 + (family == "wavy") * 10,
        }
        for family in ["cubic", "wavy"]
        for condition in ["clean", "perturbed"]
    ]
    d["chronology"] = (
        "Designed after examining the original eight generic golden-angle results and their weak downstream overlap. Exploratory additional stress cohort, not a prespecified independent confirmation."
    )
    d["truth_arrangement"] = (
        "Rotate each complete leaf rigidly around its own attachment from golden angle to alternating azimuth i*pi. Distichous geometry is synthetic and not a measured maize developmental sequence."
    )
    d["unchanged_fitter_sha256"] = core.sha(BASE / "v5_recovery_benchmark.py")
    return d


def make_distichous(family):
    g = core.make_truth(5, family, 0)
    for i in range(5):
        original = i * 2.39996323 + 0.44
        delta = i * np.pi - original
        c = np.cos(delta)
        s = np.sin(delta)
        r = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        origin = g["midribs"][i, 0].copy()
        for key in ["vertices", "midribs", "generating_control_points"]:
            g[key][i] = (g[key][i] - origin) @ r.T + origin
    return g


def effect_errors(pred, ref):
    off = ~np.eye(len(ref), dtype=bool)
    error = np.abs(pred - ref)[off]
    den = float(np.abs(ref[off]).sum())
    return {
        "absolute_mean_error": float(error.mean()),
        "absolute_max_error": float(error.max()),
        "truth_mean_effect": float(np.abs(ref[off]).mean()),
        "truth_max_effect": float(np.abs(ref).max()),
        "magnitude_relative_l1_error": float(error.sum() / den) if den > 0 else None,
        "definition": "L1 sum absolute effect error / L1 sum absolute true effects; null when truth magnitude is zero. Not mean pointwise relative error.",
    }


def run():
    d = stress_design()
    DEST.mkdir(parents=True, exist_ok=True)
    p = DEST / "predeclared_design.json"
    if p.exists() and json.loads(p.read_text(encoding="utf-8")) != d:
        raise RuntimeError("Stress design changed")
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    start = core.utc()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    print("BEGIN", start, "device", device, "cases", len(d["cases"]), flush=True)
    with threadpool_limits(limits=2):
        for case in d["cases"]:
            folder = DEST / case["id"]
            folder.mkdir(exist_ok=True)
            begin = time.time()
            print("CASE", case["id"], flush=True)
            try:
                g = make_distichous(case["truth_family"])
                obs, clean = core.observations_from_truth(g, case, d)
                initial = core.initialize_from_observations(obs, d["camera"])
                np.savez_compressed(
                    folder / "synthetic_inputs.npz",
                    **{"truth_" + k: v for k, v in g.items()},
                    **{"observed_" + k: v for k, v in obs.items()},
                    clean_masks=clean,
                )
                np.savez_compressed(folder / "initial_parameters.npz", **initial)
                fitted, history = core.fit_case(obs, d["camera"], initial, d["iterations"], device)
                np.savez_compressed(folder / "fitted_parameters.npz", **fitted)
                write_csv(folder / "loss.csv", history)
                ref, refbeams = core.release_matrix(g["vertices"])
                arrays = {"truth_release": ref, "truth_beam_release": refbeams}
                metrics = {}
                for label, pars in [("initial", initial), ("fitted", fitted)]:
                    v, mid, faces = core.surface_from_parameters(pars)
                    m = core.geometry_metrics(v, mid, g["vertices"], g["midribs"], faces)
                    masks = core.render_masks(v, obs["angles"], d["camera"], d["shape"])
                    ious = core.mask_metrics(masks, clean)
                    effects, beams = core.release_matrix(v)
                    m.update(
                        {
                            "mean_train_clean_mask_iou": float(ious[d["train_views"]].mean()),
                            "mean_withheld_clean_mask_iou": float(ious[d["test_views"]].mean()),
                            "per_view_clean_mask_iou": ious.tolist(),
                            "release_mean_abs_error": float(
                                abs(effects - ref)[~np.eye(5, dtype=bool)].mean()
                            ),
                            "release_max_abs_error": float(abs(effects - ref).max()),
                            "release_magnitude_errors": effect_errors(effects, ref),
                            "zenith_release_magnitude_errors": effect_errors(beams[0], refbeams[0]),
                        }
                    )
                    metrics[label] = m
                    arrays.update(
                        {
                            label + "_vertices": v,
                            label + "_midribs": mid,
                            label + "_masks": masks,
                            label + "_beam_release": beams,
                        }
                    )
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
                    "metrics": metrics,
                    "seconds": time.time() - begin,
                    "units": "Fixed synthetic world gauge with parameter scale100, not actual shoot height or physical dimensions",
                }
                results.append(row)
                (folder / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                print(
                    "RESULT",
                    case["id"],
                    {
                        k: {
                            x: round(v[x], 4)
                            for x in [
                                "surface_symmetric_distance",
                                "mean_area_relative_abs_error",
                                "mean_withheld_clean_mask_iou",
                                "release_mean_abs_error",
                            ]
                        }
                        for k, v in metrics.items()
                    },
                    "truthmax",
                    round(float(ref.max()), 4),
                    "seconds",
                    round(time.time() - begin, 1),
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
                (folder / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                print(row["traceback"], flush=True)
    summary = {
        "title": "Exploratory distichous known-answer inverse stress cohort",
        "status": "completed"
        if all(r["status"] == "completed" for r in results)
        else "completed_with_failures",
        "started_utc": start,
        "ended_utc": core.utc(),
        "design": d,
        "results": results,
        "input_sha256": {
            str(q.relative_to(BASE)).replace("\\", "/"): core.sha(q)
            for q in [
                Path(__file__),
                BASE / "v5_recovery_benchmark.py",
                BASE / "test_v5_recovery_occlusion.py",
                BASE / "organ_inverse.py",
                BASE / "anatomical_ribbons.py",
                BASE / "bootstrap.py",
            ]
        },
        "claim": "Same frozen fitting configuration on four additional synthetic alternating-azimuth shoots with privileged supplied paths; not independent real-plant validation.",
    }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("END", summary["ended_utc"], summary["status"], flush=True)


if __name__ == "__main__":
    run()
