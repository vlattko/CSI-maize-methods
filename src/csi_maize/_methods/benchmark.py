"""Fixed-penalty ridge and training-mean baseline with held-out genotypes."""

import json
import numpy as np
import openpyxl
from common import ROOT, OUT, config, reference, read_csv, write_csv, sample_key


def ridge_predict(xtrain, ytrain, xtest, penalty):
    if penalty <= 0:
        raise ValueError("Penalty must be positive")
    mean = xtrain.mean(axis=0)
    scale = xtrain.std(axis=0)
    scale[scale < 1e-12] = 1
    x = (xtrain - mean) / scale
    z = (xtest - mean) / scale
    ymean = ytrain.mean(axis=0)
    # Dual formulation is small at this sample size. All fitting is training-only.
    weights = np.linalg.solve(x @ x.T + penalty * np.eye(len(x)), ytrain - ymean)
    return ymean + z @ x.T @ weights


def fold_indices(groups):
    groups = np.asarray(groups)
    for group in sorted(set(groups)):
        train = np.where(groups != group)[0]
        test = np.where(groups == group)[0]
        if len(train) < 3:
            raise ValueError("Insufficient training samples for held-out genotype")
        yield group, train, test


def run(spectra=False):
    cfg = config()
    refs = reference()
    features = read_csv(OUT / "features.csv")
    if len(features) < 12 or len({r["genotype"] for r in features}) < 3:
        raise ValueError("At least 12 feature sessions and 3 genotypes required")
    targets = list(cfg["targets"])
    lookup = {key: {t: float(row[t]) for t in targets} for key, row in refs.items()}
    if spectra:
        excluded = {tuple(k) for k in cfg["exclude_spectral_keys"]}
        for filename in cfg["spectral_sources"]:
            wb = openpyxl.load_workbook(ROOT / filename, read_only=True, data_only=True)
            rows = iter(wb.worksheets[0].values)
            header = next(rows)
            names = [filename.removesuffix(".xlsx") + "__" + str(v) for v in header[4:]]
            targets.extend(names)
            for raw in rows:
                key = tuple(str(v) for v in raw[:4])
                if key in lookup and key not in excluded:
                    lookup[key].update(zip(names, map(float, raw[4:])))
            wb.close()
        features = [r for r in features if all(t in lookup[sample_key(r)] for t in targets)]
    y = np.array([[lookup[sample_key(r)][t] for t in targets] for r in features])
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite target values")
    groups = [r["genotype"] for r in features]
    metrics = []
    predictions = []
    folds = []
    for modality in ["mean_baseline", "rgb", "noir", "combined"]:
        cols = sorted(
            c
            for c in features[0]
            if (c.startswith("rgb__") or c.startswith("noir__"))
            and (modality == "combined" or c.startswith(modality + "__"))
        )
        x = np.array([[float(r[c]) for c in cols] for r in features])
        pred = np.empty_like(y)
        for group, train, test in fold_indices(groups):
            if modality == "mean_baseline":
                predicted = np.tile(y[train].mean(axis=0), (len(test), 1))
            else:
                predicted = ridge_predict(x[train], y[train], x[test], cfg["ridge_penalty"])
            pred[test] = predicted
            folds.append(
                {
                    "model": modality,
                    "held_out_genotype": group,
                    "training_sessions": len(train),
                    "test_sessions": len(test),
                    "train_test_genotype_overlap": False,
                }
            )
        for j, target in enumerate(targets):
            error = pred[:, j] - y[:, j]
            ss = float(((y[:, j] - y[:, j].mean()) ** 2).sum())
            metrics.append(
                {
                    "model": modality,
                    "target": target,
                    "n_sessions": len(y),
                    "n_genotypes": len(set(groups)),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(abs(error))),
                    "r2": float(1 - (error**2).sum() / ss) if ss > 0 else "",
                    "status": "exploratory_pipeline_smoke_test_unvalidated_foreground",
                }
            )
            for i, r in enumerate(features):
                predictions.append(
                    {
                        "session": r["session"],
                        "held_out_genotype": r["genotype"],
                        "model": modality,
                        "target": target,
                        "observed": float(y[i, j]),
                        "predicted": float(pred[i, j]),
                    }
                )
    prefix = "spectral_" if spectra else ""
    write_csv(OUT / (prefix + "metrics.csv"), metrics)
    write_csv(OUT / (prefix + "predictions.csv"), predictions)
    write_csv(OUT / (prefix + "folds.csv"), folds)
    summary = {
        "status": "smoke_test_only_not_publication_results",
        "sessions": len(features),
        "treatments": sorted(set(r["treatment"] for r in features)),
        "targets": len(targets),
        "outer_group": "genotype",
        "ridge_penalty_fixed": cfg["ridge_penalty"],
        "limitations": [
            "Foreground masks are unvalidated heuristics",
            "Current identified coverage is control only",
            "No confidence intervals or tuned model selection",
            "No claims about treatment transfer or direct fluorescence imaging",
        ],
    }
    (OUT / (prefix + "benchmark_summary.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
