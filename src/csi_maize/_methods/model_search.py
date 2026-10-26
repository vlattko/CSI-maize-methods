"""Nested grouped model comparisons with training-only preprocessing."""

from bootstrap import ROOT, OUT, read_csv, write_csv, publication_case, method_config
import argparse, json, time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from threadpoolctl import threadpool_limits

TARGETS = ["ojip__Fv/Fm", "ojip__Pi_Abs", "nir__NDVI"]


def estimators():
    def scaled(regressor):
        return TransformedTargetRegressor(
            regressor=Pipeline([("scale", StandardScaler()), ("model", regressor)]),
            transformer=StandardScaler(),
        )

    return {
        "ridge": (scaled(Ridge()), {"regressor__model__alpha": [1.0, 10.0, 100.0, 1000.0]}),
        "pls": (scaled(PLSRegression(scale=False)), {"regressor__model__n_components": [1, 2, 3]}),
        "rbf_kernel": (
            scaled(KernelRidge(kernel="rbf")),
            {"regressor__model__alpha": [0.1, 1.0, 10.0], "regressor__model__gamma": [0.001, 0.01]},
        ),
        "extra_trees": (
            ExtraTreesRegressor(n_estimators=128, random_state=73, n_jobs=1),
            {"min_samples_leaf": [2, 5], "max_features": [0.5, 1.0]},
        ),
    }


def metrics(y, pred):
    residual = y - pred
    denominator = np.sum((y - y.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(abs(residual))),
        "r2": float(1 - np.sum(residual**2) / denominator) if denominator > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--split", choices=["genotype", "plant_label"], default="genotype")
    args = parser.parse_args()
    provenance = json.loads((OUT / "sam_traits/summary.json").read_text(encoding="utf-8"))
    expected = method_config().get("expected_acquisitions", 41) if publication_case() else None
    if provenance["status"] != "exploratory_image_traits" or (
        expected is not None and provenance["sessions_processed"] != int(expected)
    ):
        raise ValueError("Finish the complete SAM trait run before model comparison")
    rows = read_csv(OUT / "sam_traits/session_traits.csv")
    reference = read_csv(ROOT / "research_audit/reference_master.csv")
    targets = {
        tuple(r[k] for k in ["treatment", "measurement", "genotype", "replicate"]): r
        for r in reference
    }
    rows = [
        r
        for r in rows
        if tuple(r[k] for k in ["treatment", "measurement", "genotype", "replicate"]) in targets
    ]
    if len(rows) < 20:
        raise ValueError("Fewer than 20 complete QC sessions; investigate masks before modelling")
    # Raw skeleton topology is not sufficiently stable for physiological models.
    feature_names = sorted(
        k for k in rows[0] if k.startswith(("rgb__", "noir__")) and "skeleton_" not in k
    )
    x = np.array([[float(r[k]) for k in feature_names] for r in rows])
    sets = {}
    sets["rgb_traits"] = x[:, [k.startswith("rgb__") for k in feature_names]]
    sets["noir_traits"] = x[:, [k.startswith("noir__") for k in feature_names]]
    sets["combined_traits"] = x
    if args.embeddings:
        data = np.load(OUT / "dinov2_embeddings.npz")
        lookup = {str(s): i for i, s in enumerate(data["sessions"])}
        if any(r["session"] not in lookup for r in rows):
            raise ValueError("Embedding coverage is incomplete")
        embeddings = data["features"][[lookup[r["session"]] for r in rows]]
        sets["dinov2_crops"] = embeddings
        sets["traits_plus_dinov2"] = np.concatenate([x, embeddings], axis=1)
    if not all(np.isfinite(v).all() for v in sets.values()):
        raise ValueError("Nonfinite predictors")
    groups = np.array(
        [
            r["genotype"] if args.split == "genotype" else r["genotype"] + "__" + r["replicate"]
            for r in rows
        ]
    )
    outer = (
        list(LeaveOneGroupOut().split(x, groups=groups))
        if args.split == "genotype"
        else list(GroupKFold(5).split(x, groups=groups))
    )
    all_metrics = []
    all_predictions = []
    traces = []
    rng = np.random.default_rng(73)
    folder = OUT / ("nested_" + args.split + ("_embeddings" if args.embeddings else ""))
    folder.mkdir(exist_ok=True)
    for target in TARGETS:
        y = np.array(
            [
                float(
                    targets[
                        tuple(r[k] for k in ["treatment", "measurement", "genotype", "replicate"])
                    ][target]
                )
                for r in rows
            ]
        )
        baseline = np.empty(len(y))
        round_baseline = np.empty(len(y))
        for train, test in outer:
            baseline[test] = np.mean(y[train])
            for i in test:
                same = train[[rows[j]["measurement"] == rows[i]["measurement"] for j in train]]
                round_baseline[i] = y[same].mean() if len(same) else y[train].mean()
        plans = [
            ("training_mean", "none", baseline),
            ("round_mean", "metadata_round_only", round_baseline),
        ]
        for feature_set, features in sets.items():
            for architecture, (estimator, grid) in estimators().items():
                pred = np.empty(len(y))
                started = time.time()
                for fold, (train, test) in enumerate(outer):
                    if set(groups[train]) & set(groups[test]):
                        raise AssertionError("Group leakage")
                    inner = GroupKFold(min(4, len(set(groups[train]))))
                    search = GridSearchCV(
                        estimator,
                        grid,
                        cv=inner,
                        scoring="neg_mean_squared_error",
                        n_jobs=1,
                        error_score="raise",
                    )
                    with threadpool_limits(limits=2):
                        search.fit(features[train], y[train], groups=groups[train])
                        pred[test] = np.asarray(search.predict(features[test])).ravel()
                    traces.append(
                        {
                            "target": target,
                            "features": feature_set,
                            "architecture": architecture,
                            "outer_fold": fold,
                            "train_groups": ";".join(sorted(set(groups[train]))),
                            "test_groups": ";".join(sorted(set(groups[test]))),
                            "train_sessions": ";".join(rows[j]["session"] for j in train),
                            "test_sessions": ";".join(rows[j]["session"] for j in test),
                            "selected_parameters": json.dumps(search.best_params_),
                            "inner_mse": float(-search.best_score_),
                        }
                    )
                plans.append((architecture, feature_set, pred.copy()))
                print(
                    target,
                    feature_set,
                    architecture,
                    metrics(y, pred),
                    "seconds",
                    round(time.time() - started, 1),
                    flush=True,
                )
        for architecture, feature_set, pred in plans:
            outcome = {
                "target": target,
                "architecture": architecture,
                "features": feature_set,
                "split": args.split,
                "n_sessions": len(rows),
                "n_groups": len(set(groups)),
                **metrics(y, pred),
            }
            diffs = []
            # Paired cluster bootstrap of existing outer predictions; not new validation.
            for _ in range(1000):
                sampled = rng.choice(sorted(set(groups)), size=len(set(groups)), replace=True)
                ix = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
                diffs.append(
                    np.sqrt(np.mean((y[ix] - pred[ix]) ** 2))
                    - np.sqrt(np.mean((y[ix] - baseline[ix]) ** 2))
                )
            outcome.update(
                delta_rmse_vs_mean=float(outcome["rmse"] - metrics(y, baseline)["rmse"]),
                delta_rmse_bootstrap_low=float(np.percentile(diffs, 2.5)),
                delta_rmse_bootstrap_high=float(np.percentile(diffs, 97.5)),
            )
            all_metrics.append(outcome)
            for i, row in enumerate(rows):
                all_predictions.append(
                    {
                        "session": row["session"],
                        "genotype": row["genotype"],
                        "measurement": row["measurement"],
                        "replicate": row["replicate"],
                        "target": target,
                        "architecture": architecture,
                        "features": feature_set,
                        "observed": float(y[i]),
                        "predicted": float(pred[i]),
                    }
                )
        write_csv(folder / "metrics.csv", all_metrics)
        write_csv(folder / "predictions.csv", all_predictions)
        write_csv(folder / "fold_provenance.csv", traces)
    summary = {
        "sessions": len(rows),
        "targets": TARGETS,
        "split": args.split,
        "feature_sets": {k: v.shape[1] for k, v in sets.items()},
        "outer_folds": len(outer),
        "inner_cv": "grouped within each outer training partition",
        "target_standardization": "training-only for linear, PLS and kernel models",
        "segmentation": provenance,
        "feature_exclusions": ["raw skeleton topology"],
        "notes": [
            "Exploratory architecture comparison; selecting a winning outer result requires independent confirmation",
            "Six genotype clusters give imprecise uncertainty intervals",
            "Plant-label groups conservatively combine genotype/replicate across rounds; physical identity unverified",
            "All current identified sessions are control samples",
            "Automated mask QC is not human-annotated segmentation validation",
        ],
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
