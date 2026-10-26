"""Nested grouped prediction of native leaf transmittance grids, not HSI cubes."""

from bootstrap import BASE, ROOT, OUT, read_csv, write_csv
import json
import numpy as np
import openpyxl
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV
from model_search import estimators
from threadpoolctl import threadpool_limits


def main():
    rows = read_csv(OUT / "sam_traits/session_traits.csv")
    groups = np.array([r["genotype"] for r in rows])
    sessions = np.array([r["session"] for r in rows])
    names = sorted(k for k in rows[0] if k.startswith(("rgb__", "noir__")) and "skeleton_" not in k)
    x = np.array([[float(r[k]) for k in names] for r in rows])
    sets = {
        "rgb_traits": x[:, [k.startswith("rgb__") for k in names]],
        "noir_traits": x[:, [k.startswith("noir__") for k in names]],
        "combined_traits": x,
    }
    dino = np.load(OUT / "dinov2_embeddings.npz")
    lookup = {str(s): i for i, s in enumerate(dino["sessions"])}
    sets["dinov2_crops"] = dino["features"][[lookup[s] for s in sessions]]
    folder = OUT / "nested_spectra"
    folder.mkdir(exist_ok=True)
    all_metrics = []
    all_folds = []
    summaries = []
    excluded = {
        tuple(r)
        for r in json.loads((BASE / "config.json").read_text(encoding="utf-8"))[
            "exclude_spectral_keys"
        ]
    }
    key_fields = ["treatment", "measurement", "genotype", "replicate"]
    keys = [tuple(r[k] for k in key_fields) for r in rows]
    for band in ["UVIS", "NIR"]:
        source = ROOT / (band + "_P6_transmitance.xlsx")
        book = openpyxl.load_workbook(source, read_only=True, data_only=True)
        it = iter(book.worksheets[0].values)
        header = next(it)
        wavelength = np.array(header[4:], float)
        raw = {tuple(str(v) for v in row[:4]): np.array(row[4:], float) for row in it}
        book.close()
        if any(key in excluded for key in keys):
            raise ValueError("Excluded spectral sample unexpectedly entered selected cohort")
        y = np.array([raw[key] for key in keys])
        outer = list(LeaveOneGroupOut().split(x, groups=groups))
        baseline = np.empty_like(y)
        round_mean = np.empty_like(y)
        for train, test in outer:
            baseline[test] = y[train].mean(axis=0)
            for i in test:
                same = train[[rows[j]["measurement"] == rows[i]["measurement"] for j in train]]
                round_mean[i] = y[same].mean(axis=0) if len(same) else y[train].mean(axis=0)
        predictions = {"training_mean__none": baseline, "round_mean__metadata": round_mean}
        for feature_set, features in sets.items():
            for architecture in ["ridge", "pls"]:
                estimator, grid = estimators()[architecture]
                pred = np.empty_like(y)
                for fold, (train, test) in enumerate(outer):
                    search = GridSearchCV(
                        estimator,
                        grid,
                        cv=GroupKFold(4),
                        scoring="neg_mean_squared_error",
                        n_jobs=1,
                        error_score="raise",
                    )
                    with threadpool_limits(limits=2):
                        search.fit(features[train], y[train], groups=groups[train])
                        pred[test] = search.predict(features[test])
                    all_folds.append(
                        {
                            "band": band,
                            "features": feature_set,
                            "architecture": architecture,
                            "fold": fold,
                            "test_genotype": groups[test[0]],
                            "train_sessions": ";".join(sessions[train]),
                            "test_sessions": ";".join(sessions[test]),
                            "selected_parameters": json.dumps(search.best_params_),
                        }
                    )
                predictions[architecture + "__" + feature_set] = pred
                print(
                    band,
                    architecture,
                    feature_set,
                    "RMSE",
                    float(np.sqrt(np.mean((pred - y) ** 2))),
                    flush=True,
                )
        baseline_rmse = np.sqrt(np.mean((baseline - y) ** 2, axis=0))
        for label, pred in predictions.items():
            architecture, feature_set = label.split("__", 1)
            rmse = np.sqrt(np.mean((pred - y) ** 2, axis=0))
            ratio = rmse / np.maximum(baseline_rmse, 1e-12)
            summaries.append(
                {
                    "band": band,
                    "architecture": architecture,
                    "features": feature_set,
                    "n_sessions": len(rows),
                    "pooled_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
                    "median_wavelength_rmse_ratio": float(np.median(ratio)),
                    "wavelengths_better_than_mean": int((ratio < 1).sum()),
                    "wavelengths": len(wavelength),
                }
            )
            for wave, error, relative in zip(wavelength, rmse, ratio):
                all_metrics.append(
                    {
                        "band": band,
                        "architecture": architecture,
                        "features": feature_set,
                        "wavelength_source_header": float(wave),
                        "rmse": float(error),
                        "rmse_ratio_vs_mean": float(relative),
                        "n_sessions": len(rows),
                    }
                )
        np.savez_compressed(
            folder / (band + "_predictions.npz"),
            sessions=sessions,
            wavelengths=wavelength,
            observed=y,
            **predictions,
        )
    write_csv(folder / "metrics.csv", all_metrics)
    write_csv(folder / "band_summary.csv", summaries)
    write_csv(folder / "fold_provenance.csv", all_folds)
    (folder / "summary.json").write_text(
        json.dumps(
            {
                "sessions": len(rows),
                "bands": ["UVIS", "NIR"],
                "wavelengths_per_band": 256,
                "outer_split": "six held-out genotypes",
                "inner_split": "four grouped training folds",
                "quantity": "native source transmittance values",
                "status": "exploratory_unvalidated_segmentation",
                "limitations": [
                    "No spatial hyperspectral image is reconstructed",
                    "All identified samples are controls",
                    "Model selection is exploratory",
                    "Wavelength interpretation needs instrument metadata",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
