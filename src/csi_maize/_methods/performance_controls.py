"""Post-hoc fixed Ridge controls: diagnostic specificity, not causal identification.

Design fixed before this diagnostic's execution, AFTER inspecting previous results.
No architecture, alpha, feature-family or outer-result selection is performed here.
"""

from common import workspace_relative

from bootstrap import ROOT, BASE, OUT, read_csv, write_csv
import hashlib
import json
import platform
from datetime import datetime, timezone
import numpy as np
import openpyxl
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from threadpoolctl import threadpool_limits

TARGETS = ["ojip__Fv/Fm", "ojip__Pi_Abs"]
KEY_FIELDS = ["treatment", "measurement", "genotype", "replicate"]
GEOMETRY = {
    "silhouette_area_px",
    "area_fraction",
    "visible_height_px",
    "silhouette_width_px",
    "height_fraction",
    "width_fraction",
    "perimeter_px",
    "convex_hull_area_px",
    "solidity",
    "eccentricity",
    *[f"stratum_{i}_width_fraction" for i in range(1, 6)],
}
OPTICAL = {
    *[f"{c}_{stat}" for c in "rgb" for stat in ["median", "iqr", "chromaticity_median"]],
    "normalized_excess_green",
    "green_red_normalized_difference",
    "blue_red_normalized_difference",
    "hue_median",
    "saturation_median",
    "yellow_hue_fraction",
}


def partition_fields(names):
    families = {"geometry": [], "optical": [], "excluded": []}
    for name in names:
        parts = name.split("__")
        family = "excluded"
        if len(parts) == 3 and parts[0] in ["rgb", "noir"] and parts[2] in ["median", "max", "iqr"]:
            if parts[1] in GEOMETRY:
                family = "geometry"
            elif parts[1] in OPTICAL:
                family = "optical"
        families[family].append(name)
    return families


def build_design(rows, train, test, numeric, numeric_names, metadata):
    """Fit categorical vocabulary and every scaling parameter only on train."""
    blocks_train, blocks_test, names = [], [], []
    audit = {"metadata_categories": []}
    if metadata:
        raw = np.array([[r["measurement"], r["genotype"]] for r in rows], dtype=object)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        blocks_train.append(encoder.fit_transform(raw[train]))
        blocks_test.append(encoder.transform(raw[test]))
        names.extend(encoder.get_feature_names_out(["round", "genotype"]).tolist())
        audit["metadata_categories"] = [a.tolist() for a in encoder.categories_]
    if numeric is not None:
        blocks_train.append(numeric[train])
        blocks_test.append(numeric[test])
        names.extend(numeric_names)
    a, b = np.concatenate(blocks_train, axis=1), np.concatenate(blocks_test, axis=1)
    scaler = StandardScaler()
    a = scaler.fit_transform(a)
    b = scaler.transform(b)
    audit.update(
        feature_names=names, scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist()
    )
    return a, b, audit


def score(y, p):
    residual = y - p
    denom = np.sum((y - y.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "mean_error_pred_minus_observed": float(np.mean(p - y)),
        "r2": float(1 - np.sum(residual**2) / denom) if denom > 0 else None,
    }


def source_targets(rows):
    """Check both targets in the original workbook against keyed reference data."""
    path = ROOT / "OJIP_P6_computed_data.xlsx"
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    data = iter(book.worksheets[0].values)
    header = next(data)
    original = {}
    for index, row in enumerate(data, 2):
        key = tuple(str(v).strip() for v in row[:4])
        if key in original:
            raise ValueError("Duplicate original target key")
        original[key] = (index, dict(zip(header, row)))
    book.close()
    reference = {}
    for row in read_csv(ROOT / "research_audit/reference_master.csv"):
        key = tuple(row[k] for k in KEY_FIELDS)
        if key in reference:
            raise ValueError("Duplicate reference key")
        reference[key] = row
    y = {target: [] for target in TARGETS}
    provenance = []
    for row in rows:
        key = tuple(row[k] for k in KEY_FIELDS)
        index, values = original[key]
        for target in TARGETS:
            value = float(values[target.split("__", 1)[1]])
            if abs(value - float(reference[key][target])) > 1e-12:
                raise ValueError("Reference value differs from original workbook")
            y[target].append(value)
            provenance.append(
                {
                    "session": row["session"],
                    "target": target,
                    "source_workbook": path.name,
                    "source_row": index,
                    "source_header": target.split("__", 1)[1],
                    "observed": value,
                    **{k: row[k] for k in KEY_FIELDS},
                }
            )
    return {k: np.array(v) for k, v in y.items()}, provenance


def matching_folds(rows, split):
    groups = np.array(
        [
            r["genotype"] if split == "genotype" else r["genotype"] + "__" + r["replicate"]
            for r in rows
        ]
    )
    x = np.zeros((len(rows), 1))
    folds = (
        list(LeaveOneGroupOut().split(x, groups=groups))
        if split == "genotype"
        else list(GroupKFold(5).split(x, groups=groups))
    )
    previous = read_csv(OUT / ("nested_" + split + "_embeddings/fold_provenance.csv"))
    observed = np.zeros(len(rows), dtype=int)
    for index, (train, test) in enumerate(folds):
        assert not set(groups[train]) & set(groups[test])
        assert not set(train) & set(test)
        observed[test] += 1
        expected = [r for r in previous if int(r["outer_fold"]) == index]
        assert expected
        actual_train = {rows[i]["session"] for i in train}
        actual_test = {rows[i]["session"] for i in test}
        assert all(
            set(r["train_sessions"].split(";")) == actual_train
            and set(r["test_sessions"].split(";")) == actual_test
            for r in expected
        )
    assert np.all(observed == 1)
    return groups, folds


def main():
    folder = OUT / "performance_controls"
    folder.mkdir(exist_ok=True)
    rows = read_csv(OUT / "sam_traits/session_traits.csv")
    assert len(rows) == len({r["session"] for r in rows}) == 33
    assert len({tuple(r[k] for k in KEY_FIELDS) for r in rows}) == 33
    assert {r["treatment"] for r in rows} == {"C"}
    assert all(int(r["valid_views"]) == 80 for r in rows)
    fields = [k for k in rows[0] if k.startswith(("rgb__", "noir__"))]
    families = partition_fields(fields)
    assert len(families["geometry"]) == len(families["optical"]) == 90
    matrices = {
        k: np.array([[float(r[f]) for f in families[k]] for r in rows])
        for k in ["geometry", "optical"]
    }
    dino = np.load(OUT / "dinov2_embeddings.npz")
    lookup = {str(s): i for i, s in enumerate(dino["sessions"])}
    assert len(lookup) == len(dino["sessions"])
    embeddings = dino["features"][[lookup[r["session"]] for r in rows]]
    assert embeddings.shape == (33, 768)
    assert all(np.isfinite(a).all() for a in [*matrices.values(), embeddings])
    names_dino = [f"dinov2_{i:03d}" for i in range(768)]
    joined = np.concatenate([matrices["geometry"], matrices["optical"]], axis=1)
    joined_names = families["geometry"] + families["optical"]
    designs = {
        "metadata": (None, [], True),
        "geometry": (matrices["geometry"], families["geometry"], False),
        "optical": (matrices["optical"], families["optical"], False),
        "metadata_geometry": (matrices["geometry"], families["geometry"], True),
        "metadata_optical": (matrices["optical"], families["optical"], True),
        "metadata_geometry_optical": (joined, joined_names, True),
        "dinov2": (embeddings, names_dino, False),
        "metadata_dinov2": (embeddings, names_dino, True),
    }
    targets, target_audit = source_targets(rows)
    predictions, metrics, fold_rows, preprocessing, fold_metrics = [], [], [], [], []
    for split in ["genotype", "plant_label"]:
        groups, folds = matching_folds(rows, split)
        all_pred = {
            target: {
                name: np.empty(len(rows)) for name in ["training_mean", "round_mean", *designs]
            }
            for target in TARGETS
        }
        for fold, (train, test) in enumerate(folds):
            fold_rows.append(
                {
                    "split": split,
                    "outer_fold": fold,
                    "train_sessions": ";".join(rows[i]["session"] for i in train),
                    "test_sessions": ";".join(rows[i]["session"] for i in test),
                    "train_groups": ";".join(sorted(set(groups[train]))),
                    "test_groups": ";".join(sorted(set(groups[test]))),
                }
            )
            for target, y in targets.items():
                all_pred[target]["training_mean"][test] = y[train].mean()
                for i in test:
                    same = [j for j in train if rows[j]["measurement"] == rows[i]["measurement"]]
                    all_pred[target]["round_mean"][i] = y[same].mean() if same else y[train].mean()
            for name, (numeric, numeric_names, metadata) in designs.items():
                a, b, audit = build_design(rows, train, test, numeric, numeric_names, metadata)
                preprocessing.append(
                    {"split": split, "outer_fold": fold, "features": name, **audit}
                )
                for target, y in targets.items():
                    # Scalar target scaling is also fitted within this outer train.
                    sy = StandardScaler()
                    scaled_y = sy.fit_transform(y[train, None]).ravel()
                    with threadpool_limits(limits=2):
                        model = Ridge(alpha=10.0, solver="svd").fit(a, scaled_y)
                        pred = sy.inverse_transform(model.predict(b)[:, None]).ravel()
                    all_pred[target][name][test] = pred
            for target, y in targets.items():
                for name, p in all_pred[target].items():
                    fold_metrics.append(
                        {
                            "split": split,
                            "outer_fold": fold,
                            "target": target,
                            "features": name,
                            "n_sessions": len(test),
                            **score(y[test], p[test]),
                        }
                    )
        for target, y in targets.items():
            for name, pred in all_pred[target].items():
                metrics.append(
                    {
                        "split": split,
                        "target": target,
                        "features": name,
                        "architecture": "baseline"
                        if name in ["training_mean", "round_mean"]
                        else "fixed_ridge",
                        "alpha": "" if name in ["training_mean", "round_mean"] else 10.0,
                        "n_sessions": len(rows),
                        "n_groups": len(set(groups)),
                        **score(y, pred),
                        "delta_rmse_vs_training_mean": score(y, pred)["rmse"]
                        - score(y, all_pred[target]["training_mean"])["rmse"],
                        "delta_rmse_vs_metadata": score(y, pred)["rmse"]
                        - score(y, all_pred[target]["metadata"])["rmse"],
                    }
                )
                for i, row in enumerate(rows):
                    predictions.append(
                        {
                            "split": split,
                            "target": target,
                            "features": name,
                            "session": row["session"],
                            "group": groups[i],
                            **{k: row[k] for k in KEY_FIELDS},
                            "observed": float(y[i]),
                            "predicted": float(pred[i]),
                        }
                    )
    write_csv(folder / "predictions.csv", predictions)
    write_csv(folder / "metrics.csv", metrics)
    write_csv(folder / "fold_metrics.csv", fold_metrics)
    write_csv(folder / "fold_provenance.csv", fold_rows)
    write_csv(folder / "target_source_provenance.csv", target_audit)
    (folder / "feature_families.json").write_text(
        json.dumps(
            {
                **families,
                "geometry_definitions": sorted(GEOMETRY),
                "optical_definitions": sorted(OPTICAL),
                "metadata": ["measurement(round)", "genotype"],
                "dino_dimensions": names_dino,
                "exclusion_rationale": "Exclude skeleton topology, centroids (acquisition position), all censoring flags and saturation clipping quality proxy; whitelist only semantic geometry/optical fields.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (folder / "preprocessing_provenance.json").write_text(
        json.dumps(preprocessing, indent=2), encoding="utf-8"
    )
    # Verify what was serialized, not only the arrays used to generate it.
    saved = read_csv(folder / "predictions.csv")
    for metric in read_csv(folder / "metrics.csv"):
        subset = [
            r for r in saved if all(r[k] == metric[k] for k in ["split", "target", "features"])
        ]
        assert len(subset) == len({r["session"] for r in subset}) == 33
        actual = {
            r["session"]: float(r["observed"])
            for r in target_audit
            if r["target"] == metric["target"]
        }
        assert all(abs(float(r["observed"]) - actual[r["session"]]) < 1e-12 for r in subset)
        rmse = np.sqrt(
            np.mean([(float(r["predicted"]) - float(r["observed"])) ** 2 for r in subset])
        )
        assert abs(rmse - float(metric["rmse"])) < 1e-12
    sources = [
        BASE / "performance_controls.py",
        BASE / "test_performance_controls.py",
        OUT / "sam_traits/session_traits.csv",
        OUT / "sam_traits/summary.json",
        OUT / "dinov2_embeddings.npz",
        ROOT / "research_audit/reference_master.csv",
        ROOT / "OJIP_P6_computed_data.xlsx",
        OUT / "nested_genotype_embeddings/fold_provenance.csv",
        OUT / "nested_plant_label_embeddings/fold_provenance.csv",
    ]
    manifest = {
        "status": "post_hoc_falsification_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sessions": 33,
        "targets": TARGETS,
        "controls_only": True,
        "splits": ["genotype", "plant_label"],
        "fixed_model": {"architecture": "Ridge", "alpha": 10.0, "solver": "svd", "tuning": False},
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "openpyxl": openpyxl.__version__,
        },
        "target_range": {
            k: {
                "min": float(v.min()),
                "max": float(v.max()),
                "mean": float(v.mean()),
                "sample_sd": float(v.std(ddof=1)),
            }
            for k, v in targets.items()
        },
        "predictor_sets": list(designs),
        "feature_counts_without_metadata": {k: len(v[1]) for k, v in designs.items()},
        "metadata_encoding": "OneHotEncoder fitted only to outer training rows; unseen categories ignored; no target encoding",
        "preprocessing": "All predictor columns and scalar targets standardized using outer training rows only",
        "folds": "Exactly matched every stored outer train/test session set in existing analyses",
        "verification": {
            "metric_rows_recomputed": len(metrics),
            "prediction_rows": len(saved),
            "unique_sessions_per_metric": 33,
            "fold_rows_checked": len(fold_rows),
            "source_target_cells_checked": len(target_audit),
            "group_overlap": False,
        },
        "limitations": [
            "Designed after inspecting earlier results: post-hoc, not preregistered or confirmatory",
            "No new architecture or hyperparameter search; do not select a winner from these diagnostics",
            "Prediction differences do not estimate causal effects; geometry may mediate biological differences",
            "Metadata adjustment diagnoses one specificity question and may remove biologically meaningful signal",
            "All 33 sessions selected by all-view automatic QC; manual mask accuracy unvalidated",
            "Only identified controls in two rounds; narrow target range; physical repeated-plant identity unverified",
            "Six genotype clusters or 24 replicate-label groups, not independent views",
            "Metadata+genotype is unavailable for a truly unknown genotype; unknown onehot categories ignored",
            "Fixed alpha is a diagnostic choice; failure does not establish information-theoretic absence of signal",
            "Equal alpha does not equalize effective complexity across different feature dimensions and correlations",
            "Optical descriptors are processed camera responses, not calibrated pigment measurements",
        ],
        "source_sha256": {
            workspace_relative(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    }
    (folder / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
