"""E11: fixed cross-view self-supervision on frozen image embeddings.

All views of each held-out genotype stay outside the learned projection head.
The experiment tests a small linear contrastive head, not a new foundation model.
"""

from common import workspace_relative

from bootstrap import BASE, ROOT, OUT, read_csv, write_csv, model_directory
from experiment_lab import RUN, utc, sha256_file, append_event
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from performance_controls import source_targets, score, build_design


def positive_mask(labels):
    labels = np.asarray(labels)
    result = labels[:, None] == labels[None, :]
    np.fill_diagonal(result, False)
    return result


def view_blocks(features):
    features = np.asarray(features)
    coefficients = np.fft.rfft(features, axis=1) / features.shape[1]
    return features.mean(axis=1), np.concatenate(
        [2 * abs(coefficients[:, i]) for i in [1, 2]], axis=1
    )


def train_head(features, epochs=150, dimensions=32, seed=20261024):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    x = torch.tensor(features.reshape(-1, features.shape[-1]), dtype=torch.float32)
    labels = np.repeat(np.arange(len(features)), features.shape[1])
    positives = torch.tensor(positive_mask(labels))
    if not bool(positives.any(1).all()):
        raise ValueError("Each visit must provide multiple positive views")
    head = torch.nn.Linear(x.shape[1], dimensions, bias=False)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.001)
    diagonal = torch.eye(len(x), dtype=torch.bool)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = F.normalize(head(x), dim=1)
        logits = z @ z.T / 0.1
        logits = logits.masked_fill(diagonal, -1e9)
        loss = (
            torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~positives, -1e9), 1)
        ).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return head.weight.detach().numpy(), history


def transformed(x, weight):
    z = x @ weight.T
    return z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)


def extract_views(records, folder):
    cache = folder / "frozen_view_embeddings.npz"
    paths = []
    for record in records:
        entry = []
        for camera in ["rgb", "noir"]:
            selected = sorted(
                (OUT / "sam_traits/crops").glob(
                    record["session"].split("/")[-1] + "__" + camera + "__*.png"
                )
            )
            if len(selected) != 8:
                raise ValueError("Expected exactly eight cached SAM crops per camera")
            indices = [int(p.stem.rsplit("__", 1)[-1]) for p in selected]
            if indices != list(range(0, 40, 5)):
                raise ValueError("Unexpected angular sampling: " + str(indices))
            entry.append(selected)
        paths.append(entry)
    source_hashes = {
        workspace_relative(p): sha256_file(p)
        for cameras in paths
        for subset in cameras
        for p in subset
    }
    cache_meta = folder / "embedding_cache_manifest.json"
    if cache.exists() and cache_meta.exists():
        manifest = json.loads(cache_meta.read_text(encoding="utf-8"))
        saved = np.load(cache)
        if manifest["input_sha256"] == source_hashes and list(saved["sessions"]) == [
            r["session"] for r in records
        ]:
            print("Reuse verified per-view embeddings", flush=True)
            return saved["features"], source_hashes
    from transformers import AutoImageProcessor, AutoModel

    directory = model_directory("dinov2-small")
    processor = AutoImageProcessor.from_pretrained(directory, use_fast=False, local_files_only=True)
    model, loading = AutoModel.from_pretrained(
        directory, output_loading_info=True, local_files_only=True
    )
    if any(
        loading[k] for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
    ):
        raise ValueError(str(loading))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    torch.set_num_threads(2)
    features = []
    for i, entry in enumerate(paths):
        vectors = []
        for subset in entry:
            images = []
            for path in subset:
                with Image.open(path) as image:
                    images.append(
                        ImageOps.pad(image.convert("RGB"), (224, 224), color=(124, 116, 104))
                    )
            inputs = processor(
                images=images, do_resize=False, do_center_crop=False, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                z = F.normalize(model(**inputs).last_hidden_state[:, 0, :], dim=1).cpu().numpy()
            vectors.append(z)
        features.append(vectors)
        print("Frozen DINO views", i + 1, "/", len(records), flush=True)
    features = np.array(features)
    np.savez_compressed(
        cache, features=features, sessions=np.array([r["session"] for r in records])
    )
    cache_meta.write_text(
        json.dumps(
            {
                "created_utc": utc(),
                "model": "facebook/dinov2-small",
                "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
                "shape": list(features.shape),
                "device": device,
                "input_sha256": source_hashes,
            },
            indent=2,
        )
    )
    return features, source_hashes


def main():
    started = utc()
    clock = time.time()
    folder = RUN / "e11_learning"
    folder.mkdir(parents=True, exist_ok=True)
    append_event(
        RUN / "events.jsonl",
        "E11",
        "started",
        {
            "command": "additive_learning.py",
            "seed": 20261024,
            "epochs": 150,
            "head_dimensions": 32,
            "temperature": 0.1,
        },
    )
    records = read_csv(OUT / "sam_traits/session_traits.csv")
    features, input_hashes = extract_views(records, folder)
    rgb = features[:, 0]
    old = np.load(OUT / "dinov2_embeddings.npz")
    lookup = {str(s): i for i, s in enumerate(old["sessions"])}
    pooled = features.mean(axis=2).reshape(len(records), -1)
    previous = old["features"][[lookup[r["session"]] for r in records]]
    pool_delta = float(np.max(abs(pooled - previous)))
    if pool_delta > 2e-5:
        raise ValueError(
            "Per-view pooling does not reproduce prior frozen embeddings: " + str(pool_delta)
        )
    targets, target_audit = source_targets(records)
    write_csv(folder / "target_source_provenance.csv", target_audit)
    groups = np.array([r["genotype"] for r in records])
    predictions = []
    retrieval = []
    folds = []
    losses = []
    models = {}
    names = [
        "metadata",
        "frozen_rgb_384",
        "random_projection_32",
        "contrastive_32",
        "contrastive_plus_angular_96",
        "metadata_contrastive_plus_angular_96",
    ]
    all_pred = {
        target: {name: np.empty(len(records)) for name in ["training_mean", *names]}
        for target in targets
    }
    for fold, (train, test) in enumerate(LeaveOneGroupOut().split(rgb, groups=groups)):
        assert not set(groups[train]) & set(groups[test])
        weight, history = train_head(rgb[train])
        rng = np.random.default_rng(20261024)
        random = rng.normal(size=(32, rgb.shape[-1])).astype(np.float32) / np.sqrt(rgb.shape[-1])
        latent = transformed(rgb, weight)
        random_latent = transformed(rgb, random)
        mean, angular = view_blocks(latent)
        designs = {
            "frozen_rgb_384": rgb.mean(axis=1),
            "random_projection_32": random_latent.mean(axis=1),
            "contrastive_32": mean,
            "contrastive_plus_angular_96": np.concatenate([mean, angular], axis=1),
        }
        models["fold_" + str(fold)] = weight
        folds.append(
            {
                "fold": fold,
                "train_sessions": ";".join(records[i]["session"] for i in train),
                "test_sessions": ";".join(records[i]["session"] for i in test),
                "train_groups": ";".join(sorted(set(groups[train]))),
                "test_groups": ";".join(sorted(set(groups[test]))),
                "head_training_views": len(train) * 8,
            }
        )
        losses.extend(
            {"fold": fold, "epoch": i + 1, "loss": value} for i, value in enumerate(history)
        )
        for name, z in [
            ("frozen_rgb_384", rgb),
            ("random_projection_32", random_latent),
            ("contrastive_32", latent),
        ]:
            gallery = z[test][:, [0, 2, 4, 6]].mean(axis=1)
            gallery /= np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
            for local, index in enumerate(test):
                for angle in [1, 3, 5, 7]:
                    similarity = gallery @ z[index, angle]
                    best = int(np.argmax(similarity))
                    candidates = [
                        j
                        for j, t in enumerate(test)
                        if records[t]["measurement"] == records[index]["measurement"]
                    ]
                    within = max(candidates, key=lambda j: similarity[j])
                    retrieval.append(
                        {
                            "fold": fold,
                            "representation": name,
                            "query_session": records[index]["session"],
                            "query_view": angle * 5,
                            "selected_session": records[test[best]]["session"],
                            "correct": int(best == local),
                            "candidate_visits": len(test),
                            "chance": 1 / len(test),
                            "within_round_correct": int(within == local),
                            "within_round_candidates": len(candidates),
                            "within_round_chance": 1 / len(candidates),
                        }
                    )
        for target, y in targets.items():
            all_pred[target]["training_mean"][test] = np.mean(y[train])
            for name, x in designs.items():
                scaler = StandardScaler()
                a = scaler.fit_transform(x[train])
                b = scaler.transform(x[test])
                with threadpool_limits(limits=2):
                    model = Ridge(alpha=10.0, solver="svd").fit(a, y[train])
                all_pred[target][name][test] = model.predict(b)
            for name, numeric in [
                ("metadata", None),
                ("metadata_contrastive_plus_angular_96", designs["contrastive_plus_angular_96"]),
            ]:
                a, b, _ = build_design(
                    records,
                    train,
                    test,
                    numeric,
                    []
                    if numeric is None
                    else ["contrastive_angular_" + str(i) for i in range(numeric.shape[1])],
                    True,
                )
                with threadpool_limits(limits=2):
                    model = Ridge(alpha=10.0, solver="svd").fit(a, y[train])
                all_pred[target][name][test] = model.predict(b)
        print(
            "Contrastive fold",
            fold,
            "held genotype",
            ",".join(set(groups[test])),
            "loss",
            history[0],
            "=>",
            history[-1],
            flush=True,
        )
    metrics = []
    for target, y in targets.items():
        for name, pred in all_pred[target].items():
            metrics.append(
                {
                    "target": target,
                    "representation": name,
                    "split": "leave_genotype_out",
                    "n_visits": len(records),
                    **score(y, pred),
                }
            )
            predictions.extend(
                {
                    "target": target,
                    "representation": name,
                    "session": r["session"],
                    "genotype": r["genotype"],
                    "observed": float(y[i]),
                    "predicted": float(pred[i]),
                }
                for i, r in enumerate(records)
            )
    retrieval_metrics = []
    for name in ["frozen_rgb_384", "random_projection_32", "contrastive_32"]:
        subset = [r for r in retrieval if r["representation"] == name]
        retrieval_metrics.append(
            {
                "representation": name,
                "queries": len(subset),
                "recall_at_1": float(np.mean([r["correct"] for r in subset])),
                "mean_chance": float(np.mean([r["chance"] for r in subset])),
                "within_round_recall_at_1": float(
                    np.mean([r["within_round_correct"] for r in subset])
                ),
                "within_round_mean_chance": float(
                    np.mean([r["within_round_chance"] for r in subset])
                ),
            }
        )
    for name, rows in [
        ("physiology_metrics.csv", metrics),
        ("physiology_predictions.csv", predictions),
        ("retrieval_metrics.csv", retrieval_metrics),
        ("retrieval_queries.csv", retrieval),
        ("fold_provenance.csv", folds),
        ("training_loss.csv", losses),
    ]:
        write_csv(folder / name, rows)
    np.savez_compressed(folder / "contrastive_heads.npz", **models)
    sources = [
        BASE / "additive_learning.py",
        BASE / "test_additive_learning.py",
        OUT / "sam_traits/session_traits.csv",
        OUT / "dinov2_embeddings.npz",
        ROOT / "OJIP_P6_computed_data.xlsx",
    ]
    sources.extend((model_directory("dinov2-small")).glob("*.safetensors"))
    input_hashes.update({workspace_relative(p): sha256_file(p) for p in sources})
    learned = next(r for r in retrieval_metrics if r["representation"] == "contrastive_32")
    baseline = retrieval_metrics[0]
    findings = [
        f"Cross-view visit retrieval: frozen RGB {baseline['recall_at_1']:.3f}, learned 32-dimensional contrastive head {learned['recall_at_1']:.3f}; chance {learned['mean_chance']:.3f}.",
        f"Within-round retrieval: frozen {baseline['within_round_recall_at_1']:.3f}, contrastive {learned['within_round_recall_at_1']:.3f}. This does not validate physical identity across visits.",
    ]
    for target in targets:
        before = next(
            r for r in metrics if r["target"] == target and r["representation"] == "frozen_rgb_384"
        )
        after = next(
            r
            for r in metrics
            if r["target"] == target and r["representation"] == "contrastive_plus_angular_96"
        )
        findings.append(
            f"{target}: fixed-Ridge genotype-held-out RMSE {before['rmse']:.6f} for frozen RGB versus {after['rmse']:.6f} for contrastive plus angular summaries; exploratory comparison, no winner selection."
        )
        metadata = next(
            r for r in metrics if r["target"] == target and r["representation"] == "metadata"
        )
        adjusted = next(
            r
            for r in metrics
            if r["target"] == target
            and r["representation"] == "metadata_contrastive_plus_angular_96"
        )
        findings.append(
            f"Metadata specificity control for {target}: metadata-only RMSE {metadata['rmse']:.6f}, metadata plus contrastive/angular RMSE {adjusted['rmse']:.6f}. Controls were added after the first execution and are post-hoc."
        )
    summary = {
        "id": "E11",
        "title": "Cross-view self-supervised representation trial",
        "status": "completed_exploratory",
        "question": "Does a small rotation-supervised head improve cross-view visit retrieval or source-linked physiological prediction?",
        "started_utc": started,
        "finished_utc": utc(),
        "runtime_seconds": time.time() - clock,
        "methods": {
            "frozen_backbone": "DINOv2-small, existing local checkpoint",
            "views_per_camera": 8,
            "cameras_extracted": ["rgb", "noir"],
            "head_uses": "RGB only",
            "head": "linear 384 to 32, AdamW lr .01, 150 epochs, temperature .1",
            "positives": "different views within the same training visit",
            "negatives": "other training visits",
            "outer_evaluation": "leave one genotype out; all visits and views of test genotypes excluded from head training",
            "regression": "fixed Ridge alpha10, training-only scaling",
            "angular_features": "phase-invariant first and second Fourier amplitudes on nominal eight-view schedule",
            "selection": "all fixed representations reported; no tuning or winner selection",
            "post_hoc_control_addendum": "Metadata-only and metadata+learned representations added after first run; genotype and round onehot/scaling fit on outer training only",
        },
        "metrics": {
            "n_visits": len(records),
            "view_embeddings": int(np.prod(features.shape[:3])),
            "max_difference_from_previous_pooling": pool_delta,
            "retrieval": retrieval_metrics,
            "physiology": metrics,
        },
        "findings": findings,
        "limitations": [
            "Only 33 QC-complete control visits in two rounds and six genotypes; not 528 independent plants",
            "Same-visit retrieval can exploit acquisition conditions and is not verified biological identity",
            "NoIR embeddings extracted for reproducibility but not used in the RGB contrastive head",
            "Different head dimensions change effective capacity; random projection is included as a control",
            "The experiment trains only a small projection head, not the frozen backbone",
            "Automatic mask QC and physiological linkage limitations remain; exploratory development data reused",
        ],
        "next_requirements": [
            "Independent visits/cohorts and repeated locked-setting scans to test biological specificity and acquisition invariance"
        ],
        "input_sha256": input_hashes,
        "artifacts": [
            workspace_relative(p)
            for p in sorted(folder.iterdir())
            if p.is_file() and p.name != "summary.json"
        ],
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    append_event(
        RUN / "events.jsonl",
        "E11",
        "finished",
        {
            "summary_sha256": sha256_file(folder / "summary.json"),
            "runtime_seconds": summary["runtime_seconds"],
            "status": summary["status"],
        },
    )
    print(
        json.dumps(
            {"id": "E11", "findings": findings, "runtime_seconds": summary["runtime_seconds"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
