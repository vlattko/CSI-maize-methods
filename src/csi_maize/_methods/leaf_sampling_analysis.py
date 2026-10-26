"""Posthoc angular acquisition-design comparison, not independent validation."""

from common import workspace_relative

import numpy as np
from bootstrap import ROOT, BASE, read_csv, write_csv
from leaf_expanded_analyses import DEST, angular_sequence, dump, sha, load


def support_score(counts, threshold=8):
    return float(np.mean(np.minimum(np.asarray(counts) / threshold, 1)))


def greedy_qualified(qualified, threshold=8):
    count = np.zeros(qualified.shape[1], int)
    order = []
    curve = []
    while len(order) < len(qualified):
        choice = max(
            [i for i in range(len(qualified)) if i not in order],
            key=lambda i: (
                support_score(count + qualified[i], threshold),
                int(np.min(count + qualified[i])),
                -i,
            ),
        )
        count += qualified[choice]
        order.append(choice)
        curve.append(support_score(count, threshold))
    return order, curve


def main():
    rows = read_csv(DEST / "source_support_all40.csv")
    sessions = list(dict.fromkeys(r["session"] for r in rows))
    curves = []
    summary = []
    bincounts = []
    rng = np.random.default_rng(20261024)
    for session in sessions:
        rr = [r for r in rows if r["session"] == session]
        ids = list(dict.fromkeys(r["leaf_id"] for r in rr))
        q = np.array(
            [
                [
                    next(
                        r["qualified"] == "True"
                        for r in rr
                        if r["view"] == str(v) and r["leaf_id"] == leaf
                    )
                    for leaf in ids
                ]
                for v in range(40)
            ]
        )
        greedy, gscore = greedy_qualified(q)
        orders = [
            ("angular_offset", offset, [(i + offset) % 40 for i in angular_sequence(40)])
            for offset in range(40)
        ]
        orders += [("random_permutation", i, rng.permutation(40).tolist()) for i in range(100)]
        orders += [("posthoc_greedy", 0, greedy)]
        for design, rep, order in orders:
            assert len(set(order)) == 40
            count = np.zeros(len(ids), int)
            for k, view in enumerate(order):
                count += q[view]
                curves.append(
                    {
                        "session": session,
                        "design": design,
                        "design_id": rep,
                        "views": k + 1,
                        "added_view": view,
                        "support_score": support_score(count),
                        "minimum_qualified_views": int(count.min()),
                        "leaves_with_eight_views": int(np.count_nonzero(count >= 8)),
                    }
                )
        for threshold in [2, 5, 10]:
            bins = np.zeros((len(ids), 10), bool)
            for r in rr:
                if r["qualified"] == "True":
                    bins[ids.index(r["leaf_id"])] |= (
                        np.array([int(r[f"bin_{b}_pixels"]) for b in range(10)]) >= threshold
                    )
            bincounts.append(
                {
                    "session": session,
                    "bin_pixel_threshold": threshold,
                    "supported_bins": int(bins.sum()),
                    "possible_bins": len(ids) * 10,
                    "fraction": float(bins.mean()),
                }
            )
        subset = [r for r in curves if r["session"] == session and r["views"] in [4, 8, 12, 16]]
        values = []
        for k in [4, 8, 12, 16]:
            value = {"views": k}
            for design in ["angular_offset", "random_permutation", "posthoc_greedy"]:
                a = [
                    r["support_score"] for r in subset if r["views"] == k and r["design"] == design
                ]
                value[design] = {
                    "median": float(np.median(a)),
                    "min": min(a),
                    "max": max(a),
                    "p05": float(np.percentile(a, 5)),
                    "p95": float(np.percentile(a, 95)),
                }
            values.append(value)
        summary.append(
            {
                "session": session,
                "local_leaves": ids,
                "qualified_counts_all40": q.sum(0).tolist(),
                "maximum_eight_view_support_score": support_score(q.sum(0)),
                "all_leaves_reach_eight_views": bool(np.all(q.sum(0) >= 8)),
                "design_budget_summaries": values,
            }
        )
    write_csv(DEST / "acquisition_design_curves.csv", curves)
    write_csv(DEST / "parameter_bin_threshold_sensitivity.csv", bincounts)
    result = {
        "random_seed": 20261024,
        "angular_offsets": 40,
        "random_permutations": 100,
        "threshold_qualified_views": 8,
        "sessions": summary,
        "interpretation": "Operational posthoc support threshold; intervals describe design variability, not biological uncertainty. Greedy uses observed source masks after geometry fitting.",
        "input_sha256": {
            workspace_relative(p): sha(p)
            for p in [
                BASE / "leaf_sampling_analysis.py",
                BASE / "test_leaf_sampling.py",
                DEST / "source_support_all40.csv",
            ]
        },
    }
    dump(DEST / "sampling_summary.json", result)
    print(result["sessions"])


if __name__ == "__main__":
    main()
