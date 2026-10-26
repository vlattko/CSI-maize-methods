"""Acquisition-level summaries, segmented exemplars and portable fusion investigation."""

from run_channel_fusion import DEST, OLD, ROOT, BASE, OUT, raw, load_prediction
from channel_fusion_core import *
from multimodal_leaf_core import dump, sha
from bootstrap import read_csv, write_csv
from PIL import Image
import json, io, base64, collections, html, sys
from fractions import Fraction
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYS = [CHANNELS[i] + "__" + CHANNELS[j] for i, j in PAIRS]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#f6f6ef",
        "axes.facecolor": "#f6f6ef",
        "savefig.facecolor": "#f6f6ef",
        "pdf.fonttype": 42,
    }
)


def finite(values):
    return np.array(
        [float(v) for v in values if v is not None and v != "" and np.isfinite(float(v))]
    )


def med(values):
    v = finite(values)
    return float(np.median(v)) if len(v) else None


def encimage(array):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def crop_bounds(mask):
    yy, xx = np.where(mask)
    return (
        max(0, int(xx.min()) - 12),
        max(0, int(yy.min()) - 12),
        min(mask.shape[1], int(xx.max()) + 13),
        min(mask.shape[0], int(yy.max()) + 13),
    )


def crop(a, b):
    return a[b[1] : b[3], b[0] : b[2]]


def b64file(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(DEST / "figures" / (name + ".png"), dpi=160, bbox_inches="tight")
    fig.savefig(DEST / "figures" / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def title(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", pad=14)


def analyse():
    summary = json.loads((DEST / "summary.json").read_text(encoding="utf-8"))
    frames = json.loads((DEST / "frame_registration.json").read_text(encoding="utf-8"))["records"]
    targets = json.loads((DEST / "shared_targets.json").read_text(encoding="utf-8"))["records"]
    good = [t for t in targets if t["pixel_hypothesis"]]
    geometric = [t for t in targets if t["geometry_candidate"]]
    sessions = sorted({f["session"] for f in frames})
    names = [s.split("/")[-1].replace("capture_session_", "") for s in sessions]
    (DEST / "figures").mkdir(exist_ok=True)
    per = []
    for session, name in zip(sessions, names):
        ff = [f for f in frames if f["session"] == session]
        tt = [t for t in good if t["session"] == session]
        gg = [t for t in geometric if t["session"] == session]
        per.append(
            dict(
                session=session,
                label=name,
                frames_with_geometry_candidate=sum(f["supported_targets"] > 0 for f in ff),
                frames_with_pixel_hypothesis=sum(f["pixel_hypotheses"] > 0 for f in ff),
                geometry_candidates=len(gg),
                pixel_hypotheses=len(tt),
                median_affine_IoU=med([f["whole_mask_iou"] for f in ff]),
                median_opposite_angle_IoU=med([f["opposite_view_fixed_transform_iou"] for f in ff]),
                persistent_2px_at_least_40=sum(t["persistent_core_2px_pixels"] >= 40 for t in tt),
                persistent_4px_at_least_40=sum(t["persistent_core_4px_pixels"] >= 40 for t in tt),
            )
        )
    write_csv(DEST / "acquisition_summary.csv", per)
    fig, axs = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [1.5, 1]})
    x = np.arange(len(sessions))
    axs[0].bar(
        x,
        [p["frames_with_geometry_candidate"] for p in per],
        color="#127b72",
        label="At least one geometric candidate",
    )
    axs[0].bar(
        x,
        [p["frames_with_pixel_hypothesis"] for p in per],
        width=0.45,
        color="#d8b653",
        label="At least one common-support hypothesis",
    )
    axs[0].set_ylim(0, 42)
    axs[0].set_ylabel("Views out of 40")
    axs[0].legend(loc="lower right", fontsize=8)
    title(axs[0], "Coverage is qualified separately for each acquisition")
    axs[1].plot(
        x, [p["median_affine_IoU"] for p in per], "o-", color="#127b72", label="Same filename index"
    )
    axs[1].plot(
        x,
        [p["median_opposite_angle_IoU"] for p in per],
        ".--",
        color="#bb643b",
        label="Opposite-angle fixed-transform control",
    )
    axs[1].axhline(0.65, color="gray", ls=":", lw=1)
    axs[1].set_ylabel("Median whole-mask IoU")
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.set_xticks(x, names, rotation=90, fontsize=7)
        ax.grid(axis="y", alpha=0.15)
    save(fig, "01_correspondence_coverage")
    signal = []
    for session, name in zip(sessions, names):
        ff = [f for f in frames if f["session"] == session]
        for c in ["rgb", "noir"]:
            for j, ch in enumerate("RGB"):
                signal.append(
                    dict(
                        session=session,
                        label=name,
                        channel=c + "_" + ch,
                        DN_median=med([f[c + "_channel_qc"]["median"][j] for f in ff]),
                        dark_fraction_median=med(
                            [f[c + "_channel_qc"]["dark_fraction"][j] for f in ff]
                        ),
                        clip_fraction_median=med(
                            [f[c + "_channel_qc"]["clipped_fraction"][j] for f in ff]
                        ),
                    )
                )
    write_csv(DEST / "channel_signal_summary.csv", signal)
    exposure = []
    for f in frames:

        def ex(c, key):
            try:
                return float(Fraction(f[c + "_EXIF"][key]))
            except (KeyError, ValueError, ZeroDivisionError):
                return None

        rt, nt = ex("rgb", "ExposureTime"), ex("noir", "ExposureTime")
        ri, ni = ex("rgb", "ISOSpeedRatings"), ex("noir", "ISOSpeedRatings")
        exposure.append(
            dict(
                session=f["session"],
                view=f["view"],
                rgb_exposure_seconds=rt,
                noir_exposure_seconds=nt,
                noir_to_rgb_exposure_ratio=nt / rt if rt and nt else None,
                rgb_ISO=ri,
                noir_ISO=ni,
                noir_to_rgb_ISO_ratio=ni / ri if ri and ni else None,
            )
        )
    write_csv(DEST / "EXIF_exposure_audit.csv", exposure)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(sessions))
    for c, col in [("rgb", "#127b72"), ("noir", "#b96d3c")]:
        axs[0].plot(
            x,
            [
                med([r[c + "_exposure_seconds"] for r in exposure if r["session"] == s])
                for s in sessions
            ],
            "o-",
            color=col,
            label=c.upper(),
            alpha=0.8,
            ms=3,
        )
    axs[0].set_ylabel("Median shutter time (seconds)")
    axs[0].legend()
    title(axs[0], "Recorded acquisition settings")
    for field, col, label in [
        ("noir_to_rgb_exposure_ratio", "#127b72", "Shutter-time ratio"),
        ("noir_to_rgb_ISO_ratio", "#b96d3c", "ISO-number ratio"),
    ]:
        axs[1].plot(
            x,
            [med([r[field] for r in exposure if r["session"] == s]) for s in sessions],
            "o-",
            color=col,
            label=label,
            alpha=0.8,
            ms=3,
        )
    axs[1].set_ylabel("NoIR / RGB ratio")
    axs[1].legend(fontsize=8)
    title(axs[1], "Equal settings do not establish equal response")
    for ax in axs:
        ax.set_xticks(x, names, rotation=90, fontsize=7)
    save(fig, "09_EXIF_settings")
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, field, label in zip(
        axs,
        ["DN_median", "dark_fraction_median", "clip_fraction_median"],
        ["Recorded DN", "Fraction at DN ≤1", "Fraction at DN ≥254"],
    ):
        values = [finite([r[field] for r in signal if r["channel"] == ch]) for ch in CHANNELS]
        ax.boxplot(values, tick_labels=CHANNELS, showfliers=False)
        ax.tick_params(axis="x", rotation=70)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.15)
    title(axs[0], "Signal before interpretation")
    title(axs[1], "Dark-floor susceptibility")
    title(axs[2], "Clipping audit")
    save(fig, "02_channel_signal")

    def balanced_index(key, kind):
        vals = []
        for session in sessions:
            ts = [t for t in (good if kind == "pixel" else geometric) if t["session"] == session]
            vv = [
                t["common_support_indices"][key]["pixel_median"]
                if kind == "pixel"
                else t["organ_summary"]["nd_of_separate_visible_medians"][key]
                for t in ts
            ]
            vals.append(med(vv))
        return med(vals)

    pixel_matrix = np.full((6, 6), np.nan)
    organ_matrix = np.full((6, 6), np.nan)
    indexsummary = []
    for i, j in PAIRS:
        key = CHANNELS[i] + "__" + CHANNELS[j]
        pv = balanced_index(key, "pixel")
        ov = balanced_index(key, "organ")
        pixel_matrix[i, j] = pv if pv is not None else np.nan
        organ_matrix[i, j] = ov if ov is not None else np.nan
        indexsummary.append(
            dict(
                index=key,
                group="RGB" if j < 3 else "NoIR" if i >= 3 else "cross-camera",
                common_support_targets_with_40_valid_pixels=sum(
                    t["common_support_indices"][key]["n"] >= 40 for t in good
                ),
                organ_summary_targets_valid=sum(
                    t["organ_summary"]["nd_of_separate_visible_medians"][key] is not None
                    for t in geometric
                ),
                pixel_acquisitions_available=len(
                    {
                        t["session"]
                        for t in good
                        if t["common_support_indices"][key]["pixel_median"] is not None
                    }
                ),
                organ_acquisitions_available=len(
                    {
                        t["session"]
                        for t in geometric
                        if t["organ_summary"]["nd_of_separate_visible_medians"][key] is not None
                    }
                ),
                acquisition_balanced_pixel_ND_median=pv,
                acquisition_balanced_organ_ND_median=ov,
            )
        )
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#e3e6df")
    for ax, m, label in zip(
        axs,
        [organ_matrix, pixel_matrix],
        ["Separate visible organ medians", "Shared support under affine hypothesis"],
    ):
        im = ax.imshow(m, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(6), CHANNELS, rotation=65)
        ax.set_yticks(range(6), CHANNELS)
        title(ax, label)
        for i, j in PAIRS:
            if np.isfinite(m[i, j]):
                r = next(r for r in indexsummary if r["index"] == CHANNELS[i] + "__" + CHANNELS[j])
                n = (
                    r["organ_acquisitions_available"]
                    if label.startswith("Separate")
                    else r["pixel_acquisitions_available"]
                )
                ax.text(j, i, f"{m[i, j]:.2f}\n{n} acq", ha="center", va="center", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.24, top=0.86, wspace=0.50)
    cax = fig.add_axes([0.91, 0.27, 0.015, 0.49])
    fig.colorbar(im, cax=cax, label="ND(row channel − column channel)")
    # Save without tight_layout because these axes share a manually placed colorbar.
    fig.savefig(DEST / "figures/03_channel_pairs.png", dpi=160, bbox_inches="tight")
    fig.savefig(DEST / "figures/03_channel_pairs.pdf", bbox_inches="tight")
    plt.close(fig)
    perturb = read_csv(DEST / "registration_perturbations.csv")
    grouped = collections.defaultdict(list)
    for r in perturb:
        grouped[r["target_id"], r["index"], r.get("support_mode") or "adaptive"].append(r)
    spatial = []
    for (tid, key, mode), rr in grouped.items():
        baseline = next(r for r in rr if int(r["dx"]) == int(r["dy"]) == 0)
        base = baseline["pixel_median"]
        shifts = [r for r in rr if abs(int(r["dx"])) + abs(int(r["dy"])) > 0]
        two = [r for r in shifts if abs(int(r["dx"])) + abs(int(r["dy"])) <= 2]

        def maximum(rows):
            vals = finite([r["pixel_median"] for r in rows])
            return float(np.max(np.abs(vals - float(base)))) if len(vals) and base != "" else None

        spatial.append(
            dict(
                target_id=tid,
                session=rr[0]["session"],
                index=key,
                support_mode=mode,
                max_abs_delta_2px=maximum(two),
                max_abs_delta_4px=maximum(shifts),
                assignment_retention_2px=sum(r["assignment_retained"] == "True" for r in two)
                / max(1, len(two)),
                baseline_n=int(baseline["n"]),
                missing_shift_fraction=sum(r["pixel_median"] == "" for r in shifts)
                / max(1, len(shifts)),
            )
        )
    write_csv(DEST / "spatial_sensitivity_summary.csv", spatial)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    xx = np.arange(15)
    for mode, color, label in [
        ("adaptive", "#d69b3b", "Changing shared support"),
        ("persistent_2px_core", "#127b72", "Fixed persistent core"),
    ]:
        values = []
        for key in KEYS:
            peracq = [
                med(
                    [
                        r["max_abs_delta_2px"]
                        for r in spatial
                        if r["index"] == key and r["support_mode"] == mode and r["session"] == s
                    ]
                )
                for s in sessions
            ]
            values.append(med(peracq))
        axs[0].plot(
            xx, [np.nan if v is None else v for v in values], "o-", color=color, label=label
        )
    axs[0].set_xticks(xx, [k.replace("__", " / ") for k in KEYS], rotation=90, fontsize=7)
    axs[0].set_ylabel("Median of acquisition medians of target max |ΔND|")
    axs[0].legend(fontsize=8)
    title(axs[0], "Sensitivity to ±2 processing pixels")
    axs[1].hist(
        [t["persistent_core_2px_pixels"] / t["common_interior_pixels"] for t in good],
        bins=25,
        color="#127b72",
        alpha=0.8,
        label="±2 pixel core",
    )
    axs[1].hist(
        [t["persistent_core_4px_pixels"] / t["common_interior_pixels"] for t in good],
        bins=25,
        histtype="step",
        lw=2,
        color="#c57037",
        label="±4 pixel core",
    )
    axs[1].set_xlabel("Persistent / baseline common pixels")
    axs[1].set_ylabel("Candidate targets, repeated views")
    axs[1].legend(fontsize=8)
    title(axs[1], "Thin structures can lose their common core")
    save(fig, "04_spatial_sensitivity")
    gainrows = read_csv(DEST / "relative_DN_scaling.csv")
    gaincurves = []
    for key in KEYS:
        if key.split("__")[0].startswith("noir") or not key.split("__")[1].startswith("noir"):
            continue
        for gain in POLICY["relative_noir_DN_scales"]:
            rr = [
                r
                for r in gainrows
                if r["index"] == key and float(r["relative_noir_DN_scale"]) == gain
            ]
            v = [med([r["pixel_median"] for r in rr if r["session"] == s]) for s in sessions]
            gaincurves.append(
                dict(index=key, relative_noir_DN_scale=gain, acquisition_balanced_median=med(v))
            )
    write_csv(DEST / "DN_scaling_summary.csv", gaincurves)
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in dict.fromkeys(r["index"] for r in gaincurves):
        rr = [r for r in gaincurves if r["index"] == key]
        ax.plot(
            [r["relative_noir_DN_scale"] for r in rr],
            [
                np.nan
                if r["acquisition_balanced_median"] is None
                else r["acquisition_balanced_median"]
                for r in rr
            ],
            "o-",
            label=key.replace("__", " / "),
            ms=3,
        )
    ax.axvline(1, color="gray", ls=":")
    ax.set_xlabel("Counterfactual NoIR DN multiplier (no clipping, fixed baseline-valid samples)")
    ax.set_ylabel("Median of acquisition median ND")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    title(ax, "Independent camera scaling does not cancel in a cross-camera index")
    save(fig, "05_relative_camera_scaling")
    # Pure algebra demonstration, distinct from camera-calibration claims.
    compositions = []
    for f in frames:
        v = np.array(f["rgb_channel_qc"]["median"] + f["noir_channel_qc"]["median"], float)
        if np.all(np.isfinite(v) & (v > 1) & (v < 254)):
            lv = np.log(v)
            compositions.append(lv - lv.mean())
    rank = {
        "eligible_frame_summaries": len(compositions),
        "maximum_log_ratio_degrees_of_freedom": 5,
        "interpretation": "Measured compositional variation in processed camera DN, not independent spectral bands.",
    }
    if len(compositions) > 5:
        x = np.array(compositions)
        sv = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
        rank["singular_values"] = sv.tolist()
        rank["variance_fraction"] = (sv**2 / (sv**2).sum()).tolist()
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].bar(range(1, 7), rank["variance_fraction"], color="#127b72")
        axs[0].set_xlabel("Centered log-ratio component")
        axs[0].set_ylabel("Variance fraction")
        title(axs[0], "Six channels carry at most five relative coordinates")
        axs[1].axis("off")
        axs[1].text(
            0.02,
            0.94,
            "NDᵢⱼ = tanh[½ log(Cᵢ/Cⱼ)]\n\nFor one positive six-channel vector:\n15 ratios, 5 relative coordinates.\n\nPairwise spatial medians may vary\nin more dimensions because their\nsamples and nonlinear summaries differ.\nThey do not create new spectral bands.",
            va="top",
            fontsize=11,
        )
        save(fig, "06_feature_redundancy")
    else:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        values = np.array(
            [f["rgb_channel_qc"]["median"] + f["noir_channel_qc"]["median"] for f in frames], float
        )
        counts = np.sum((values > 1) & (values < 254), axis=0)
        axs[0].bar(range(6), counts, color="#127b72")
        axs[0].set_xticks(range(6), CHANNELS, rotation=60)
        axs[0].set_ylabel("Frames with channel median in (1,254) DN")
        title(axs[0], "Joint signal availability limits decomposition")
        axs[1].axis("off")
        axs[1].text(
            0.02,
            0.94,
            f"{len(compositions)} eligible six-channel frame vectors.\nEmpirical six-channel CLR SVD omitted.\n\nFor one positive six-channel vector,\n15 ND ratios have 5 relative coordinates.\n\nPairwise spatial medians can vary\nin more statistical dimensions.\nThey do not create new spectral bands.",
            va="top",
            fontsize=11,
        )
        save(fig, "06_feature_redundancy")
    dump(DEST / "compositional_rank.json", rank)
    # Compare the two summary estimands on identical common-support samples.
    differences = []
    for key in KEYS:
        values = []
        for s in sessions:
            rr = [t["common_support_indices"][key] for t in good if t["session"] == s]
            values.append(
                med(
                    [
                        abs(r["pixel_median"] - r["ratio_of_medians"])
                        for r in rr
                        if r["pixel_median"] is not None and r["ratio_of_medians"] is not None
                    ]
                )
            )
        differences.append(med(values))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(15), [np.nan if v is None else v for v in differences], color="#b76c42")
    ax.set_xticks(range(15), [k.replace("__", " / ") for k in KEYS], rotation=75, fontsize=8)
    ax.set_ylabel("Acquisition-balanced median absolute difference")
    title(ax, "Median of ratios and ratio of medians answer different questions")
    save(fig, "07_estimand_difference")
    support = read_csv(DEST / "target_support_audit.csv")
    gates = read_csv(DEST / "gate_sensitivity.csv")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for field, color, label in [
        ("geometry_candidates", "#127b72", "Geometric candidates"),
        ("shared_support_hypotheses", "#c58b33", "Common-support hypotheses"),
    ]:
        gatevals = sorted({float(r["leaf_iou_gate"]) for r in gates})
        axs[0].plot(
            gatevals,
            [sum(int(r[field]) for r in gates if float(r["leaf_iou_gate"]) == g) for g in gatevals],
            "o-",
            color=color,
            label=label,
        )
    axs[0].axvline(0.5, color="gray", ls=":")
    axs[0].set_xlabel("Leaf-mask IoU gate, filtering frozen proposals")
    axs[0].set_ylabel("Retained candidate targets")
    axs[0].legend(fontsize=8)
    title(axs[0], "Gate sensitivity is not an accuracy curve")
    erode = []
    for k in [1, 3, 5, 7]:
        rr = [
            r
            for r in support
            if int(r["erosion_kernel"]) == k and r["geometry_candidate"] == "True"
        ]
        erode.append(sum(int(r["common_pixels"]) >= 40 for r in rr))
    axs[1].bar([1, 3, 5, 7], erode, width=1.2, color="#127b72")
    axs[1].set_xticks([1, 3, 5, 7])
    axs[1].set_xlabel("Square erosion kernel width, processing pixels")
    axs[1].set_ylabel("Geometric candidates with ≥40 common pixels")
    title(axs[1], "Interior selection affects thin-leaf coverage")
    save(fig, "08_support_and_gates")
    spatialbase = [
        r for r in spatial if r["index"] == "rgb_R__noir_B" and r["support_mode"] == "adaptive"
    ]
    retention = med([r["assignment_retention_2px"] for r in spatialbase])
    analysis = dict(
        summary=summary,
        acquisitions=per,
        index_summary=indexsummary,
        channel_summary=signal,
        rank=rank,
        persistent_2px_at_least_40=sum(t["persistent_core_2px_pixels"] >= 40 for t in good),
        persistent_4px_at_least_40=sum(t["persistent_core_4px_pixels"] >= 40 for t in good),
        median_assignment_retention_2px=retention,
        spatial_acquisition_balanced_RGB_R_NoIR_B=med(
            [
                med([r["max_abs_delta_2px"] for r in spatialbase if r["session"] == s])
                for s in sessions
            ]
        ),
        paired_equal_EXIF_shutter=sum(
            r["rgb_exposure_seconds"] == r["noir_exposure_seconds"] for r in exposure
        ),
        paired_equal_EXIF_ISO=sum(r["rgb_ISO"] == r["noir_ISO"] for r in exposure),
    )
    dump(DEST / "analysis.json", analysis)
    write_csv(DEST / "index_summary.csv", indexsummary)
    return analysis, targets, frames


def exemplars(targets):
    good = [t for t in targets if t["pixel_hypothesis"]]
    ss = sorted({t["session"] for t in good})
    selected = [ss[i] for i in np.linspace(0, len(ss) - 1, min(12, len(ss)), dtype=int)]
    traits = {
        (r["session"], r["camera"], int(r["view"])): r
        for r in read_csv(OUT / "sam_traits/view_traits.csv")
    }
    examples = []
    (DEST / "specimens").mkdir(exist_ok=True)
    for session in selected:
        t = max(
            [x for x in good if x["session"] == session], key=lambda x: x["common_interior_pixels"]
        )
        v = t["view"]
        a, am, _ = raw(traits[session, "rgb", v])
        b, bm, _ = raw(traits[session, "noir", v])
        rp = load_prediction(session, "rgb")[v]
        npred = load_prediction(session, "noir")[v]
        rm = next(m for r, m in rp if int(r["detection_index"]) == t["rgb_detection_index"])
        nm = next(m for r, m in npred if int(r["detection_index"]) == t["noir_detection_index"])
        ri = interior(rm & am)
        ni = interior(nm & bm)
        matrix = np.array(t["registration_source_to_target"], np.float32)
        wb, fov = warp_to_rgb(b.astype(np.float32), matrix, am.shape)
        wn = warp_to_rgb(ni.astype(np.uint8), matrix, am.shape, True)[0] > 0
        common = ri & wn & fov
        rgb = a.copy()
        rgb[~(rm & am)] = 0
        nir = b.copy()
        nir[~(nm & bm)] = 0
        registered = np.clip(wb, 0, 255).astype(np.uint8)
        registered[~common] = 0
        rgbbounds = crop_bounds(rm | wn)
        noirbounds = crop_bounds(nm)
        maps = {}
        prefix = t["target_id"]
        for i, j in PAIRS:
            six = np.concatenate([a.astype(float), wb.astype(float)], axis=2)
            val = normalized_difference(six[:, :, i], six[:, :, j])
            valid = common & np.isfinite(val)
            colors = np.zeros((*am.shape, 3), np.uint8)
            colors[valid] = (plt.get_cmap("RdYlGn")((val[valid] + 1) / 2)[:, :3] * 255).astype(
                np.uint8
            )
            key = CHANNELS[i] + "__" + CHANNELS[j]
            maps[key] = encimage(crop(colors, rgbbounds))
        np.savez_compressed(
            DEST / "specimens" / (prefix + ".npz"),
            rgb=a,
            noir_registered=wb,
            common_support=common,
            source_rgb_mask=rm,
            source_noir_mask=nm,
            affine_rgb_to_noir=matrix,
        )
        # Scientific specimen artwork is derived directly from the data, never generated imagery.
        Image.fromarray(np.concatenate([rgb, nir, registered], axis=1)).save(
            DEST / "specimens" / (prefix + ".png")
        )
        examples.append(
            dict(
                target=dict(
                    t,
                    display_crop_RGB_xyxy=list(rgbbounds),
                    display_crop_NoIR_xyxy=list(noirbounds),
                ),
                rgb=encimage(crop(rgb, rgbbounds)),
                noir=encimage(crop(nir, noirbounds)),
                registered=encimage(crop(registered, rgbbounds)),
                maps=maps,
            )
        )
    dump(
        DEST / "specimen_manifest.json",
        {
            "selection": "At most 12 evenly spaced acquisitions; largest common-support target within each. Illustrative selection, not evidence of anatomical accuracy.",
            "targets": [e["target"]["target_id"] for e in examples],
        },
    )
    return examples


REFERENCES = [
    (
        "Stamford JD, Vialet-Chabrand S, Cameron I, Lawson T (2023). Development of an accurate low cost NDVI imaging system for assessing plant health. Plant Methods 19, 9.",
        "https://doi.org/10.1186/s13007-023-00981-8",
        "Direct dual-camera precedent: alignment plus reference-based calibration. Their sensor and filter differ from this archive; their conversion is not transferred here.",
    ),
    (
        "Burggraaff O et al. (2019). Standardized spectral and radiometric calibration of consumer cameras. Optics Express 27, 19075–19101.",
        "https://doi.org/10.1364/OE.27.019075",
        "Consumer camera response, JPEG processing and camera-specific calibration motivate the radiometric audit.",
    ),
    (
        "Gitelson AA, Kaufman YJ, Stark R, Rundquist D (2002). Novel algorithms for remote estimation of vegetation fraction. Remote Sensing of Environment 80, 76–87.",
        "https://doi.org/10.1016/S0034-4257(01)00289-9",
        "Visible reflectance indices provide context; processed camera-DN indices do not inherit calibrated vegetation-fraction interpretation.",
    ),
    (
        "Raspberry Pi. Camera Module 2 NoIR specifications. Accessed 24 October 2026.",
        "https://www.raspberrypi.com/products/pi-noir-camera-v2/",
        "NoIR identifies removal of the IR-blocking filter, not isolation of three NIR wavelengths.",
    ),
    (
        "Raspberry Pi. Camera software documentation. Accessed 24 October 2026.",
        "https://www.raspberrypi.com/documentation/computers/camera_software.html",
        "RAW acquisition and camera processing controls are priorities for validation captures.",
    ),
]


def build_report(analysis, examples):
    s = analysis["summary"]
    ng = s["geometry_candidates"]
    npix = s["pixel_hypotheses"]
    dark = med(
        [r["dark_fraction_median"] for r in analysis["channel_summary"] if r["channel"] == "noir_G"]
    )
    allvalid = s["all_six_valid_pixel_targets"]
    text = f"""CSI-MAIZE | Shared targets and six-channel response
Executed 24 October 2026. Authors of the study: Vlatko Galic and Maja Mazur.

Purpose and capability
A candidate leaf can become a meeting point for geometry, visible colour and longpass-camera response. This experiment creates a local target keyed by acquisition, view, RGB detection and NoIR detection. It does not establish persistent physical leaf identity. It adds an auditable measurement layer to the frozen RGB/NoIR detector experiment, using the original recorded channels rather than the triplicated-blue training ablation.

Archive and correspondence
All {s["pairs"]} filename-index pairs from {s["acquisitions"]} acquisitions and {s["observations"]} frozen camera-local candidate observations were audited. There were {s["proposal_targets"]} one-to-one proposals at IoU >=0.25. Geometric gates retained {ng} candidate targets. A total of {npix} passed common-interior engineering gates. None has independent human-reviewed cross-camera anatomical identity. Shared weak teachers make prediction agreement non-independent. Same-index phase was the best silhouette-profile hypothesis in {s["nominal_same_index_phase_best_acquisitions"]}/{s["acquisitions"]} acquisitions; this is not proof of synchronization. Every unmatched and rejected observation is retained in observation_states.csv.

Method
Whole-plant SAM silhouettes were resized with nearest-neighbour interpolation to 512 x 385. A moment-initialized affine transform was refined by ECC only if whole-mask IoU improved. Assignment used original predicted instance shapes, without clipping prediction mistakes away. A Hungarian proposal was retained as a geometric candidate only if whole-mask IoU >=0.65, instance IoU >=0.50, and row and column margins >=0.10. These deliberately conservative thresholds were frozen before reading index results. They are engineering rules, not calibrated accuracy estimates. For feature extraction only, each instance was intersected with SAM foreground and eroded with a 3 x 3 kernel. NoIR was sampled bilinearly on the RGB grid with inverse affine mapping; masks used nearest neighbours and out-of-frame samples were excluded. Common support required >=40 pixels and at least 0.35 of the smaller visible interior. Affine overlap cannot resolve non-planar parallax, different occlusion order or camera-dependent visibility.

The supplementary support audit retains common-support fractions separately for both cameras, prediction foreground incompatibility, and erosion kernels of width 1, 3, 5 and 7. A leaf-IoU gate sweep from 0.25 to 0.80 filters frozen Hungarian proposals without changing assignments. These analyses show threshold dependence; they do not select an optimal threshold by any unmeasured biological accuracy.

Three measurement products
1. Camera-local pixelwise indices are available for all frozen observations where signal is adequate.
2. Separate visible-organ channel medians can be combined for geometric candidate matches. They need not describe identical tissue patches.
3. Shared-support pixelwise indices are exported as affine sampling hypotheses for sensitivity analysis. Matching physical tissue is not established by the current archive.

Feature definitions and signal exclusions
Channels are rgb_R, rgb_G, rgb_B, noir_R, noir_G and noir_B, named by recorded camera output. For all 15 unique pairs, ND(a,b)=(a-b)/(a+b). Channel values <=1 or >=254 and pair sums <12 DN were excluded. At least 40 valid pixels were required for a pixel summary. Pairwise support counts are exported; one dark channel does not silently remove all other pairs. We export median pixel ND, its 10th and 90th percentiles, and ND of the two medians on the same pair-valid pixels. These are distinguished from ND of separate visible-organ medians. RGB/NoIR local GCC is G/(R+G+B), and normalized ExG is 3G/(R+G+B)-1, applied to valid recorded channels. Their camera-DN use is exploratory. These are not calibrated NDVI, reflectance, chlorophyll or photosynthetic performance estimates. Three longpass Bayer outputs are not three established spectral bands.

Signal result
The acquisition-balanced median dark fraction in whole-plant NoIR green was {dark:.3f}. Only {allvalid}/{npix} common-support target hypotheses had at least 40 jointly valid pixels in all six channels. Exposure time was recovered from EXIF for {s["EXIF_exposure_available_images"]}/{s["images"]} images. Paired shutter times were equal in {analysis["paired_equal_EXIF_shutter"]}/{s["pairs"]} pairs and paired ISO numbers in {analysis["paired_equal_EXIF_ISO"]}/{s["pairs"]}. Equal settings do not establish equal spectral response or output gain. Availability of some EXIF fields does not recover white-balance gains, spectral response or a reflectance reference. Signal limitations are part of the result, especially for any green-channel ratio.

Spatial and camera-scaling experiments
The fixed affine was translated by +/-2 and +/-4 pixels along each RGB-grid axis, with no refitting and no shift selected by colour. Instance assignments and shared-support indices were recomputed. Both adaptive support and the fixed intersection of supports under the +/-2-pixel perturbations were analysed. A total of {analysis["persistent_2px_at_least_40"]}/{npix} targets retained a persistent core of >=40 pixels under the small shifts; {analysis["persistent_4px_at_least_40"]} retained that many under the larger shifts. Empty cores and missing indices remain explicit. Ranges are sensitivity measures, not confidence intervals. Separately, original baseline-valid NoIR DN was multiplied by 0.5, 0.75, 1, 1.25, 1.5 and 2 without clipping. This is a mathematical stress test, not exposure correction. Same-camera common gain cancels; independent camera scaling does not. A ratio therefore cannot be interpreted as physiology merely because it is normalized.

Aggregation and redundancy
Plots of index values first take a median across candidate targets within an acquisition, then a median across available acquisitions, preventing dense views from dominating these descriptive summaries. There are no independent-pixel significance tests. Forty views are technical repeats and anatomical candidates remain unreviewed. Acquisition-balanced summaries are still not independent biological replicate inference. For a single positive six-channel vector, ND_ij=tanh[0.5 log(C_i/C_j)] and the 15 ratios have five relative degrees of freedom. This bound does not apply to the vector of 15 exported spatial medians: nonlinear pairwise aggregation and different valid supports can introduce more statistical dimensions. Neither creates new spectral bands. The algebra illustration and signal-availability audit use frame-level whole-foreground channel medians rather than verified leaf tissue; they are not proof of independent spectral information. No frame vector passed the joint six-channel signal rules, so the empirical compositional decomposition was omitted.

What this adds downstream
The target schema can join a validated leaf identity, geometric surface and camera-local response without conflating them. A later canonical leaf surface could accumulate only observations with verified visibility and camera calibration, retaining angle, illumination and uncertainty as covariates. That enables within-leaf response trajectories, spatial heterogeneity, senescence-front hypotheses and joint geometry/physiology models. The present addition establishes provenance, local candidate correspondence, channel-quality checks and failure-aware feature extraction. It does not yet establish longitudinal leaf physiology.

Decisive next validation
Capture identified leaves and spectrally characterized neutral references with fixed pose, RAW output, recorded exposure/gain, deliberate exposure and illumination changes, and both cameras geometrically calibrated. Annotate anatomical correspondence on independent plants and difficult occlusions. Compare RGB-only, NoIR-only and combined features using the same reviewed targets and grouped held-out plants, including brightness and morphology baselines. Match any fluorescence, pigment or spectrometer endpoint by explicit plant/leaf/visit metadata. An index must demonstrate stability on unchanged tissue and added predictive value before being promoted to a physiological trait. Existing unverified plant/visit joins are not repaired by adding more indices.

Provenance and preservation
Raw images, frozen predictions, trained weights and earlier manuscripts were preserved. run_configuration.json hashes input metadata, prediction masks and code. Original images were SHA256 checked during the run; frame registration records bind mask hashes, source names and EXIF. All thresholds, exclusions, assignment states and perturbation rows are exported. Representative figures are derived from recorded imagery on black backgrounds. Exemplars were selected by evenly spaced acquisition names, then largest common support, solely to illustrate the measurement interface.

Scientific resources
""" + "\n\n".join(title + "\n" + url + "\n" + note for title, url, note in REFERENCES)
    (DEST / "INVESTIGATION_shared_targets.txt").write_text(text, encoding="utf-8", newline="\n")
    figs = [
        (
            "01_correspondence_coverage",
            "Every acquisition is audited",
            "Coverage and the fixed-transform opposite-angle control. Opposite views can share silhouette shape; this control cannot validate anatomy.",
        ),
        (
            "02_channel_signal",
            "Which channels carry usable signal?",
            "Each box summarizes acquisition medians of whole-plant foreground. A low-signal channel can generate an extreme ratio without revealing plant condition.",
        ),
        (
            "03_channel_pairs",
            "One feature library, two estimands",
            "Signed ND is row minus column. Each value is a median of acquisition medians. Grey cells are unused combinations or missing data. Separate visible medians and pixel hypotheses need different interpretations.",
        ),
        (
            "04_spatial_sensitivity",
            "Does a small misregistration change the answer?",
            "Fixed-transform shifts are in 512 x 385 processing pixels. The persistent core avoids changing which RGB pixels are sampled; it does not prove corresponding NoIR tissue.",
        ),
        (
            "05_relative_camera_scaling",
            "Normalization does not remove independent camera gain",
            "The NoIR DN multiplier is a counterfactual mathematical intervention, with fixed baseline-valid samples and no clipping. It is not a reconstruction of physical camera exposure.",
        ),
        (
            "06_feature_redundancy",
            "15 indices do not create 15 independent bands",
            "The six-channel centered log-ratio decomposition uses eligible whole-foreground frame medians. The sixth relative component is algebraically redundant.",
        ),
        (
            "07_estimand_difference",
            "Choose the statistic before the interpretation",
            "Both statistics use the same baseline pair-valid samples. Their numerical difference is why they are exported separately.",
        ),
    ]
    figs.append(
        (
            "08_support_and_gates",
            "Which choices determine coverage?",
            "The two-sided support audit preserves each camera fraction and varies interior erosion. Gate sweeps filter frozen proposals; they are not accuracy estimates.",
        )
    )
    if analysis["rank"]["eligible_frame_summaries"] <= 5:
        figs[5] = (
            figs[5][0],
            figs[5][1],
            "Insufficient six-channel frame vectors survived the signal rules, so empirical CLR decomposition was omitted. Five relative coordinates applies to a single positive six-channel vector, not to all aggregated spatial features.",
        )
    figs.append(
        (
            "09_EXIF_settings",
            "What do the recorded camera settings establish?",
            "Shutter time and ISO were recovered from original JPEG EXIF. Equality of recorded settings cannot establish equal gain, spectral response, tone mapping or reflectance calibration.",
        )
    )
    sections = "".join(
        '<article class="figure"><div class="eyebrow">EXPERIMENT '
        + str(i + 1).zfill(2)
        + "</div><h2>"
        + name
        + '</h2><img loading="lazy" src="'
        + b64file(DEST / "figures" / (file + ".png"))
        + '" alt="'
        + html.escape(name)
        + '"><p>'
        + desc
        + "</p></article>"
        for i, (file, name, desc) in enumerate(figs)
        if (DEST / "figures" / (file + ".png")).exists()
    )
    refs = "".join(
        '<li><a href="'
        + url
        + '">'
        + html.escape(title)
        + "</a><p>"
        + html.escape(note)
        + "</p></li>"
        for title, url, note in REFERENCES
    )
    cards = (
        '<div class="metrics">'
        + "".join(
            '<div class="metric"><strong>' + str(v) + "</strong><span>" + label + "</span></div>"
            for v, label in [
                (s["pairs"], "filename-index pairs audited"),
                (ng, "geometric candidate targets"),
                (npix, "common-support hypotheses"),
                (analysis["persistent_2px_at_least_40"], "targets with ≥40 persistent pixels"),
            ]
        )
        + "</div>"
    )
    payload = json.dumps(examples, ensure_ascii=False, allow_nan=False).replace("</", r"<\/")
    a_json = json.dumps(analysis, ensure_ascii=False, allow_nan=False).replace("</", r"<\/")
    report = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CSI maize | Shared leaf targets</title><style>
 :root{--ink:#173b37;--muted:#526c65;--paper:#f6f6ef;--line:#d6ded2;--gold:#d6ad45}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,sans-serif}header,main,footer{max-width:1180px;margin:auto;padding:35px 34px}header{padding-top:70px}.eyebrow{font-size:12px;letter-spacing:2px;font-weight:700;color:#667c65}h1{font:clamp(36px,6vw,70px)/1.07 Georgia,serif;letter-spacing:-2px;max-width:950px;margin:20px 0}h2{font:32px/1.2 Georgia,serif;margin:14px 0 24px}p{max-width:900px}.lead{font-size:21px;max-width:850px;color:var(--muted)}nav{display:flex;gap:25px;flex-wrap:wrap;border-block:1px solid var(--line);padding:16px 0;margin:35px 0}a{color:#08675d}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:35px 0}.metric{padding:20px;border-top:3px solid var(--gold);background:#e9eee3}.metric strong{display:block;font-size:34px}.metric span{font-size:13px}.notice{border-left:4px solid #ba783d;padding:16px 22px;background:#f0e8d7}.figure{padding:45px 0;border-top:1px solid var(--line)}.figure>img{width:100%;height:auto}.figure>p{font-size:14px;color:var(--muted)}.controls{display:flex;gap:15px;flex-wrap:wrap;margin:20px 0}label{font-size:13px;font-weight:600}select{display:block;max-width:90vw;padding:10px;margin-top:6px;color:var(--ink);background:white;border:1px solid #9db3a3;border-radius:4px;font-size:14px}.specimens{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.specimens img{width:100%;background:#000}.specimens figcaption{font-size:12px;color:var(--muted)}figure{margin:0}.legend{height:10px;background:linear-gradient(to right,#a50026,#ffffbf,#006837);max-width:280px;margin-top:20px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}.stats div{border:1px solid var(--line);padding:14px}.stats b{display:block;font-size:22px}.details{font-size:13px;overflow-wrap:anywhere}details{margin:25px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#e9eee3;padding:22px;font:13px/1.7 ui-monospace,monospace}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:8px;text-align:left}li{margin:20px 0}li p{font-size:14px;color:var(--muted)}.roadmap{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}.roadmap div{padding:20px;background:#e9eee3}.roadmap h3{font:23px Georgia,serif}button{padding:10px 16px;background:#17695f;color:white;border:0;border-radius:3px;cursor:pointer}@media(max-width:750px){header,main,footer{padding:25px 18px}.metrics,.specimens,.roadmap{grid-template-columns:repeat(2,1fr)}.stats{grid-template-columns:1fr}h1{letter-spacing:-1px}}
 </style></head><body><header><div class="eyebrow">CSI-MAIZE · RESEARCH ADDITION · 24 OCTOBER 2026</div><h1>A shared leaf target.<br>Six recorded channels.<br>Every assumption visible.</h1><p class="lead">Connecting organ geometry, visible colour and longpass-camera response, with explicit uncertainty about which leaf and which tissue each camera sees.</p><nav><a href="#explorer">Target explorer</a><a href="#experiments">Archive experiments</a><a href="#method">Investigation</a><a href="#next">Validation path</a><a href="#references">References</a></nav>"""
        + cards
        + """<p class="notice"><b>Executed measurement audit.</b> The targets below are geometric correspondence hypotheses. Their ratios describe processed camera DN. Anatomical identity, calibrated reflectance and chlorophyll response require independent validation.</p></header><main><section id="explorer"><div class="eyebrow">INTERACTIVE SPECIMENS</div><h2>One candidate, complementary views</h2><p>Select an acquisition example and channel pair. The original camera-local segmented leaves remain visible beside the registered common-support image and exploratory index map. Black indicates excluded background or missing support.</p><div class="controls"><label>Illustrative candidate<select id="case"></select></label><label>Signed normalized difference<select id="index"></select></label></div><div class="specimens"><figure><img id="rgb" alt="Segmented RGB candidate"><figcaption>RGB, original camera grid</figcaption></figure><figure><img id="noir" alt="Segmented NoIR candidate"><figcaption>NoIR, original recorded channels</figcaption></figure><figure><img id="registered" alt="NoIR on shared support in RGB coordinates"><figcaption>NoIR on affine common support</figcaption></figure><figure><img id="map" alt="Exploratory recorded-DN index map"><figcaption id="maplabel"></figcaption></figure></div><div class="legend"></div><span class="details">−1 · signed index · +1. Black: excluded, dark, clipped or unsupported.</span><div class="stats" id="stats"></div><p id="identity" class="details"></p><details><summary>Candidate measurements and provenance</summary><pre id="provenance"></pre></details></section><section id="experiments">"""
        + sections
        + """</section><section id="method" class="figure"><div class="eyebrow">FORMATTED INVESTIGATION</div><h2>What this layer establishes</h2><p>A target now has traceable camera observations, local descriptors, candidate correspondence, channel quality and sensitivity diagnostics. A future leaf identity can attach to this schema, while retaining different visible tissue and acquisition effects.</p><button id="download">Download investigation text</button><details><summary>Read the full method, findings and limits</summary><pre>"""
        + html.escape(text)
        + """</pre></details><details><summary>Feature availability across all 15 channel pairs</summary><div style="overflow:auto"><table><thead><tr><th>Pair (first minus second)</th><th>Group</th><th>Common-support targets ≥40 valid pixels</th><th>Organ summaries valid</th></tr></thead><tbody>"""
        + "".join(
            "<tr><td>"
            + r["index"]
            + "</td><td>"
            + r["group"]
            + "</td><td>"
            + str(r["common_support_targets_with_40_valid_pixels"])
            + "</td><td>"
            + str(r["organ_summary_targets_valid"])
            + "</td></tr>"
            for r in analysis["index_summary"]
        )
        + """</tbody></table></div></details></section><section id="next" class="figure"><div class="eyebrow">FROM CANDIDATES TO LEAF PHYSIOLOGY</div><h2>The next experiments have a clear job</h2><div class="roadmap"><div><h3>Establish identity</h3><p>Independent annotations, hard occlusions, anatomical leaf rank and verified plant/visit links.</p></div><div><h3>Calibrate response</h3><p>RAW, spectral references, measured filter response, camera geometry and deliberate exposure changes on unchanged tissue.</p></div><div><h3>Test added value</h3><p>Compare RGB, NoIR and fusion on identical held-out plants, with brightness and morphology baselines and explicitly linked physiological targets.</p></div></div></section><section id="references" class="figure"><h2>Scientific resources</h2><ol>"""
        + refs
        + """</ol></section></main><footer><p>CSI-MAIZE · Vlatko Galic and Maja Mazur · Agricultural Institute Osijek.<br>Original data and previous manuscripts preserved. This portable report contains the figures and illustrative specimens; machine-readable tables and reproducible code accompany the experiment package.</p></footer><script>
 const EXAMPLES=EXAMPLES_JSON;const ANALYSIS=ANALYSIS_JSON;const TEXT=TEXT_JSON;
 const caseSelect=document.getElementById('case'),indexSelect=document.getElementById('index');
 EXAMPLES.forEach((e,i)=>{let o=new Option(e.target.target_id,i);caseSelect.add(o)});
 Object.keys(EXAMPLES[0].maps).forEach(k=>indexSelect.add(new Option(k.replace('__',' − '),k)));
 indexSelect.value='rgb_R__noir_B';
 function fmt(x){return x===null||x===undefined?'Missing':Number(x).toFixed(3)}
 function render(){let e=EXAMPLES[Number(caseSelect.value)],t=e.target,k=indexSelect.value,f=t.common_support_indices[k],org=t.organ_summary.nd_of_separate_visible_medians[k];['rgb','noir','registered'].forEach(id=>document.getElementById(id).src=e[id]);document.getElementById('map').src=e.maps[k];document.getElementById('maplabel').textContent=k.replace('__',' − ')+' (processed DN)';document.getElementById('stats').innerHTML='<div>Separate visible medians<b>'+fmt(org)+'</b>ND of camera-local channel medians</div><div>Affine pixel hypothesis<b>'+fmt(f.pixel_median)+'</b>'+f.n+' valid common-support pixels</div><div>Geometric support<b>'+fmt(t.leaf_iou)+'</b>leaf-mask IoU; not anatomical accuracy</div>';document.getElementById('identity').textContent=t.session+' | view '+t.view+' | RGB candidate '+t.rgb_detection_index+' / NoIR candidate '+t.noir_detection_index+' | persistent ±2 px core: '+t.persistent_core_2px_pixels+' pixels. This ID is local to one view.';document.getElementById('provenance').textContent=JSON.stringify(t,null,2)}
 caseSelect.addEventListener('change',render);indexSelect.addEventListener('change',render);render();document.getElementById('download').onclick=()=>{let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([TEXT],{type:'text/plain;charset=utf-8'}));a.download='CSI_maize_shared_target_investigation.txt';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
 </script></body></html>"""
    )
    report = report.replace(
        ".specimens img{width:100%;background:#000}",
        ".specimens img{width:100%;background:#000;aspect-ratio:4/3;object-fit:contain}",
    )
    report = (
        report.replace("EXAMPLES_JSON", payload)
        .replace("ANALYSIS_JSON", a_json)
        .replace("TEXT_JSON", json.dumps(text, ensure_ascii=False).replace("</", r"<\/"))
    )
    (DEST / "CSI_maize_shared_leaf_targets.html").write_text(report, encoding="utf-8", newline="\n")
    dump(
        DEST / "report_manifest.json",
        dict(
            report_sha256=sha(DEST / "CSI_maize_shared_leaf_targets.html"),
            report_code_sha256=sha(__file__),
            figures=[p.name for p in (DEST / "figures").glob("*.png")],
            examples=len(examples),
        ),
    )


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        analysis = json.loads((DEST / "analysis.json").read_text(encoding="utf-8"))
        targets = json.loads((DEST / "shared_targets.json").read_text(encoding="utf-8"))["records"]
    else:
        analysis, targets, frames = analyse()
    build_report(analysis, exemplars(targets))
    print("REPORT", DEST / "CSI_maize_shared_leaf_targets.html", flush=True)
