"""Whole-plant RGB/NoIR overlap and guarded spatial camera-DN features.

Uses frozen transforms and matches; does not fit registration to index values.
Run in the declared workspace using the installed method environment.
"""

from common import workspace_relative

from bootstrap import BASE, ROOT, OUT, read_csv, write_csv, validate_rotation_records
from run_channel_fusion import DEST as FUSION, OLD, raw, load_prediction
from channel_fusion_core import np, cv2, interior, warp_to_rgb
from multimodal_leaf_core import sha, dump
from spatial_fusion_core import (
    index_definitions,
    spatial_indices,
    index_statistics,
    overlap_labels,
    blend_projections,
)
from pathlib import Path
from PIL import Image
import base64, collections, io, json, time, html
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEST = OUT / "spatial_fusion/2026-10-26"
DEFS = index_definitions()
FIXED_VIEWS = (0, 10, 20, 30)
REPORT_NAME = "CSI_maize_RGB_NoIR_spatial_atlas.html"
COLORS = np.array([[0, 0, 0], [219, 77, 181], [43, 196, 217], [235, 235, 235]], np.uint8)
REFERENCES = [
    dict(
        title="Bendig et al. (2015). Combining UAV-based plant height, visible and near infrared vegetation indices for biomass monitoring in barley.",
        url="https://doi.org/10.1016/j.jag.2015.02.012",
        role="MGRVI and RGBVI formula provenance; no transfer of the original biomass validation to these maize images.",
    ),
    dict(
        title="Gitelson et al. (2002). Novel algorithms for remote estimation of vegetation fraction.",
        url="https://doi.org/10.1016/S0034-4257(01)00289-9",
        role="VARI formula provenance; applied here to JPEG DN rather than calibrated reflectance.",
    ),
    dict(
        title="RGB-based indices for estimating cover crop biomass, nitrogen content, and carbon:nitrogen ratio.",
        url="https://doi.org/10.1002/agj2.21657",
        role="Definitions of normalized ExG, ExGR and GLI; original study-specific validation does not validate the current extension.",
    ),
    dict(
        title="Raspberry Pi camera documentation.",
        url="https://www.raspberrypi.com/documentation/accessories/camera.html",
        role="NoIR hardware has no infrared blocking filter; the actual longpass response and channel calibration remain unmeasured here.",
    ),
]


def png_data(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def bounds(mask, pad=9):
    y, x = np.where(mask)
    if not len(x):
        return (0, 0, mask.shape[1], mask.shape[0])
    return (
        max(0, int(x.min()) - pad),
        max(0, int(y.min()) - pad),
        min(mask.shape[1], int(x.max()) + pad + 1),
        min(mask.shape[0], int(y.max()) + pad + 1),
    )


def cut(a, box):
    return a[box[1] : box[3], box[0] : box[2]]


def map_image(values, definition):
    valid = np.isfinite(values)
    out = np.zeros((*values.shape, 3), np.uint8)
    if valid.any():
        v = (values[valid] - definition["vmin"]) / (definition["vmax"] - definition["vmin"])
        out[valid] = (plt.get_cmap("coolwarm")(np.clip(v, 0, 1))[:, :3] * 255).astype(np.uint8)
    return out


def prepare_frame(frame, traits, predictions, candidates):
    session, view = frame["session"], frame["view"]
    records = {c: traits[session, c, view] for c in ("rgb", "noir")}
    loaded = {c: raw(records[c]) for c in ("rgb", "noir")}
    a, am, _ = loaded["rgb"]
    b, bm, _ = loaded["noir"]
    for c in ("rgb", "noir"):
        assert records[c]["source"] == frame[c + "_source"]
        assert sha(ROOT / records[c]["mask"]) == frame[c + "_mask_sha256"]
    matrix = np.asarray(frame["affine_source_to_target"], np.float32)
    wb, field = warp_to_rgb(b.astype(np.float32), matrix, am.shape)
    wbm = warp_to_rgb(bm.astype(np.uint8), matrix, am.shape, True)[0] > 0
    wbi = warp_to_rgb(interior(bm).astype(np.uint8), matrix, am.shape, True)[0] > 0
    ari = interior(am)
    valid_b = ((b > 1) & (b < 254)).astype(np.float32)
    valid_b = warp_to_rgb(valid_b, matrix, am.shape)[0] > 0.999999
    channel_valid = np.concatenate([(a > 1) & (a < 254), valid_b & field[..., None]], axis=2)
    cross = np.zeros(am.shape, bool)
    ids = []
    rp = {int(r["detection_index"]): m for r, m in predictions["rgb"][view]}
    npr = {int(r["detection_index"]): m for r, m in predictions["noir"][view]}
    for t in candidates:
        assert (t["session"], t["view"]) == (session, view)
        assert all(
            t[c + "_source"] == records[c]["source"]
            and t[c + "_source_sha256"] == records[c]["source_sha256"]
            for c in ("rgb", "noir")
        )
        if not t["pixel_hypothesis"]:
            continue
        np.testing.assert_allclose(matrix, t["registration_source_to_target"], atol=0, rtol=0)
        ri = interior(rp[t["rgb_detection_index"]] & am)
        ni = interior(npr[t["noir_detection_index"]] & bm)
        wni = warp_to_rgb(ni.astype(np.uint8), matrix, am.shape, True)[0] > 0
        common = ri & wni & field
        assert int(common.sum()) == t["common_interior_pixels"]
        cross |= common
        ids.append(t["target_id"])
    box = bounds(am | (wbm & field))
    six = cut(np.concatenate([a.astype(np.float32), wb], axis=2), box)
    supports = dict(rgb=cut(ari, box), noir=cut(wbi & field, box), cross=cut(cross, box))
    cv = cut(channel_valid, box)
    maps = spatial_indices(six, supports, cv)
    rm, nm = cut(am, box), cut(wbm & field, box)
    union = rm | nm
    overlap = int((rm & nm).sum())
    unionsum = int(union.sum())
    joint = supports["cross"] & cv.all(axis=2) & np.all((six > 1) & (six < 254), axis=2)
    # A six-channel availability diagnostic also requires the sum guards of all ND pairs.
    joint &= np.min(six, axis=2) + np.partition(six, 1, axis=2)[..., 1] >= 12
    meta = dict(
        session=session,
        view=view,
        nominal_degrees=view * 9,
        phase_best_shift=frame["phase_best_shift"],
        registration_status=frame["status"],
        frozen_whole_iou=frame["whole_mask_iou"],
        rgb_grid_iou=overlap / unionsum if unionsum else None,
        rgb_pixels=int(rm.sum()),
        noir_registered_pixels=int(nm.sum()),
        overlap_pixels=overlap,
        cross_support_pixels=int(cross.sum()),
        accepted_leaf_candidates=len(ids),
        joint_six_valid_pixels=int(joint.sum()),
        target_ids=ids,
        rgb_source=records["rgb"]["source"],
        noir_source=records["noir"]["source"],
        rgb_source_sha256=records["rgb"]["source_sha256"],
        noir_source_sha256=records["noir"]["source_sha256"],
        rgb_mask_sha256=frame["rgb_mask_sha256"],
        noir_mask_sha256=frame["noir_mask_sha256"],
        affine_rgb_to_noir=matrix.tolist(),
        crop_RGB_xyxy=list(box),
        processing_shape_HW=list(am.shape),
        rgb_EXIF=frame["rgb_EXIF"],
        noir_EXIF=frame["noir_EXIF"],
        rgb_qc_issues=records["rgb"]["qc_issues"],
        noir_qc_issues=records["noir"]["qc_issues"],
    )
    stats = []
    for d in DEFS:
        row = index_statistics(maps[d["id"]], supports[d["support"]])
        values = maps[d["id"]]
        row.update(
            session=session,
            view=view,
            index=d["id"],
            support=d["support"],
            display_below=int(np.sum(np.isfinite(values) & (values < d["vmin"]))),
            display_above=int(np.sum(np.isfinite(values) & (values > d["vmax"]))),
        )
        stats.append(row)
    native = b.copy()
    native[~bm] = 0
    native = cut(native, bounds(bm))
    return dict(
        meta=meta,
        six=six,
        supports=supports,
        channel_valid=cv,
        maps=maps,
        stats=stats,
        rgb_mask=rm,
        noir_registered_mask=nm,
        noir_native=native,
    )


def save_specimen(item, selection):
    meta = item["meta"]
    name = Path(meta["session"]).name + f"__v{meta['view']:03}"
    meta["selection"] = selection
    folder = DEST / "specimens" / name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "spatial_arrays.npz",
        six_channels=item["six"],
        channel_valid=item["channel_valid"],
        rgb_support=item["supports"]["rgb"],
        noir_support=item["supports"]["noir"],
        cross_support=item["supports"]["cross"],
        rgb_mask=item["rgb_mask"],
        noir_registered_mask=item["noir_registered_mask"],
        crop_RGB_xyxy=np.array(meta["crop_RGB_xyxy"]),
        affine_rgb_to_noir=np.asarray(meta["affine_rgb_to_noir"], np.float32),
    )
    a = item["six"][..., :3].astype(np.uint8)
    b = np.clip(item["six"][..., 3:], 0, 255).astype(np.uint8)
    a[~item["rgb_mask"]] = 0
    b[~item["noir_registered_mask"]] = 0
    agreement = COLORS[overlap_labels(item["rgb_mask"], item["noir_registered_mask"])]
    support_image = np.zeros_like(a)
    support_image[item["rgb_mask"] | item["noir_registered_mask"]] = [35, 35, 35]
    support_image[item["supports"]["cross"]] = [233, 188, 73]
    images = dict(
        rgb=a,
        noir_native=item["noir_native"],
        noir_registered=b,
        overlap=agreement,
        support=support_image,
    )
    for key, image in images.items():
        Image.fromarray(image).save(folder / (key + ".png"))
    map_data = {}
    for d in DEFS:
        image = map_image(item["maps"][d["id"]], d)
        Image.fromarray(image).save(folder / (d["id"] + ".png"))
        map_data[d["id"]] = png_data(image)
    dump(folder / "provenance.json", dict(meta=meta, stats=item["stats"]))
    return dict(
        id=name,
        meta=meta,
        stats={r["index"]: r for r in item["stats"]},
        images={k: png_data(v) for k, v in images.items()},
        maps=map_data,
    )


def check_inputs():
    config = json.loads((FUSION / "run_configuration.json").read_text(encoding="utf-8"))
    for path, digest in config["input_sha256"].items():
        assert sha(ROOT / path) == digest, path
    binding = json.loads((FUSION / "dependency_binding.json").read_text(encoding="utf-8"))
    for path, record in binding["bindings"].items():
        assert sha(ROOT / path) == record["sha256"], path
    paths = [
        FUSION / "frame_registration.json",
        FUSION / "shared_targets.json",
        OUT / "sam_traits/view_traits.csv",
        OLD / "paired_frames.csv",
        FUSION / "run_configuration.json",
        FUSION / "dependency_binding.json",
        BASE / "spatial_fusion_core.py",
        BASE / "run_spatial_fusion.py",
        BASE / "spatial_fusion_viewer.html",
    ]
    return {workspace_relative(p): sha(p) for p in paths}


def median(values):
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


def figure_outputs(examples, frames, index_rows, acquisition_rows):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(fig, name):
        fig.savefig(DEST / "figures" / (name + ".png"), dpi=300, bbox_inches="tight")
        fig.savefig(DEST / "figures" / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    # Deterministic highest accepted-support acquisition among those with a joint-six eligible frame.
    eligible = [e for e in examples if e["meta"]["joint_six_valid_pixels"] >= 40] or examples
    exemplar = max(
        eligible,
        key=lambda e: (e["meta"]["joint_six_valid_pixels"], e["meta"]["cross_support_pixels"]),
    )
    session = exemplar["meta"]["session"]
    fixed = [
        e for e in examples if e["meta"]["session"] == session and e["meta"]["view"] in FIXED_VIEWS
    ]
    fig, axs = plt.subplots(4, 4, figsize=(11, 12), layout="constrained")
    for row, e in enumerate(fixed):
        folder = DEST / "specimens" / e["id"]
        with np.load(folder / "spatial_arrays.npz") as z:
            blend = blend_projections(
                z["six_channels"][..., :3],
                z["six_channels"][..., 3:],
                z["rgb_mask"],
                z["noir_registered_mask"],
            )
        for col, key in enumerate(["rgb", "noir_registered", "blend", "overlap"]):
            ax = axs[row, col]
            ax.imshow(blend if key == "blend" else Image.open(folder / (key + ".png")))
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    ["RGB", "Registered NoIR", "50:50 projection blend", "Silhouette agreement"][
                        col
                    ]
                )
        axs[row, 0].set_ylabel(
            f"Index {e['meta']['view']}\nnominal {e['meta']['nominal_degrees']}°"
        )
    fig.suptitle(
        Path(session).name
        + "\nFour fixed nominal indices; magenta RGB only, cyan NoIR only, white overlap",
        fontsize=12,
    )
    save(fig, "01_same_plant_overlap")
    folder = DEST / "specimens" / exemplar["id"]
    fig, axs = plt.subplots(5, 3, figsize=(10, 13), layout="constrained")
    for ax, d in zip(axs.flat, DEFS[:15]):
        ax.imshow(Image.open(folder / (d["id"] + ".png")))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            d["id"].replace("ND_", "").replace("__", " − ")
            + f"\nn={exemplar['stats'][d['id']]['valid_pixels']}",
            fontsize=8,
        )
    fig.suptitle(
        exemplar["id"]
        + "\nAll 15 pairwise ND maps: blue −1, neutral 0, red +1; black missing\nAvailability-selected example, not a typical biological sample",
        fontsize=11,
    )
    save(fig, "02_fifteen_channel_pair_maps")
    fig, axs = plt.subplots(3, 4, figsize=(12, 10), layout="constrained")
    for ax, d in zip(axs.flat, DEFS[15:]):
        ax.imshow(Image.open(folder / (d["id"] + ".png")))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            d["id"]
            + f"  [{d['vmin']}, {d['vmax']}]\nn={exemplar['stats'][d['id']]['valid_pixels']}",
            fontsize=9,
        )
    fig.suptitle(
        exemplar["id"]
        + "\nSix RGB formulas and six exploratory multichannel descriptors\nPanel-specific fixed scales; black missing. No calibrated physiological interpretation.",
        fontsize=11,
    )
    save(fig, "03_rgb_and_multichannel_maps")
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 8.5), layout="constrained")
    panels = [
        ("rgb", "A  Segmented RGB"),
        ("noir_registered", "B  NoIR on RGB grid"),
        ("overlap", "C  Silhouette agreement"),
        ("support", "D  Accepted common interior"),
        ("ND_rgb_G__noir_B", "E  Two-channel ND"),
        ("PRODUCT4", "F  Four-channel product balance"),
    ]
    for ax, (key, label) in zip(axs.flat, panels):
        ax.imshow(Image.open(folder / (key + ".png")))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, loc="left", fontsize=10)
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-1, 1), cmap="coolwarm"),
        ax=list(axs[1, 1:]),
        orientation="horizontal",
        fraction=0.06,
        pad=0.02,
        label="Camera-DN descriptor (E,F); black missing",
        ticks=[-1, 0, 1],
    )
    fig.suptitle(
        exemplar["id"] + "\nNominal camera pair; availability-selected example", fontsize=11
    )
    save(fig, "05_observation_extension")
    sessions = sorted({f["session"] for f in frames})
    matrix = np.array(
        [
            [
                next(
                    r["qualified_views"]
                    for r in acquisition_rows
                    if r["session"] == s and r["index"] == d["id"]
                )
                for d in DEFS
            ]
            for s in sessions
        ]
    )
    fig, ax = plt.subplots(figsize=(13, 12), layout="constrained")
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=40, aspect="auto")
    ax.set_yticks(
        range(len(sessions)),
        [Path(s).name.replace("capture_session_", "") for s in sessions],
        fontsize=8,
    )
    ax.set_xticks(
        range(len(DEFS)),
        [
            d["id"]
            .replace("ND_", "")
            .replace("rgb_", "R:")
            .replace("noir_", "N:")
            .replace("__", "/")
            for d in DEFS
        ],
        rotation=90,
        fontsize=7,
    )
    ax.axvline(14.5, color="white", lw=1)
    ax.axvline(20.5, color="white", lw=1)
    fig.colorbar(im, ax=ax, label="Views with ≥40 valid support pixels, out of 40", shrink=0.5)
    ax.set_title(
        f"Feature availability is part of the result\n{len(sessions)} acquisitions × {len(DEFS)} definitions; repeated views are technical observations"
    )
    save(fig, "04_feature_availability")
    return exemplar["id"]


def investigation(summary, acquisition_rows):
    weak = [r for r in summary["index_summary"] if "noir_G" in r["id"]]
    text = f"""RGB–NoIR projection overlap and spatial camera-DN features
Vlatko Galic and Maja Mazur
Agricultural Institute Osijek, Južno predgrađe 17, 31000 Osijek, Croatia

Materials and methods

We extended the circular-imaging workflow with registered projections and spatial descriptors of six recorded channels: RGB red, green and blue, and the three corresponding processed outputs from the longpass NoIR camera. We paired images by acquisition identifier and filename index. We audited {summary["pairs"]} nominal pairs from {summary["acquisitions"]} acquisitions, retaining all 40 indices per acquisition. These are repeated angular observations, not independent biological replicates. Filename pairing does not establish synchronized exposure, measured rotation or corresponding tissue.

We retained the previously estimated affine RGB-to-NoIR transforms and sampled NoIR onto the RGB reference grid using the inverse affine transform. All calculations used the frozen 512 × 385 processing grid. We used bilinear sampling for NoIR channel values and nearest-neighbour sampling for masks, excluding samples without complete interpolation support inside the source frame. We did not re-estimate transforms to improve the appearance or distribution of index values. For each acquisition, we retained its existing phase audit; {summary["same_index_best_phase_acquisitions"]} of {summary["acquisitions"]} acquisitions favoured same-index alignment in that diagnostic. We did not silently shift the camera pairing.

We displayed segmented RGB, original NoIR, registered NoIR, a variable-opacity projection blend and a silhouette agreement image. The agreement image distinguishes RGB-only pixels, NoIR-only pixels and their intersection. Whole-plant overlap is distinct from the more restrictive support used for cross-camera calculations. We computed RGB-only descriptors inside the eroded RGB plant mask. We computed NoIR-only descriptors from the interpolated NoIR channels inside the sampled eroded NoIR mask. This defines a common-grid measurement: calculating a nonlinear ratio after interpolation is not equivalent to interpolating a native-camera ratio. For cross-camera descriptors we used the union of accepted candidate leaf common interiors under the frozen correspondence rules. This union does not establish independently verified anatomical correspondence.

We excluded channel values ≤1 or ≥254 DN, applied the frozen minimum sum of 12 DN to two-channel normalized differences, and retained separate validity masks for every definition. Before affine sampling, we marked NoIR processing-grid channels for the same dark and clipping exclusions, then required all nonzero-weight interpolation contributors to pass. Thus interpolation cannot rescue a dark channel. These guards do not reverse JPEG processing or establish sensor noise or detection limits. We did not impute missing values or require all six channels for a descriptor that uses fewer channels.

We evaluated 27 definitions: all 15 unordered channel pairs with signed difference (Cᵢ−Cⱼ)/(Cᵢ+Cⱼ); six RGB formulas (normalized ExG, normalized ExGR, GLI, VARI, MGRVI and RGBVI); and six exploratory multichannel descriptors. The latter comprise camera-sum contrast, green-share contrast, chromatic composition distance, geometric-mean camera contrast, a specified four-channel product balance and a paired GLI-formula contrast. We applied RGB literature formulas to processed DN and did not transfer validation from their source studies. We treated the multichannel descriptors as algebraic hypotheses, without assigning them physiological specificity. The complete formula catalogue and channel requirements accompany the supplementary methods. Where the validity rules remain satisfied, formula values depend only on relative channel values under common positive scaling; their joint algebraic information is therefore at most five-dimensional at a single positive six-channel vector. Separate per-feature masks and nonlinear summaries do not imply independent spectral bands.

For VARI we excluded |G+R−B|<12 DN. We retained numerical values beyond the plotted interval and recorded their display saturation counts. We used fixed scales for each definition across acquisitions, with individual ranges for normalized ExG and ExGR. Missing pixels are black; nonzero colours do not indicate statistical significance. We summarized medians, interquartile ranges and 10th–90th percentiles only when at least 40 pixels remained. These are descriptive spatial distributions, not confidence intervals. We first summarized angular views within each acquisition and reported acquisition coverage; we did not treat pixels or angular views as biological replication.

For visualization we retained indices 0, 10, 20 and 30 in every acquisition, corresponding to nominal quarter rotations under the indexing convention. Where different, we also included the view with greatest joint-six-channel support, provided at least one such pixel existed. This is an availability-selected diagnostic and cannot be interpreted as typical plant condition. We saved the six-channel floating-point samples, interpolation-validity masks, geometric supports and coordinate transforms for numerical reproduction.

Results

Across all {summary["pairs"]} pairs, {summary["frames_with_cross_support"]} frames contained accepted cross-camera candidate support. Joint-six-channel availability was markedly lower: {summary["frames_with_40_joint_six_pixels"]} frames from {summary["acquisitions_with_40_joint_six_pixels"]} acquisitions retained at least 40 pixels passing all channel and pair-sum guards. The portable atlas contains {summary["displayed_frames"]} displayed frames, while numerical coverage summaries retain every nominal pair. Missing NoIR-green signal restricts both pairwise and more complex descriptors; additional algebra does not restore absent information.

The overlap panels expose disagreement between camera silhouettes, whereas the support panels reveal which regions contributed to cross-camera features. We treat spatial contrast as a recorded-image pattern conditional on projection, segmentation, lighting, exposure, filter transmission and camera processing. We did not estimate chlorophyll concentration, calibrated reflectance, NDVI, fluorescence or leaf physiological performance from these maps.

Placement in the manuscript

The main Materials and methods should retain the image join, transform direction, interpolation, support rules, signal exclusions, formula families and replication policy. One compact figure should connect the two camera views, their overlap, accepted support and representative two-channel and multichannel maps. The complete formula table, 27-map atlases, all-acquisition coverage, phase diagnostics and machine-readable measurements belong in the supplement and data repository. These additions extend the observation layer of the architectural assay; the central question remains whether image-supported leaf representations preserve the endpoint of the virtual experiment.

References

"""
    text += "\n\n".join(r["title"] + "\n" + r["url"] + "\n" + r["role"] for r in REFERENCES)
    return text


def main():
    started = time.time()
    DEST.mkdir(parents=True, exist_ok=True)
    for folder in ("figures", "specimens"):
        (DEST / folder).mkdir(exist_ok=True)
    input_hashes = check_inputs()
    executed_source = DEST / "executed_source"
    executed_source.mkdir(exist_ok=True)
    for source in [
        BASE / "run_spatial_fusion.py",
        BASE / "spatial_fusion_core.py",
        BASE / "spatial_fusion_viewer.html",
    ]:
        (executed_source / source.name).write_bytes(source.read_bytes())
    frames = json.loads((FUSION / "frame_registration.json").read_text(encoding="utf-8"))["records"]
    targets = json.loads((FUSION / "shared_targets.json").read_text(encoding="utf-8"))["records"]
    traits_list = read_csv(OUT / "sam_traits/view_traits.csv")
    traits = {(r["session"], r["camera"], int(r["view"])): r for r in traits_list}
    pairs = read_csv(OLD / "paired_frames.csv")
    acquisitions = validate_rotation_records(traits_list)
    if (
        len(frames) != len(pairs)
        or len(traits) != len(traits_list)
        or len(traits_list) != 2 * len(pairs)
    ):
        raise ValueError("Spatial fusion requires unique paired camera/view records")
    if len({(f["session"], f["view"]) for f in frames}) != len(pairs):
        raise ValueError("Duplicate spatial frame identities")
    pair_lookup = {(r["session"], int(r["view"])): r for r in pairs}
    groups = collections.defaultdict(list)
    for t in targets:
        groups[t["session"], t["view"]].append(t)
    rows, frame_rows, examples = [], [], []
    sessions = sorted({f["session"] for f in frames})
    for number, session in enumerate(sessions, 1):
        predictions = {c: load_prediction(session, c) for c in ("rgb", "noir")}
        local_frames = sorted(
            [f for f in frames if f["session"] == session], key=lambda f: f["view"]
        )
        assert [f["view"] for f in local_frames] == list(range(40))
        best, fixed = None, {}
        for f in local_frames:
            p = pair_lookup[session, f["view"]]
            for c in ("rgb", "noir"):
                assert (
                    p[c + "_source"] == f[c + "_source"] == traits[session, c, f["view"]]["source"]
                )
                assert p[c + "_source_sha256"] == traits[session, c, f["view"]]["source_sha256"]
            item = prepare_frame(f, traits, predictions, groups[session, f["view"]])
            rows.extend(item["stats"])
            frame_rows.append(item["meta"].copy())
            if f["view"] in FIXED_VIEWS:
                fixed[f["view"]] = item
            if (
                best is None
                or item["meta"]["joint_six_valid_pixels"] > best["meta"]["joint_six_valid_pixels"]
            ):
                best = item
        for v, item in sorted(fixed.items()):
            examples.append(save_specimen(item, "fixed nominal index"))
        if best["meta"]["view"] not in FIXED_VIEWS and best["meta"]["joint_six_valid_pixels"] > 0:
            examples.append(
                save_specimen(best, "highest joint-six-channel support; availability-selected")
            )
        print(
            f"{number}/{len(sessions)} {Path(session).name}: best joint-six={best['meta']['joint_six_valid_pixels']}",
            flush=True,
        )
    assert len(rows) == len(pairs) * len(DEFS)
    acquisition_rows = []
    for session in sessions:
        for d in DEFS:
            rr = [r for r in rows if r["session"] == session and r["index"] == d["id"]]
            acquisition_rows.append(
                dict(
                    session=session,
                    index=d["id"],
                    qualified_views=sum(r["qualified"] for r in rr),
                    median_of_qualified_view_medians=median([r["median"] for r in rr]),
                    median_eligible_pixels=median([r["eligible_pixels"] for r in rr]),
                    median_valid_pixels=median([r["valid_pixels"] for r in rr]),
                )
            )
    summary = dict(
        status="completed_camera_DN_spatial_audit",
        pairs=len(frames),
        acquisitions=len(sessions),
        definitions=len(DEFS),
        displayed_frames=len(examples),
        frames_with_cross_support=sum(f["cross_support_pixels"] > 0 for f in frame_rows),
        frames_with_40_joint_six_pixels=sum(f["joint_six_valid_pixels"] >= 40 for f in frame_rows),
        acquisitions_with_40_joint_six_pixels=len(
            {f["session"] for f in frame_rows if f["joint_six_valid_pixels"] >= 40}
        ),
        same_index_best_phase_acquisitions=len(
            {f["session"] for f in frame_rows if f["phase_best_shift"] == 0}
        ),
        index_summary=[
            dict(
                id=d["id"],
                qualified_frames=sum(r["qualified"] for r in rows if r["index"] == d["id"]),
                qualified_acquisitions=sum(
                    r["qualified_views"] > 0 for r in acquisition_rows if r["index"] == d["id"]
                ),
                balanced_median=median(
                    [
                        r["median_of_qualified_view_medians"]
                        for r in acquisition_rows
                        if r["index"] == d["id"]
                    ]
                ),
            )
            for d in DEFS
        ],
        interpretation="Exploratory processed-DN maps under explicit support hypotheses; no calibrated physiology or anatomical identity.",
    )
    write_csv(DEST / "frame_index_statistics.csv", rows)
    write_csv(DEST / "acquisition_index_statistics.csv", acquisition_rows)
    dump(DEST / "frame_provenance.json", frame_rows)
    dump(DEST / "index_definitions.json", DEFS)
    exemplar = figure_outputs(examples, frame_rows, rows, acquisition_rows)
    summary["figure_exemplar"] = exemplar
    summary["elapsed_seconds"] = round(time.time() - started, 2)
    dump(DEST / "summary.json", summary)
    text = investigation(summary, acquisition_rows)
    (DEST / "INVESTIGATION_spatial_fusion.txt").write_text(text, encoding="utf-8")
    payload = dict(
        definitions=DEFS,
        examples=examples,
        summary=summary,
        frames=frame_rows,
        acquisition_rows=acquisition_rows,
        references=REFERENCES,
        default_example=next(i for i, e in enumerate(examples) if e["id"] == exemplar),
    )
    template = (BASE / "spatial_fusion_viewer.html").read_text(encoding="utf-8")
    report = template.replace(
        "__PAYLOAD__",
        json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/"),
    )
    report = report.replace("__INVESTIGATION__", html.escape(text))
    for name in (
        "01_same_plant_overlap",
        "02_fifteen_channel_pair_maps",
        "03_rgb_and_multichannel_maps",
        "04_feature_availability",
    ):
        report = report.replace(
            "__FIG_" + name + "__",
            "data:image/png;base64,"
            + base64.b64encode((DEST / "figures" / (name + ".png")).read_bytes()).decode(),
        )
    (DEST / REPORT_NAME).write_text(report, encoding="utf-8")
    dump(
        DEST / "run_manifest.json",
        dict(
            input_sha256=input_hashes,
            definitions=DEFS,
            image_hashes_checked=len(traits_list),
            mask_hashes_checked=len(traits_list),
            source_fusion_unchanged=True,
            elapsed_seconds=round(time.time() - started, 2),
            output_sha256={
                p.name: sha(p)
                for p in [
                    DEST / "summary.json",
                    DEST / "frame_index_statistics.csv",
                    DEST / REPORT_NAME,
                ]
            },
        ),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
