"""Retrieve visible shoot morphology and camera descriptors over full rotations.

Pixel units refer to the processing image, not calibrated physical dimensions.
Skeleton endpoints and segments are explicitly proxies, not validated leaf counts.
"""

from common import workspace_relative

from bootstrap import ROOT, OUT, read_csv, write_csv
import argparse, json, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)


def load_image(path, width=1024):
    with Image.open(path) as source:
        native = source.size
        source.draft("RGB", (width, width))
        source.load()
        image = source.convert("RGB")
        image.thumbnail((width, width))
    return np.array(image), native


def segment(image, camera):
    array = image.astype(float)
    h, w = array.shape[:2]
    r, g, b = np.moveaxis(array, -1, 0)
    roi = np.zeros((h, w), bool)
    roi[int(0.02 * h) :, int(0.05 * w) : int(0.95 * w)] = True
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if camera == "rgb":
        candidate = (
            roi
            & (hsv[:, :, 0] >= 10)
            & (hsv[:, :, 0] <= 95)
            & (hsv[:, :, 1] > 70)
            & (hsv[:, :, 2] > 30)
            & (g > b * 1.5)
        )
    elif camera == "noir":
        threshold, _ = cv2.threshold(
            b[roi].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # This acquisition has a blue background vignette reaching ~60 DN.
        # Otsu alone can select that vignette when a small shoot covers <1%.
        # A conservative 70-DN seed floor is acquisition-specific, not calibrated reflectance.
        candidate = roi & (b > max(70, threshold)) & (b - r > 4) & (b - g > 4)
    else:
        raise ValueError("Unknown camera")
    # Close subpixel fragmentation while retaining a record of visible geometry.
    joined = cv2.morphologyEx(
        candidate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    choices = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 20:
            continue
        ys, xs = np.where(labels == index)
        if np.any((ys > 0.6 * h) & (xs > 0.2 * w) & (xs < 0.8 * w)):
            choices.append((stats[index, cv2.CC_STAT_AREA], index))
    if not choices:
        raise ValueError("No lower-central foreground candidate")
    selected = labels == max(choices)[1]
    if selected.sum() > roi.sum() * 0.5:
        raise ValueError("Foreground occupies implausibly large fraction")
    return selected


def prune_skeleton(skeleton, min_branch_length=12):
    """Remove short terminal spurs; length is processing pixels, not an organ label.

    Retain unbranched paths and junction pixels. This utility is deliberately
    separate from production measurements until segmentation has been reviewed.
    """
    result = np.asarray(skeleton, dtype=bool).copy()
    for _ in range(3):
        degree = (
            ndi.convolve(result.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
            - result
        )
        remove = set()
        for y, x in zip(*np.where(result & (degree == 1))):
            path = [(int(y), int(x))]
            previous = None
            current = path[0]
            length = 0.0
            while True:
                cy, cx = current
                adjacent = [
                    (ny, nx)
                    for ny in range(max(0, cy - 1), min(result.shape[0], cy + 2))
                    for nx in range(max(0, cx - 1), min(result.shape[1], cx + 2))
                    if (ny, nx) != current and (ny, nx) != previous and result[ny, nx]
                ]
                if len(adjacent) != 1:
                    break
                nxt = adjacent[0]
                length += float(np.hypot(nxt[0] - cy, nxt[1] - cx))
                if degree[nxt] >= 3:
                    if length < min_branch_length:
                        remove.update(path)
                    break
                if length >= min_branch_length:
                    break
                path.append(nxt)
                previous, current = current, nxt
        if not remove:
            break
        for pixel in remove:
            result[pixel] = False
    return result


def measure_mask(image, mask, camera):
    if mask.sum() < 10:
        raise ValueError("Insufficient foreground pixels")
    ys, xs = np.where(mask)
    h, w = mask.shape
    pixels = image[mask].astype(float)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boundary = np.concatenate(contours)
    hull = cv2.convexHull(boundary)
    hull_area = float(cv2.contourArea(hull))
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    # Fill ratio uses continuous contour area to avoid a discretization >1.
    contour_area = sum(cv2.contourArea(c) for c in contours)
    skeleton = skeletonize(mask)
    neighbors = (
        ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
        - skeleton
    )
    tips = skeleton & (neighbors == 1)
    junctions = skeleton & (neighbors >= 3)
    tip_count = ndi.label(tips, np.ones((3, 3)))[1]
    junction_count = ndi.label(junctions, np.ones((3, 3)))[1]
    # Cardinal and diagonal undirected graph edges, counted once.
    length = float(
        np.sum(skeleton[:, :-1] & skeleton[:, 1:]) + np.sum(skeleton[:-1, :] & skeleton[1:, :])
    )
    length += float(
        np.sqrt(2)
        * (
            np.sum(skeleton[:-1, :-1] & skeleton[1:, 1:])
            + np.sum(skeleton[1:, :-1] & skeleton[:-1, 1:])
        )
    )
    covariance = np.cov(np.array([xs, ys]))
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
    eccentricity = float(np.sqrt(1 - eigenvalues[0] / max(eigenvalues[1], 1e-10)))
    values = {
        "silhouette_area_px": int(mask.sum()),
        "area_fraction": float(mask.mean()),
        "visible_height_px": int(ys.max() - ys.min() + 1),
        "silhouette_width_px": int(xs.max() - xs.min() + 1),
        "height_fraction": float((ys.max() - ys.min() + 1) / h),
        "width_fraction": float((xs.max() - xs.min() + 1) / w),
        "perimeter_px": float(perimeter),
        "convex_hull_area_px": hull_area,
        "solidity": float(contour_area / max(hull_area, 1)),
        "eccentricity": eccentricity,
        "skeleton_length_px": length,
        "skeleton_endpoint_proxy": tip_count,
        "skeleton_junction_proxy": junction_count,
        "centroid_x_fraction": float(xs.mean() / w),
        "centroid_y_fraction": float(ys.mean() / h),
        "bottom_censored": int(ys.max() >= h - 2),
        "top_censored": int(ys.min() <= 1),
        "side_censored": int(xs.min() <= 1 or xs.max() >= w - 2),
    }
    chrom = pixels / np.maximum(pixels.sum(axis=1, keepdims=True), 1)
    for i, channel in enumerate("rgb"):
        values[channel + "_median"] = float(np.median(pixels[:, i]))
        values[channel + "_iqr"] = float(
            np.percentile(pixels[:, i], 75) - np.percentile(pixels[:, i], 25)
        )
        values[channel + "_chromaticity_median"] = float(np.median(chrom[:, i]))
    r, g, b = chrom.T
    values["normalized_excess_green"] = float(np.median(2 * g - r - b))
    values["green_red_normalized_difference"] = float(np.median((g - r) / np.maximum(g + r, 0.01)))
    values["blue_red_normalized_difference"] = float(np.median((b - r) / np.maximum(b + r, 0.01)))
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[mask]
    values["hue_median"] = float(np.median(hsv[:, 0]) * 2)
    values["saturation_median"] = float(np.median(hsv[:, 1]) / 255)
    values["yellow_hue_fraction"] = float(np.mean((hsv[:, 0] < 30) & (hsv[:, 0] >= 10)))
    values["saturation_clip_fraction"] = float(np.mean(np.max(pixels, axis=1) >= 250))
    # Median widths in vertical shoot strata provide interpretable architecture.
    for i in range(5):
        band = mask[
            int(ys.min() + (ys.max() - ys.min() + 1) * i / 5) : int(
                ys.min() + (ys.max() - ys.min() + 1) * (i + 1) / 5
            )
        ]
        widths = []
        for line in band:
            indices = np.flatnonzero(line)
            if len(indices):
                widths.append(indices[-1] - indices[0] + 1)
        values[f"stratum_{i + 1}_width_fraction"] = float(np.median(widths) / w) if widths else 0.0
    return values, skeleton


def annotate(image, mask, skeleton, values, title):
    draw = image.copy()
    draw[~mask] = (draw[~mask] * 0.18).astype(np.uint8)
    draw[skeleton] = [255, 80, 210]
    canvas = Image.fromarray(draw)
    p = ImageDraw.Draw(canvas)
    ys, xs = np.where(mask)
    p.rectangle(
        (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), outline="#e7b46b", width=2
    )
    p.rectangle((0, 0, canvas.width, 91), fill="#172a23")
    for i, line in enumerate(
        [
            title,
            f"Visible extent {values['visible_height_px']} px | area {values['silhouette_area_px']} px²",
            f"Skeleton {values['skeleton_length_px']:.1f} px | endpoint proxy {values['skeleton_endpoint_proxy']}",
            f"Bottom truncated: {'yes' if values['bottom_censored'] else 'no'} | processing pixels; no metric scale",
        ]
    ):
        p.text((12, 8 + i * 20), line, fill="white")
    return canvas


def process(task):
    session, camera, path, record = task
    try:
        image, native = load_image(path)
        mask = segment(image, camera)
        values, skeleton = measure_mask(image, mask, camera)
        idx = int(path.stem.rsplit("_", 1)[1])
        stem = session.name + "__" + camera + "__" + str(idx).zfill(3)
        Image.fromarray((mask * 255).astype(np.uint8)).save(OUT / "masks" / (stem + ".png"))
        if idx % 5 == 0:
            ys, xs = np.where(mask)
            pad = 20
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(image.shape[1], int(xs.max()) + pad + 1)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(image.shape[0], int(ys.max()) + pad + 1)
            crop = image[y0:y1, x0:x1].copy()
            cropmask = mask[y0:y1, x0:x1]
            crop[~cropmask] = [124, 116, 104]
            Image.fromarray(crop).save(OUT / "crops" / (stem + ".png"))
        if idx == 0:
            annotate(image, mask, skeleton, values, session.name + " | " + camera).save(
                OUT / "annotations" / (stem + ".jpg"), quality=90
            )
        return {
            "session": record["session"],
            **{k: record[k] for k in ["treatment", "measurement", "genotype", "replicate"]},
            "camera": camera,
            "view": idx,
            "path": workspace_relative(path),
            "processing_width": image.shape[1],
            "processing_height": image.shape[0],
            "native_width": native[0],
            "native_height": native[1],
            **values,
        }, None
    except Exception as error:
        return None, {
            "session": record["session"],
            "camera": camera,
            "path": str(path),
            "error": str(error),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    for folder in ["masks", "crops", "annotations"]:
        (OUT / folder).mkdir(exist_ok=True)
    sessions = [r for r in read_csv(OUT / "sessions.csv") if r["eligible"] == "True"]
    if args.limit:
        sessions = sessions[: args.limit]
    rows = []
    errors = []
    aggregates = []
    for number, record in enumerate(sessions, 1):
        session = ROOT / record["session"]
        tasks = [
            (session, cam, path, record)
            for cam in ["rgb", "noir"]
            for path in sorted((session / cam).glob("*.jpg"))
        ]
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(process, tasks))
        good = [r for r, e in outputs if r]
        bad = [e for r, e in outputs if e]
        rows.extend(good)
        errors.extend(bad)
        entry = {
            k: record[k] for k in ["session", "treatment", "measurement", "genotype", "replicate"]
        }
        entry["valid_views"] = len(good)
        for camera in ["rgb", "noir"]:
            camera_rows = [r for r in good if r["camera"] == camera]
            if not camera_rows:
                continue
            numeric = [
                k
                for k in camera_rows[0]
                if k
                not in [
                    "session",
                    "treatment",
                    "measurement",
                    "genotype",
                    "replicate",
                    "camera",
                    "view",
                    "path",
                    "processing_width",
                    "processing_height",
                    "native_width",
                    "native_height",
                ]
            ]
            for key in numeric:
                vals = np.array([r[key] for r in camera_rows], float)
                for stat, value in [
                    ("median", np.median(vals)),
                    ("max", vals.max()),
                    ("iqr", np.percentile(vals, 75) - np.percentile(vals, 25)),
                ]:
                    entry[camera + "__" + key + "__" + stat] = float(value)
        if len(good) == len(tasks):
            aggregates.append(entry)
        print(
            f"SESSION {number}/{len(sessions)} {session.name}: {len(good)}/{len(tasks)} views",
            flush=True,
        )
    write_csv(OUT / "view_traits.csv", rows)
    write_csv(OUT / "session_traits.csv", aggregates)
    write_csv(OUT / "trait_errors.csv", errors, ["session", "camera", "path", "error"])
    summary = {
        "processed_sessions": len(sessions),
        "complete_sessions": len(aggregates),
        "measured_views": len(rows),
        "failed_views": len(errors),
        "views_per_complete_session": 80,
        "processing_max_dimension": 1024,
        "trait_definitions": "Visible pixel geometry, camera colour descriptors and unvalidated skeleton proxies; no calibrated physical height, biochemical chlorophyll or validated organ counts",
        "bottom_censored_view_count": sum(r["bottom_censored"] for r in rows),
    }
    (OUT / "trait_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
