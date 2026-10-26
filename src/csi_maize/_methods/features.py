"""Deliberately simple, uncalibrated foreground features for a feasibility run."""

from common import workspace_relative

import json
import numpy as np
from PIL import Image, ImageDraw
from common import ROOT, OUT, config, build_index, write_csv


def plant_component(mask):
    """Keep the largest connected candidate reaching the lower central image.

    This encodes the current upright-plant acquisition geometry, not a validated
    general segmentation model. Background patches in the upper image are omitted.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask)
    candidates = []
    for flat in np.flatnonzero(mask):
        y, x = divmod(int(flat), w)
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        pixels = []
        anchored = False
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            if cy >= 0.6 * h and 0.25 * w <= cx <= 0.75 * w:
                anchored = True
            for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if anchored:
            candidates.append(pixels)
    selected = np.zeros_like(mask)
    if candidates:
        pixels = max(candidates, key=len)
        yy, xx = zip(*pixels)
        selected[yy, xx] = True
    return selected


def select_views(paths, count):
    paths = sorted(paths)
    if len(paths) < count:
        raise ValueError("Fewer source images than requested views")
    return [paths[i] for i in np.floor(np.arange(count) * len(paths) / count).astype(int)]


def otsu(values):
    hist = np.bincount(values.astype(np.uint8).ravel(), minlength=256).astype(float)
    weights = np.cumsum(hist)
    means = np.cumsum(hist * np.arange(256))
    total = weights[-1]
    score = (means[-1] * weights - means * total) ** 2 / np.maximum(weights * (total - weights), 1)
    score[(weights == 0) | (weights == total)] = 0
    return int(np.argmax(score))


def measure(path, camera, width):
    with Image.open(path) as original:
        original.load()  # Full decode catches truncated image streams.
        im = original.convert("RGB")
        im.thumbnail((width, width))
    a = np.asarray(im, dtype=float)
    h, w = a.shape[:2]
    r, g, b = np.moveaxis(a, -1, 0)
    roi = np.zeros((h, w), bool)
    roi[int(0.03 * h) : int(0.98 * h), int(0.08 * w) : int(0.92 * w)] = True
    if camera == "rgb":
        mask = roi & (g > 1.08 * r) & (g > 1.08 * b) & ((2 * g - r - b) > 12)
    else:
        # Camera-specific intensity segmentation, never transferred from RGB.
        threshold = max(otsu(b[roi]), 12)
        mask = roi & (b > threshold) & ((b - r) > 4) & ((b - g) > 4)
    mask = plant_component(mask)
    count = int(mask.sum())
    if count < 30 or count > roi.sum() * 0.6:
        raise ValueError("Foreground heuristic failed: " + str(path))
    pixels = a[mask]
    yy, xx = np.where(mask)
    med = np.median(pixels, axis=0)
    denom = max(float(med.sum()), 1)
    values = {
        "area_fraction": float(mask.mean()),
        "height_fraction": float((yy.max() - yy.min() + 1) / h),
        "width_fraction": float((xx.max() - xx.min() + 1) / w),
        "r_median": float(med[0]),
        "g_median": float(med[1]),
        "b_median": float(med[2]),
        "r_chromaticity": float(med[0] / denom),
        "g_chromaticity": float(med[1] / denom),
        "b_chromaticity": float(med[2] / denom),
        "saturated_fraction": float((pixels.max(axis=1) >= 250).mean()),
    }
    preview = a.astype(np.uint8).copy()
    preview[~mask] = (preview[~mask] * 0.2).astype(np.uint8)
    return values, Image.fromarray(preview), Image.fromarray((mask * 255).astype(np.uint8))


def extract():
    cfg = config()
    records = build_index()
    results = []
    errors = []
    viewrows = []
    panels = []
    for record in records:
        if not record["eligible"]:
            continue
        row = dict(record)
        success = True
        for camera in ("rgb", "noir"):
            paths = select_views(
                (ROOT / record["session"] / camera).glob("*.jpg"), cfg["feature_views_per_camera"]
            )
            measurements = []
            for i, path in enumerate(paths):
                try:
                    values, preview, _ = measure(path, camera, cfg["thumbnail_width"])
                    measurements.append(values)
                    viewrows.append(
                        {
                            "session": record["session"],
                            "camera": camera,
                            "path": workspace_relative(path),
                            **values,
                        }
                    )
                    if i == 0 and len(panels) < 12:
                        panels.append((record["session"].split("/")[-1] + " " + camera, preview))
                except Exception as exc:
                    errors.append(
                        {
                            "session": record["session"],
                            "camera": camera,
                            "path": workspace_relative(path),
                            "error": str(exc),
                        }
                    )
                    success = False
            if len(measurements) == len(paths):
                for name in measurements[0]:
                    values = np.array([m[name] for m in measurements])
                    row[camera + "__" + name + "__mean"] = float(values.mean())
                    row[camera + "__" + name + "__view_sd"] = float(values.std())
        if success:
            results.append(row)
    write_csv(OUT / "features.csv", results, list(results[0]) if results else ["session"])
    write_csv(OUT / "feature_errors.csv", errors, ["session", "camera", "path", "error"])
    write_csv(OUT / "view_features.csv", viewrows)
    if panels:
        canvas = Image.new("RGB", (1000, 270 * ((len(panels) + 1) // 2)), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (title, im) in enumerate(panels):
            im.thumbnail((490, 240))
            x = (i % 2) * 500
            y = (i // 2) * 270
            canvas.paste(im, (x, y + 25))
            draw.text((x + 5, y + 5), title, fill="black")
        canvas.save(OUT / "foreground_preview.jpg")
    summary = {
        "stage": "prototype_unvalidated_masks",
        "eligible_sessions": sum(r["eligible"] for r in records),
        "feature_sessions": len(results),
        "failed_views": len(errors),
        "views_per_camera": cfg["feature_views_per_camera"],
    }
    (OUT / "feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
