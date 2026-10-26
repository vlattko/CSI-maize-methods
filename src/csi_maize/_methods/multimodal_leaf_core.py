"""Small audited primitives for dual-camera leaf observations and label provenance."""

from bootstrap import BASE, ROOT, OUT
import sys, os, json, hashlib
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

DEST = OUT / "multimodal_leaf/2026-10-24"
os.environ["YOLO_CONFIG_DIR"] = str(DEST / "runtime_config")
os.environ["YOLO_AUTOINSTALL"] = "false"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )


def pair_records(rows):
    grouped = {}
    for r in rows:
        key = (r["session"], int(r["view"]))
        grouped.setdefault(key, {})
        if r["camera"] in grouped[key]:
            raise ValueError("Duplicate session/camera/view: " + str(key))
        grouped[key][r["camera"]] = r
    pairs = []
    missing = []
    for key, streams in sorted(grouped.items()):
        if "rgb" in streams and "noir" in streams:
            pairs.append(dict(session=key[0], view=key[1], **streams))
        else:
            missing.append(dict(session=key[0], view=key[1], present=list(streams)))
    return pairs, missing


def plant_group(r):
    if not r.get("genotype") or not r.get("replicate"):
        raise ValueError("Unknown plant labels cannot define a repeated-plant split")
    return "|".join(str(r[k]) for k in ["treatment", "genotype", "replicate"])


def split_group(group):
    n = int(hashlib.sha256(("CSI-maize-20261024|" + group).encode()).hexdigest()[:8], 16) % 10
    return "test" if n < 2 else "validation" if n < 4 else "train"


def require_reviewed(record):
    if (
        record.get("review_status") != "human_reviewed_complete"
        or not record.get("reviewer")
        or "objects" not in record
    ):
        raise ValueError(
            "Supervised export requires a named reviewer and complete corrected instance labels"
        )


def export_polygon(mask, min_fraction=0.9):
    mask = np.asarray(mask, np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Empty instance")
    contour = max(contours, key=cv2.contourArea)
    retained = np.zeros_like(mask)
    cv2.drawContours(retained, [contour], -1, 1, -1)
    if (mask & retained).sum() / mask.sum() < min_fraction:
        raise ValueError(
            "Disconnected visible pieces require adjudication; refusing an artificial bridge"
        )
    polygon = cv2.approxPolyDP(contour, 0.55, True).reshape(-1, 2)
    if len(polygon) < 3:
        raise ValueError("Degenerate polygon")
    return polygon.astype(float) / np.array([mask.shape[1], mask.shape[0]])


def assign_tracks(cost, gate):
    cost = np.asarray(cost, float)
    if not cost.size:
        return []
    # Dummy rows/columns permit leaving every real observation unmatched.
    n, m = cost.shape
    a = np.full((n + m, n + m), gate / 2)
    a[n:, m:] = 0
    a[:n, :m] = np.where(cost <= gate, cost, 1e6)
    rr, cc = linear_sum_assignment(a)
    return [(int(i), int(j)) for i, j in zip(rr, cc) if i < n and j < m and cost[i, j] <= gate]


def mask_iou(a, b):
    return float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))


def affine_align(source_mask, target_mask):
    """Image-based affine source→target; no camera-calibration interpretation."""
    a = source_mask.astype(np.uint8)
    b = target_mask.astype(np.uint8)

    def moments(x):
        y, z = np.where(x)
        return np.array([z.mean(), y.mean()]), np.array([max(z.std(), 1), max(y.std(), 1)])

    ca, sa = moments(a)
    cb, sb = moments(b)
    scale = np.clip(sb / sa, 0.6, 1.6)
    initial = np.array(
        [[scale[0], 0, cb[0] - ca[0] * scale[0]], [0, scale[1], cb[1] - ca[1] * scale[1]]],
        np.float32,
    )
    inv = cv2.invertAffineTransform(initial)
    aa = cv2.GaussianBlur(a.astype(np.float32), (9, 9), 2)
    bb = cv2.GaussianBlur(b.astype(np.float32), (9, 9), 2)
    status = "moment_initialisation"
    try:
        _, w = cv2.findTransformECC(
            bb,
            aa,
            inv,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 80, 1e-5),
            None,
            5,
        )
        forward = cv2.invertAffineTransform(w)
        test = cv2.warpAffine(a, forward, (b.shape[1], b.shape[0]), flags=cv2.INTER_NEAREST) > 0
        initial_warp = (
            cv2.warpAffine(a, initial, (b.shape[1], b.shape[0]), flags=cv2.INTER_NEAREST) > 0
        )
        if mask_iou(test, b) > mask_iou(initial_warp, b):
            initial = forward
            status = "affine_ECC"
    except cv2.error:
        pass
    warped = cv2.warpAffine(a, initial, (b.shape[1], b.shape[0]), flags=cv2.INTER_NEAREST) > 0
    return initial, mask_iou(warped, b > 0), status
