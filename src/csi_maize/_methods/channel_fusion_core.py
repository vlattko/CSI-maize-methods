"""Exploratory camera-DN features; no reflectance or anatomical identity assumptions."""

from multimodal_leaf_core import np, cv2, assign_tracks
from itertools import combinations

CHANNELS = ("rgb_R", "rgb_G", "rgb_B", "noir_R", "noir_G", "noir_B")
PAIRS = list(combinations(range(6), 2))
POLICY = dict(
    whole_iou_min=0.65,
    leaf_iou_min=0.50,
    assignment_margin_min=0.10,
    min_common_interior_pixels=40,
    min_interior_overlap_fraction=0.35,
    dark_dn_excluded_at_or_below=1,
    clipped_dn_excluded_at_or_above=254,
    erosion_kernel=3,
    min_pair_DN_sum=12,
    spatial_offsets_px=[-4, -2, 0, 2, 4],
    relative_noir_DN_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
)


def warp_to_rgb(noir, forward_rgb_to_noir, shape, nearest=False):
    """Sample NoIR in RGB coordinates. Border support is explicit, never black data."""
    forward = np.asarray(forward_rgb_to_noir, np.float32)
    inv = cv2.invertAffineTransform(forward)
    size = (shape[1], shape[0])
    flag = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    moved = cv2.warpAffine(noir, inv, size, flags=flag, borderMode=cv2.BORDER_CONSTANT)
    # Linear interpolation requires all contributors to be inside the original frame.
    support = cv2.warpAffine(
        np.ones(noir.shape[:2], np.float32), inv, size, flags=flag, borderMode=cv2.BORDER_CONSTANT
    )
    return moved, support >= 1 - 1e-7


def interior(mask):
    return (
        cv2.erode(
            np.asarray(mask, np.uint8),
            np.ones((3, 3), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )


def match_instances(ious, min_iou=0.5, margin=0.1):
    """Hungarian proposal with independent mutual-best/margin rejection; never colour."""
    ious = np.asarray(ious, float)
    if not ious.size:
        return []
    matches = assign_tracks(1 - ious, 1 - min_iou)
    rows = []
    for i, j in matches:
        row_alt = np.delete(ious[i], j)
        col_alt = np.delete(ious[:, j], i)
        rmargin = float(ious[i, j] - (row_alt.max() if row_alt.size else 0))
        cmargin = float(ious[i, j] - (col_alt.max() if col_alt.size else 0))
        rows.append(
            dict(
                i=i,
                j=j,
                iou=float(ious[i, j]),
                row_margin=rmargin,
                col_margin=cmargin,
                accepted=bool(rmargin >= margin and cmargin >= margin),
            )
        )
    return rows


def normalized_difference(a, b, guard=True):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    den = a + b
    valid = np.isfinite(a) & np.isfinite(b) & (den > 0)
    if guard:
        valid &= (a > 1) & (a < 254) & (b > 1) & (b < 254) & (den >= 12)
    return np.divide(a - b, den, out=np.full(np.broadcast(a, b).shape, np.nan), where=valid)


def pair_features(samples, min_pixels=40):
    samples = np.asarray(samples, float).reshape(-1, 6)
    out = {}
    for i, j in PAIRS:
        a, b = samples[:, i], samples[:, j]
        nd = normalized_difference(a, b)
        valid = np.isfinite(nd)
        n = int(valid.sum())
        values = nd[valid]
        out[CHANNELS[i] + "__" + CHANNELS[j]] = dict(
            n=n,
            pixel_median=float(np.median(values)) if n >= min_pixels else None,
            pixel_q10=float(np.quantile(values, 0.1)) if n >= min_pixels else None,
            pixel_q90=float(np.quantile(values, 0.9)) if n >= min_pixels else None,
            ratio_of_medians=float(normalized_difference(np.median(a[valid]), np.median(b[valid])))
            if n >= min_pixels
            else None,
        )
    return out


def organ_features(rgb_samples, noir_samples, min_pixels=40):
    """Separate visible interiors, each channel separately guarded; no pixel correspondence."""
    values = list(rgb_samples.T) + list(noir_samples.T)
    medians = []
    counts = []
    for a in values:
        good = np.isfinite(a) & (a > 1) & (a < 254)
        n = int(good.sum())
        counts.append(n)
        medians.append(float(np.median(a[good])) if n >= min_pixels else None)
    nd = {}
    for i, j in PAIRS:
        value = (
            float(normalized_difference(medians[i], medians[j]))
            if medians[i] is not None and medians[j] is not None
            else np.nan
        )
        nd[CHANNELS[i] + "__" + CHANNELS[j]] = value if np.isfinite(value) else None
    return dict(
        channel_medians=medians, channel_valid_counts=counts, nd_of_separate_visible_medians=nd
    )
