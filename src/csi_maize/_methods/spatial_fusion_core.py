"""Spatial camera-DN features with explicit, family-specific geometric support.

No NoIR channel is designated a calibrated NIR band. All calculations use the
frozen 512 x 385 processing grid, with NoIR channels sampled onto RGB coordinates.
"""

from channel_fusion_core import CHANNELS, PAIRS, np


def index_definitions():
    definitions = []
    for i, j in PAIRS:
        a, b = CHANNELS[i], CHANNELS[j]
        family = "rgb" if j < 3 else "noir" if i >= 3 else "cross"
        definitions.append(
            dict(
                id=f"ND_{a}__{b}",
                label=f"ND({a}, {b})",
                formula=f"({a} - {b}) / ({a} + {b})",
                channels=[i, j],
                support=family,
                kind="ND",
                vmin=-1,
                vmax=1,
                reference="Pairwise camera-DN heuristic",
            )
        )
    rgb = [
        ("ExG", "(2G - R - B) / (R + G + B)", -1, 2, "Normalized excess green"),
        (
            "ExGR",
            "(3G - 2.4R - B) / (R + G + B)",
            -2.4,
            3,
            "Normalized excess green minus excess red",
        ),
        ("GLI", "(2G - R - B) / (2G + R + B)", -1, 1, "Green leaf index"),
        ("VARI", "(G - R) / (G + R - B)", -1, 1, "Visible atmospherically resistant index formula"),
        ("MGRVI", "(G² - R²) / (G² + R²)", -1, 1, "Modified green-red vegetation index"),
        ("RGBVI", "(G² - R B) / (G² + R B)", -1, 1, "Red-green-blue vegetation index"),
    ]
    for name, formula, lo, hi, label in rgb:
        definitions.append(
            dict(
                id=name,
                label=label,
                formula=formula,
                channels=[0, 1] if name == "MGRVI" else [0, 1, 2],
                support="rgb",
                kind="RGB",
                vmin=lo,
                vmax=hi,
                reference="Bendig et al. 2015"
                if name in ("MGRVI", "RGBVI")
                else "Gitelson et al. 2002"
                if name == "VARI"
                else "RGB colour-index literature; see report references",
            )
        )
    multichannel = [
        (
            "CAMSUM6",
            "Camera sum contrast",
            "(ΣNoIR - ΣRGB) / (ΣNoIR + ΣRGB)",
            list(range(6)),
            -1,
            1,
        ),
        ("GREENSHARE6", "Green-share contrast", "G/ΣRGB - noir_G/ΣNoIR", list(range(6)), -1, 1),
        (
            "CHROMDIST6",
            "Chromatic composition distance",
            "sqrt(Σc (RGB_c/ΣRGB - NoIR_c/ΣNoIR)² / 2)",
            list(range(6)),
            0,
            1,
        ),
        (
            "GMEAN6",
            "Geometric-mean camera contrast",
            "(GM(NoIR) - GM(RGB)) / (GM(NoIR) + GM(RGB))",
            list(range(6)),
            -1,
            1,
        ),
        (
            "PRODUCT4",
            "Four-channel product balance",
            "(noir_B × rgb_G - rgb_R × noir_R) / (noir_B × rgb_G + rgb_R × noir_R)",
            [0, 1, 3, 5],
            -1,
            1,
        ),
        (
            "GLICONTRAST6",
            "Paired GLI-formula contrast",
            "(GLI(RGB) - GLI(NoIR)) / 2",
            list(range(6)),
            -1,
            1,
        ),
    ]
    for name, label, formula, channels, lo, hi in multichannel:
        definitions.append(
            dict(
                id=name,
                label=label,
                formula=formula,
                channels=channels,
                support="cross",
                kind="multichannel",
                vmin=lo,
                vmax=hi,
                reference="Exploratory algebraic descriptor; no validated physiological interpretation",
            )
        )
    return definitions


def spatial_indices(six, supports, channel_valid=None):
    """Return float32 maps; every unsupported or invalid pixel is NaN.

    Optional channel_valid records validity before interpolation. For NoIR we
    require every nonzero-weight contributor of the affine interpolation to
    be valid, not just the brightness of the interpolated result.
    """
    six = np.asarray(six, dtype=np.float32)
    if six.ndim != 3 or six.shape[-1] != 6:
        raise ValueError("Expected H x W x 6 recorded channels")
    support = {key: np.asarray(supports[key], bool) for key in ("rgb", "noir", "cross")}
    if any(value.shape != six.shape[:2] for value in support.values()):
        raise ValueError("Support masks must share the RGB reference grid")
    signal = np.isfinite(six) & (six > 1) & (six < 254)
    if channel_valid is not None:
        if np.shape(channel_valid) != six.shape:
            raise ValueError("Channel validity must be H x W x 6")
        signal &= np.asarray(channel_valid, bool)
    r, g, b = [six[..., i] for i in range(3)]
    output = {}
    for definition in index_definitions():
        key = definition["id"]
        valid = support[definition["support"]] & np.all(
            signal[..., definition["channels"]], axis=-1
        )
        if definition["kind"] == "ND":
            i, j = definition["channels"]
            numerator = six[..., i] - six[..., j]
            denominator = six[..., i] + six[..., j]
            valid &= denominator >= 12
        elif definition["kind"] == "RGB":
            valid &= np.sum(six[..., definition["channels"]], axis=-1) >= 12
            if key == "ExG":
                numerator, denominator = 2 * g - r - b, r + g + b
            elif key == "ExGR":
                numerator, denominator = 3 * g - 2.4 * r - b, r + g + b
            elif key == "GLI":
                numerator, denominator = 2 * g - r - b, 2 * g + r + b
            elif key == "VARI":
                numerator, denominator = g - r, g + r - b
                valid &= np.abs(denominator) >= 12
            elif key == "MGRVI":
                numerator, denominator = g * g - r * r, g * g + r * r
            elif key == "RGBVI":
                numerator, denominator = g * g - r * b, g * g + r * b
        else:
            a, bcam = six[..., :3], six[..., 3:]
            suma, sumb = a.sum(axis=-1), bcam.sum(axis=-1)
            safea = np.maximum(suma, 1)
            safeb = np.maximum(sumb, 1)
            ca, cb = a / safea[..., None], bcam / safeb[..., None]
            valid &= np.sum(six[..., definition["channels"]], axis=-1) >= 12
            if len(definition["channels"]) == 6:
                valid &= (suma >= 12) & (sumb >= 12)
            if key == "CAMSUM6":
                numerator, denominator = sumb - suma, sumb + suma
            elif key == "GREENSHARE6":
                numerator, denominator = ca[..., 1] - cb[..., 1], np.ones_like(suma)
            elif key == "CHROMDIST6":
                numerator, denominator = (
                    np.sqrt(np.sum((ca - cb) ** 2, axis=-1) / 2),
                    np.ones_like(suma),
                )
            elif key == "GMEAN6":
                ga = np.cbrt(np.prod(a, axis=-1))
                gb = np.cbrt(np.prod(bcam, axis=-1))
                numerator, denominator = gb - ga, gb + ga
            elif key == "PRODUCT4":
                p, q = six[..., 5] * g, r * six[..., 3]
                numerator, denominator = p - q, p + q
                valid &= (six[..., 5] + g >= 12) & (r + six[..., 3] >= 12)
            elif key == "GLICONTRAST6":
                dr, dn = 2 * g + r + six[..., 2], 2 * six[..., 4] + six[..., 3] + six[..., 5]
                numerator = (2 * g - r - six[..., 2]) / np.maximum(dr, 1) - (
                    2 * six[..., 4] - six[..., 3] - six[..., 5]
                ) / np.maximum(dn, 1)
                denominator = np.full_like(suma, 2)
                valid &= (suma >= 12) & (sumb >= 12)
        valid &= np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
        output[key] = np.divide(
            numerator, denominator, out=np.full(six.shape[:2], np.nan, np.float32), where=valid
        )
    return output


def index_statistics(values, eligible, min_pixels=40):
    eligible = np.asarray(eligible, bool)
    good = eligible & np.isfinite(values)
    v = np.asarray(values)[good]
    n = int(good.sum())
    total = int(eligible.sum())
    row = dict(
        eligible_pixels=total,
        valid_pixels=n,
        valid_fraction=n / total if total else None,
        qualified=n >= min_pixels,
        median=None,
        q10=None,
        q25=None,
        q75=None,
        q90=None,
        iqr=None,
    )
    if n >= min_pixels:
        q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
        row.update(
            q10=float(q[0]),
            q25=float(q[1]),
            median=float(q[2]),
            q75=float(q[3]),
            q90=float(q[4]),
            iqr=float(q[3] - q[1]),
        )
    return row


def overlap_labels(rgb_mask, noir_mask):
    return np.asarray(rgb_mask, np.uint8) + 2 * np.asarray(noir_mask, np.uint8)


def blend_projections(rgb, noir_registered, rgb_mask, noir_mask, alpha=0.5):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    a = np.asarray(rgb, np.float32) * np.asarray(rgb_mask, bool)[..., None]
    b = np.asarray(noir_registered, np.float32) * np.asarray(noir_mask, bool)[..., None]
    return np.clip((1 - alpha) * a + alpha * b, 0, 255).astype(np.uint8)
