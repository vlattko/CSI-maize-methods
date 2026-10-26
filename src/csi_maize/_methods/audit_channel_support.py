"""Two-sided support and erosion/gate sensitivity; frozen assignments unchanged."""

from run_channel_fusion import *


def main():
    targets = json.loads((DEST / "shared_targets.json").read_text(encoding="utf-8"))["records"]
    traits = {
        (r["session"], r["camera"], int(r["view"])): r
        for r in read_csv(OUT / "sam_traits/view_traits.csv")
    }
    rows = []
    for session in sorted({t["session"] for t in targets}):
        pp = {c: load_prediction(session, c) for c in ["rgb", "noir"]}
        for view in range(40):
            tt = [t for t in targets if t["session"] == session and t["view"] == view]
            if not tt:
                continue
            masks = {
                c: np.asarray(
                    Image.open(ROOT / traits[session, c, view]["mask"]).resize(
                        (512, 385), Image.Resampling.NEAREST
                    )
                )
                > 0
                for c in ["rgb", "noir"]
            }
            det = {
                c: {int(r["detection_index"]): m for r, m in pp[c][view]} for c in ["rgb", "noir"]
            }
            for t in tt:
                rm = det["rgb"][t["rgb_detection_index"]]
                nm = det["noir"][t["noir_detection_index"]]
                matrix = np.array(t["registration_source_to_target"], np.float32)
                for kernel in [1, 3, 5, 7]:
                    ri = (
                        cv2.erode(
                            (rm & masks["rgb"]).astype(np.uint8),
                            np.ones((kernel, kernel), np.uint8),
                            borderType=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        > 0
                    )
                    ni = (
                        cv2.erode(
                            (nm & masks["noir"]).astype(np.uint8),
                            np.ones((kernel, kernel), np.uint8),
                            borderType=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        > 0
                    )
                    wn, fov = warp_to_rgb(ni.astype(np.uint8), matrix, ri.shape, True)
                    wn = wn > 0
                    common = ri & wn & fov
                    if kernel == 3:
                        assert int(common.sum()) == t["common_interior_pixels"]
                    rows.append(
                        dict(
                            target_id=t["target_id"],
                            session=session,
                            view=view,
                            erosion_kernel=kernel,
                            RGB_interior_pixels=int(ri.sum()),
                            NoIR_warped_interior_pixels=int(wn.sum()),
                            common_pixels=int(common.sum()),
                            common_fraction_RGB=float(common.sum() / max(1, ri.sum())),
                            common_fraction_NoIR=float(common.sum() / max(1, wn.sum())),
                            RGB_prediction_outside_SAM_fraction=float(
                                (rm & ~masks["rgb"]).sum() / max(1, rm.sum())
                            ),
                            NoIR_prediction_outside_SAM_fraction=float(
                                (nm & ~masks["noir"]).sum() / max(1, nm.sum())
                            ),
                            geometry_candidate=t["geometry_candidate"],
                            leaf_iou=t["leaf_iou"],
                        )
                    )
    write_csv(DEST / "target_support_audit.csv", rows)
    gates = []
    for session in sorted({t["session"] for t in targets}):
        for gate in [0.25, 0.35, 0.5, 0.65, 0.8]:
            eligible = [
                t
                for t in targets
                if t["session"] == session
                and t["whole_mask_iou"] >= 0.65
                and min(t["assignment_row_margin"], t["assignment_col_margin"]) >= 0.1
                and t["leaf_iou"] >= gate
            ]
            gates.append(
                dict(
                    session=session,
                    leaf_iou_gate=gate,
                    geometry_candidates=len(eligible),
                    shared_support_hypotheses=sum(
                        t["common_interior_pixels"] >= 40 and t["common_interior_fraction"] >= 0.35
                        for t in eligible
                    ),
                    interpretation="Sensitivity within frozen Hungarian proposals, not reassigned or accuracy-calibrated.",
                )
            )
    write_csv(DEST / "gate_sensitivity.csv", gates)
    dump(
        DEST / "support_audit_manifest.json",
        dict(
            rows=len(rows),
            targets=len(targets),
            code_sha256=sha(__file__),
            scope="Both camera support fractions retained; erosion is varied only for measurement support, not to improve assignment. Gates filter frozen proposals, not a new matching optimization.",
        ),
    )
    print("SUPPORT_AUDIT", len(rows), "rows")


if __name__ == "__main__":
    main()
