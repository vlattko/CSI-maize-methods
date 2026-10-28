# Interpolation-validity impact audit

The full-cohort pairing runner and common-support camera-DN controls used `warped_valid > 0.999`. With OpenCV's tabulated bilinear weights, a single invalid source contributor can have weight 1/1024. Its remaining validity is 0.9990234375, so this guard admits it. The stricter comparison rejects that case. Identity interpolation retains valid pixels whose invalid neighbours have zero weight.

This audit keeps frozen source bytes, 512×385 decoding/resizing, segmentation masks, nominal ECC matrices, overlap gates and 27 formula definitions. It changes only the validity comparison to `>= 1-1e-7`. Every reconstructed inherited statistic must reproduce the stored record before inclusion. It hashes and snapshots the existing scientific operator. The original study outputs remain unchanged.

Run `python -m unittest discover -s THIS_FOLDER -p test_validity.py -v`, then:

```
python validity_audit.py --repo PUBLIC_OR_ORIGINAL_REPO --intake RESOLVED_RAW_ROOT --run FROZEN_RUN --output NEW_AUDIT --max-accepted 12
python validity_audit.py --repo PUBLIC_OR_ORIGINAL_REPO --intake RESOLVED_RAW_ROOT --run FROZEN_RUN --output NEW_FULL_AUDIT
python audit_summary.py --audit NEW_FULL_AUDIT --output NEW_SUMMARY.json
python field_support_audit.py --audit NEW_FULL_AUDIT --repo PUBLIC_OR_ORIGINAL_REPO --intake RESOLVED_RAW_ROOT --run FROZEN_RUN --output NEW_FIELD_AUDIT.json
python audit_rejected.py --repo PUBLIC_OR_ORIGINAL_REPO --intake RESOLVED_RAW_ROOT --run FROZEN_RUN --output NEW_REJECTED_SUPPLEMENT
```

The scientific source must match the frozen pairing configuration. Use the resolved raw root explicitly: the local intake `IMAGES` junction points outside the intake folder, so the audit's containment guard rejects that mapping. The raw-root binding is still verified against every frozen source SHA.

There are 1440 planned anchors. The primary comparison includes the 1212 accepted nominal ECC anchors and retains the other anchors in coverage. The supplemental runner replays all 58 rejected nominal ECC anchors, keeping their cross-camera support empty while correcting same-camera NoIR formulas. Together these cover all 1270 available nominal ECC registrations. The 170 unavailable nominal registrations remain unavailable. Control registrations are not re-evaluated. Per-formula lost-pixel counts overlap; they cannot be interpreted as distinct pixels or organs. These are JPEG-DN and teacher-support sensitivity results, without biological accuracy claims.

The accepted-anchor run reproduced all 32724 inherited formula rows exactly before comparison. It found 2149 changed count rows and 18 qualification losses. The rejected-anchor supplement reproduced all 1566 inherited rows and found 19 changed count rows and two further qualification losses. RGB-only formulas were unchanged. The common-support channel table supplies strict camera-DN medians and IQRs as well as formula statistics. All 7272 inherited common-support channel rows were independently matched to the frozen common-DN table. A separate field-support audit on all 1212 accepted anchors found 1034 altered frame-border pixels but no changes inside the eroded NoIR or common supports.

Inherited locations are `channel_fusion_core.warp_to_rgb` frame support, `pairing_run.describe` channel validity, and `pairing_dn_controls.main` channel validity. The public `run_spatial_fusion.py` channel guard already uses `> 0.999999`. Frozen investigation snapshots and outputs should retain their recorded operators. Future corrected executions require a new operator identity and explicit provenance; existing results should describe their actual interpolation tolerance.
