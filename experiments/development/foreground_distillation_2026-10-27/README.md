# Fast maize foreground

This investigation trains a single `plant_foreground` class from operationally accepted SAM2.1 masks. It predicts a whole-plant foreground for DIY image processing. Joint training means one three-channel model trained on a mixture of RGB and recorded NoIR frames; inference takes either camera image independently. It does not require paired cameras or perform six-channel fusion. The class does not separate leaves, identify organs or track their identities. Individual leaf instances still require reviewed instance or landmark annotations.

The existing CUDA runtime and local Ultralytics8.4.187 dependency are used without package installation or model downloads. The checked offline adapter is imported from `CSI-maize-methods/src/csi_maize`; initialization assets are verified against its asset manifest. Original extraction operators and input masks are unchanged.

## Use an available model

Pilot candidates appear in `deployment/pilot_joint` and `deployment/pilot_rgb`. Final candidates appear in `deployment/final_joint`, `deployment/final_joint_all40` and `deployment/final_rgb` only after the frozen final test completes. Read the candidate's model card; a pilot has validation evidence only.

```powershell
$Repo = '/path/to/prepared-workspace'
$Run = '/path/to/foreground-output'
$Tool = '/path/to/source-checkout/experiments/development/foreground_distillation_2026-10-27'
python "$Tool/infer_foreground.py" `
  --repo $Repo --weights "$Run/deployment/pilot_joint/best.pt" `
  --model-card "$Run/deployment/pilot_joint/model_card.json" `
  --source 'path/to/camera_image.jpg' --output "$Run/my_foreground" --device 1
```

The CLI writes a binary mask, a black-background image and an inference manifest. Mask and image use the released processing decode, whose maximum dimension is1024pixels; they are not native3280pixel masks. Multiple source paths can follow `--source`. `--device cpu` also works. A model card binds exact weight bytes and the validation-selected confidence. A reused output directory refuses a different model.

The cache producer accepts the same explicit repository/intake/teacher/output paths and a model card:

```powershell
python "$Tool/cache_foreground.py" `
  --repo $Repo --intake '/path/to/intake' `
  --teacher-run '/path/to/full-cohort-run' --output $Run `
  --model-card "$Run/deployment/pilot_joint/model_card.json" --batch-size 8
```

`student_cache/<model_name>/frames.csv` joins to source images through the four identity keys, session, camera, view and source path. `student_mask` is relative to that cache root. Per-frame records and mask hashes support resume; `status.json` becomes completed only after every expected available source is present. `acquisition_coverage.csv` retains both cameras for all360reference identities, including zero-image cases. `--quarters-only` and `--limit` produce clearly separate, partial cache directories. Inference does not require a SAM mask and does not reject a source because its SAM teacher failed QC.

`--imgsz 960` is an explicit inference sensitivity. Its pilot validation check improved thin-region support but did not solve teacher-terminal preservation. The primary model comparisons use640pixels. A1280pixel pilot sensitivity lost prediction coverage, so it is not an automatic quality upgrade. Final model cards state their actual inference size.

## What was frozen before test

The genotype partition is B73/Mo17/Os024445/Os622448 for training, Os6251/15 for validation and Os7798/2 for test. All treatment/replicate visits and views remain together. This is one held-out-genotype test, not six-fold cross-validation.

The models are RGB-only and joint RGB/NoIR quarter-view pilots (5epochs), their25epoch fixed-quarter fits, and one joint all40-view model (5epochs, with at most5additional validation-gated epochs). Inputs, split lists, teacher polygon roundtrip checks, configurations, logs, weights and execution source snapshots are retained. All-angle export uses maximum640pixel JPEG95/subsampling0 training images and normalized polygons from the original processing1024teacher masks. For the all-angle fit, per-epoch checkpoint selection uses the fixed-quarter views of the validation genotype; postfit extension and confidence choices use every available validation angle.

`evaluation_protocol.json` declares threshold choices and all-angle extension rules before final test. Model choices use validation only. Final scoring uses every available angle from the test genotype, with accepted teachers and rejected-teacher diagnostics reported separately. Whole-mask agreement is supplemented by teacher skeleton endpoint and thin-region preservation; these are unreviewed geometry proxies, not biological leaf counts.

The pinned COCO potted-plant class and the previous camera-specific weak instance models provide foreground baselines. The latter were trained on89images from one candidate of the validation genotype, so their validation diagnostics overlap their development material. Their foreground unions do not establish leaf-instance accuracy. All training/evaluation conditions retain the same new test genotype.

## Limits and reproducibility

Teacher agreement inherits SAM errors and training-label polygon approximations. Holes and disconnected fragments are audited rather than treated as independent expert annotations. RGB and recorded NoIR are camera DNs. The models do not estimate calibrated reflectance, leaf physiology or3D anatomy.

Real timing distinguishes warmed model prediction from raw-JPEG decode, digest checking and output writes. Do not interpret Ultralytics training-validation per-image timings as an end-to-end DIY camera frame rate.

Run the known-answer checks with the existing runtime:

```powershell
python "$Tool/test_foreground.py"
```

The numerical model assets and Ultralytics runtime carry upstream AGPL-3.0 terms or an applicable separate upstream license. Preserve upstream attribution. This repository does not relicense pretrained weights as original project material.

## Public source layout

The `--repo` argument remains a prepared outer workspace. It expects the checked
segmentation initialization at `prototype/research_v2/yolo26n-seg.pt`, the AMP
asset at `prototype/weights/yolo26n.pt`, and either an installed `csi_maize`
package or `CSI-maize-methods/src`. Ultralytics must be installed at the pinned
version or available in the optional prepared `prototype/research_v2/yolo_deps`
directory. A bare public checkout does not supply that workspace or any weights.
No download fallback is provided.

The public cache runner resolves a relative card `weights` name beside its model
card, then checks the declared digest. Direct inference retains its explicit
`--weights` argument. The added public regression makes 19 tests; the original
frozen research suite contained 18. Original and public SHA256 hashes are listed
in `../source_provenance.json`; the executed private source was not overwritten.
