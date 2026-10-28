# One capture from images to spatial observations

This source extension connects foreground inference, image traits, rotational terminal hypotheses, training-angle camera alignment and spatial camera-DN maps. It does not require study spreadsheets or infer experimental labels. Missing slots, unknown plant identity, failed inference and unsupported alignment remain explicit.

```sh
python run_digitization.py --input /path/to/capture --session session.json \
  --output /path/to/new-results --repo /path/to/CSI-maize-methods \
  --model-card /path/to/model_card.json --provider unet --device cuda:0
```

The model card must bind local checkpoint bytes. Use `--provider yolo` with an appropriate plant-foreground card. A U-Net pixel threshold and YOLO object confidence have different meanings. Overrides are optional and recorded. `--plan` prints the two commands without creating outputs. The inference and CPU fusion stages run in separate processes; the runner records their commands, exit states and output-manifest hashes.

The capture folder contains `rgb/` and `noir/` images with declared zero-based indices. A minimal `session.json` is:

```json
{"sample_id":"capture-001","plant_uid":null,"steps":40,"cameras":{"rgb":"rgb","noir":"noir"}}
```

The first stage supports other declared step counts for traits and source inspection. The current rotational tracker and fusion camera model require a uniform 40-index sequence and abstain for unsupported geometry. Nominal indices are not measured servo telemetry or calibrated physical angles.

The second stage can also process an existing completed capture:

```sh
python capture_fusion.py --capture /path/to/completed-capture \
  --raw /path/to/original-capture --repo /path/to/CSI-maize-methods \
  --output /path/to/new-fusion-results
python -m unittest discover -v
```

Before fitting, we verify the capture manifest, sample/camera/index bindings, original image bytes and mask bytes. Phase, direction and camera complexity use indices other than 2 modulo 4 in both cameras. A shared affine is compared with a first-harmonic affine model using training-only cross-validation. Withheld angles are scored after camera selection.

Matrices map RGB pixels to NoIR pixels. We rescale the selected matrix from the 256 by 192 camera grid to the established 512 by 385 atlas grid, then sample NoIR into RGB coordinates. The camera diagnostic and atlas-grid overlap are retained separately. Cross-camera formulas require a training-accepted camera and the established atlas-grid foreground/common-support gates. Abstaining reports retain the unregistered source foregrounds and withhold fused projections and maps.

The 27 formula definitions and numerical formula operators are reused from the public methods. Their support is formula-specific. We exclude dark/clipped channels before interpolation and require a warped validity of at least `1 - 1e-7`, so even a nonzero contributor of weight `1/1024` invalidates the corresponding channel. This is stricter than the original full-cohort runner's `>0.999` tolerance. Original outputs remain archived; the separate validity audit measures the effect of this correction. NoIR channels are camera output mixtures, not calibrated NIR bands.

Each fusion run retains all planned pair rows and all 27 formula rows per index, including unavailable observations. Compressed arrays retain unclipped numerical values and NaNs for invalid support. Black-background previews use the recorded fixed display ranges. The HTML is self-contained; CSV, numeric maps, fit diagnostics, formula definitions and hashes remain alongside it.

Required sibling source folders are `new_session_pipeline_development_2026-10-27` and `camera_geometry_development_2026-10-27`, with the tracker/U-Net siblings documented by the capture stage. This layout is preserved in the source distribution. Model weights and private images are supplied separately.

Operational checks include three complete captures with the U-Net provider and one complete CPU run with the validation-selected YOLO provider. These checks establish execution and source binding, not expert segmentation, anatomical leaf tracking, metric geometry, chlorophyll concentration or physiological prediction.
