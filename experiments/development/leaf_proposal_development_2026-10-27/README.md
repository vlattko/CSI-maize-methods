# Track-supported region annotation bootstrap

This pilot creates unreviewed region proposals from real RGB and NoIR images. It uses observed terminal paths from the new tracking graphs and a locally cached SAM2.1 Hiera Small checkpoint. It does not convert plant foreground masks or track IDs into anatomical leaf labels.

The fixed pilot represents15 treatment-by-visit strata. Indices0 and20 in each camera give60 camera frames. It retains86 ready track-region requests, seven failed prompt transfers and three frames without usable prompts. For each ready request it compares a local box with the same box plus three distal positive points and up to three other-terminal negatives. Three SAM alternatives per mode yield516 raw masks and516 separate parent-clipped masks.

Prompt paths are oriented by their observed terminal. Only the distal80% of the terminal-to-last-junction branch supplies positives; shared stem paths are excluded. NoIR requests use an accepted camera surrogate to find a nearby observed NoIR terminal, then derive the prompt from the NoIR mask's own branch. This is a weak correspondence hypothesis, not an organ or tissue identification.

The tracking grid is256x192; SAM prompts and region masks use the source processing resolution, generally1024x769. Pixel centres are converted explicitly. Positive prompts must lie inside the exact processing-resolution foreground; an unsupported point is refined to nearby foreground within six processing pixels or the prompt abstains. Coordinates before refinement and distances remain in the record.

## Review outputs

- `pilot/frame_index.json`: all60 planned camera frames, four-key identity plus session, source/parent hashes, dimensions, review image and selected proposal IDs for both modes.
- `pilot/regions.json` and CSV: all516 alternatives with track-node candidate ID, camera/index, mode/alternative, model score, prompts, raw and clipped mask paths/hashes, outside-parent fraction, parent coverage and prompt support.
- `pilot/review_gallery.html`: self-contained black-background review gallery. Six panels show prompts, raw predictions, clipped regions and coverage gaps. Raw pixels outside the parent are red. Expand each frame to inspect every multimask alternative.
- `pilot/review_predictions_coco.json`:86 highest-model-score point-prompted predictions with uncompressed column-major RLE, processing-resolution review images and `review_required=true`. Their category is `unreviewed_terminal_region`. Model score selects a display hypothesis and is not expert anatomical accuracy.
- `pilot/frame_overlap_coverage.csv`: complete frame/mode coverage, gaps, overlap and region counts, including zero-prompt frames.
- `pilot/qa/all_strata_view_0.png` and view20: zoomed contact sheets covering all60 pilot frames.
- `verification.json` and `development_findings.json`: independent input/output, coordinate and COCO-mask checks, ablation summaries and the explicit extension judgment.

No prediction is accepted automatically as a training label. The inspected masks include distal fragments, visible blade regions and short basal branches. Median foreground union coverage is low. These outputs support human annotation review and region taxonomy decisions; the pilot does not justify complete-leaf pseudo-labels across the cohort.

## Provenance and correction history

The numerical run records actual model assets, source/model versions, FP32/GPU0 settings, image-embedding reuse, exact source-image and parent-mask hashes, graph/camera bindings and source snapshots. The seven unit tests cover distal path geometry, negative prompts, matching abstention, clipping accounting, COCO RLE and exact processing-coordinate support. The independent audit rehashes all60 sources, parents and1032 output masks and decodes all86 exported COCO masks.

The first bounded run used direct grid scaling. QA found40 of258 positive points outside the exact processing mask. That complete run, its masks, source snapshot, plan and logs are preserved under `pilot_coordinate_v1` and matching suffixed files. The corrected pilot uses the same60 frames and86 requests, with zero outside-parent positives. This is a coordinate correction, not a silent replacement of an initial failed diagnostic.

## Running

Use the existing CUDA-clean numerical environment, cached model and completed upstream outputs. No package or model downloads are required. Source and output roots are explicit arguments:

```powershell
python -m unittest discover -s . -p test_leaf_proposals.py -v
python leaf_plan.py --tracking TRACKING_OUTPUT --camera-lookup CAMERA_OUTPUT/periodic/full/session_camera_lookup.json --full-run FULL_COHORT_OUTPUT --output LEAF_OUTPUT
python leaf_run.py --plan LEAF_OUTPUT/pilot_plan.json --intake INTAKE --full-run FULL_COHORT_OUTPUT --model CACHED_SAM_DIRECTORY --output LEAF_OUTPUT
python leaf_exports.py --output LEAF_OUTPUT --full-run FULL_COHORT_OUTPUT
python verification/leaf_verify.py --output LEAF_OUTPUT --intake INTAKE --full-run FULL_COHORT_OUTPUT
python verification/leaf_findings.py --output LEAF_OUTPUT --full-run FULL_COHORT_OUTPUT
```

`leaf_run.py` exposes physical GPU0 only, uses local model files and FP32, reuses each image embedding across prompt modes and bounds CPU numerical threads to two. The model is not fine-tuned. NumPy, Pillow, OpenCV, SciPy/scikit-image, PyTorch and Transformers come from the existing environment. Hash-based resume requires the exact source/configuration and intact masks; immutable numerical snapshots preserve the executed versions when export scripts later change.
