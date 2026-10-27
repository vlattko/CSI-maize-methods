# Full cohort protocol

This extension processes the complete keyed image and assay population. It preserves partial acquisitions, per-view failures, missing organ hypotheses and unsuccessful camera alternatives. The original source-assisted surface and fixed-receiver experiments remain available through the main package workflow.

The protocol was evaluated on 360 reference acquisitions, 27,126 identified decoded frames, five recorded visits and 72 repeated-label candidates. Repeated labels do not independently establish physical plant continuity. Operational foreground checks do not measure anatomical segmentation accuracy. See [the methods and validation record](../../docs/full-cohort.md).

## Inputs and environment

Use a source checkout and Python 3.12. Install the main CUDA requirements and package learning extra in a separate environment as described in [reproduction](../../docs/reproduction.md). The coordinator uses one CUDA device, `cuda:0`, because the diagnostic and frozen encoder currently select that device. Original all-frame extraction also supports explicit GPU shards when invoked separately. CPU stages limit their numerical threads.

The archive must contain `IMAGES/<round>/<session>/{rgb,noir}/...jpg` and the original `.xlsx` files, including nested workbook folders. The intake scans workbook locations and rejects duplicate basenames. It decodes complete JPEG pixels and retains empty and damaged files as failed observations. Do not repair originals or substitute black frames.

Supply the pinned SAM 2.1 Hiera Small and DINOv2-small assets locally. The main package asset registry records official revisions and SHA256 values. Asset preparation is a separate, explicit download operation. This coordinator enables offline model loading and never downloads model weights or external datasets.

Copy `cohort_config.example.json` to a location outside the raw archive and replace each path. Paths can be absolute or relative to that configuration file. `models` is the folder containing `sam2.1-hiera-small`; `dino` points directly to the DINOv2-small checkpoint folder. `raw`, `intake` and `run` must be separate, non-nested locations. Start with new empty intake and output folders.

```sh
python experiments/full_cohort/cohort_pipeline.py --config cohort_config.json --target cohort --plan
python experiments/full_cohort/cohort_pipeline.py --config cohort_config.json --target cohort
```

This route runs intake, SAM diagnostic and all-frame extraction, independent extraction audits, 20/30/40-view summaries, angular profiles, terminal hypotheses, geometry adequacy and known-answer tests, paired-camera controls, the 27-formula spatial atlas, matched DN summaries, frozen image descriptors, grouped prediction controls, numerical audits and component reports. It keeps the 360-row reference population even when image eligibility is lower.

## Independent geometry and depth components

The `full` target adds Pheno4D component tests, the retained DA3 diagnostic and the integrated HTML. It requires both inputs before starting. Pheno4D files remain external evaluation data and are not redistributed here.

```sh
python experiments/full_cohort/external_pheno4d.py --output /path/to/Pheno4D --download last_two_per_plant
python experiments/full_cohort/cohort_pipeline.py --config cohort_config.json --target full --plan
python experiments/full_cohort/cohort_pipeline.py --config cohort_config.json --target full
```

The retrieval command explicitly contacts the original provider, selects all seven maize plants' last two labelled scans and checks archive CRC and saved SHA256 values. The coordinator itself makes no retrieval request. An external rendered silhouette with supplied foreground and known cameras is a component test; it does not validate uncalibrated CSI reconstruction or segmentation.

For DA3, create the following local asset layout:

```text
da3-assets/
  vendor/Depth-Anything-3/src/...
  models/DA3-SMALL/config.json
  models/DA3-SMALL/model.safetensors
  da3_deps/...
```

Use the official Depth Anything 3 source commit and checkpoint revision in `depth_assets.json`. The manifest lists the exact executed source and model hashes. The optional numerical overlay is recorded in `requirements/da3-overlay-py312.txt`; install it with `--no-deps --target /path/to/da3-assets/da3_deps`, outside the main environment. It supplies NumPy 1.26 and compatible dependencies for this vendor API. Install `plyfile` into the same overlay before running. Do not install a second OpenCV distribution into the main environment. Optional Gaussian-splatting acceleration is not used. The overlay snapshot documents the observed environment, not hashed installation wheels or a guarantee across drivers.

All five depth/camera hypotheses remain in the results. Index 30 never enters the three-view model input. Poor held-out foreground consistency is retained rather than selecting an attractive point cloud.

## Verification and interruption

The coordinator binds original image/workbook bytes, model assets, external inputs, source and package versions. It records stage commands and a central output manifest. An operating-system lock permits only one coordinator per output folder and is released when its process exits. A plan prints commands without acquiring a lock, creating outputs or loading models.

```sh
python experiments/full_cohort/cohort_pipeline.py --config cohort_config.json --target cohort --resume
```

Verified resume rehashes recorded inputs and outputs before skipping completed stages. Missing masks, changed embeddings, different weights or source edits cause refusal. An owned interrupted intake or diagnostic is preserved in a sibling `interrupted` directory before a clean retry. The driver never adopts or renames an unrelated preexisting nonempty intake.

Recovery is conservative. If an interrupted later writer already replaced an output recorded by an earlier stage, resume refuses that changed result and identifies it. Use a fresh output workspace, or restore the recorded output before resuming. The driver does not silently accept partially rewritten results. Individual scientific producers expose their own explicit cache/resume contracts; preserve their executed source and configurations when using those contracts.

After forcibly terminating the coordinator, confirm that its producer processes have stopped before resuming. A parent process can exit while a child continues; the parent-held lock does not manage surviving process trees. Each workflow must use its own intake as well as its own run folder. Recovery is not automatic under those two unsupported sharing conditions.

Original Windows extraction encountered transient file-sharing errors during atomic progress replacement. The operational launcher retries only those sharing failures, with a finite limit. It leaves the numerical extractor and model operator unchanged. Regression tests cover this failure mode.

## Tests and reports

```sh
python -m pytest tests experiments/full_cohort
node experiments/full_cohort/test_report_browser.mjs /path/to/output/CSI_maize_full_data_methods.html
```

The numerical component stages were executed on the original archive and independently audited. The new coordinator assembles those tested commands; its contracts, argument parsing, dependency ordering, ownership and resume guards have separate tests. Do not describe this assembly as a second fresh all-image execution. Dataset-dependent runs require the original inputs and models; the package's controlled CPU example remains available without them.

The HTML embeds its image previews and scientific figures. Missing images have explicit placeholders. The keyed explorer covers all 72 label combinations, five visits and both cameras, with measurement export. A companion atlas retains every formula's valid-pixel support. Neither map colours nor completed execution establish chlorophyll calibration, true leaf identity or physiological causation.

Data and reference outputs will be made available via Zenodo after publication. The repository contains original methods code under MIT, with third-party terms retained separately.
