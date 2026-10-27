# Third-party software and pretrained assets

The MIT licence applies to the original code in this repository. Dependencies and pretrained assets retain their own licences. No model weights, proprietary fonts or third-party source trees are redistributed here.

| Component | Role | Upstream terms |
|---|---|---|
| Meta SAM 2.1 | Prompted plant foreground proposals | [Apache 2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE) |
| Depth Anything 3 | Optional relative-depth and camera adequacy diagnostic | [Official source and terms](https://github.com/ByteDance-Seed/Depth-Anything-3) and the pinned model card in the cohort asset manifest |
| Meta DINOv2 | Optional frozen image descriptors | [Apache 2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) |
| Ultralytics YOLO26 | Optional weak-label leaf segmentation and tracking comparisons | [AGPL 3.0 or an applicable Enterprise licence](https://www.ultralytics.com/license) |
| Matplotlib and bundled DejaVu Sans | Plots and offline label rendering | [Matplotlib licence and font notices](https://matplotlib.org/stable/project/license.html) |
| PyTorch, OpenCV, NumPy, SciPy and other packages | Numerical and image processing | The licence notices included in their installed distributions |

The original-code licence does not remove Ultralytics' conditions for a combined application or its trained models. The YOLO workflow and resulting weights remain subject to those conditions. This repository provides the corresponding training and inference procedures; users must comply with the upstream terms applicable to their use.

`src/csi_maize/assets_manifest.json` records model sources, pinned revisions, file sizes and SHA256 values. Asset preparation verifies every file before use. The YOLO AMP-check checkpoint is recorded separately from segmentation checkpoints. Plotting uses Matplotlib's bundled DejaVu Sans; the workflow does not fetch or redistribute Arial.

Third-party works should also be cited where their methods or models are used. These notices are separate from the authors' software citation in `CITATION.cff`.

The optional full-cohort depth manifest and environment overlay are recorded separately under `experiments/full_cohort/depth_assets.json` and `requirements/da3-overlay-py312.txt`. External Pheno4D data are retrieved from the original provider and are not redistributed. Cohort plots use installed Times New Roman when available with an open serif fallback; proprietary font files are not included.
