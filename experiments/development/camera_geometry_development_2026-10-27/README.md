# Rotation-conditioned paired-camera development

This development layer reads the existing full-cohort extraction without editing it. It produces a reusable image-coordinate alignment lookup, evaluation diagnostics and source-overlap illustrations. A mask alignment is not physical camera calibration, corresponding tissue or an independent spectral measurement.

## Shared camera surrogate

`camera_geometry.py` contains the numerical operators. A session fits one RGB-to-NoIR affine to its training angles. It starts with the coefficient-wise median moment affine and optimizes symmetric sampled silhouette-boundary distances with robust least squares, retaining the moment affine if training median IoU does not improve.

Training angles are all indices except 2 modulo 4. Ten test angles remain unused by fitting, phase/sign selection and deployment eligibility. Phase hypotheses have signs +/-1 and offsets 0,4,...,36, so each hypothesis preserves the withheld index set in both cameras. The four-index resolution is deliberate. The three best coarse moment hypotheses and nominal mapping receive shared refinement. A nonnominal mapping requires a matched training median IoU improvement of 0.03, a best/runner-up margin of 0.015 and improvement in at least two of three training-index strata. These gates do not establish physical angle truth.

The nominal baseline is always retained. Deployment acceptance requires at least 20 training pairs, training median IoU at least 0.65 and a finite positive-determinant matrix. An abstaining matrix remains a labelled hypothesis for inspection. Held-out scores are diagnostic fields and never determine this acceptance.

## First-harmonic competitor

The separate `periodic/` package regresses six affine coefficients on `1,sin(2*pi*i/40),cos(2*pi*i/40)`. Teachers are per-view ECC transforms fitted only at outer training angles. Valid teachers require converged ECC, IoU at least 0.45 and determinant/conditioning/scale/translation guards. Weighted ridge uses fixed penalty 0.1 on harmonic coefficients and six Huber reweighting iterations based on two-pixel probe-displacement residuals. No additional harmonic order is searched.

Three inner training-angle folds compare periodic predictions with a shared affine refitted without the inner validation angles. Periodic deployment requires mean inner-fold median paired IoU gain at least 0.02, positive gains in at least two folds, at least 20 accepted teachers, training median IoU at least 0.65 and valid query matrices at all40 indices. Otherwise deployment falls back to the shared model and its abstention status. Per-view adaptive test ECC is retained as a separate comparator, explicitly using test masks.

## Outputs and UI contract

The shared run emits `full/session_camera_lookup.json` and CSV with one row per360 assay acquisitions. Each row includes four-key identity and session, `selected_sign`, `selected_phase`, `phase_status`, `deployment_status`, `deployment_abstain`, reasons, nominal/shared affine baselines, grid dimensions, training support and held-out score/count fields. `alignment_lookup.csv` also contains all40 planned indices per session, including unavailable fits.

The periodic run emits the same identity/status fields under `periodic/full/session_camera_lookup.json`, with `deployment_method`, selection diagnostics, harmonic coefficients and per-index `view_matrices_rgb_to_noir[str(i)]` / `view_matrices_noir_to_rgb[str(i)]`. `periodic/full/alignment_lookup.csv` is the flat40-index contract and separately retains unselected periodic candidate matrices/guards.

All fitted matrices use a 256 x192 pixel grid. A forward matrix maps RGB pixel coordinates into the NoIR grid. To render NoIR on the reference RGB grid, pass that forward matrix to `cv2.warpAffine` with `WARP_INVERSE_MAP`, or use the provided inverse matrix with regular `warpAffine`. Change source/target grids with `camera_geometry.rescale_matrix`; source and target dimensions may differ. A matrix alone never certifies foreground availability at a queried angle: consult the source/mask availability and deployment status.

Every numerical session has an atomic record with a payload digest, configuration signature, actual mask hashes, source hash bindings, all mapping candidates and all retained failures. Source snapshots and upstream table/configuration hashes bind the run. Hash-based resume accepts only matching intact records. Never overwrite the original full-cohort pairing outputs with these results.

## Execution

Use the existing CPU numerical environment. Run from this source directory, supplying absolute paths appropriate to the installation:

```powershell
python -m unittest discover -s . -p test_camera_geometry.py -v
python -m unittest discover -s periodic -p test_periodic_camera.py -v
python camera_geometry_run.py --intake INTAKE --full-run COMPLETED_FULL_COHORT --output DEVELOPMENT_OUTPUT --mode pilot
python camera_geometry_exports.py --output DEVELOPMENT_OUTPUT/pilot --intake INTAKE --full-run COMPLETED_FULL_COHORT
python camera_geometry_run.py --intake INTAKE --full-run COMPLETED_FULL_COHORT --output DEVELOPMENT_OUTPUT --mode full
python camera_geometry_exports.py --output DEVELOPMENT_OUTPUT/full --intake INTAKE --full-run COMPLETED_FULL_COHORT
python periodic/periodic_run.py --shared DEVELOPMENT_OUTPUT/pilot --intake INTAKE --full-run COMPLETED_FULL_COHORT --output DEVELOPMENT_OUTPUT/periodic/pilot
python periodic/periodic_run.py --shared DEVELOPMENT_OUTPUT/full --intake INTAKE --full-run COMPLETED_FULL_COHORT --output DEVELOPMENT_OUTPUT/periodic/full
```

The primary numerical dependencies are NumPy, SciPy, OpenCV and Pillow; figures require Matplotlib. Each numerical process is bounded to at most two threads and disables CUDA. Run full numerical stages sequentially to avoid CPU oversubscription. PIL does not repair truncated originals for illustrations.

## Development evidence

Eight shared-layer tests cover known affine recovery on unseen angles, reversed sequence/phase recovery, test-mask mutation invariance, ambiguous phase, empty training support, grid conversion, a genuine OpenCV ECC error and deployment gating independent of test scores. Five periodic tests cover outlier-robust harmonic recovery, test-angle teacher rejection, invalid affine guards, inner-validation acceptance/decline and outer-test-mask mutation invariance in teacher/inner fitting. The 15-stratum shared pilot completed separately before the full run. The additional periodic pilot preserves all 15 cases and original failures.
