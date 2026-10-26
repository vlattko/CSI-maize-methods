# Numerical method and fixed study protocol

The publication workflow is a prescribed computational case. It fixes the source
acquisitions, assisted observations, camera family, geometry prior, optimization
schedule, weak-label recipes and virtual experiments described below. Running
it on new files does not by itself provide a new plant's organ identities,
calibrated camera, true hidden geometry or longitudinal leaf correspondence.

`src/csi_maize/stages.py` is the authoritative dependency catalogue. The runner
executes each command in a fresh workspace whose input bindings, configuration,
source hashes and stage results are recorded. Raw mode derives results from
declared original inputs. Supplied masks, checkpoints or saved numerical results
belong to explicitly selected replay modes and must not be represented as a new
raw end-to-end fit.

## Data and foregrounds

The join key is `(treatment, measurement, genotype, replicate)`. Duplicate keys,
conflicting filename/manual identity evidence and paths outside the workspace or
registered raw bindings are rejected. A matching computational key does not
establish correct physical labeling or assay timing. Unresolved identities stay
unresolved; capture-order sorting does not create a plant identity.

The publication cohort contains 41 complete paired acquisitions, each with RGB
and NoIR indices 0 through 39. Forty angular views of one acquisition are
technical observations, not independent biological replicates. A configurable
new cohort may change the acquisition count, but the present circular geometry
and camera-fusion kernels still require all 40 indices in both streams and reject
incomplete rotations. The earlier baseline index can enumerate other allowed
view counts; that does not make them valid input to the 40-view methods.

SAM 2.1 uses a pinned small checkpoint and bounding boxes derived from
conservative camera-specific color seeds. Images are processed at a maximum
dimension of 1024 pixels; the fixed case uses a 1024 × 769 processing frame.
The comparison stage also evaluates box and point prompting on quarter views of
two declared examples. It is separate from all-view mask extraction.

Operational QC rejects a model score below 0.85, mask area fraction outside
`(0.0005, 0.15)`, seed coverage below 0.85, mask/seed area ratio outside
`(0.6, 2.5)`, or more than 0.12 dark-pixel fraction. These are engineering checks,
not measured segmentation accuracy. Session-level traits require all 80 camera
views to pass; every failed view remains in the audit table. Skeleton endpoints
are not leaf counts, and processing pixels do not provide physical dimensions.

## Assisted geometry and camera

The late example is `capture_session_C2_5-1`, with five source-assisted local blade
hypotheses initialized in fitting views 0 and 20. The early example,
`capture_session_C1_5-1`, supplies three local blade hypotheses. The source
coordinates are part of the prescribed assistance, not independent reference
annotations. Projected skeleton junctions can be overlap points rather than
botanical insertions. Unobserved paths and competing identities retain their
uncertainty flags.

Fitting uses indices `view % 4 != 2` (30 views). The ten interleaved indices are
development-held-out views that have been inspected during method development;
they are not an untouched benchmark. Uniform nominal turn angles, a single
rotation axis, the image-center principal point and a fitted conditional
perspective camera are assumptions. The model gauge is not a measured metric
scale.

The late inverse fit uses seed 201, 900 initial optimization steps, followed by
600 triangle-renderer steps with an explicit warm start from the saved initial
fit. The early fit uses seed 2410 and 750 steps. The cubic ribbon prior represents
prescribed organs and fills hidden surfaces through assumptions; it does not
discover organ count or independently validate attachments. The finite E08
experiment evaluates every declared candidate against fitting evidence, retaining
the accepted models and rejected scores. Its finite range is a sensitivity
analysis, not a confidence interval. No candidate is chosen using the
development-held-out score.

Source-appearance sampling uses frontmost positive-depth ownership, reciprocal
depth interpolation and eroded organ/SAM interiors on the 512 × 385 observation
grid. It excludes very dark or clipped recorded values and retains per-view
support. Appearance remains conditional on the fitted ownership and uncalibrated
camera processing. Candidate early/late rank-order maps are enumerated rather
than declared as physical longitudinal identities.

## Weak instance segmentation and tracking

Teacher masks come from the two source-assisted RGB organ models intersected
with SAM foreground. NoIR proposals use a whole-mask-gated affine transfer.
These are explicitly weak labels. The two teacher acquisitions are assigned to
development training. The conservative scientific grouping keeps treatment,
genotype and replicate labels together across rounds and cameras; its fixed
hash salt is `CSI-maize-20261024|`.

Six YOLO26 segmentation recipes are executed: nano and small models on RGB,
recorded NoIR channels, and triplicated NoIR blue. Each requests 30 epochs,
768-pixel training size, batch 8, seed 20261024, deterministic mode, AdamW,
initial learning rate 0.001 and patience 12. The source recipe fixes the remaining
augmentation parameters. Validation uses other angles of the same two teacher
acquisitions. Its mask AP measures weak-teacher agreement, not independent
anatomical accuracy. Pretrained resources are fetched and checksum-verified by
the resource stage before training; training does not silently fetch a different
checkpoint.

The highest within-acquisition weak mask AP50–95 selects one model per camera,
including the blue-channel ablation as a candidate. Inference uses confidence
0.25, NMS IoU 0.7 and at most 12 detections per image. Overlap and circular
trackers operate independently in each camera. Official ByteTrack receives the
same prefiltered observations. Local prediction IDs, track length and nominal
harmonic residuals are diagnostics, not verified identities, botanical counts or
longitudinal tracking accuracy.

## Cross-camera and spatial observation

Images are paired by acquisition identifier and filename index. Shape-profile
phase diagnostics are retained without silently shifting pairings; they do not
prove synchronization. Affine registration uses moment initialization, ECC
refinement with 80 iterations and tolerance `1e-5`, and retains refinement only
when it improves whole-mask IoU. No transform is refit to improve a color index.

Instance proposals use IoU at least 0.25. Qualification requires whole-mask IoU
at least 0.65, leaf-mask IoU at least 0.50, and mutual assignment margins at least
0.10. Pixel hypotheses further require at least 40 common interior pixels and
common-interior fraction at least 0.35. A 3 × 3 erosion removes uncertain
boundaries. These thresholds are engineering gates, not calibrated anatomical
accuracy. Unmatched and rejected observations remain in the exports.

Recorded DN values at or below 1 or at or above 254 are excluded. Pairwise
normalized differences use a minimum channel sum of 12 DN. The spatial extension
requires all nonzero-weight NoIR interpolation contributors to satisfy source
validity and keeps a separate validity mask for every formula. Numerical values
outside plotted intervals are retained. There is no imputation.

The 27 spatial definitions comprise 15 unordered channel-pair normalized
differences, six RGB formulas, and six exploratory multichannel descriptors.
`index_definitions.json` records the exact algebra, channels, support and fixed
display ranges. For each acquisition the report shows nominal indices 0, 10, 20
and 30, plus an additional highest-joint-support view when appropriate. That
selection is an availability diagnostic, not a typical biological example.
Numerical coverage retains every pair. Pixels and views are not biological
replicates; acquisition-balanced summaries preserve missing signal.

The support perturbation uses offsets `(0,0)`, `(±2,0)`, `(0,±2)`, `(±4,0)` and
`(0,±4)` pixels. Relative NoIR DN scales are 0.5, 0.75, 1, 1.25, 1.5 and 2.
These are counterfactual sensitivity tests, not exposure correction. The
camera-pair challenge uses ten anchors, indices 0:4:36, nominal pairs, +10 and
+20 index shifts, and a deterministic different-genotype/same-round bijection.
Every condition gets identical moment/ECC opportunities. Negative-pair pass
rates test capture-pair specificity; they are not biological false-positive rates
or proof of nominal anatomical correspondence.

Cache reuse binds actual image, mask, prediction, phase and configuration hashes
plus the complete local helper-source import closure. A recorded source hash
mismatch, changed signature or altered cached payload is rejected. This binding
is computational and does not require an old delivery archive.

## Virtual removal, repositioning and beam accounting

The virtual assay uses opaque double-sided triangle surfaces under nine designed
beams: zenith plus elevation 45° at azimuths 0:45:315°. Removal leaves retained
receiver geometry unchanged. Receiver-relative exposure is normalized by its
isolated projected footprint for each direction; whole-shoot quantities use the
declared common height gauge. Equal-direction weights are a designed convention,
not a measured illumination distribution.

Removal/subset experiments run 256- and 512-pixel grids and record numerical
differences and allocation budgets. Source-support and view-sampling stages are
posthoc computational investigations, not evidence of prospective acquisition
savings.

Repositioning rotates local L4 while preserving L2 and all retained geometry.
Yaw runs from −60° to +60° in 10° steps; elevation from −30° to +30° in 5° steps.
Both sweeps run all nine beams, two grids and every configuration accepted by
the geometry-family stage. A fixed domain encloses every configured pose. The
two sweeps contain a duplicate zero pose. For K accepted configurations the
export contains 468K rows and 450K distinct configured states. The historical
family had eight configurations; a fresh fit can accept a different family
under the same criteria, and its IDs and count are retained in the summaries. The full sweeps are retained rather than selecting a
beneficial angle. Reversibility, unchanged receivers and beam-budget identities
are checked separately from biological interpretation. There is no contact,
collision, elasticity, radiative transfer or physiological model.

## Known-answer recovery

The original cohort has eight cases: three or five supplied leaves, cubic or
out-of-family wavy truth, each clean or perturbed. A separately designed
exploratory distichous cohort adds four five-leaf cases. The experiment supplies
known cameras, nominal angles, local organ IDs, a straight stem axis, and four
complete projected midrib paths including hidden portions. Truth controls are
excluded from initialization and fitting. These privileged observations make it
a component recovery experiment rather than an end-to-end validation of raw
image segmentation, discovery, correspondence and calibration.

The synthetic grid is 128 × 160 with 16 views. Twelve views fit the model and
four interleaved views evaluate clean-mask agreement. Assisted paths use indices
0, 4, 8 and 12. The original observation seed is
`20261024 + 100*leaf_count + 10*(wavy_truth)`; the stress cohort uses its separately
declared seed schedule. Fitting runs 350 fixed Adam steps, with control/attachment
learning rate 0.002 and width/twist learning rate 0.012. Fit surfaces use 32 × 5
samples; evaluation uses 64 × 9. The last iterate is retained without selecting
on evaluation outcomes.

Perturbation combines path noise with standard deviation 0.65 pixels,
alternating one-pixel mask dilation/erosion, and a fixed rectangular occlusion on
every third fitting view. Geometry errors, clean withheld-mask agreement and the
directed neighbor-release endpoint are reported separately. Improved image
agreement is not assumed to improve endpoint recovery. All configured failures
and outcomes must remain visible.

## Optional prediction comparisons

The `learning` stages add the earlier unvalidated color-feature baseline, frozen
DINOv2 vectors, nested grouped phenotype/spectrum prediction and fixed posthoc
metadata/geometry/optical controls. They address different generalization
questions from the architectural assay. Training-only standardization, grouping
and native UVIS/NIR wavelength grids must be preserved. These comparisons do
not establish physiological specificity, calibrated chlorophyll, treatment
transfer or independent biological validation.

Model versions and immutable revisions are machine-readable resources. Seeds
and parameter schedules identify the intended computation; floating-point
hardware/library differences can still affect numerical optimizers and raster
boundary decisions. Verify numerical invariants and saved metric replay, and
record any departure from the prescribed case rather than relabeling it as an
identical reproduction.
