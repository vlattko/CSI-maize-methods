# Source-only development experiments

The programs in [`experiments/development`](../experiments/development/README.md)
extend the archived full-cohort analysis. Existing numerical outputs remain
unchanged; the declared strict interpolation correction applies to fresh 0.3
runs. They are retrospective research tools. No private inputs, generated
workbenches, model weights or reviewed anatomical annotations are included.

Use a source checkout or unpack the source distribution. These experiments are
included in the source distribution, not installed as application modules in the
wheel. Run commands from the repository root unless a branch README says
otherwise. Keep all dated sibling directories and `experiments/full_cohort`
together; imports use their relative positions.

## Environment and tests

Python 3.12 and the base package dependencies support the numerical camera,
kernel, tracking, perspective, sprites and saved-pose diagnostics. Node.js 22 or
later is required for the workbench tests. Install the test tools with:

```sh
python -m pip install -e ".[test]"
python experiments/development/run_tests.py --report runs/development-tests.json
```

The runner launches each branch in an independent Python subprocess, bounds
numerical threads to two and explicitly imports this checkout's `src`. This
avoids both generic module-name collisions and an unrelated editable install.
It also exercises public CLI imports with `--help`, compiles all copied Python
sources and checks that the two HTML files remain unfilled templates.

The base environment may omit PyTorch. Only suites with explicitly declared
missing optional dependencies are skipped, with the reason in the report.
Installed dependencies that fail to import cause a failure. To require the
optional CPU foreground and topology tests as well:

```sh
python -m pip install -e ".[geometry,test]"
python experiments/development/run_tests.py --require-optional --report runs/development-tests.json
```

Use the documented CPU or CUDA requirement files if a pinned environment is
needed. Installing dependencies is separate from downloading model assets. The
tests are offline, use synthetic fixtures and do not invoke model inference.
The leaf-review suite has 17 Node behavior tests, including incomplete-frame
export rejection, explicit verified-empty gating, source-hash validation and
native-coordinate editing. The rotation fixture tests the template using
synthetic identities and visits. Neither provides real-browser visual QA.

## Programs and input roles

| Directory | Program and purpose | Required existing inputs |
| --- | --- | --- |
| `camera_geometry_development_2026-10-27` | Shared session affine and a training-selected first-harmonic competitor | Full-cohort extraction records/masks, session reference and source intake |
| `cyclic_kernel_development_2026-10-27` | Pooled, ordered and finite-group circular kernel comparisons | Full-cohort reference/eligibility, frozen DINO quarter features and extraction records |
| `depth_pose_development_2026-10-27` | Audit saved three-view learned-pose outputs | Original debug intake table/masks and saved depth/pose arrays |
| `foreground_distillation_2026-10-27` | Whole-plant YOLO foreground distillation, cache and common evaluation | Prepared source/teacher manifests, local hash-bound initialization assets and frozen selection cards |
| `leaf_proposal_development_2026-10-27` | Bounded terminal-supported SAM2.1 region proposals | Tracking graphs, camera lookup, full-cohort inputs and an explicit local SAM checkpoint directory |
| `leaf_review_workbench_2026-10-27` | Build a portable native-image annotation tool | Verified proposal bundle and original source intake |
| `longitudinal_tracking_development_2026-10-27` | Conditional path/candidate association and external controls | Full-cohort masks/reference, intake and local Pheno4D scans |
| `new_session_pipeline_development_2026-10-27` | Inventory, masks, 39 traits, conditional tracks and a new-capture report | One explicit capture folder, session JSON and a local hash-bound foreground model card |
| `perspective_turntable_development_2026-10-27` | Conditional perspective/orthographic point-model comparison | Full-cohort extraction and completed tracking output |
| `rotation_workbench_2026-10-27` | Source sprites and a forty-index inspection workbench | Full-cohort inputs, tracking graphs and optionally camera lookup |
| `topology_foreground_development_2026-10-27` | Matched U-Net foreground arms with and without soft-clDice | Explicit prepared frame manifest, unchanged dense teacher masks and selection gate |

Each directory has a README and its executable programs accept explicit input
and output paths. For example:

```sh
python experiments/development/camera_geometry_development_2026-10-27/camera_geometry_run.py --help
python experiments/development/cyclic_kernel_development_2026-10-27/benchmark.py --help
python experiments/development/leaf_review_workbench_2026-10-27/build_workbench.py --help
```

The SAM proposal runner requires the `learning` extra, including PyTorch and
Transformers, and an already cached compatible SAM2.1 checkpoint. Its model load
uses local files only. It exposes physical GPU 0 as in the executed pilot; the
helper tests need no GPU or model. Topology training/inference requires PyTorch
but no pretrained checkpoint; its unit tests construct a small model on CPU.
Saved-pose auditing loads NumPy arrays, not a depth model.

Foreground training/inference requires the pinned Ultralytics learning runtime
and explicit local initialization assets. Its `--repo` argument denotes a
prepared outer workspace, not a bare public checkout: it expects segmentation
initialization at `prototype/research_v2/yolo26n-seg.pt` and the checked AMP asset
at `prototype/weights/yolo26n.pt`. The application package must be installed or
available below `CSI-maize-methods/src`. Stage the verified assets explicitly
under that layout; this source release supplies none. Direct inference takes
`--weights` explicitly. In the public cache producer, a relative model-card
`weights` name is resolved beside the card and its SHA256 is checked. This is a
public portability correction, not a change to the frozen research execution.
The added regression covers relative/absolute names and changed checkpoint bytes.

The new-session CLI accepts `--repo` pointing directly at this public source
checkout. Its base tests use a controlled known-mask provider, explicitly bind
the public trait kernel and need no checkpoint or PyTorch. Real U-Net inference
requires the PyTorch extra and the sibling topology provider; optional YOLO
inference requires the learning runtime. Their model-card thresholds retain
distinct pixel-probability and detector-confidence semantics. The capture does
not require the old study inventory, assay workbook or genotype map. Missing
declared slots remain explicit; a nonuniform or non-40-view capture can still
produce traits while its current tracker abstains.

## Scientific and annotation limits

Camera lookup matrices map an explicit RGB grid into an explicit NoIR grid.
Deployment abstention and nominal baselines remain available. Training angles
select phase, sign and complexity; the withheld-angle adaptive ECC comparator
uses the withheld masks and is reported separately. A silhouette overlap is
not tissue correspondence, physical camera calibration or spectral calibration.

The public version 0.3 paired-channel validity correction is separate from the
frozen version 0.2 research executions. The earlier `> 0.999` tolerance could
admit a clipped or dark bilinear contributor with weight `1/1024`; corrected
paired-descriptor, matched-DN and image-field guards require `>= 1 - 1e-7`.
Nearest-neighbor behavior and the already stricter spatial channel guard are
unchanged. A known-answer test uses a `1/32`-pixel shift in both axes and checks
that the affected cross-channel formula is missing while unaffected formulas
remain available. A separate border test distinguishes bilinear support from
nearest-neighbor support. Historical cohort numbers retain their original
operator scope; corrected reruns require separate hashes and results. No frozen
execution source or old output is overwritten by this public correction.

The circular kernel is invariant to the eight tested quarter shifts/directions,
not arbitrary rotations. Its null shuffles training trajectories within matched
strata and is one diagnostic seed, not a significance test. Perspective and
tracking outputs remain conditional image hypotheses. The tracking graph does
not authenticate physical identity across acquisition dates.

SAM outputs are unreviewed regions. A complete visible-blade label requires a
human decision, and a training image must also be explicitly complete-reviewed.
Partial regions and unreviewed images cannot become implicit negatives. The
portable review tool saves audit JSON separately from eligible reviewed COCO.
Its source-hash guards protect resume; no real rater decisions are bundled.
Original private source assets belong beside a generated workbench only in an
authorized private workflow, never in a public source release.

Foreground models are compared with fixed teacher masks. Their pixel
probabilities, SAM proposal scores and YOLO object confidences have different
meanings. Dense-mask versus polygon teachers are a representation confound in
operational comparisons; these branches do not isolate a network-architecture
effect or establish expert leaf truth.

## Provenance and distribution

`experiments/development/source_provenance.json` records the original relative
source names and SHA256 hashes alongside the public copies. Portability changes
and added documentation are listed explicitly. Original execution snapshots
remain in the private research archive; this release contains current readable
source, not cached claims of a new rerun.

The source distribution includes Python, JavaScript, HTML templates, READMEs and
licenses. Weights, images, masks, raw scans, generated HTML payloads, execution
snapshots and internal plans are excluded. Development CI uses the isolated
runner; the main package/full-cohort tests remain separate. New-session capture
is included as source after its frozen code and three real smoke captures were
independently checked; those private captures and model assets are excluded.
Project code uses the root MIT license. Third-party model and data terms remain
separate; consult `THIRD_PARTY_NOTICES.md` before preparing external inputs.

## One capture and confirmed longitudinal context

The [single-capture bridge](../experiments/development/capture_fusion_development_2026-10-27/README.md)
connects the foreground CLI, 39 image traits, conditional terminal hypotheses,
training-angle camera selection and 27 spatial channel formulas. Its runner
keeps GPU inference and CPU geometry in separate processes, records stage
manifests and produces offline reports. The three U-Net smoke captures and a
complete CPU YOLO run were source/mask checked independently. Their private
images, weights and generated pages are not part of this source release.

The [identity overlay](../experiments/development/longitudinal_confirmation_2026-10-28/README.md)
records the experimenter's confirmation that repeated treatment/genotype/replicate
labels identify the same pots. The current 72-pot study uses folder-declared
ages of 13, 19, 20, 22 and 25 DAS. Raw timestamps remain unchanged. This
confirmation supports same-pot comparisons, not anatomical leaf identity.
The newly supplied timetable places priming at 6–11 DAS and the severe challenge
at 14–19 DAS, with measurements before warming on day 19. The first image follows
priming for P. The priming-night footnote conflicts with the worksheet cells and
experimenter; both values remain recorded. Scheduled dates differ between
treatments and may confound treatment with calendar run. They do not replace
camera timestamps; exact execution hours still require the complete method.

Absolute and first-observation-relative trajectories preserve missing values
and unequal elapsed days. Image support fractions are not calibrated growth
measurements. Treatment summaries use within-genotype whole-pot bootstrap
resampling, retaining the five visits of each sampled pot together.

The same branch now includes read-only timetable extraction and a confirmed
whole-aboveground fresh/dry mass crosswalk. Genotype order is read independently
within each six-row replicate block. The endpoint benchmark compares fixed image
shape summaries, temporal summaries, metadata and mean baselines under nested
group-disjoint validation. Its independent verifier checks original workbook
cells and replays the preprocessing, inner tuning and outer predictions using
explicit linear algebra. Generated data, confirmations and reports remain
private; they are not bundled in this source release.

## Corrected channel validity and downstream sensitivity

The [validity audit](../experiments/development/channel_validity_audit_2026-10-27/README.md)
reproduces archived statistics before applying the strict contributor guard.
Across 1,270 available nominal anchors, 20 formula/view records lose their
minimum-pixel qualification. Some sparse predictors and selected regularization
values consequently change; the correction is not numerically negligible in
every model. The [strict sensitivity protocol](../experiments/development/strict_index_sensitivity_2026-10-27/README.md)
recomputes the same 12 grouped prediction experiments with corrected channel
and formula summaries, retaining the original splits and hyperparameter grid.
Original results remain associated with their archived source version.

The [foreground tracking comparison](../experiments/development/foreground_tracking_sensitivity_2026-10-27/README.md)
substitutes source-bound masks on the exact teacher-accepted view set. It retains
empty masks and compares coverage as well as conditional image agreement.
The held-out U-Net comparison reuses frozen test predictions and checks their
pixel metrics before tracking, with no new inference, threshold selection or
anatomical accuracy claim.
