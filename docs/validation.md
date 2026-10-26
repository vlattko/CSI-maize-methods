# Validation scope

Software verification and biological validation answer different questions. The numerical suite checks projection, renderer coverage, depth ownership, fixed-receiver accounting, interpolation guards and reproducible execution. The controlled example supplies cameras, organ labels, foreground and projected paths; its successful fit does not validate automatic anatomical discovery.

## Current execution status

The controlled CPU example and numerical suites have passed locally on Windows with Python 3.12. An original-data attempt exposed a junction-path integration defect, which was corrected and regression-tested. A corrected fresh run of the publication and optional learning stages is in progress. Real-image geometry fitting, weak-label training and all downstream experiments have not yet completed together in one verified run. There is no completed end-to-end original-data reproduction claim. This status must be updated from a completed run manifest, rather than inferred from historical results or a passing CPU example.

## Execution levels

| Level | Inputs | What it checks |
|---|---|---|
| Unit/invariant suite | Analytic geometry, small arrays, temporary workbooks and files | Defined numerical and provenance behaviour, including failure cases |
| Controlled end-to-end example | Procedural RGB/NoIR images and explicit truth inputs | Image re-reading, descriptors, actual inverse updates, registration, spatial maps, removal/reposition and report generation |
| Raw study protocol | Original images/workbooks, supplied observations, pinned pretrained models | Complete declared study computation, with stage-specific logs and result identities |
| Stored-result inspection | Verified derived outputs | Numerical integrity and self-contained presentation; not fresh raw recomputation |

## Independent metadata check

The new workbook producer was compared with the existing reference tables using all 14 original workbooks. All 360 reference rows match after path normalization, including the scientific columns. The 73-row variable dictionary and workbook-hash table are byte-identical. All 255 UVIS inconsistency records retain their identity/wavelength/cell evidence and finite numerical errors; an undefined infinite error is now strict JSON `null` with a reason. Fresh source/session manifests are produced from the configured folders.

## Regression coverage

The suite includes oversized projected triangles, exact and near depth ties, per-organ nearest-layer preservation, output/source tampering, missing or altered checkpoints, truncated event history, unsafe source bindings, changed runtime environments and false-success analysis summaries. Valid in-bounds triangle outputs and gradients were compared with the original numerical function on CPU and CUDA. Saved manuscript scenes satisfied the original small-triangle condition; the new guard is not evidence that their historical results were wrong.

The controlled example executes three- and five-leaf cases without a model download. It retains foreground, camera and organ-label assistance explicitly, saves procedural input hashes and runs the same numerical operators used by the method. A CUDA-only parity test may be skipped in the clean CPU suite; the test report must state that skip.

## Interpretation boundaries

- Source-assisted real-image surfaces are hypotheses, not independently measured anatomy or metric leaf dimensions.
- Development-view image consistency has been inspected during method development and is not an untouched anatomical test set.
- Synthetic recovery supplies privileged paths and identities. Its endpoint errors must be evaluated separately from image agreement.
- Equal beam weights and finite geometry families are designed sensitivity scenarios, not measured daily irradiance or biological posterior distributions.
- Camera-DN descriptors are not calibrated reflectance, narrow NIR bands, chlorophyll measurements or fluorescence predictions.
- The source-contributor validity used for spatial maps is stricter than the earlier post-interpolation target summaries; the two availability counts are not interchangeable.
- Neither a completed computational run nor agreement between weak teachers establishes persistent physical leaf identity or corresponding tissue.

Data and reference outputs will be made available via Zenodo after publication. The deposit will supply permanent identifiers. Until that record exists, the repository makes no claim of a published complete dataset.


## Test accounting and CI artifacts

Local source verification passed 117 public pytest tests and seven subtests. The canonical CPU suite executed through the bridge recorded 128 tests, 127 successful tests, one CUDA-unavailable skip and four successful subtests, with no failures or errors. These counts describe different discovery levels and are not added together.

The public pytest suite contains a bridge that discovers every shipped canonical `test_*.py` module and explicitly includes `v5_pair_controls_test.py`, whose filename is outside that discovery pattern. It writes strict JSON with discovered/run/successful test counts, individual skips and reasons, failures/errors, successful/failed subtest counts, module names, elapsed time and Python/PyTorch/CUDA availability. A skipped CUDA parity case is reported explicitly and is not included in the successful-test count.

Set `CSI_MAIZE_CANONICAL_SUMMARY` to a destination JSON path to keep that record outside pytest's temporary directory. CI uses `runs/ci-canonical-summary.json` and also uploads `runs/ci-pytest.xml`. Both are retained alongside the installed-wheel controlled example's report, verification and run manifest. The JSON summary remains available if the canonical suite fails; a failure before test collection can prevent creation of that artifact and does not establish any numerical-test count.

The CI workflow targets Python 3.12 on Windows and Ubuntu CPU runners. The Windows CPU requirements snapshot was checked locally; the Ubuntu job is a portability check whose execution is established by its CI result, not by this document. CI builds the wheel and source archive, checks bundled web/model-registry assets and PEP 639 license metadata, installs the wheel, and runs the controlled example through an isolated installed-package import. No original images or pretrained models are required for that example. A source-only test run does not establish that packaging checks or GitHub CI have run.

## Environment scope

`requirements/cpu-py312.txt` records the locally checked CPU environment and `requirements/cuda128-py312-windows.txt` records the Windows CUDA 12.8 scientific dependency closure. The dependency-metadata audit checked the reachable CPU/test and Windows CUDA publication/learning dependencies against the pinned files and found no missing or unsatisfied constraints. These files use regular `opencv-python`, consistently with package metadata and Ultralytics. Do not install a second headless OpenCV distribution into the same environment.

The CUDA snapshot excludes unrelated document-authoring packages. A fresh installation of this lock in an isolated Python 3.12 environment passed `pip check` and completed the four-stage original-data SAM comparison workflow: workbook ingestion, indexing, verified model preparation and real-image inference. Its 109 generated outputs and original inputs passed manifest verification. This check is separate from the complete publication/learning run, which remains in progress. The version snapshots do not contain wheel hashes or pin a particular NVIDIA driver and are not a promise of bit-identical computation, including repeat inverse fits on the same machine.

## Refitting and stored-model verification

Two fresh inverse fits were compared using matching raw inventories, all-view SAM masks, observation paths, fitted cameras, numerical source snapshots and recorded package versions. The resulting leaf parameters were not bit-identical. Small surface changes can affect directional screening more strongly than an average image-agreement score suggests. This comparison is a computational repeatability check, not independent anatomical validation or an estimate of biological variation.

A run manifest identifies the particular source, inputs and fitted states used for its virtual experiments. Verification of that stored execution is different from obtaining identical parameters in a new optimization. Report actual accepted candidate IDs and endpoint ranges, retain fitted parameters with published results, and inspect endpoint stability separately from mask agreement. We do not attribute the observed differences to a specific numerical component without further evidence.

## Declared-output audit

All 69 output paths declared by the numerical stage catalog were found in the original local evidence archive. The missing-output list was empty. For this archive comparison, baseline/index outputs were located under the original `prototype/outputs`; the portable producers correctly emit them under the materialized research output root. This presence audit checks producer/path consistency only. It does not certify that the new source has recomputed those outputs or that an output satisfies scientific validation.

The models hook declares no static stage-output list; its independently verified model-asset manifest is generated dynamically. The architecture report is an additional CLI terminal outside the numerical catalog and is covered by report/example integration tests.
