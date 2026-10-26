# Reproduction

## Environment

The validated interpreter is Python 3.12. Create a fresh virtual environment; do not install a package into a raw-data directory. On Windows activate `.venv\Scripts\Activate.ps1`; on Linux/macOS use `source .venv/bin/activate`.

CPU numerical example and tests:

```sh
python -m pip install -r requirements/cpu-py312.txt
python -m pip install -e ".[test]"
python -m pytest tests -q
csi-maize example --output runs/example --steps 30
csi-maize verify --run runs/example
```

For the raw study workflow, install SAM/YOLO dependencies too:

```sh
python -m pip install -e ".[learning]"
```

The study's Windows CUDA 12.8 dependency closure is pinned in `requirements/cuda128-py312-windows.txt`. Install it in a separate Python 3.12 environment with a compatible NVIDIA driver, then install this repository. The CPU and CUDA lock files describe different environments; do not combine them. Platform-specific PyTorch builds and drivers must match. A CPU installation can execute numerical stages but is substantially slower for the complete image/model workflow.

The optional YOLO component has separate upstream licence conditions described in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Plotting uses bundled DejaVu Sans. Pretrained model binaries are not shipped in this repository.

## Raw input layout

The data directory is the extracted study archive, not the repository directory. It contains the original workbooks at its top level and the indexed capture directories, for example:

```text
data/
  Genotype_list.xlsx
  OJIP_P6_computed_data.xlsx
  UVIS_P6_computed_data.xlsx
  NIR_P6_computed_data.xlsx
  NIR_P6_transmitance.xlsx
  UVIS_P6_transmitance.xlsx
  ...remaining original workbooks...
  CSI-maize_podaci/IMAGES/
    Control1_13das/capture_session_C1_5-1/{rgb,noir}/
    Control2_19das/capture_session_C2_5-1/{rgb,noir}/
  Stress1_13das/
  Stress2_19das/
  about/
```

The complete workbook schema is described in [metadata.md](metadata.md). Keep native spectral grids and the four identity keys. Timestamp-only sessions require an evidence-backed identity mapping; chronological sorting is not a mapping. Laboratory/biomass records are not joined to image targets without a sampling crosswalk.

The prescribed study profile uses 41 eligible acquisitions, each with indices 0–39 from both cameras. `13das` and `19das` are documentary labels, not independently verified visit ages. Other named or unresolved directories remain in the inventory. Count-based eligibility is distinct from segmentation quality.

The real-image geometry demonstrations use the supplied distal seeds in two views and the case-specific camera/shape assumptions documented in [methods.md](methods.md). These inputs are explicit in the corresponding method modules. A new plant requires new observation review; changing a cohort count does not infer its organ identities.

## Initialize and inspect a run

```sh
csi-maize init --data-root /path/to/data --work-dir runs/publication --models-root /path/to/cache --device cuda:0
csi-maize doctor --config runs/publication/run_config.json
csi-maize plan --config runs/publication/run_config.json
```

Use quoted absolute paths when they contain spaces. `init` refuses a nonempty unowned work directory. It copies the workbooks, registers raw-image directory bindings and materializes an immutable source/runtime snapshot. The `runtime` and `prototype/research_v2` directories are generated execution infrastructure; users edit the public package and create a new run rather than editing those snapshots.

The method's dated output identifiers are compatibility names for the reference protocol. They do not assert an acquisition date and do not require a particular computer or external directory layout.

## Prepare assets explicitly

```sh
csi-maize assets --config runs/publication/run_config.json --download
```

This is the only command that permits model downloads. It fetches official pinned sources and verifies file sizes and SHA256 values before staging. Omit `--download` to require an existing complete cache. Modified existing assets are rejected rather than overwritten. The default publication assets include SAM 2.1 small, YOLO26 nano/small segmentation and the separate AMP-check detection checkpoint. The learning profile additionally needs DINOv2 small.

Analysis scripts run with outbound network access disabled inside their Python process. Local numerical/data-loader IPC remains available. Missing assets therefore fail at a declared boundary rather than triggering an incidental download. Model preparation neither executes model code nor loads pickle weights.

## Execute and verify

```sh
csi-maize run --config runs/publication/run_config.json
csi-maize verify --run runs/publication
```

The default terminal is `paper`, which includes raw metadata/indexing, SAM foreground/traits, assisted organ observations and fitting, geometry alternatives, virtual removal, source support, view budgets, weak-label model training/inference, tracking comparison, camera controls, channel/spatial fusion, known-answer recovery and reversible repositioning. It builds an architecture-first self-contained report at `reports/CSI_maize_architecture.html` within the run.

Each stage has declared prerequisites, commands and required outputs. A subprocess exit code alone is insufficient: failed/partial summary statuses are rejected. Logs are under `logs/`, and `events.jsonl` has an anchored hash chain. The run manifest records actual dependency versions, input/source identities, materialized configuration, checkpoint verification and derived-file hashes.

To select a terminal or inspect a branch:

```sh
csi-maize plan --config runs/publication/run_config.json --target synthetic-stress
csi-maize run --config runs/publication/run_config.json --target synthetic-stress
```

This executes only that dependency closure. A completed requested branch is not a completed full study. Use `--profile learning` at initialization to add baseline, frozen-descriptor, phenotype/spectrum and control comparisons to the `paper` terminal.

## Interruptions and reuse

```sh
csi-maize run --config runs/publication/run_config.json --resume
```

Resume checks the original input bytes, registered binding topology, copied workbooks/configuration, staged code, dependency environment, model weights, event history and recorded outputs before reuse. Changed inputs, code or dependencies require a new run. Partial and failed stages remain recorded; they are not promoted to completed analyses.

`csi-maize verify --run ... --outputs-only` validates stored results and execution records without requiring the original external image/model inputs. Its output states that original inputs were not verified. This is an inspection/replay boundary, not raw-data recomputation. `csi-maize report --run ...` builds a display from available outputs and labels missing stages explicitly; it does not execute missing analyses.

## Storage and execution time

Real-image execution needs the raw data, roughly 300 MB of declared pretrained assets, and space for masks, predictions, fitted geometry, logs and all numerical outputs. Runtime depends on GPU, image I/O and fitting settings. The controlled example is suitable for CI; the complete archive workflow is not substituted by that example. The current execution evidence is kept in [validation.md](validation.md).
