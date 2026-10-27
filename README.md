# CSI-MAIZE

**Data and reference outputs will be made available via Zenodo after publication.**
The deposit link will be added here.
<!-- ZENODO_RECORD_URL: replace this line with the verified public deposit link. -->

Circular maize imaging and leaf-scale architectural experiments.

**Vlatko Galic and Maja Mazur** · Agricultural Institute Osijek, Croatia

These methods hold one leaf fixed, remove or rotate a neighbouring leaf, and calculate the change in projected exposure under prescribed parallel beams. Image agreement, known-answer geometry recovery and neighbour effects are evaluated separately. Registered RGB–NoIR images and guarded camera-DN descriptors provide supporting observations.

## Run the controlled example

Use Python **3.12** in a new environment. The example needs no dataset, pretrained weights or network connection after installation.

```sh
python -m venv .venv
# Activate .venv using your operating system's usual command.
python -m pip install -r requirements/cpu-py312.txt
python -m pip install -e ".[test]"
csi-maize example --output runs/example --steps 30
csi-maize verify --run runs/example
```

Open `runs/example/reports/CSI_maize_architecture.html`. The HTML embeds its mesh, figures and supporting atlas. It uses procedural plants with explicitly supplied foreground, organ identities, cameras and paths. It exercises the numerical pipeline; it does not stand in for a trained segmentation model or real-plant validation.

## Full cohort and external component tests

The [full-cohort protocol](experiments/full_cohort/README.md) processes all keyed acquisitions, including partial rotations, with explicit observation coverage. It includes grouped prediction controls, registered channel descriptors and known-geometry tracking and exposure tests. Its [evidence and limitations](docs/full-cohort.md) distinguish component validation from automatic anatomical reconstruction.

## Recompute the source assisted study from its inputs

The raw-data route requires the original images, spreadsheets and supplied observation seeds, plus the pinned model assets. The Zenodo link above will identify the archival inputs after publication, when the deposit is available. We do not substitute cached results for a raw run.

```sh
csi-maize init --data-root /path/to/data --work-dir runs/publication --models-root /path/to/model-cache
csi-maize assets --config runs/publication/run_config.json --download
csi-maize doctor --config runs/publication/run_config.json
csi-maize plan --config runs/publication/run_config.json
csi-maize run --config runs/publication/run_config.json
csi-maize verify --run runs/publication
```

Install the `learning` extra and the appropriate CUDA PyTorch build before this route. The complete raw workflow requires an NVIDIA CUDA device; the controlled example runs on CPU. [Reproduction instructions](docs/reproduction.md) cover CPU/CUDA environments, offline execution, individual stages, interruption/resume and the expected input layout. Asset downloads occur only through the explicit preparation command; analysis stages reject outbound network access.

Each run uses a separate workspace. Its manifest records source code, configuration, raw inputs, dependency versions, checkpoints and output hashes. Resume verifies these records before reusing results. Source folders are registered separately from generated outputs. Fresh inverse fits need not be bit-identical, even with the same inputs and recorded package versions. Retain the fitted parameters and manifest when referring to a particular virtual experiment.

## Methods and verification

- [Method stages and assumptions](docs/methods.md)
- [Workbook identities, spectra and quality flags](docs/metadata.md)
- [Reproduction and environment setup](docs/reproduction.md)
- [Validation and release evidence](docs/validation.md)
- [Third-party software and model licences](THIRD_PARTY_NOTICES.md)

JavaScript interaction checks use Node.js 22 and its standard library.

```sh
python -m pytest tests -q
node tests/web_behavior.cjs
python -m build
```

The tests include known-answer geometry, ownership accounting, interpolation validity, metadata joins, cache invalidation and a complete controlled example. They do not establish anatomical accuracy, persistent physical leaf identity, calibrated spectral response or physiological prediction. The prescribed real-data geometry demonstrations remain source-assisted case studies.

Original code is MIT licensed. The optional Ultralytics workflow and its weights retain the upstream AGPL/Enterprise conditions. Cite this software using [CITATION.cff](CITATION.cff); the dataset citation will be added after deposit.

This research was funded by the NextGenerationEU project CSI-MAIZE: Innovative strategies for early stress detection and increasing maize tolerance to cold stress.
