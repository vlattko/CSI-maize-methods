# Inspect one new capture

This CLI reads one capture folder and explicit session metadata. It requires no
study inventory, workbook, genotype mapping or assay table. Raw data stay read-only.

```json
{"sample_id":"my-shoot-visit1","plant_uid":null,"steps":40,"cameras":{"rgb":"rgb","noir":"noir"}}
```

Images use a trailing zero-based numeric index, such as `image_rgb_000.jpg`.
Missing slots remain missing. Duplicate indices, escaped paths and overlapping
camera directories are errors. An `angles_degrees` array can replace nominal
uniform angles. Traits and the report support30 or other declared step counts;
the current tracker explicitly abstains unless the rotation is uniform40-view.
Optional metadata and assays are carried only when supplied. No plant UID,
physical continuity or assay values are inferred.

```powershell
python run_capture.py --input CAPTURE --session session.json --output NEW_RESULTS --model-card foreground_model/model_card.json --device cuda:0 --repo PUBLIC_REPOSITORY
```

The default provider is the frozen foreground U-Net. A model card binds its
checkpoint hash and pixel probability threshold. Relative checkpoint paths are
resolved against the card directory. `--provider yolo` instead requires a card
with a detector confidence, class IDs and weight hash; it uses the checked
offline adapter. `--confidence` and `--pixel-threshold` cannot be interchanged.
The three complete smoke captures exercised the U-Net provider, not optional YOLO.

Required source layout, also valid under `experiments/development/`:

- This folder contains the CLI, provider, report and tests.
- Sibling `topology_foreground_development_2026-10-27/` supplies
  `infer_topology.py` and `topology_core.py`; `--provider-module` can override it.
- Sibling `longitudinal_tracking_development_2026-10-27/` supplies
  `tracking_graph_core.py` and `tracking_observations.py` for40-view tracking.
- `--repo` supplies unchanged `src/csi_maize/_methods/traits.py` and binds the
  public package before optional YOLO adapter import. Otherwise the installed
  public package supplies the numerical trait kernel.

Outputs include fixed-slot inventory and records, fully decoded source hashes,
mask hashes, the39 public traits, per-camera coverage and summaries, conditional
terminal points and path evidence, execution source snapshots and a self-contained
`report.html`. Structural usability is an engineering check independent of model
confidence. It does not validate anatomy or segmentation against human labels.

```powershell
python -m unittest test_capture -v
python audit_capture.py --capture NEW_RESULTS
```
