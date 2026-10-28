# Learned-pose diagnostic

`audit_pose.py` checks saved three-view depth/pose predictions from the earlier
debug experiment. It compares reprojections between the three training cameras
and records learned relative rotations. Training reprojection is a consistency
check, not held-out accuracy or physical camera calibration.

The original debug subset is fixed in the script: genotype Os6251/15, replicate
1, treatments C/P/S, visits 1–5, RGB and NoIR, with source indices 0/10/20.
Saved depth arrays and their adjacent JSON source bindings must already exist
under `RUN/depth_trial`. The intake must contain
`segmentation_triage/view_diagnostics.csv` and its source masks. This program does
not create depth predictions or expand that subset to the cohort.

From this directory, with the repository's base dependencies installed:

```sh
python audit_pose.py --help
python audit_pose.py --intake /path/to/intake --run /path/to/depth-run --operators ../../../src/csi_maize/_methods --output /path/to/new-pose-audit
```

The operator path above is relative to this directory. Inputs and output paths
can also be absolute. No PyTorch, checkpoint or model download is needed to audit
already saved arrays. The isolated development runner covers syntax, public
imports and `--help`; this branch has no independent numerical unit-test suite.
