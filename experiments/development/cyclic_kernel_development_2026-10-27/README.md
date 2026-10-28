# Circular image-feature comparison

This CPU experiment compares pooled, ordered and finite-group-averaged image kernels on frozen DINO quarter-view observations. The circular kernel averages four quarter shifts and both directions, giving invariance to those eight group actions. It does not provide arbitrary-angle invariance or infer a camera pose.

We fit camera-shared feature scaling, missing-view imputation, ridge regularization, kernel width and metadata weighting within the training folds. The outer partitions hold out a genotype or treatment. Inner validation holds out genotypes. All visits and replicate labels for the held-out group remain excluded from training.

The primary cohort contains 329 paired eligible acquisitions. A sensitivity analysis contains 282 acquisitions with all eight quarter-view observations. One training-only null permutes entire image trajectories within treatment and matching visit patterns. It retains fixed points and singleton strata and is not a significance test.

```sh
python benchmark.py --intake /path/to/intake --run /path/to/full-cohort --output /path/to/new-output --cohort primary
python benchmark.py --intake /path/to/intake --run /path/to/full-cohort --output /path/to/new-output --cohort complete
python -m unittest discover -v
```

Inputs are the full-cohort reference and eligibility tables, frozen recorded-camera DINO features, their extraction records and original source bindings. The runner checks feature/record hashes and records its source and configuration. Resume checks cached prediction bytes and held-out indices. Arbitrary group labels are mapped to hashed filenames.

Outputs include every out-of-fold prediction, per-group selected settings, metrics, execution source, configuration and progress. Independent audit reports in the research outputs replay all selected predictions, verify input and output hashes and reconstruct train-only null mappings. The initial primary execution used label-derived subdirectories for two genotype labels containing slashes; those contained directories were prepared operationally. Its execution source remains preserved separately. The corrected runner uses hashed group filenames for subsequent runs.

Circular structure did not consistently improve prediction over pooled features in either cohort. Treatment transfer remained weak. The shuffled control can match or exceed some measured-image models. These are retrospective development results from an already inspected cohort, not a new external test or evidence of calibrated physiological imaging.
