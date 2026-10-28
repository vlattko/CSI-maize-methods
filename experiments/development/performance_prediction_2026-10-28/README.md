# Circular-image harvest prediction

This study-specific benchmark compares projected area, circular image summaries,
growth history and supporting observations as predictors of one destructive
harvest per physical pot. The supplied CSI-maize grid has 72 pots and five visits
at 13, 19, 20, 22 and 25 days after sowing. Multiple images and repeated visits
are predictors, not additional independent harvest samples.

## Inputs

Prepare the original study inputs using the full-cohort and confirmed-identity
workflows. These programs require the following tables rather than cached model
scores:

- `confirmed_mass.csv`: the investigator-confirmed treatment/genotype/replicate
  crosswalk with fresh mass, dry mass and dry-matter fraction.
- `identity_registry.json`: the physical-pot registry for all 360 reference visits.
- `session_traits.csv` and `view_traits.csv`: camera summaries, per-view traits,
  source identities and mask quality from the full-cohort extraction.
- `reference_master.csv`: the non-destructive fluorescence and spectral records.
- `confirmed_pot_observations.csv`: the observation table for presentation
  synthesis, produced by the sibling confirmed-identity
  [`dynamics.py`](../longitudinal_confirmation_2026-10-28/dynamics.py) program.

The mass workbook must be mapped by the written genotype within each confirmed
six-genotype replicate block; internal row order differs between blocks. LAB
biochemical samples are from separate plants and are not joined to these pots.

## Recompute

Use Python 3.12 with the repository's base scientific dependencies. The numerical
programs do not download weights or run segmentation inference.

```sh
python run_performance.py --mass /data/confirmed_mass.csv \
  --registry /data/identity_registry.json --traits /data/session_traits.csv \
  --views /data/view_traits.csv --output /results/core

python run_extensions.py --features /results/core/pot_cutoff_features.csv \
  --traits /data/session_traits.csv --reference /data/reference_master.csv \
  --views /data/view_traits.csv --output /results/extensions

python ../longitudinal_confirmation_2026-10-28/dynamics.py \
  --registry /data/identity_registry.json --reference /data/reference_master.csv \
  --traits /data/session_traits.csv --output /results/observations

python synthesize.py --run /results/core --extensions /results/extensions \
  --observations /results/observations/confirmed_pot_observations.csv --output /results/synthesis

python -m unittest discover -v
python -m unittest discover -s verification -v
```

The independent verifiers in `verification` use scikit-learn without importing
the implementation. Keep both files together. They check source/input/output
hashes, every feature/cohort/fold/metric and a declared structural sample of
numerical fits, including inner tuning. For a relocated reviewer input set:

```sh
python verification/verify_new_run_sklearn.py --run /results/core \
  --source . --output /results/audit-core --input-root /reviewer/inputs
python verification/verify_extensions_sklearn.py --run /results/extensions \
  --source . --output /results/audit-extensions --final /results/core \
  --tables /reviewer/computed_exports --input-root /reviewer/inputs
```

Input rebasing preserves the original recorded keys and verifies their hashes.
It requires a unique filename match; missing or ambiguous files fail explicitly.
Numerical replay is an implementation check, not independent biological
validation.

Use a new output directory for numerical runs. The source, input and output
hashes accompany each run. Programs retain the complete configured score grid,
fold membership, selected penalties, inner errors and every pot's held-out
prediction. Presentation synthesis is separate from fitting.

## Prediction and observation policies

Features use only observations at or before the cutoff. The ratio uses the same
pot's 13-DAS area, without substituting a later baseline. The slope uses the
actual uneven observation ages and requires two qualified observations. All
history fields are omitted at 13 DAS. Single-view controls retain fixed indices
0, 10, 20 and 30; no phase is selected from its association with the target.

Current-availability and all-pot policies do not require future images. The
common-five-visit intersection deliberately uses future availability and is a
retrospective matched-population sensitivity, not an operational forecast rule.
Missing images do not imply zero mass. All-pot models use training-derived
imputation and missing indicators. Multiple correlated missing indicators can
alter ridge's effective regularization of observation-quality information.

Outer holdouts exclude an entire genotype, an entire treatment/run group, or
both simultaneously. The joint test trains on neither held-out group and leaves
the cross-arm observations unused. Inner partitions repeat the outer exclusion
principle. Imputation, scaling, category encoding and alpha selection are
training-only. The allometric reference additionally fits a training-only log
transform and residual smearing factor. Its target predictions return to the
original scale.

## Interpretation

Pooled R², subgroup bias, error and observation coverage answer different
questions. Treatment was administered in separate temporal runs in the same
chamber, so treatment and run are confounded. Group holdouts do not create an
independently collected test cohort. The analysis is retrospective model
development on a previously explored experiment.

Bootstrap contrasts resample pots within genotype/treatment strata and keep
predictions paired. They condition on saved cross-validation predictions and do
not refit models. Their intervals omit retraining and new-cohort uncertainty.
The shuffled-label model is one diagnostic, not a permutation significance test.

Camera fractions are visible projected support. NoIR channel summaries are
uncalibrated digital image measurements; the measured instrument NDVI is a
different observation. Whole-plant mass predictions do not validate anatomical
leaf identities, 3D surfaces, spectral calibration or a causal priming effect.

Original code uses the repository MIT licence. Data and reference outputs will
be deposited through Zenodo after publication; no private study files are
included in this source directory.
