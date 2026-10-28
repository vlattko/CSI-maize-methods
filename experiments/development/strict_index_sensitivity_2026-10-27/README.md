# Strict paired-index prediction sensitivity

This separate investigation reruns the frozen registered-index protocol after
correcting the bilinear contributor-validity tolerance. It reads small numeric
tables and source hashes; it does not copy the image archive, refit registration
or alter the original experiment.

The independent channel audit supplies 1,212 accepted and 58 rejected nominal
ECC anchors, with 170 unavailable slots unchanged. All 34,290 available formula
rows are replaced, including the rejected anchors' unimodal descriptors. The
matched DN control uses the 7,272 `common_interior` audit rows, not camera-interior
channel summaries. Before replacement, both frozen session tables must replay
exactly. All 360 four-key rows and the 329-row primary cohort remain present.

The original five-alpha grid, three null seeds, grouped outer partitions,
training-only imputation/scaling and inner genotype tuning are executed through
hash-verified frozen `prediction.py` and `prediction_indices.py` copies. No new
grid is selected. UVIS flag sensitivity retains the same 328 records; other
cases retain 329. The 12 experiments save fresh OOF arrays and checkpoint hashes.

From this directory in the base scientific environment, with at most two CPU
threads:

```sh
python -m unittest test_strict_features -v
python strict_run.py --intake /path/to/intake --run /path/to/original-full-cohort-run --repo /path/to/methods-checkout --audit /path/to/channel-validity-audit --protocol-source /path/to/original-frozen-protocol-source --output /path/to/new-strict-output
python strict_verify.py --intake /path/to/intake --run /path/to/original-full-cohort-run --output /path/to/new-strict-output
python strict_report.py --output /path/to/new-strict-output
```

The `--protocol-source` folder supplies the original aggregation helpers, whose
hashes must match the archived run. Use the archived originals when later public
numerical corrections differ; the current public helper's strict-guard source
hash is deliberately different from the original execution.
The predictor execution copies come from the original run. `--features-only`
builds the target-free corrected tables without fitting models. NumPy,
scikit-learn and threadpoolctl are sufficient; no GPU, checkpoint or download
is required.

Verification independently recomputes metrics, checks original/strict case and
fold identity, replays selected outer fits and verifies training-only null donor
maps. It does not independently repeat every inner grid candidate. Archived
wrong-pair and moment controls remain under their original operator scope.

The executed sensitivity changed some exploratory predictions materially:
PiAbs genotype cross-index R² −0.124795 to 0.070726; Fv/Fm genotype all-index R²
0.098898 to 0.013295. Metadata/SAM-only predictions remained bitwise identical.
The metadata+DN+cross PiAbs R² rose from 0.301162 to 0.411897, still below
metadata alone at 0.433867. Sparse recorded NoIR-green availability remains a
limitation. These retrospective representations do not establish calibrated
spectral or anatomical measurements.
