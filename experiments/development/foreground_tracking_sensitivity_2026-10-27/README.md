# Foreground substitution through cyclic tracking

This experiment substitutes a frozen fast foreground provider into the existing cyclic tracker on the exact teacher-eligible RGB source set. It consumes completed outputs and performs no image inference, assay prediction, model selection or anatomical scoring.

```
python -m unittest discover -s THIS_FOLDER -p test_sensitivity.py -v
python tracking_sensitivity.py --full-run FULL_COHORT --teacher-tracking TRACKING_RUN --cache COMPLETED_CACHE --output NEW_OUTPUT --tracking-source TRACKING_SOURCE --full-source FULL_COHORT_SOURCE
python audit_sensitivity.py --output NEW_OUTPUT --teacher-tracking TRACKING_RUN
```

The tracking source folder contains tracking_development.py, tracking_graph_core.py and tracking_observations.py; the full-cohort source folder supplies their unchanged architecture imports. All paths are explicit. Scientific dependencies are NumPy, SciPy, scikit-image, Pillow, OpenCV and threadpoolctl in the existing methods environment. No GPU, downloads or additional model weights are needed for this substitution experiment.

All 339 teacher-executed RGB acquisitions and their 13375 teacher-accepted source views were retained. Fifteen corresponding student masks were truly empty. The teacher baseline had 642 terminal hypotheses across 305 acquisitions; the student produced 608 across 292. Every first-twelve teacher fit replay matched the frozen coefficients. Across the full cohort, the frozen teacher held-out denominators and compatibility counts reproduced exactly. The output keeps 72 metadata candidate graphs and all 360 visit slots; candidate identities do not establish physical plant continuity.

The teacher has 3852 compatible observations out of 6338 source-available held-out hypothesis-view rows. The student has 3691 out of 6016. A slightly larger conditional fraction would not establish improved accuracy: the output sets and coverage differ. The median symmetric projected-point distance on the 2860 held-out rows where both models produced hypotheses was 0.048313 teacher training-height units. The remaining held-out set states remain explicit, including 190 teacher-only, 60 student-only and 280 both-empty rows. Empty sets never receive zero distance. Coordinates remain conditional on nominal rotation and an image-derived orthographic gauge.

Files include per-acquisition coverage, all planned held-out node states, per-angle setwise comparisons, frozen training-only fits and source-linked student graphs with forty angular evidence records per node. The graphs describe model hypotheses, without anatomical leaf IDs, death claims, calibrated geometric uncertainty or physiological light absorption. All outputs are sensitivity to the foreground operator; this full cohort includes training and validation acquisitions and is not a new independent accuracy test.
