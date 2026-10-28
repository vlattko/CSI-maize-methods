# Longitudinal tracking development

This executable investigation extends rotation-supported image paths to a
candidate association graph. It does not authenticate physical CSI plants or
identify anatomical leaves. Every CSI point remains conditional on a nominal
orthographic camera and training-image height gauge.

Run from this directory in an installed base environment (at most two numerical threads). Replace the input and output placeholders with local paths:

```sh
python tracking_development.py --full-run /path/to/full-cohort-run --external /path/to/Pheno4D --output /path/to/tracking-output --intake /path/to/intake --mode all
python tracking_validation.py --output /path/to/tracking-output --external /path/to/Pheno4D
python -m unittest test_tracking_graph -v
```

The public copy imports the numerical operators from `../../full_cohort`. Keep
that directory and the development sibling directories together. No private
images, external point clouds or generated graph files are distributed here.

The configuration was fixed before the first external evaluation. More angular
anchors, a support floor of max(8,25% of training views), a 6-pixel minimum branch
and joint per-view one-to-one refitting extend the prior 12-pixel/40% greedy
configuration. All v=2 modulo4 observations remain excluded from fitting.
Training residual covariance and leave-sector point shifts are sensitivity
descriptors, not calibrated anatomical uncertainty. Forty relative phase models
are profiled at 9-degree increments. Links must survive at least90% of retained
phase models, a0.10 global assignment margin and a1.2 normalized cost ceiling.
These are frozen engineering choices, not inferred identity probabilities.

Actual executed results:

- CSI:72 candidate identities,360 reference visit slots,288 adjacent pairs and
  216 gap-two pairs;339 eligible sessions,305 with hypotheses;642 nodes,
  272 selected adjacent geometry hypotheses and54 possible reappearances.
  All alternatives remain in1,490 edges. No death event is asserted.
- CSI heldout compatibility:3,852/6,338 track-view rows meet the positional gate;
  5,966 have a terminal candidate. Median nearest-terminal distance among the
  latter is0.03798 training heights, compared with0.03552 for the earlier
  377-hypothesis configuration. More coverage did not improve this median.
- External:all14 original scans from7 plants retained.62 hypotheses map one-to-one
  to61 distinct instance-dates, all within68/72 terminal-observable regions.
  These61 have training owner purity at least0.8; one additional mixed/unmapped
  hypothesis is retained as a failure. Heldout correctID+gate is519/620;
  a target terminal is available in525rows. These denominators are not replaced
  by the higher conditional519/525 fraction.
- All61 mapped points lie closer to correct than wrong observed instance point
  support. Median correct-support distance0.001431,90th percentile0.003426 and
  maximum0.005389 scan heights. These are not anatomical tip errors.
- Graph association recovers26/36 true cross-date links with26 predictions.
  Centroids recover25/36; simple invariant geometry recovers27/36. The graph is
  therefore not uniformly better. The prior silhouette configuration recovered
  16/36. With a90-degree phase perturbation, graph26/36 versus centroid13/36;
  with a hidden later node as well, graph21/36 versus centroid10/36.
- Geometry still accepts127/166 possible links across42 different-physical-plant
  controls. Experimental identity is an external prerequisite, not a property
  that the graph has learned to establish.
- Eight numeric synthetic perturbation scenarios recover29/30 known common
  links with29 predictions. A missing-middle scenario retains4 gap-two links.
  These are designed component tests, not biological validation.

`csi/index.json` indexes72 source-linked graphs; `external/index.json` indexes
the7 known-geometry graphs. Each node contains40 explicit angle states,
source/mask paths and digests, predicted image fractions and observed skeleton
polylines. A path is a visible image skeleton, not a measured3D midrib. States
distinguish missing source, rejected mask, censored candidates and absent terminal
support. New later observations have unresolved birth status. Empty visit slots
and unmatched alternatives remain present.

`configuration.json`, `execution_sources/`, `experiment_log.jsonl`, frozen
training models, scoring maps, `audit.json` and the independent external audit
retain provenance. Old full-cohort outputs were read-only. The final rerun
corrected a displayed external shared-gauge coordinate field and added per-angle
censor counts; neither changed association calculations or fitted parameters.
