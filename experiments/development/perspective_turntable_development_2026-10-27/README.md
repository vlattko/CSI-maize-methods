# Conditional perspective turntable points

These programs compare nominal orthographic and perspective point surrogates
using observed terminal paths. Focal length, elevation, roll and phase remain
conditional model choices. Sparse image paths do not establish a calibrated
camera, leaf surfaces or anatomical identity.

`perspective_points.py` provides projection and fitting operators.
`perspective_run.py` reads the completed full-cohort extraction and longitudinal
tracking output. `perspective_export.py` exports selected results into the graph
contract; `perspective_audit.py` checks the saved run against its upstream input.
The dated tracking sibling and `../../full_cohort` must remain available.

Run from this directory in the base CPU environment:

```sh
python -m unittest test_perspective_points -v
python perspective_run.py --full-run /path/to/full-cohort-run --tracking /path/to/tracking-output --output /path/to/new-perspective-output
python perspective_export.py --output /path/to/new-perspective-output --tracking /path/to/tracking-output
python perspective_audit.py --output /path/to/new-perspective-output --full-run /path/to/full-cohort-run
```

The nine known-answer tests cover perspective recovery, orthographic recovery,
phase, withheld-angle fitting guards, positive depth, the harmonic comparator,
retained alternatives and local identifiability diagnostics. They require no
private images, model checkpoint or GPU.
