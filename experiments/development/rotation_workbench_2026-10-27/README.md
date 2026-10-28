# Forty-index rotation workbench

`sprites.py` builds source-linked foreground/original image sprites from a
completed extraction. `build_workbench.py` combines those sprites, the candidate
tracking graph, the acquisition reference table and an optional paired-camera
lookup. Original frame indices remain 0–39; a missing source retains its slot.

`workbench.html` is the program template, with an unfilled payload marker. It
contains no private image data. A generated workbench embeds compressed private
assets and can be opened without a server or network requests. Its foreground
and camera overlays are image-support hypotheses, not anatomical correspondence
or authenticated cross-date identity.

Run from this directory in the base CPU environment:

```sh
python -m unittest test_sprites -v
node test_workbench.mjs workbench.html
python sprites.py --intake /path/to/intake --run /path/to/full-cohort-run --output /path/to/new-sprites --workers 2
python build_workbench.py --sprites /path/to/new-sprites/sprites/sprite_manifest.json --tracking /path/to/tracking-output/csi/index.json --reference /path/to/session-reference.csv --camera /path/to/camera-output/periodic/full/session_camera_lookup.json --output /path/to/workbench.html
```

Use `--help` for argument details and supply the actual verified sprite manifest
and reference table from the preceding stages. The paths above show the input
roles; they do not include data. The Node fixture creates 72 synthetic candidate
identities and 360 visit slots in memory when the template payload is empty.
It verifies selection keys, indices, affine direction, phase, abstention,
payload release and offline behavior. It is not visual browser QA.
