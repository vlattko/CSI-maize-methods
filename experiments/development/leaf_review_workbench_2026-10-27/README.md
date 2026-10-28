# Local leaf-region annotation workbench

Open `leaf_review_workbench.html` with its `assets/originals/` directory beside it. It uses60 native original JPEGs and embeds the proposal metadata, all raw/clipped masks and all three alternatives for both prompt modes. It requires no server, fetch/XHR, external library or network connection. The optional `leaf_review_workbench_standalone.html` embeds the same native originals and is substantially larger; it is separate from the primary load path. Both variants use the same dataset ID, so review-session JSON can move between them.

These are private source images. The bundle is for the authorized private Drive/workspace workflow, not a public code repository.

## Review

1. Enter your rater ID. Navigate the60 source images with the selector or arrow keys. Fit, zoom and pan operate on the original image grid. Switch between the original source and black parent foreground as needed.
2. Select a region and inspect box-only or point-prompted alternatives. Raw masks and clipped masks are separate previews. Adopting a different mask resets acceptance and image completeness.
3. Mark a region accepted, rejected or uncertain. Accepted partial regions remain review notes and block training export. Complete visible-blade instances need an explicit extent decision and acceptance.
4. Draw a missing instance with **Draw a manual polygon**: click vertices, press Enter to finish. Drag polygon vertices to edit; double-click an edge to insert a vertex. Edits reset acceptance and completeness. You can annotate outside the parent foreground using the original image.
5. Confirm **All visible leaves reviewed** only after every visible blade is represented and every proposal/manual region is resolved. A missing proposal is not a negative. A negative image additionally requires **Verified no visible leaf**, with no accepted instances and all proposals resolved.
6. Save a review-session JSON to retain all decisions, partial regions, notes and provenance. Import validates dataset/frame/candidate IDs, four-key identity, source hashes and native dimensions. There is no implicit autosave or automatic acceptance.

Reviewed JSON/COCO export includes only eligible complete-reviewed images and accepted complete visible-blade instances. Partial, uncertain and unreviewed image states remain exclusively in the review-session audit file. Unreviewed predictions are provided separately as `unreviewed_predictions.json`. The COCO category is `complete_visible_blade`; source track IDs remain provenance and do not authenticate anatomical or cross-date identity.

## Coordinates and provenance

Manual and edited polygons use continuous pixel-edge coordinates in the full native original image, normally3280×2464. Zoom/pan never changes saved coordinates. Model masks remain explicitly processing-resolution proposals; mask pixel centres are mapped to the native coordinate grid. COCO image dimensions and asset filenames remain native and stable in both portable and standalone versions.

`asset_manifest.json` binds every native asset and delivered HTML to a SHA256 digest. `source_bindings.json` binds all60 frame identities/source hashes and516 raw/clipped proposal alternatives. The original model scores and prompts remain separate from review judgments. No real anatomical review or accepted training annotation is preloaded or delivered.

## Verification status

The Node suite tests core export/import rules and the actual app event handlers against a synthetic offline DOM fixture. It covers defaults, partial/incomplete export blocking, verified-empty gating, polygon coordinates, cancelled edits, alternative invalidation and mismatched source imports. Synthetic fixture decisions are never delivered as real reviewed ground truth.

Real-browser visual/layout QA is unverified: browser automation rejected the local `file://` URL under its URL security policy. No server or alternate browser workaround was attempted. The delivered tool is prepared for the user's own browser inspection; this limitation is recorded in `validation.json`.

To rebuild with the existing numerical environment:

```powershell
node --test review_core.test.js review_app.test.js
python build_workbench.py --proposals VERIFIED_PROPOSAL_OUTPUT --intake INTAKE --output WORKBENCH_OUTPUT --standalone
python verify_bundle.py --output WORKBENCH_OUTPUT
```

The code uses Node's built-in test/VM modules and the existing Python NumPy, OpenCV and Pillow installation. It uses no GPU and does not train a model.
