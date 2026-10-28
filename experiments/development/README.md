# Development investigations

These are source-only research programs, separate from the installed application
and the frozen full-cohort operators in [`../full_cohort`](../full_cohort/README.md).
Keep the dated sibling directories together: some programs consume tracking or
camera helpers from a sibling. The tracking runner also uses `../full_cohort`.

See [`../../docs/development.md`](../../docs/development.md) for dependencies,
input contracts, commands and test scope. From the repository root:

```sh
python experiments/development/run_tests.py
python experiments/development/run_tests.py --require-optional --report runs/development-tests.json
```

Each Python branch runs in a separate subprocess, with this checkout's `src`
explicitly first on the import path. Node.js 22 or later runs the 17 leaf-review
behavior tests and the rotation template's synthetic fixture. No test loads a
checkpoint, accesses private images or downloads model assets. An absent PyTorch
extra causes only its declared optional suite to be skipped; an installed but
broken dependency fails. Use `--require-optional` for a fully provisioned run.

The HTML files here are program templates. Generated workbenches, original
images, masks, model weights, execution snapshots and private output manifests
are excluded. Review annotations require a human rater; predictions and tracking
hypotheses are not anatomical ground truth. The native-image review workbench
has Node behavior coverage but no completed real-browser visual QA.

Original project code is MIT licensed; see [`LICENSE`](LICENSE) and the root
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) for external software
and model terms. No third-party model weights are redistributed.
