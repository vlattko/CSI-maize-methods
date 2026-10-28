# Experimenter-confirmed pot identities and declared imaging ages

The experimenter confirmed that matching treatment, genotype and replicate labels identify the same physical pots across visits. The current study contains 72 pots, 360 reference records and declared imaging ages of 13, 19, 20, 22 and 25 DAS. Pot continuity does not validate individual leaf identities or geometry-based associations.

Keep the confirmation JSON with the private study inputs. It records the authority and date, declared ages and any treatment details supplied by the experimenter. Raw spreadsheets, images and timestamps are not edited.

```sh
python registry.py --reference /path/to/reference_master.csv \
  --confirmation /path/to/experimenter_confirmation.json --output /path/to/identity-overlay
python dynamics.py --registry /path/to/identity-overlay/identity_registry.json \
  --reference /path/to/reference_master.csv --traits /path/to/session_traits.csv \
  --output /path/to/identity-overlay
python -m unittest discover -v
```

The identity overlay checks four-key uniqueness, keeps different treatments as different pots, and assigns stable within-study pot IDs. The workbench builder accepts it through `--identity-registry` and retains the original graph fits and image bindings.

The dynamics report keeps uneven day intervals and missing adjacent observations. It displays measured acquisition-level assays and qualified visible-image support. Whole-pot bootstrap resampling occurs within genotype and treatment, keeping each pot's five visits together. Both absolute measurements and changes relative to the first observation are retained. The first observation is not assumed to be an untreated baseline.

The timetable places priming at 6–11 DAS and the severe challenge at 14–19 DAS for both S and P, with warming after the 19 DAS measurements. Priming durations increase across days; their interpretation as day/night hours awaits the complete method. The first image follows priming for P. The priming-night setting remains unresolved: worksheet cells and the experimenter say 8 °C, while a footnote says 5 °C. We retain both sources. Scheduled calendar blocks differ between treatments, so treatment and run may be confounded. Schedule dates do not replace camera timestamps. Exact exposure hours and LAB tissue removal remain unresolved.

`study_metadata.py` extracts source-cell values and comments without modifying Excel files. It verifies scheduled date/DAS consistency, writes the 15-visit crosswalk and maps destructive mass by the written genotype within each confirmed six-genotype replicate block. Internal genotype order differs between blocks. A separate private confirmation must explicitly declare replicate blocks, grams and whole aboveground tissue. Never infer these from row positions alone.

```sh
python study_metadata.py --timetable /data/timetable.xlsx --mass /data/mass.xlsx \
  --genotypes /data/genotypes.xlsx --mass-confirmation /data/mass_confirmation.json \
  --output /results/metadata
python mass_validation.py --mass /results/metadata/confirmed_mass.csv \
  --registry /results/identity/identity_registry.json --traits /results/session_traits.csv \
  --output /results/mass-validation
python verify_mass.py --output /results/mass-validation
```

The destructive mass benchmark keeps all 72 pot joins and uses a common qualified endpoint RGB/NoIR subset for every model. Endpoints are fresh mass, dry mass and dry-matter fraction. Nested leave-one-genotype-out and leave-one-treatment-out ridge models compare fixed shape features, temporal summaries, metadata and training-mean baselines. Preprocessing and tuning use training pots only. A fixed shuffled-label control is diagnostic, not a significance test. The verifier independently reconstructs the preprocessing and linear algebra, replays inner alpha selection and outer predictions, and re-reads all 144 original mass cells. It requires the original private source files.

The last image precedes scheduled fresh harvest by one day. Dry-matter fraction is not relative water content. The small 0.31 g observation remains included. Whole-plant mass cannot validate leaf identity or leaf-level mass.

Projected image fractions and their per-day changes are not calibrated biomass or leaf-area growth rates. They remain affected by visibility, cropping, pose and camera geometry. Source-assisted organ geometry and automatic terminal correspondences retain their separate validation limits.
