"""Declared numerical publication DAG.

Commands execute in a materialized ``prototype/research_v2`` directory. Outputs
are paths relative to the declared workspace. ``core`` covers every numerical
method of the architectural/observation paper; ``learning`` adds the earlier
phenotypic prediction comparisons. Old manuscript history and cloud delivery
are deliberately absent. The dated paths identify the frozen study, not dates
inferred for new acquisitions.
"""

OUT = "prototype/research_v2/outputs/"
LEAF = OUT + "leaf_dynamics/2026-10-24/"
WEAK = OUT + "multimodal_leaf/2026-10-24/"
FUSION = OUT + "channel_fusion/2026-10-24/"


def _stage(dependencies, commands, outputs, description, profile="core"):
    return {
        "dependencies": dependencies,
        "commands": commands,
        "outputs": outputs,
        "profile": profile,
        "description": description,
    }


STAGES = {
    "ingest": _stage(
        [],
        [["@metadata"]],
        [
            "research_audit/reference_master.csv",
            "research_audit/computed_variable_dictionary.csv",
            "research_audit/uvis_consistency_issues.json",
        ],
        "Create checked four-key reference joins directly from original workbooks.",
    ),
    "source-audit": _stage(
        ["ingest"],
        [["source_audit.py"]],
        [
            "research_audit/image_manifest.csv",
            "research_audit/session_manifest.csv",
            "research_audit/audit_summary.json",
        ],
        "Audit original workbook/image containers and explicit sample keys.",
    ),
    "index": _stage(
        ["ingest"],
        [["baseline.py", "index"]],
        [OUT + "sessions.csv", OUT + "identity_mapping_template.csv"],
        "Index named or explicitly mapped sessions; preserve unresolved identities.",
    ),
    "models": _stage(
        [], [["@models"]], [], "Fetch and verify declared pretrained checkpoint resources."
    ),
    "sam-comparison": _stage(
        ["index", "models"],
        [["sam_segmentation.py"]],
        [OUT + "sam_comparison/metrics.csv"],
        "Compare color-seeded SAM prompting on the declared quarter-view cases.",
    ),
    "segmentation": _stage(
        ["index", "models"],
        [["sam_traits.py", "--batch-size", "8"]],
        [
            OUT + "sam_traits/view_traits.csv",
            OUT + "sam_traits/session_traits.csv",
            OUT + "sam_traits/summary.json",
        ],
        "Generate all-view SAM foregrounds and operational QC; no annotated accuracy claim.",
    ),
    "organ-observations": _stage(
        ["segmentation"],
        [["organ_observations.py"]],
        [OUT + "organ_observations/capture_session_C2_5-1/observations.json"],
        "Extract source-assisted local organ paths from declared fitting views.",
    ),
    "camera-fit": _stage(
        ["organ-observations"],
        [["organ_camera.py"]],
        [OUT + "organ_observations/capture_session_C2_5-1/camera_tip_fit.json"],
        "Fit the conditional shared perspective camera against provisional tip observations.",
    ),
    "geometry-initial": _stage(
        ["camera-fit"],
        [["organ_inverse.py", "--iterations", "900"]],
        [OUT + "organ_inverse/capture_session_C2_5-1/organ_model.npz"],
        "Fit source-assisted late organ surfaces with the frozen 900-step initial configuration.",
    ),
    "geometry": _stage(
        ["geometry-initial"],
        [
            [
                "organ_inverse.py",
                "--iterations",
                "600",
                "--name",
                "organ_inverse_triangles",
                "--triangles",
                "--warm-start",
            ]
        ],
        [
            OUT + "organ_inverse_triangles/capture_session_C2_5-1/organ_model.npz",
            OUT + "organ_inverse_triangles/capture_session_C2_5-1/browser_mesh.json",
            OUT + "organ_inverse_triangles/capture_session_C2_5-1/summary.json",
        ],
        "Refine the saved late model with the frozen triangle renderer and warm start.",
    ),
    "geometry-family": _stage(
        ["geometry"],
        [["experiment_lab.py"], ["additive_geometry.py", "--experiment", "E08"]],
        [
            OUT + "additive_experiments/2026-10-24/e08_ambiguity/accepted_continuous_meshes.npz",
            OUT + "additive_experiments/2026-10-24/e08_ambiguity/candidate_scores.csv",
        ],
        "Evaluate every declared finite geometry alternative using fitting evidence only.",
    ),
    "leaf-models": _stage(
        ["geometry-family"],
        [["leaf_dynamics_geometry.py", "--iterations", "750"]],
        [
            LEAF + "geometry/earlier_organ_model.npz",
            LEAF + "geometry/summary.json",
            LEAF + "geometry/leaf_source_appearance_by_view.csv",
            LEAF + "geometry/fixed_shape_neighbour_removal.csv",
        ],
        "Fit the early assisted case and compute ownership-conditioned appearance, light and candidate correspondences.",
    ),
    "virtual-removal": _stage(
        ["leaf-models"],
        [["leaf_virtual_experiments.py"]],
        [
            LEAF + "jeb_vision_v2/experiments/summary.json",
            LEAF + "jeb_vision_v2/experiments_512/summary.json",
            LEAF + "jeb_vision_v2/numerical_stability_summary.json",
        ],
        "Run fixed-receiver neighbor removal, subset budgets and 256/512-grid stability.",
    ),
    "source-support": _stage(
        ["virtual-removal"],
        [["leaf_expanded_analyses.py"]],
        [
            LEAF + "jeb_expanded_v3/expanded_analysis_summary.json",
            LEAF + "jeb_expanded_v3/source_support_all40.csv",
        ],
        "Replay all-view local-organ source support and illumination/normalization sensitivity.",
    ),
    "view-sampling": _stage(
        ["source-support"],
        [["leaf_sampling_analysis.py"]],
        [LEAF + "jeb_expanded_v3/sampling_summary.json"],
        "Compare declared posthoc angular and support-based sampling sequences.",
    ),
    "weak-labels": _stage(
        ["leaf-models"],
        [["prepare_multimodal_leaf.py"]],
        [
            WEAK + "preparation_summary.json",
            WEAK + "paired_frames.csv",
            WEAK + "phase_audit.csv",
            WEAK + "proposals.json",
        ],
        "Create explicitly weak RGB/NoIR proposals from both geometry teachers and preserve group splits.",
    ),
    "weak-training": _stage(
        ["weak-labels", "models"],
        [
            [
                "train_multimodal_leaf.py",
                "--allow-weak-labels",
                "--epochs",
                "30",
                "--scales",
                "n,s",
            ],
            ["train_noir_blue_leaf.py"],
        ],
        [WEAK + "training_summary.json"],
        "Run all six frozen weak-teacher training recipes before per-camera model selection.",
    ),
    "leaf-inference": _stage(
        ["weak-training"],
        [["infer_multimodal_leaf.py"]],
        [
            WEAK + "inference_summary.json",
            WEAK + "leaf_predictions.csv",
            WEAK + "tracking_diagnostics.csv",
            WEAK + "cross_camera_diagnostics.csv",
        ],
        "Survey all rotations with selected weak models and independent camera-local circular/overlap tracks.",
    ),
    "tracking-comparison": _stage(
        ["leaf-inference"],
        [["benchmark_leaf_bytetrack.py"]],
        [WEAK + "bytetrack_diagnostics.csv"],
        "Replay official ByteTrack on the same prefiltered observations.",
    ),
    "channel-fusion": _stage(
        ["leaf-inference"],
        [["run_channel_fusion.py"]],
        [
            FUSION + "summary.json",
            FUSION + "run_configuration.json",
            FUSION + "dependency_binding.json",
            FUSION + "frame_registration.json",
            FUSION + "shared_targets.json",
        ],
        "Compute frozen-rule cross-camera hypotheses and bind actual data plus complete helper source hashes.",
    ),
    "channel-support": _stage(
        ["channel-fusion"],
        [["audit_channel_support.py"]],
        [FUSION + "target_support_audit.csv", FUSION + "gate_sensitivity.csv"],
        "Audit common interiors, validity exclusions and frozen engineering gates.",
    ),
    "channel-report": _stage(
        ["channel-support"],
        [["report_channel_fusion.py"]],
        [FUSION + "analysis.json", FUSION + "CSI_maize_shared_leaf_targets.html"],
        "Summarize acquisition-balanced camera-DN features, perturbations and signal availability.",
    ),
    "pair-controls": _stage(
        ["channel-fusion"],
        [["v5_pair_controls.py"], ["v5_pair_controls_verify.py"]],
        [
            LEAF + "jeb_v5/pair_controls/summary.json",
            LEAF + "jeb_v5/pair_controls/frame_records.json",
            LEAF + "jeb_v5/pair_controls/nominal_replay.json",
        ],
        "Challenge nominal and deliberate mismatch pairs with equal moment/ECC registration effort.",
    ),
    "spatial-fusion": _stage(
        ["channel-fusion"],
        [["run_spatial_fusion.py"]],
        [
            OUT + "spatial_fusion/2026-10-26/summary.json",
            OUT + "spatial_fusion/2026-10-26/frame_index_statistics.csv",
            OUT + "spatial_fusion/2026-10-26/run_manifest.json",
            OUT + "spatial_fusion/2026-10-26/CSI_maize_RGB_NoIR_spatial_atlas.html",
        ],
        "Produce all 27 guarded spatial descriptors, complete coverage and an offline observation atlas.",
    ),
    "synthetic-recovery": _stage(
        [],
        [["v5_recovery_benchmark.py"]],
        [LEAF + "jeb_v5/recovery/summary.json"],
        "Run eight known-answer assisted inverse cases and retain geometry plus endpoint errors.",
    ),
    "synthetic-stress": _stage(
        ["synthetic-recovery"],
        [["v5_recovery_occlusion.py"]],
        [LEAF + "jeb_v5/recovery_distichous/summary.json"],
        "Run four declared exploratory distichous stress fits with the same frozen inverse configuration.",
    ),
    "reposition": _stage(
        ["geometry-family", "virtual-removal"],
        [["v5_reposition.py"]],
        [
            LEAF + "jeb_v5/reposition_fixed_domain/summary.json",
            LEAF + "jeb_v5/reposition_fixed_domain/all_reposition_states.json",
            LEAF + "jeb_v5/reposition_fixed_domain/two_grid_differences.csv",
        ],
        "Run every fixed-domain rigid-neighbor intervention while preserving receiver geometry and beam budgets.",
    ),
    "baseline-features": _stage(
        ["index"],
        [["baseline.py", "features"], ["baseline.py", "benchmark"], ["baseline.py", "spectra"]],
        [OUT + "features.csv", OUT + "metrics.csv", OUT + "spectral_metrics.csv"],
        "Replay the original unvalidated foreground/scalar/native-spectrum baselines.",
        "learning",
    ),
    "embeddings": _stage(
        ["segmentation", "models"],
        [["pretrained_features.py", "embeddings"]],
        [OUT + "dinov2_embeddings.npz"],
        "Extract frozen DINOv2 image vectors with source/model provenance.",
        "learning",
    ),
    "phenotype-models": _stage(
        ["embeddings"],
        [
            ["model_search.py", "--embeddings"],
            ["model_search.py", "--embeddings", "--split", "plant_label"],
        ],
        [
            OUT + "nested_genotype_embeddings/summary.json",
            OUT + "nested_plant_label_embeddings/summary.json",
        ],
        "Run nested grouped phenotype prediction under the two declared generalization questions.",
        "learning",
    ),
    "spectral-models": _stage(
        ["embeddings", "ingest"],
        [["spectral_models.py"]],
        [OUT + "nested_spectra/summary.json"],
        "Predict native UVIS/NIR grids with train-only nested model selection.",
        "learning",
    ),
    "phenotype-controls": _stage(
        ["phenotype-models"],
        [["performance_controls.py"]],
        [OUT + "performance_controls/summary.json"],
        "Replay fixed posthoc metadata, geometry, optical and embedding controls.",
        "learning",
    ),
}
