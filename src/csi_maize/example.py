"""Offline CPU example using procedural inputs and the canonical numerical kernels.

This is an assisted known-answer demonstration. Masks, organ labels, cameras
and complete projected paths are supplied synthetic truth. SAM and YOLO are
not run, and these results never stand in for the publication acquisitions.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run_example(output: Path, steps: int = 30) -> dict:
    """Create a fresh no-download procedural run and return its verified report."""
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 350:
        raise ValueError("steps must be an integer between 1 and 350")
    root = Path(output).expanduser().absolute()
    package = Path(__file__).parent
    if root.resolve().is_relative_to(package.resolve()):
        raise ValueError("Example output must be outside the installed source package")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Example requires a new or empty output directory")
    root.mkdir(parents=True, exist_ok=True)
    base = root / "prototype/research_v2"
    base.mkdir(parents=True)
    (base / "outputs").mkdir()
    for source in (package / "_methods").iterdir():
        if source.is_file() and source.suffix in {".py", ".json", ".html", ".js", ".css"}:
            shutil.copyfile(source, base / source.name)
    # Freeze the package used for report/provenance authoring as well as kernels.
    shutil.copytree(
        package, root / "runtime/csi_maize", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (base / "config.json").write_text(
        json.dumps(dict(case="synthetic_example", expected_acquisitions=None)), encoding="utf-8"
    )
    (root / "prototype/config.json").write_bytes((base / "config.json").read_bytes())
    (root / ".csi_workspace.json").write_text(
        json.dumps(dict(schema_version=1, owner="csi-maize-methods", mode="synthetic_example")),
        encoding="utf-8",
    )
    env = dict(
        os.environ,
        CSI_MAIZE_WORKSPACE=str(root),
        PYTHONPATH=str(root / "runtime"),
        PYTHONUTF8="1",
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        YOLO_AUTOINSTALL="false",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    logs = root / "logs"
    logs.mkdir()
    with (logs / "example.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "csi_maize.example",
                "--worker",
                str(root),
                "--steps",
                str(steps),
            ],
            cwd=base,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError("Synthetic example failed; inspect " + str(logs / "example.log"))
    verification = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    return dict(
        workspace=str(root),
        report=str(root / "reports/CSI_maize_architecture.html"),
        verification=verification,
    )


def _worker(root: Path, steps: int):
    """Run in a subprocess so flat legacy imports cannot affect the caller."""
    base = root / "prototype/research_v2"
    sys.path.insert(0, str(base))
    import csv
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from threadpoolctl import threadpool_limits
    from csi_maize.provenance import append_event, digest, file_manifest, utc, write_json
    from csi_maize.report import build_report, _ResourceCheck
    from csi_maize.config import RunConfig
    from csi_maize.engine import raw_hashes
    from csi_maize.workspace import _bind_directory
    import v5_recovery_benchmark as recovery
    from v5_recovery_occlusion import make_distichous
    from organ_camera import project as camera_project
    from leaf_dynamics_geometry import frontmost_ownership
    from traits import measure_mask
    from multimodal_leaf_core import affine_align, mask_iou
    from channel_fusion_core import warp_to_rgb, interior
    from spatial_fusion_core import (
        index_definitions,
        spatial_indices,
        index_statistics,
        overlap_labels,
    )
    from leaf_virtual_experiments import layered_raster, compose_owners, screening_bookkeeping
    from v5_reposition import rigid_reposition, domain_anchors, raster_fixed, transition_budget

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.manual_seed(20261024)
    out = base / "outputs"
    leaf = out / "leaf_dynamics/2026-10-24"
    definitions = index_definitions()
    design = recovery.design()
    angles = np.arange(16) * 2 * np.pi / 16
    matrix_truth = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]], np.float32)
    sample_indices = np.rint(np.linspace(0, 63, 16)).astype(int)
    cross_indices = np.arange(0, 9, 2)
    cases = ["example_3leaf", "example_5leaf"]
    inputs = root / "inputs"
    inputs.mkdir()
    example_config = RunConfig(
        data_root=inputs,
        work_dir=root,
        source_folders=tuple(cases),
        mode="example",
        profile="example",
        device="cpu",
        expected_acquisitions=None,
    )
    write_json(root / "run_config.json", example_config.to_dict())
    write_json(root / "input_bindings.json", dict(bindings=[]))
    declaration = dict(
        mode="synthetic_example",
        seed=20261024,
        device="cpu",
        steps=steps,
        cases=cases,
        views=16,
        shape=design["shape"],
        camera=design["camera"],
        provided_inputs=[
            "procedural RGB/NoIR PNG images",
            "known foreground masks",
            "known local organ labels",
            "known cameras and angles",
            "four complete projected paths including hidden portions",
        ],
        neural_models_executed=False,
        models_downloaded=False,
        evaluation_surface=[16, 5],
        fit_surface=[32, 5],
        example_pose_degrees=[-30, 0, 30],
        example_pose_grid=96,
        interpretation="Procedural assisted component demonstration; not raw archive processing or independent biological validation.",
    )
    write_json(inputs / "provided_truth.json", declaration)
    source_files = [
        path
        for folder in [base, root / "runtime"]
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".json", ".html", ".css", ".js"}
        and "__pycache__" not in path.parts
    ]
    sources = file_manifest(root, source_files)
    fingerprint = hashlib.sha256(
        json.dumps(dict(declaration=declaration, sources=sources), sort_keys=True).encode()
    ).hexdigest()
    state = dict(
        schema_version=1,
        mode="synthetic_example",
        status="running",
        fingerprint=fingerprint,
        created_utc=utc(),
        stages={},
        source_sha256=sources,
        declaration=declaration,
        staged_code_sha256=sources,
        materialized_input_sha256={},
        context=dict(config=example_config.to_dict(), raw_sha256={}),
        runtime=dict(
            python=sys.version, numpy=np.__version__, torch=torch.__version__, device="cpu"
        ),
        output_sha256={},
    )
    traces = {}
    frames = []
    fitted_cases = {}
    checks = {}

    def save_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def stage(name, operation):
        started = time.monotonic()
        state["stages"][name] = dict(status="running", started_utc=utc(), mode="synthetic_example")
        write_json(root / "run_manifest.json", state)
        append_event(
            root / "events.jsonl", "stage_started", dict(stage=name, mode="synthetic_example")
        )
        try:
            operation()
        except BaseException as error:
            state["status"] = "failed"
            state["stages"][name].update(status="failed", error=str(error))
            write_json(root / "run_manifest.json", state)
            append_event(root / "events.jsonl", "stage_failed", dict(stage=name, error=str(error)))
            raise
        state["stages"][name].update(
            status="completed", elapsed_seconds=round(time.monotonic() - started, 3)
        )
        append_event(
            root / "events.jsonl", "stage_completed", dict(stage=name, mode="synthetic_example")
        )
        write_json(root / "run_manifest.json", state)
        print(name, state["stages"][name]["elapsed_seconds"], flush=True)

    def image_record(path, array):
        Image.fromarray(array).save(path)
        return dict(path=path.relative_to(root).as_posix(), sha256=digest(path))

    def generate():
        records = []
        bindings = []
        for name, count in zip(cases, [3, 5]):
            folder = inputs / name
            folder.mkdir()
            kind = _bind_directory(folder, root / name)
            bindings.append(dict(logical_path=name, source_path=str(folder), kind=kind))
            truth = recovery.make_truth(count, "cubic") if count == 3 else make_distichous("cubic")
            vertices = truth["vertices"][:, sample_indices][:, :, cross_indices]
            mids = truth["midribs"][:, sample_indices]
            faces_local = recovery.ribbon_faces(16, 5)
            faces = np.concatenate([faces_local + organ * 80 for organ in range(count)])
            face_organs = np.repeat(np.arange(count), len(faces_local))
            xyz = vertices.reshape(-1, 3)
            projected_paths = recovery.project_np(
                truth["midribs"].reshape(-1, 3),
                angles[design["assisted_path_views"]],
                design["camera"],
            ).reshape(4, count, 64, 2)
            paths = np.array(
                [[recovery.resample_path(path) for path in view] for view in projected_paths]
            )
            np.savez_compressed(
                folder / "provided_geometry.npz", vertices=vertices, midribs=mids, faces=faces_local
            )
            masks = []
            for view, angle in enumerate(angles):
                camera = design["camera"]
                xy, depth = camera_project(
                    xyz,
                    np.full(len(xyz), angle),
                    camera["elevation_deg"],
                    camera["distance"],
                    camera["roll_deg"],
                    center=(camera["principal_x"], camera["principal_y"]),
                    basal=(camera["axis_x"], camera["base_y"]),
                )
                np.testing.assert_allclose(
                    xy, recovery.project_np(xyz, np.array([angle]), camera)[0], atol=1e-10
                )
                labels = (
                    frontmost_ownership(xy, depth, faces, face_organs, tuple(design["shape"])) + 1
                )
                mask = labels > 0
                masks.append(mask)
                yy, xx = np.indices(mask.shape)
                rgb = np.zeros((*mask.shape, 3), np.uint8)
                noir_reference = np.zeros_like(rgb)
                rgb[..., 0] = np.where(mask, 45 + labels * 5 + xx // 16, 0)
                rgb[..., 1] = np.where(mask, 105 + labels * 6 + yy // 16, 0)
                rgb[..., 2] = np.where(mask, 28 + labels * 3 + (view % 4), 0)
                noir_reference[..., 0] = np.where(mask, 90 + labels * 4, 0)
                noir_reference[..., 1] = np.where(mask, 62 + labels * 3 + xx // 24, 0)
                noir_reference[..., 2] = np.where(mask, 138 + labels * 4 + yy // 24, 0)
                noir = cv2.warpAffine(
                    noir_reference,
                    matrix_truth,
                    (mask.shape[1], mask.shape[0]),
                    flags=cv2.INTER_NEAREST,
                )
                noir_labels = cv2.warpAffine(
                    labels.astype(np.uint8),
                    matrix_truth,
                    (mask.shape[1], mask.shape[0]),
                    flags=cv2.INTER_NEAREST,
                )
                for camera_name, image, local_labels in [
                    ("rgb", rgb, labels),
                    ("noir", noir, noir_labels),
                ]:
                    stem = folder / f"{camera_name}_{view:02}"
                    source = image_record(stem.with_suffix(".png"), image)
                    mask_record = image_record(
                        stem.with_name(stem.name + "_mask.png"),
                        (local_labels > 0).astype(np.uint8) * 255,
                    )
                    label_record = image_record(
                        stem.with_name(stem.name + "_labels.png"), local_labels.astype(np.uint8)
                    )
                    records.append(
                        dict(
                            session=name,
                            camera=camera_name,
                            view=view,
                            nominal_angle_rad=float(angle),
                            source=source["path"],
                            source_sha256=source["sha256"],
                            mask=mask_record["path"],
                            mask_sha256=mask_record["sha256"],
                            labels=label_record["path"],
                            labels_sha256=label_record["sha256"],
                            label_status="provided_procedural_truth",
                        )
                    )
            np.savez_compressed(
                folder / "provided_observations.npz",
                paths=paths,
                path_views=np.array(design["assisted_path_views"]),
                train_views=np.array(design["train_views"]),
                masks=np.array(masks)[design["train_views"]],
                angles=angles,
                clean_masks=np.array(masks),
            )
        write_json(
            inputs / "image_manifest.json",
            dict(
                mode="synthetic_example", records=records, affine_rgb_to_noir=matrix_truth.tolist()
            ),
        )
        write_json(root / "input_bindings.json", dict(bindings=bindings))
        checks["generated_image_pairs"] = len(records) // 2
        state["provided_input_sha256"] = file_manifest(root, [inputs])
        state["materialized_input_sha256"] = state["provided_input_sha256"]
        state["context"]["raw_sha256"] = raw_hashes(example_config)

    def image_traits():
        records = json.loads((inputs / "image_manifest.json").read_text(encoding="utf-8"))[
            "records"
        ]
        rows = []
        for record in records:
            source, mask_path = root / record["source"], root / record["mask"]
            if (
                digest(source) != record["source_sha256"]
                or digest(mask_path) != record["mask_sha256"]
            ):
                raise ValueError("Provided procedural input changed")
            image = np.array(Image.open(source).convert("RGB"))
            mask = np.array(Image.open(mask_path)) > 0
            values, _ = measure_mask(image, mask, record["camera"])
            rows.append(
                dict(
                    record,
                    **values,
                    mask_origin="provided_procedural_truth",
                    segmentation_model="not_executed",
                )
            )
        save_csv(out / "sam_traits/view_traits.csv", rows)
        write_json(
            out / "sam_traits/summary.json",
            dict(
                mode="synthetic_example",
                status="provided_procedural_masks",
                views=len(rows),
                neural_segmentation_executed=False,
                image_hashes_verified=len(rows),
            ),
        )
        traces["traits"] = len(rows)

    def fit():
        results = []
        for name in cases:
            folder = inputs / name
            with np.load(folder / "provided_observations.npz") as saved:
                observation = {
                    key: saved[key] for key in ("paths", "path_views", "train_views", "angles")
                }
                clean = saved["clean_masks"]
            # The actual saved masks, not a new hidden rendering, drive the fit.
            records = json.loads((inputs / "image_manifest.json").read_text(encoding="utf-8"))[
                "records"
            ]
            mask_rows = sorted(
                [r for r in records if r["session"] == name and r["camera"] == "rgb"],
                key=lambda r: r["view"],
            )
            observed = np.array([np.array(Image.open(root / row["mask"])) > 0 for row in mask_rows])
            np.testing.assert_array_equal(observed, clean)
            observation["masks"] = observed[observation["train_views"]]
            initial = recovery.initialize_from_observations(observation, design["camera"])
            fitted, history = recovery.fit_case(
                observation, design["camera"], initial, steps, "cpu"
            )
            if not all(np.isfinite(value).all() for value in fitted.values()):
                raise RuntimeError("Nonfinite fitted parameters")
            result_folder = leaf / "jeb_v5/recovery" / name
            result_folder.mkdir(parents=True)
            np.savez_compressed(result_folder / "initial_parameters.npz", **initial)
            np.savez_compressed(result_folder / "fitted_parameters.npz", **fitted)
            save_csv(result_folder / "loss.csv", history)
            with np.load(folder / "provided_geometry.npz") as saved:
                truth_vertices, truth_mid, faces = (
                    saved["vertices"],
                    saved["midribs"],
                    saved["faces"],
                )
            truth_release, _ = recovery.release_matrix(truth_vertices)
            metrics = {}
            arrays = dict(
                truth_vertices=truth_vertices, truth_midribs=truth_mid, truth_release=truth_release
            )
            for label, parameters in [("initial", initial), ("fitted", fitted)]:
                vertices, mids, _ = recovery.surface_from_parameters(
                    parameters, samples=16, cross=5
                )
                metric = recovery.geometry_metrics(vertices, mids, truth_vertices, truth_mid, faces)
                predicted_masks = recovery.render_masks(
                    vertices, angles, design["camera"], tuple(design["shape"])
                )
                ious = recovery.mask_metrics(predicted_masks, observed)
                release, beams = recovery.release_matrix(vertices)
                off = ~np.eye(len(vertices), dtype=bool)
                metric.update(
                    mean_withheld_clean_mask_iou=float(ious[design["test_views"]].mean()),
                    mean_train_clean_mask_iou=float(ious[design["train_views"]].mean()),
                    release_mean_abs_error=float(np.abs(release - truth_release)[off].mean()),
                )
                metrics[label] = metric
                arrays.update(
                    {
                        label + "_vertices": vertices,
                        label + "_midribs": mids,
                        label + "_release": release,
                        label + "_beam_release": beams,
                    }
                )
            np.savez_compressed(result_folder / "recovery_arrays.npz", **arrays)
            results.append(
                dict(
                    case=dict(
                        id=name, leaves=len(truth_vertices), condition="provided_clean_masks"
                    ),
                    status="completed",
                    metrics=metrics,
                )
            )
            fitted_cases[name] = dict(
                parameters=fitted,
                vertices=arrays["fitted_vertices"],
                midribs=arrays["fitted_midribs"],
                faces=faces,
            )
            traces[name] = dict(
                steps=len(history),
                first_loss=history[0]["loss"],
                last_loss=history[-1]["loss"],
                changed_control_norm=float(np.linalg.norm(fitted["control"] - initial["control"])),
            )
        write_json(
            leaf / "jeb_v5/recovery/summary.json",
            dict(mode="synthetic_example", status="completed", results=results, design=declaration),
        )

    def scene_with_stem(case):
        vertices = case["vertices"]
        count = len(vertices)
        parameters = case["parameters"]
        args = [
            torch.tensor(parameters[key], dtype=torch.float64)
            for key in ["control", "attachment", "widths", "twists"]
        ]
        with torch.no_grad():
            _, _, _, stem, _ = recovery.scene(
                args[0] / 100,
                args[1] / 100,
                torch.zeros(4, dtype=torch.float64),
                args[2] / 100,
                args[3],
                100,
                16,
                5,
            )
        stem = stem.numpy()[np.rint(np.linspace(0, 99, 8)).astype(int)]
        flat = np.concatenate([vertices.reshape(-1, 3), stem.reshape(-1, 3)])
        faces = [case["faces"] + i * 80 for i in range(count)]
        stem_faces = []
        start = count * 80
        for row in range(7):
            for col in range(12):
                a, b = start + row * 12 + col, start + (row + 1) * 12 + col
                c, d = start + (row + 1) * 12 + (col + 1) % 12, start + row * 12 + (col + 1) % 12
                stem_faces.extend([[a, b, c], [a, c, d]])
        faces.append(np.array(stem_faces))
        owners = np.r_[np.repeat(np.arange(count), 80), np.full(len(stem) * 12, count)]
        return flat, np.concatenate(faces), owners

    def interventions():
        case = fitted_cases["example_5leaf"]
        vertices, faces, organs = scene_with_stem(case)
        count, stem_id = 5, 5
        face_organs = organs[faces[:, 0]]
        mesh_folder = out / "organ_inverse_triangles/example_5leaf"
        mesh_folder.mkdir(parents=True)
        write_json(
            mesh_folder / "browser_mesh.json",
            dict(
                mode="synthetic_example",
                vertices=vertices.tolist(),
                faces=faces.tolist(),
                organs=[
                    dict(leaf_id=f"Example leaf {i + 1}", vertex_start=i * 80, vertex_count=80)
                    for i in range(count)
                ]
                + [
                    dict(
                        leaf_id="Provided straight stem",
                        vertex_start=count * 80,
                        vertex_count=len(vertices) - count * 80,
                    )
                ],
            ),
        )
        write_json(
            mesh_folder / "summary.json",
            dict(mode="synthetic_example", status="bounded_assisted_fit", units="synthetic gauge"),
        )
        directions = [np.array([0.0, 1.0, 0.0])] + [
            np.array([np.cos(a), 1.0, np.sin(a)]) / np.sqrt(2) for a in np.arange(8) * np.pi / 4
        ]
        removal = []
        for resolution in [256, 512]:
            for direction_id, direction in enumerate(directions):
                layers, _ = layered_raster(vertices, faces, face_organs, direction, resolution)
                full = compose_owners(layers, range(count + 1))
                for receiver in range(count):
                    bookkeeping = screening_bookkeeping(
                        layers, receiver, list(range(count)), stem_id
                    )
                    for donor in range(count):
                        if donor == receiver:
                            continue
                        after = compose_owners(layers, [i for i in range(count + 1) if i != donor])
                        gain = int(np.sum(after == receiver) - np.sum(full == receiver))
                        removal.append(
                            dict(
                                resolution=resolution,
                                direction=direction_id,
                                removed_leaf=f"leaf_{donor + 1}",
                                receiver_leaf=f"leaf_{receiver + 1}",
                                rescue_fraction=gain / max(1, bookkeeping["footprint_pixels"]),
                                receiver_unchanged=True,
                            )
                        )
        save_csv(
            leaf / "jeb_vision_v2/experiments_512/directed_neighbour_effects.csv",
            [r for r in removal if r["resolution"] == 512],
        )
        save_csv(
            leaf / "jeb_vision_v2/experiments/directed_neighbour_effects.csv",
            [r for r in removal if r["resolution"] == 256],
        )
        edges = []
        for donor in range(count):
            for receiver in range(count):
                if donor == receiver:
                    continue
                row = dict(removed_leaf=f"leaf_{donor + 1}", receiver_leaf=f"leaf_{receiver + 1}")
                values = [
                    r["rescue_fraction"]
                    for r in removal
                    if r["resolution"] == 512
                    and r["removed_leaf"] == row["removed_leaf"]
                    and r["receiver_leaf"] == row["receiver_leaf"]
                ]
                row.update(
                    baseline_mean512=float(np.mean(values)),
                    grid_min_mean512=None,
                    grid_max_mean512=None,
                    all_configurations_resolved_positive=False,
                )
                edges.append(row)
        write_json(
            leaf / "jeb_vision_v2/numerical_stability_summary.json",
            dict(
                mode="synthetic_example",
                edge_summaries=edges,
                grids=[256, 512],
                configurations=1,
                interpretation="One example geometry; no eight-model finite family was executed.",
            ),
        )
        donor, receiver = 3, 1
        donor_mask = organs == donor
        pivot = case["vertices"][donor, 0].mean(0)
        pose_angles = [-30, 0, 30]
        poses = [
            rigid_reposition(vertices, donor_mask, pivot, [0, 1, 0], angle) for angle in pose_angles
        ]
        budgets, responses = [], []
        receiver_unchanged = True
        closure = 0.0
        for direction_id, direction in enumerate(directions):
            anchors = domain_anchors(poses, direction)
            layers, pixel_area = raster_fixed(vertices, faces, face_organs, direction, 96, anchors)
            before = compose_owners(layers, range(count + 1))
            footprint = int(np.isfinite(layers[receiver]).sum())
            for angle, pose in zip(pose_angles, poses):
                moved_layers, moved_area = raster_fixed(
                    pose, faces, face_organs, direction, 96, anchors
                )
                receiver_unchanged &= (
                    np.array_equal(pose[~donor_mask], vertices[~donor_mask])
                    and np.array_equal(layers[receiver], moved_layers[receiver])
                    and pixel_area == moved_area
                )
                after = compose_owners(moved_layers, range(count + 1))
                budget = transition_budget(before, after, count + 1, pixel_area)
                closure = max(closure, abs(budget["closure"]))
                delta = int(np.sum(after == receiver) - np.sum(before == receiver)) / max(
                    1, footprint
                )
                budgets.append(
                    dict(
                        direction=direction_id,
                        angle_degrees=angle,
                        receiver_change_fraction=delta,
                        **budget,
                    )
                )
        for angle in pose_angles:
            rows = [r for r in budgets if r["angle_degrees"] == angle]
            responses.append(
                dict(
                    candidate_id=0,
                    resolution=96,
                    mode="yaw_degrees",
                    angle_degrees=angle,
                    equal9_receiver_change_fraction=float(
                        np.mean([r["receiver_change_fraction"] for r in rows])
                    ),
                    equal9_whole_change_H2=float(np.mean([r["whole_shoot_change"] for r in rows]))
                    / float(np.ptp(vertices[:, 1])) ** 2,
                )
            )
        pose_folder = leaf / "jeb_v5/reposition_fixed_domain"
        save_csv(pose_folder / "direction_averaged_responses.csv", responses)
        write_json(
            pose_folder / "all_reposition_states.json",
            dict(mode="synthetic_example", records=budgets),
        )
        write_json(
            pose_folder / "summary.json",
            dict(
                mode="synthetic_example",
                max_transition_closure_H2=closure,
                receiver_unchanged=bool(receiver_unchanged),
                directions=9,
                poses=3,
                grid=96,
            ),
        )
        checks.update(
            receiver_invariant=bool(receiver_unchanged),
            max_budget_closure=closure,
            removal_relations=len(edges),
            reposition_states=len(budgets),
        )

    def png_data(array):
        stream = io.BytesIO()
        Image.fromarray(np.asarray(array, np.uint8)).save(stream, format="PNG")
        return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")

    def observation():
        rows = json.loads((inputs / "image_manifest.json").read_text(encoding="utf-8"))["records"]
        lookup = {(r["session"], r["view"], r["camera"]): r for r in rows}
        folder = out / "spatial_fusion/synthetic_example"
        folder.mkdir(parents=True)
        statistics, examples = [], []
        max_error, min_iou = 0.0, 1.0
        finite_spatial = True
        colors = np.array([[0, 0, 0], [219, 77, 181], [43, 196, 217], [235, 235, 235]], np.uint8)
        for name in cases:
            for view in range(16):
                pair = [lookup[name, view, camera] for camera in ("rgb", "noir")]
                images = [np.array(Image.open(root / r["source"]).convert("RGB")) for r in pair]
                masks = [np.array(Image.open(root / r["mask"])) > 0 for r in pair]
                matrix, agreement, status = affine_align(masks[0], masks[1])
                max_error = max(max_error, float(np.max(np.abs(matrix - matrix_truth))))
                registered, field = warp_to_rgb(
                    images[1].astype(np.float32), matrix, masks[0].shape
                )
                registered_mask = (
                    warp_to_rgb(masks[1].astype(np.uint8), matrix, masks[0].shape, True)[0] > 0
                )
                iou = mask_iou(masks[0], registered_mask)
                min_iou = min(min_iou, iou)
                supports = dict(
                    rgb=interior(masks[0]),
                    noir=warp_to_rgb(
                        interior(masks[1]).astype(np.uint8), matrix, masks[0].shape, True
                    )[0]
                    > 0,
                )
                supports["cross"] = supports["rgb"] & supports["noir"] & field
                six = np.concatenate([images[0].astype(np.float32), registered], axis=-1)
                valid = np.concatenate(
                    [
                        (images[0] > 1) & (images[0] < 254),
                        warp_to_rgb(
                            ((images[1] > 1) & (images[1] < 254)).astype(np.float32),
                            matrix,
                            masks[0].shape,
                        )[0]
                        > 0.999999,
                    ],
                    axis=-1,
                )
                maps = spatial_indices(six, supports, valid)
                finite_spatial &= bool(
                    np.isfinite(six).all()
                    and all(not np.isinf(values).any() for values in maps.values())
                )
                frame_stats = {}
                for definition in definitions:
                    values = maps[definition["id"]]
                    stat = index_statistics(values, supports[definition["support"]])
                    stat.update(
                        display_below=int(
                            np.sum(np.isfinite(values) & (values < definition["vmin"]))
                        ),
                        display_above=int(
                            np.sum(np.isfinite(values) & (values > definition["vmax"]))
                        ),
                    )
                    statistics.append(dict(session=name, view=view, index=definition["id"], **stat))
                    frame_stats[definition["id"]] = stat
                np.savez_compressed(
                    folder / f"{name}_{view:02}.npz",
                    six_channels=six,
                    channel_valid=valid,
                    rgb_support=supports["rgb"],
                    noir_support=supports["noir"],
                    cross_support=supports["cross"],
                    **{"map_" + key: value for key, value in maps.items()},
                )
                frames.append(
                    dict(
                        session=name,
                        view=view,
                        known_affine=matrix_truth.tolist(),
                        estimated_affine=matrix.tolist(),
                        rgb_grid_iou=iou,
                        registration_status=status,
                        cross_support_pixels=int(supports["cross"].sum()),
                        sources=[r["source"] for r in pair],
                    )
                )
                if view in [0, 4, 8, 12]:
                    mapped = {}
                    for definition in definitions:
                        values = maps[definition["id"]]
                        scale = np.clip(
                            (values - definition["vmin"])
                            / (definition["vmax"] - definition["vmin"]),
                            0,
                            1,
                        )
                        image = np.zeros((*values.shape, 3), np.uint8)
                        good = np.isfinite(values)
                        p = scale[good]
                        image[good] = np.column_stack(
                            [
                                255 * np.minimum(2 * p, 1),
                                245 * (1 - abs(2 * p - 1)),
                                255 * np.minimum(2 * (1 - p), 1),
                            ]
                        ).astype(np.uint8)
                        mapped[definition["id"]] = png_data(image)
                    examples.append(
                        dict(
                            id=f"{name}_{view:02}",
                            session=name,
                            view=view,
                            images=dict(
                                rgb=png_data(images[0]),
                                noir=png_data(images[1]),
                                registered=png_data(registered),
                                overlap=png_data(colors[overlap_labels(masks[0], registered_mask)]),
                            ),
                            maps=mapped,
                            stats=frame_stats,
                        )
                    )
        save_csv(folder / "frame_index_statistics.csv", statistics)
        write_json(folder / "frame_provenance.json", frames)
        write_json(folder / "index_definitions.json", definitions)
        write_json(
            folder / "summary.json",
            dict(
                mode="synthetic_example",
                pairs=len(frames),
                acquisitions=2,
                definitions=27,
                provided_organ_labels=True,
                neural_detection_executed=False,
                minimum_registered_mask_iou=min_iou,
                maximum_affine_parameter_error=max_error,
            ),
        )
        payload = (
            json.dumps(
                dict(examples=examples, definitions=definitions),
                allow_nan=False,
                separators=(",", ":"),
            )
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
        )
        atlas = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Procedural paired-camera example</title><style>body{font:16px system-ui;margin:2rem;background:#f6f7f0;color:#183c31}img{max-width:100%;background:#000}section{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}label{display:inline-block;margin:1rem}figure{margin:0}select{padding:.5rem}pre{white-space:pre-wrap}</style><h1>Procedural synthetic observation example</h1><p>Two known generated shoots, sixteen views each. These PNG images, foregrounds, organ labels, cameras and their affine image transform are supplied synthetic truth. No SAM, YOLO, pretrained model or real plant image is used.</p><label>Example view <select id="frame"></select></label><label>Numerical definition <select id="index"></select></label><section><figure><img id="rgb"><figcaption>Procedural RGB input</figcaption></figure><figure><img id="noir"><figcaption>Procedural NoIR input</figcaption></figure><figure><img id="registered"><figcaption>Estimated affine registration</figcaption></figure><figure><img id="overlap"><figcaption>Mask agreement</figcaption></figure><figure><img id="map"><figcaption id="formula"></figcaption></figure><pre id="statistics"></pre></section><p>Every pair and all 27 formulas are calculated; this view shows four fixed indices per shoot. Values are procedural DN descriptors, with no physiological interpretation. Unsupported map pixels remain missing.</p><script id="example-data" type="application/json">__DATA__</script><script>const D=JSON.parse(document.getElementById('example-data').textContent),q=id=>document.getElementById(id);D.examples.forEach((e,i)=>q('frame').add(new Option(e.id,i)));D.definitions.forEach(d=>q('index').add(new Option(d.id,d.id)));function draw(){let e=D.examples[+q('frame').value],d=D.definitions.find(x=>x.id===q('index').value);for(let k of ['rgb','noir','registered','overlap'])q(k).src=e.images[k];q('map').src=e.maps[d.id];q('formula').textContent=d.id+' · '+d.formula;q('statistics').textContent=JSON.stringify(e.stats[d.id],null,2)}q('frame').onchange=q('index').onchange=draw;draw();</script></html>""".replace(
            "__DATA__", payload
        )
        atlas_path = folder / "CSI_maize_RGB_NoIR_spatial_atlas.html"
        atlas_path.write_text(atlas, encoding="utf-8")
        checker = _ResourceCheck()
        checker.feed(atlas)
        checks.update(
            known_affine_alignment=bool(min_iou > 0.999 and max_error < 1e-4),
            minimum_registered_mask_iou=min_iou,
            maximum_affine_parameter_error=max_error,
            spatial_rows=len(statistics),
            atlas_self_contained=checker.safe,
            finite_spatial_outputs=bool(finite_spatial),
        )

    def report():
        target = build_report(root)
        text = target.read_text(encoding="utf-8")
        checks["self_contained_html"] = bool(
            "srcdoc=" in text
            and "data:image/png;base64," in text
            and "__DATA__" not in text
            and "__CSS__" not in text
            and checks["atlas_self_contained"]
        )

    with threadpool_limits(limits=2):
        stage("procedural-inputs", generate)
        stage("measured-input-traits", image_traits)
        stage("assisted-inverse-fitting", fit)
        stage("fixed-receiver-interventions", interventions)
        stage("affine-and-spatial-observation", observation)
        stage("self-contained-report", report)
    finite = all(
        np.isfinite(list(trace.values())).all()
        for trace in traces.values()
        if isinstance(trace, dict)
    )
    json.dumps(state, allow_nan=False)
    checks.update(
        finite_outputs=bool(finite and checks["finite_spatial_outputs"]),
        fit_trace_lengths=all(traces[name]["steps"] == steps for name in cases),
        real_inverse_parameter_updates=all(
            traces[name]["changed_control_norm"] > 0 for name in cases
        ),
    )
    passed = (
        all(
            checks[key]
            for key in [
                "finite_outputs",
                "fit_trace_lengths",
                "real_inverse_parameter_updates",
                "receiver_invariant",
                "known_affine_alignment",
                "self_contained_html",
            ]
        )
        and checks["max_budget_closure"] < 1e-8
    )
    verification = dict(
        status="passed" if passed else "failed",
        mode="synthetic_example",
        checks=checks,
        traces=traces,
        neural_models_executed=False,
        models_downloaded=False,
        provided_truth=declaration["provided_inputs"],
        browser_interaction_review="Not performed by numerical/static resource verification.",
    )
    write_json(root / "verification.json", verification)
    if not passed:
        state["status"] = "failed"
        write_json(root / "run_manifest.json", state)
        append_event(root / "events.jsonl", "verification_failed", dict(mode="synthetic_example"))
        raise RuntimeError("Synthetic example numerical verification failed")
    state.update(status="completed", completed_utc=utc())
    paths = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.name not in {"run_manifest.json", "events.jsonl", "example.log"}
        and "__pycache__" not in p.parts
    ]
    state["output_sha256"] = file_manifest(root, paths)
    event_lines = (root / "events.jsonl").read_bytes().splitlines()
    state["event_count"] = len(event_lines)
    state["event_tail_sha256"] = (
        hashlib.sha256(event_lines[-1]).hexdigest() if event_lines else None
    )
    write_json(root / "run_manifest.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    arguments = parser.parse_args()
    _worker(arguments.worker, arguments.steps)
