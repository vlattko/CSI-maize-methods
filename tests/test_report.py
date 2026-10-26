"""Public report contracts, independent of private acquisitions or checkpoints."""

import csv
import hashlib
import html
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


class ReportTests(unittest.TestCase):
    def root_json(self, root, name, value):
        (root / name).write_text(json.dumps(value), encoding="utf-8")

    def test_procedural_disclosure_is_central_for_config_and_manifest_modes(self):
        build, _ = self.api()
        for name, mode in [
            ("run_config.json", "example"),
            ("run_manifest.json", "synthetic_example"),
        ]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.root_json(root, name, dict(mode=mode, status="completed"))
                text = build(root).read_text(encoding="utf-8")
                self.assertIn("PROCEDURAL SYNTHETIC EXAMPLE", text)
                self.assertIn("complete projected paths", text)
                self.assertIn("No pretrained segmentation or detection", text)
                self.assertNotIn("Whole-plant masks are automated observations", text)

    def test_archived_results_without_run_mode_are_explicit_inspection(self):
        build, _ = self.api()
        with tempfile.TemporaryDirectory() as directory:
            text = build(Path(directory)).read_text(encoding="utf-8")
            self.assertIn("STORED RESULTS INSPECTION", text)
            self.assertIn("not a raw-data recomputation", text)
            self.assertIn('"mode":"stored_results_inspection"', text)

    def test_actual_noncomplete_raw_status_is_visible(self):
        build, _ = self.api()
        for status in ["failed", "running", "partial"]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.root_json(root, "run_manifest.json", dict(mode="raw", status=status))
                text = build(root).read_text(encoding="utf-8")
                self.assertIn(f"Execution status at report generation: {status}", text)
                self.assertIn('"mode":"raw"', text)

    def test_completed_manifest_report_is_preserved_and_rebuild_gets_new_name(self):
        build, _ = self.api()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "reports/CSI_maize_architecture.html"
            target.parent.mkdir()
            recorded = b"Recorded completed report; preserve these bytes."
            target.write_bytes(recorded)
            self.root_json(
                root,
                "run_manifest.json",
                dict(
                    mode="raw",
                    status="completed",
                    output_sha256={
                        "reports/CSI_maize_architecture.html": hashlib.sha256(recorded).hexdigest()
                    },
                ),
            )
            rebuilt = build(root)
            self.assertNotEqual(rebuilt, target)
            self.assertIn("_rebuilt", rebuilt.stem)
            self.assertEqual(target.read_bytes(), recorded)
            self.assertIn(
                "Execution status at report generation: completed",
                rebuilt.read_text(encoding="utf-8"),
            )
            with self.assertRaisesRegex(ValueError, "manifest"):
                build(root, target)
            outside = root / "inspection-copy.html"
            self.assertEqual(build(root, outside), outside)

    def test_cli_example_rebuild_keeps_disclosure_and_manifested_original(self):
        self.api()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "reports/CSI_maize_architecture.html"
            target.parent.mkdir()
            target.write_bytes(b"Frozen procedural report")
            self.root_json(
                root,
                "run_manifest.json",
                dict(
                    mode="synthetic_example",
                    status="completed",
                    output_sha256={
                        "reports/CSI_maize_architecture.html": hashlib.sha256(
                            target.read_bytes()
                        ).hexdigest()
                    },
                ),
            )
            result = subprocess.run(
                [sys.executable, "-m", "csi_maize", "report", "--run", str(root)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rebuilt = Path(result.stdout.strip())
            self.assertNotEqual(rebuilt, target)
            self.assertEqual(target.read_bytes(), b"Frozen procedural report")
            self.assertIn("PROCEDURAL SYNTHETIC EXAMPLE", rebuilt.read_text(encoding="utf-8"))

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("csi_maize.report"))
        from csi_maize.report import build_report, build_atlas_template

        return build_report, build_atlas_template

    def save_json(self, root, relative, value):
        path = root / "prototype/research_v2/outputs" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_missing_dataset_is_explicit_and_build_is_self_contained(self):
        build, _ = self.api()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = build(root)
            self.assertTrue(result.is_file())
            text = result.read_text(encoding="utf-8")
            self.assertIn("Leaf-scale architectural experiments", text)
            self.assertIn("Geometry unavailable", text)
            self.assertIn("Removal outcomes unavailable", text)
            self.assertIn("Reposition outcomes unavailable", text)
            self.assertIn("Spatial atlas unavailable", text)
            self.assertIn("Content-Security-Policy", text)
            self.assertIn("connect-src 'none'", text)
            self.assertNotIn(str(root), text)
            self.assertNotIn("1,640", text)
            self.assertNotRegex(text, r'<(?:script|link|img)[^>]+(?:src|href)=["\']https?://')
            self.assertLess(text.index('id="architecture"'), text.index('id="observations"'))

    def test_mesh_labels_and_script_payload_are_escaped_without_private_metadata(self):
        build, _ = self.api()
        hostile = '</script><img src=x onerror="alert(1)">'
        mesh = dict(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
            colors=[[20, 100, 50]] * 3,
            organs=[dict(leaf_id=hostile, vertex_start=0, vertex_count=3)],
            source="C:/Users/private-person/private-input",
            author="unapproved-author",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_json(
                root, "organ_inverse_triangles/capture_session_C2_5-1/browser_mesh.json", mesh
            )
            output = root / "custom.html"
            self.assertEqual(build(root, output), output)
            text = output.read_text(encoding="utf-8")
            self.assertNotIn(hostile, text)
            self.assertNotIn("private-person", text)
            self.assertNotIn("unapproved-author", text)
            match = re.search(
                r'<script id="architecture-data" type="application/json">(.*?)</script>', text, re.S
            )
            data = json.loads(match.group(1))
            self.assertEqual(data["mesh"]["organs"][0]["label"], hostile)
            self.assertEqual(data["mesh"]["faces"], [[0, 1, 2]])

    def test_actual_removal_reposition_and_recovery_values_are_present(self):
        build, _ = self.api()
        base = "leaf_dynamics/2026-10-24/"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_json(
                root,
                base + "jeb_vision_v2/numerical_stability_summary.json",
                dict(
                    edge_summaries=[
                        dict(
                            removed_leaf="leaf_4",
                            receiver_leaf="leaf_2",
                            baseline_mean256=0.125,
                            baseline_mean512=0.126,
                            grid_min_mean512=0.10,
                            grid_max_mean512=0.15,
                            all_eight_resolved_positive=True,
                        )
                    ]
                ),
            )
            path = (
                root
                / "prototype/research_v2/outputs"
                / base
                / "jeb_v5/reposition_fixed_domain/direction_averaged_responses.csv"
            )
            path.parent.mkdir(parents=True)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "candidate_id",
                        "resolution",
                        "mode",
                        "angle_degrees",
                        "equal9_receiver_change_fraction",
                        "equal9_whole_change_H2",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    dict(
                        candidate_id=0,
                        resolution=512,
                        mode="yaw_degrees",
                        angle_degrees=10,
                        equal9_receiver_change_fraction=0.08,
                        equal9_whole_change_H2=-0.003,
                    )
                )
            self.save_json(
                root,
                base + "jeb_v5/recovery/summary.json",
                dict(
                    results=[
                        dict(
                            case=dict(id="three_leaf_clean"),
                            status="completed",
                            metrics=dict(
                                initial=dict(
                                    mean_withheld_clean_mask_iou=0.2,
                                    surface_symmetric_distance=4,
                                    release_mean_abs_error=0.02,
                                ),
                                fitted=dict(
                                    mean_withheld_clean_mask_iou=0.75,
                                    surface_symmetric_distance=0.8,
                                    release_mean_abs_error=0.04,
                                ),
                            ),
                        )
                    ]
                ),
            )
            text = build(root).read_text(encoding="utf-8")
            self.assertIn("leaf_4", text)
            self.assertIn("0.1260", text)
            self.assertIn("0.0800", text)
            self.assertIn("-0.0030", text)
            self.assertIn("0.7500", text)
            self.assertIn("Worsened", text)
            self.assertIn("privileged", text)

    def test_removal_uses_current_family_resolution_key(self):
        build, _ = self.api()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_json(
                root,
                "leaf_dynamics/2026-10-24/jeb_vision_v2/numerical_stability_summary.json",
                dict(
                    edge_summaries=[
                        dict(
                            removed_leaf="donor",
                            receiver_leaf="receiver",
                            baseline_mean512=0.1,
                            grid_min_mean512=0.08,
                            grid_max_mean512=0.12,
                            all_configurations_resolved_positive=True,
                        )
                    ]
                ),
            )
            text = build(root).read_text(encoding="utf-8")
            self.assertIn("donor → receiver", text)
            self.assertIn("<td>Yes</td>", text)

    def test_supporting_atlas_is_embedded_with_escaped_srcdoc(self):
        build, _ = self.api()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atlas = (
                root
                / "prototype/research_v2/outputs/spatial_fusion/2026-10-26/CSI_maize_RGB_NoIR_spatial_atlas.html"
            )
            atlas.parent.mkdir(parents=True)
            # Encoded image bytes can coincidentally contain ordinary words;
            # privacy checks must inspect textual content rather than image data.
            content = '<!doctype html><html><body><h1>Supporting specimen</h1><img src="data:image/png;base64,ZmFrZQ=="><script>const x="quoted";</script></body></html>'
            atlas.write_text(content, encoding="utf-8")
            text = build(root).read_text(encoding="utf-8")
            match = re.search(r'<iframe\b[^>]*srcdoc="(.*?)"', text, re.S)
            self.assertIsNotNone(match)
            self.assertEqual(html.unescape(match.group(1)), content)
            self.assertIn('loading="lazy"', text)
            self.assertIn(
                'sandbox="allow-scripts allow-downloads allow-popups allow-popups-to-escape-sandbox"',
                text,
            )

    def test_unsafe_atlas_resources_or_private_paths_are_not_embedded(self):
        build, _ = self.api()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atlas = (
                root
                / "prototype/research_v2/outputs/spatial_fusion/2026-10-26/CSI_maize_RGB_NoIR_spatial_atlas.html"
            )
            atlas.parent.mkdir(parents=True)
            atlas.write_text(
                '<script src="https://example.org/remote.js"></script>C:/Users/private-name',
                encoding="utf-8",
            )
            text = build(root).read_text(encoding="utf-8")
            self.assertNotIn("private-name", text)
            self.assertNotIn("remote.js", text)
            self.assertIn("Spatial atlas unavailable", text)

    def test_atlas_typography_preserves_application_json_exactly(self):
        from csi_maize.report import format_atlas_html

        payload = (
            """{"text":"ctx.font='13px sans-serif'","html":"</footer>","values":[0.1,null,2]}"""
        )
        source = (
            '<html><head><style>body{background:beige}</style></head><body><footer></footer><script id="atlas-data" type="application/json">'
            + payload
            + """</script><script>ctx.font='13px sans-serif';</script></body></html>"""
        )
        formatted = format_atlas_html(source)
        self.assertIn(
            '<script id="atlas-data" type="application/json">' + payload + "</script>", formatted
        )
        self.assertEqual(format_atlas_html(formatted), formatted)
        self.assertEqual(formatted.count('id="atlas-external-navigation"'), 1)
        self.assertIn("--paper: #ffffff", formatted)
        self.assertIn("Times New Roman", formatted)

    def test_template_merges_readable_assets_and_preserves_all_population_tokens(self):
        _, template = self.api()
        text = template()
        for token in [
            "__PAYLOAD__",
            "__INVESTIGATION__",
            "__FIG_01_same_plant_overlap__",
            "__FIG_02_fifteen_channel_pair_maps__",
            "__FIG_03_rgb_and_multichannel_maps__",
            "__FIG_04_feature_availability__",
        ]:
            self.assertIn(token, text)
        self.assertNotIn("__ATLAS_CSS__", text)
        self.assertNotIn("__ATLAS_JS__", text)
        self.assertNotIn("127.0.0.1", text)
        self.assertNotIn("manuscript preserved", text)
        self.assertNotRegex(text, r'<(?:script|link)[^>]+(?:src|href)=["\']https?://')
        source = Path(importlib.util.find_spec("csi_maize.report").origin).parent / "web"
        self.assertGreater(len((source / "atlas.js").read_text(encoding="utf-8").splitlines()), 200)
        self.assertGreater(
            len((source / "atlas.css").read_text(encoding="utf-8").splitlines()), 100
        )

    def test_geometry_consistency_scores_are_labeled_as_development_diagnostics(self):
        build, _ = self.api()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_json(
                root,
                "organ_inverse_triangles/capture_session_C2_5-1/summary.json",
                dict(
                    train_indices=[0, 1, 3],
                    heldout_indices=[2],
                    triangle_render_development_heldout=dict(iou=0.71, precision=0.82, recall=0.78),
                ),
            )
            text = build(root).read_text(encoding="utf-8")
            self.assertIn("0.7100", text)
            self.assertIn("Development-view mask consistency", text)
            self.assertIn("not independent anatomy accuracy", text)


if __name__ == "__main__":
    unittest.main()
