"""Actual offline procedural inputs through fitting, interventions and report."""

import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from csi_maize.example import run_example
from csi_maize.engine import verify_run
from csi_maize.provenance import digest


class ExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="maize procedural ")
        cls.root = Path(cls.temporary.name) / "example run"
        cls.result = run_example(cls.root, steps=30)
        cls.outputs = cls.root / "prototype/research_v2/outputs"

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_mode_provenance_and_actual_iteration_traces(self):
        verification = self.result["verification"]
        self.assertEqual(verification["status"], "passed")
        self.assertEqual(verification["mode"], "synthetic_example")
        self.assertFalse(verification["neural_models_executed"])
        self.assertFalse(verification["models_downloaded"])
        for name in ["example_3leaf", "example_5leaf"]:
            trace = verification["traces"][name]
            self.assertEqual(trace["steps"], 30)
            self.assertGreater(trace["changed_control_norm"], 0)
            folder = self.outputs / "leaf_dynamics/2026-10-24/jeb_v5/recovery" / name
            with (folder / "loss.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 30)
            self.assertTrue(np.isfinite([float(r["loss"]) for r in rows]).all())
            with (
                np.load(folder / "initial_parameters.npz") as initial,
                np.load(folder / "fitted_parameters.npz") as fitted,
            ):
                self.assertFalse(np.array_equal(initial["control"], fitted["control"]))
                self.assertTrue(all(np.isfinite(fitted[key]).all() for key in fitted.files))
        self.assertFalse((self.root / "prototype/research_v2/models").exists())

    def test_traits_are_measured_from_saved_input_pixels_and_truth_masks(self):
        manifest = json.loads(
            (self.root / "inputs/image_manifest.json").read_text(encoding="utf-8")
        )
        with (self.outputs / "sam_traits/view_traits.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            traits = list(csv.DictReader(stream))
        self.assertEqual(len(traits), len(manifest["records"]))
        indexed = {(r["session"], r["camera"], int(r["view"])): r for r in traits}
        for source in manifest["records"]:
            self.assertEqual(digest(self.root / source["source"]), source["source_sha256"])
            self.assertEqual(digest(self.root / source["mask"]), source["mask_sha256"])
            mask = np.array(Image.open(self.root / source["mask"])) > 0
            image = np.array(Image.open(self.root / source["source"]).convert("RGB"))
            row = indexed[source["session"], source["camera"], source["view"]]
            self.assertEqual(int(row["silhouette_area_px"]), int(mask.sum()))
            self.assertAlmostEqual(float(row["g_median"]), float(np.median(image[mask, 1])))
            self.assertEqual(row["mask_origin"], "provided_procedural_truth")
            self.assertEqual(row["segmentation_model"], "not_executed")

    def test_saved_spatial_values_follow_the_actual_aligned_channels(self):
        folder = self.outputs / "spatial_fusion/synthetic_example"
        definitions = json.loads((folder / "index_definitions.json").read_text(encoding="utf-8"))
        frames = json.loads((folder / "frame_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(len(definitions), 27)
        self.assertEqual(len(frames), 32)
        for frame in frames:
            np.testing.assert_allclose(frame["estimated_affine"], frame["known_affine"], atol=1e-4)
            self.assertGreater(frame["rgb_grid_iou"], 0.999)
        supported = max(frames, key=lambda row: row["cross_support_pixels"])
        with np.load(folder / f"{supported['session']}_{supported['view']:02}.npz") as saved:
            six = saved["six_channels"]
            mask = saved["cross_support"]
            # Independently replay a two-channel and four-channel definition.
            expected = (six[..., 1] - six[..., 5]) / (six[..., 1] + six[..., 5] + (~mask))
            np.testing.assert_allclose(
                saved["map_ND_rgb_G__noir_B"][mask], expected[mask], atol=1e-7
            )
            p, q = six[..., 5] * six[..., 1], six[..., 0] * six[..., 3]
            expected4 = (p - q) / (p + q + (~mask))
            np.testing.assert_allclose(saved["map_PRODUCT4"][mask], expected4[mask], atol=1e-7)
            self.assertGreater(int(mask.sum()), 40)
            for definition in definitions:
                values = saved["map_" + definition["id"]]
                self.assertFalse(np.isinf(values).any())

    def test_transition_ledger_reconstructs_fixed_receiver_and_signed_budget(self):
        folder = self.outputs / "leaf_dynamics/2026-10-24/jeb_v5/reposition_fixed_domain"
        states = json.loads((folder / "all_reposition_states.json").read_text(encoding="utf-8"))[
            "records"
        ]
        self.assertEqual(len(states), 27)
        self.assertTrue(self.result["verification"]["checks"]["receiver_invariant"])
        for state in states:
            transition = np.array(state["transition_pixels"])
            self.assertEqual(transition.sum(), 96**2)
            self.assertAlmostEqual(state["closure"], 0, places=8)
            self.assertAlmostEqual(
                sum(state["organ_area_changes"]) + state["escape_change"], 0, places=8
            )
            if state["angle_degrees"] == 0:
                self.assertEqual(int(np.trace(transition)), int(transition.sum()))
                self.assertEqual(state["receiver_change_fraction"], 0)

    def test_report_embeds_numerical_results_and_explicit_example_scope(self):
        text = Path(self.result["report"]).read_text(encoding="utf-8")
        self.assertIn("PROCEDURAL SYNTHETIC EXAMPLE", text)
        self.assertIn("example_3leaf", text)
        self.assertIn("example_5leaf", text)
        self.assertIn("data:image/png;base64,", text)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("Geometry unavailable", text)
        self.assertNotIn("Removal outcomes unavailable", text)
        self.assertNotIn("Reposition outcomes unavailable", text)
        self.assertNotIn("Spatial atlas unavailable", text)

        class Resources(HTMLParser):
            def handle_starttag(parser, tag, attrs):
                data = dict(attrs)
                if "src" in data:
                    self.assertTrue(data["src"].startswith("data:"))
                if tag == "iframe":
                    self.assertIn("srcdoc", data)
                    Resources().feed(data["srcdoc"])

        Resources().feed(text)

    def test_manifest_detects_modified_procedural_input(self):
        self.assertEqual(verify_run(self.root)["mode"], "synthetic_example")
        path = self.root / "inputs/example_3leaf/rgb_00.png"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"changed")
            with self.assertRaises(ValueError):
                verify_run(self.root)
        finally:
            path.write_bytes(original)


class ExampleInputTests(unittest.TestCase):
    def test_invalid_step_counts_and_nonempty_destinations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for steps in [0, -1, 351, True, 1.5]:
                with self.assertRaises(ValueError):
                    run_example(root / "unused", steps=steps)
            (root / "existing.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                run_example(root)
            self.assertEqual((root / "existing.txt").read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
