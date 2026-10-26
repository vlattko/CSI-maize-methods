import tempfile
import unittest
from pathlib import Path
import numpy as np
from common import filename_key, safe_source
from features import select_views, measure
from benchmark import ridge_predict, fold_indices
import features
import common


class ScientificBoundaries(unittest.TestCase):
    def test_lower_plant_component_ignores_large_top_background(self):
        mask = np.zeros((100, 100), bool)
        mask[5:40, 10:90] = True
        mask[55:96, 48:52] = True
        selected = features.plant_component(mask)
        self.assertEqual(selected[:40].sum(), 0)
        self.assertEqual(selected[55:96, 48:52].sum(), 164)

    def test_view_counts_and_pair_indices(self):
        a = [Path(f"image_rgb_{i:03}.jpg") for i in range(30)]
        b = [Path(f"image_noir_{i:03}.jpg") for i in range(30)]
        self.assertTrue(common.complete_views(a, b, [30, 40]))
        b[-1] = Path("image_noir_030.jpg")
        self.assertFalse(common.complete_views(a, b, [30, 40]))

    def test_timestamp_names_are_never_guessed(self):
        self.assertIsNone(filename_key("capture_session_2025-03-26_07-03-52"))
        self.assertEqual(filename_key("capture_session_C2_5-2"), ("C", "2", "Os6251/15", "2"))

    def test_source_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            safe_source("../outside")

    def test_genotypes_never_cross_validation_boundary(self):
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        for _, train, test in fold_indices(groups):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            self.assertEqual(len(train) + len(test), len(groups))

    def test_constant_features_recover_training_mean(self):
        x = np.ones((6, 3))
        y = np.arange(6, dtype=float)[:, None]
        actual = ridge_predict(x, y, np.ones((2, 3)), 10)
        np.testing.assert_allclose(actual, [[2.5], [2.5]])

    def test_evenly_spaced_views_and_missing_data(self):
        paths = [Path(f"image_{i:03}.jpg") for i in range(40)]
        self.assertEqual(select_views(paths, 4), [paths[i] for i in [0, 10, 20, 30]])
        with self.assertRaises(ValueError):
            select_views(paths, 41)

    def test_empty_image_cannot_be_a_feature_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jpg"
            path.write_bytes(b"")
            with self.assertRaises(Exception):
                measure(path, "rgb", 640)


if __name__ == "__main__":
    unittest.main()
