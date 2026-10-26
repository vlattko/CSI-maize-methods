"""Meaningful partition-boundary tests for the post-hoc fixed diagnostic."""

import unittest
import numpy as np
from performance_controls import build_design, partition_fields


class PartitionTests(unittest.TestCase):
    def test_unseen_metadata_and_extreme_test_value_do_not_fit_preprocessing(self):
        rows = [
            {"measurement": "1", "genotype": "A"},
            {"measurement": "2", "genotype": "A"},
            {"measurement": "2", "genotype": "UNSEEN"},
        ]
        values = np.array([[1.0], [3.0], [1000000.0]])
        train, test, audit = build_design(
            rows, np.array([0, 1]), np.array([2]), values, ["x"], True
        )
        self.assertEqual(audit["metadata_categories"], [["1", "2"], ["A"]])
        self.assertEqual(audit["scaler_mean"][-1], 2.0)
        self.assertEqual(audit["scaler_scale"][-1], 1.0)
        self.assertEqual(test[0, -1], 999998.0)
        np.testing.assert_allclose(train[:, -1], [-1.0, 1.0])

    def test_semantic_partition_excludes_quality_and_acquisition_proxies(self):
        names = [
            "rgb__visible_height_px__median",
            "noir__g_chromaticity_median__iqr",
            "rgb__skeleton_length_px__median",
            "rgb__centroid_x_fraction__median",
            "rgb__bottom_censored__max",
            "rgb__saturation_clip_fraction__median",
        ]
        families = partition_fields(names)
        self.assertEqual(families["geometry"], names[:1])
        self.assertEqual(families["optical"], names[1:2])
        self.assertEqual(families["excluded"], names[2:])


if __name__ == "__main__":
    unittest.main()
