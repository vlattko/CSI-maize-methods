import unittest
from bootstrap import BASE
import numpy as np
from channel_fusion_core import (
    CHANNELS,
    PAIRS,
    warp_to_rgb,
    match_instances,
    normalized_difference,
    pair_features,
    interior,
    organ_features,
)


class FusionTests(unittest.TestCase):
    def test_forward_rgb_to_noir_is_inverted_for_sampling(self):
        rgb = np.zeros((20, 30), bool)
        rgb[5:10, 7:12] = 1
        noir = np.zeros((20, 30, 3), np.uint8)
        noir[7:12, 11:16] = [40, 80, 120]
        moved, valid = warp_to_rgb(noir, np.array([[1, 0, 4], [0, 1, 2]], float), rgb.shape)
        np.testing.assert_array_equal(moved[rgb], np.tile([40, 80, 120], (25, 1)))
        self.assertFalse(valid[-1, -1])
        self.assertTrue(valid[7, 9])

    def test_one_to_one_unequal_and_no_forced_match(self):
        selected = match_instances(np.array([[0.8, 0.1], [0.2, 0.9], [0.1, 0.2]]))
        self.assertEqual([(r["i"], r["j"]) for r in selected if r["accepted"]], [(0, 0), (1, 1)])
        self.assertEqual(match_instances(np.zeros((2, 3))), [])

    def test_subpixel_border_requires_full_linear_field_but_nearest_is_unchanged(self):
        noir = np.full((6, 6), 80, np.float32)
        matrix = np.array([[1, 0, 1 / 32], [0, 1, 1 / 32]], np.float32)
        moved, field = warp_to_rgb(noir, matrix, noir.shape)
        self.assertEqual(float(moved[5, 5]), 80 * (31 / 32) ** 2)
        self.assertFalse(field[5, 5])
        self.assertFalse(field[5, 4])
        self.assertFalse(field[4, 5])
        self.assertTrue(field[4, 4])
        nearest, nearest_field = warp_to_rgb(noir, matrix, noir.shape, nearest=True)
        self.assertEqual(float(nearest[5, 5]), 80)
        self.assertTrue(nearest_field.all())

    def test_ambiguous_matches_not_accepted(self):
        self.assertFalse(
            any(r["accepted"] for r in match_instances(np.array([[0.8, 0.78], [0.77, 0.79]])))
        )

    def test_empty_instances(self):
        self.assertEqual(match_instances(np.empty((0, 3))), [])

    def test_fifteen_unique_pairs_and_unsigned_overflow_avoided(self):
        self.assertEqual(len(CHANNELS), 6)
        self.assertEqual(len(PAIRS), 15)
        self.assertAlmostEqual(float(normalized_difference(np.uint8(10), np.uint8(30))), -0.5)

    def test_zero_dark_and_clipped_are_missing(self):
        a = np.array([0, 1, 20, 254, 100])
        b = np.array([0, 30, 10, 20, 255])
        got = normalized_difference(a, b)
        self.assertTrue(np.isnan(got[[0, 1, 3, 4]]).all())
        self.assertAlmostEqual(got[2], 1 / 3)
        self.assertTrue(np.isnan(normalized_difference(5, 5)))

    def test_pairwise_support_not_all_six_filter(self):
        samples = np.array([[30, 50, 20, 80, 0, 150], [35, 55, 25, 90, 0, 160]], float)
        got = pair_features(samples, min_pixels=2)
        self.assertEqual(got["rgb_R__rgb_G"]["n"], 2)
        self.assertEqual(got["rgb_R__noir_G"]["n"], 0)

    def test_estimands_differ(self):
        samples = np.array(
            [[10, 10, 10, 100, 30, 30], [20, 10, 10, 20, 30, 30], [100, 10, 10, 10, 30, 30]], float
        )
        got = pair_features(samples, min_pixels=1)["rgb_R__noir_R"]
        self.assertAlmostEqual(got["pixel_median"], 0.0)
        # Different constructed samples make medians occur at different pixels.
        samples[:, 3] = [30, 100, 20]
        got = pair_features(samples, min_pixels=1)["rgb_R__noir_R"]
        self.assertNotAlmostEqual(got["pixel_median"], got["ratio_of_medians"])

    def test_common_gain_cancels_relative_gain_does_not(self):
        a = np.array([20.0, 30.0])
        b = np.array([50.0, 60.0])
        np.testing.assert_allclose(normalized_difference(a, b), normalized_difference(a * 2, b * 2))
        self.assertFalse(np.allclose(normalized_difference(a, b), normalized_difference(a, b * 2)))

    def test_erosion_never_invents_support(self):
        mask = np.zeros((10, 10), bool)
        mask[2:8, 2:8] = 1
        eroded = interior(mask)
        self.assertFalse((eroded & ~mask).any())
        self.assertEqual(eroded.sum(), 16)

    def test_low_sum_organ_medians_are_null_and_json_safe(self):
        import json

        got = organ_features(np.full((40, 3), 3.0), np.full((40, 3), 4.0))
        self.assertTrue(all(v is None for v in got["nd_of_separate_visible_medians"].values()))
        json.dumps(got, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
