"""Scientific invariants for spatial display-DN indices and camera support."""

import unittest
from bootstrap import BASE
import numpy as np
from spatial_fusion_core import (
    index_definitions,
    spatial_indices,
    index_statistics,
    overlap_labels,
    blend_projections,
)


class SpatialFusionTests(unittest.TestCase):
    def specimen(self):
        six = np.tile(np.array([30, 60, 20, 40, 10, 80], dtype=np.float32), (3, 4, 1))
        supports = {k: np.ones((3, 4), bool) for k in ("rgb", "noir", "cross")}
        return six, supports

    def test_twenty_seven_indices_and_fifteen_unique_pairs(self):
        defs = index_definitions()
        self.assertEqual(len(defs), 27)
        self.assertEqual(len({d["id"] for d in defs}), 27)
        self.assertEqual(sum(d["kind"] == "ND" for d in defs), 15)
        self.assertEqual(sum(d["support"] == "cross" for d in defs), 15)

    def test_multichannel_formulas_are_exact_and_gain_behavior_is_explicit(self):
        six, support = self.specimen()
        maps = spatial_indices(six, support)
        a, b = six[0, 0, :3], six[0, 0, 3:]
        ca, cb = a / a.sum(), b / b.sum()
        ga, gb = np.prod(a) ** (1 / 3), np.prod(b) ** (1 / 3)
        expected = dict(
            CAMSUM6=20 / 240,
            GREENSHARE6=ca[1] - cb[1],
            CHROMDIST6=np.sqrt(np.sum((ca - cb) ** 2) / 2),
            GMEAN6=(gb - ga) / (gb + ga),
            PRODUCT4=0.6,
            GLICONTRAST6=(70 / 170 + 100 / 140) / 2,
        )
        for key, value in expected.items():
            self.assertAlmostEqual(float(maps[key][0, 0]), float(value), places=6, msg=key)
        brighter = six.copy()
        brighter[..., 3:] *= 2
        changed = spatial_indices(brighter, support)
        self.assertAlmostEqual(
            float(maps["CHROMDIST6"][0, 0]), float(changed["CHROMDIST6"][0, 0]), places=6
        )
        self.assertNotAlmostEqual(
            float(maps["CAMSUM6"][0, 0]), float(changed["CAMSUM6"][0, 0]), places=6
        )

    def test_signed_uint8_difference_and_exact_rgb_formulas(self):
        six, support = self.specimen()
        maps = spatial_indices(six.astype(np.uint8), support)
        self.assertAlmostEqual(float(maps["ND_rgb_R__rgb_G"][0, 0]), -1 / 3, places=6)
        expected = dict(
            ExG=70 / 110,
            ExGR=88 / 110,
            GLI=70 / 170,
            VARI=30 / 70,
            MGRVI=2700 / 4500,
            RGBVI=3000 / 4200,
        )
        for key, value in expected.items():
            self.assertAlmostEqual(float(maps[key][0, 0]), value, places=6, msg=key)

    def test_dark_noir_green_does_not_suppress_other_pairs(self):
        six, support = self.specimen()
        six[..., 4] = 1
        maps = spatial_indices(six, support)
        self.assertTrue(np.isnan(maps["ND_rgb_R__noir_G"]).all())
        self.assertTrue(np.isfinite(maps["ND_rgb_R__noir_B"]).all())
        self.assertTrue(np.isfinite(maps["GLI"]).all())

    def test_separate_support_and_no_correspondence(self):
        six, support = self.specimen()
        support["cross"][:] = False
        support["rgb"][0] = False
        support["noir"][1] = False
        maps = spatial_indices(six, support)
        self.assertTrue(np.isnan(maps["ND_rgb_R__noir_R"]).all())
        self.assertTrue(np.isnan(maps["GLI"][0]).all())
        self.assertTrue(np.isfinite(maps["GLI"][1:]).all())
        self.assertTrue(np.isnan(maps["ND_noir_R__noir_B"][1]).all())
        self.assertTrue(np.isfinite(maps["ND_noir_R__noir_B"][0]).all())

    def test_dark_clipped_and_low_sum_are_not_values(self):
        six, support = self.specimen()
        six[0, 0, 0] = 0
        six[0, 1, 0] = 254
        six[0, 2, :2] = 4
        six[0, 3, 0] = np.nan
        value = spatial_indices(six, support)["ND_rgb_R__rgb_G"]
        self.assertTrue(np.isnan(value[0]).all())

    def test_vari_denominator_guard_and_unclipped_values(self):
        six, support = self.specimen()
        six[0, 0, :3] = [20, 80, 100]
        six[0, 1, :3] = [20, 80, 82]
        maps = spatial_indices(six, support)
        self.assertTrue(np.isnan(maps["VARI"][0, 0]))
        self.assertGreater(maps["VARI"][0, 1], 1)
        self.assertAlmostEqual(float(maps["VARI"][0, 1]), 60 / 18, places=5)

    def test_statistics_keep_empty_and_small_samples_explicit(self):
        a = np.full((2, 2), np.nan)
        row = index_statistics(a, np.zeros((2, 2), bool))
        self.assertEqual(row["valid_pixels"], 0)
        self.assertIsNone(row["median"])
        self.assertIsNone(row["valid_fraction"])
        a[0, 0] = 0
        row = index_statistics(a, np.ones((2, 2), bool))
        self.assertEqual(row["valid_fraction"], 0.25)
        self.assertIsNone(row["median"])
        row = index_statistics(a, np.ones((2, 2), bool), min_pixels=1)
        self.assertEqual(row["median"], 0)
        import json

        json.dumps(row, allow_nan=False)

    def test_statistics_never_include_values_outside_eligible(self):
        value = np.array([[0.2, 1e6], [0.2, 1e6]])
        support = np.array([[True, False], [True, False]])
        row = index_statistics(value, support, min_pixels=1)
        self.assertEqual(row["valid_pixels"], 2)
        self.assertEqual(row["median"], 0.2)

    def test_overlap_has_four_classes_in_rgb_coordinates(self):
        rgb = np.array([[0, 1], [0, 1]], bool)
        noir = np.array([[0, 0], [1, 1]], bool)
        np.testing.assert_array_equal(overlap_labels(rgb, noir), [[0, 1], [2, 3]])

    def test_blend_ignores_unsegmented_background(self):
        rgb = np.full((2, 2, 3), 200, np.uint8)
        noir = np.full((2, 2, 3), 40, np.float32)
        rm = np.array([[0, 1], [0, 1]], bool)
        nm = np.array([[0, 0], [1, 1]], bool)
        got = blend_projections(rgb, noir, rm, nm, 0.25)
        np.testing.assert_array_equal(got[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(got[1, 1], [160, 160, 160])
        np.testing.assert_array_equal(got[0, 1], [150, 150, 150])
        np.testing.assert_array_equal(got[1, 0], [10, 10, 10])

    def test_shape_mismatch_rejected(self):
        six, support = self.specimen()
        support["cross"] = np.ones((2, 4), bool)
        with self.assertRaises(ValueError):
            spatial_indices(six, support)

    def test_preinterpolation_guard_cannot_be_rescued_by_bright_neighbour(self):
        six, support = self.specimen()
        channel_valid = np.ones_like(six, bool)
        channel_valid[0, 0, 4] = False
        maps = spatial_indices(six, support, channel_valid=channel_valid)
        self.assertTrue(np.isnan(maps["ND_rgb_R__noir_G"][0, 0]))
        self.assertTrue(np.isfinite(maps["ND_rgb_R__noir_B"][0, 0]))

    def test_multichannel_requires_signal_in_each_camera(self):
        six, support = self.specimen()
        six[..., 3:] = 3
        maps = spatial_indices(six, support)
        for key in ("CAMSUM6", "GREENSHARE6", "CHROMDIST6", "GMEAN6", "GLICONTRAST6"):
            self.assertTrue(np.isnan(maps[key]).all(), key)

    def test_actual_half_pixel_warp_keeps_dark_contributor_invalid(self):
        from channel_fusion_core import warp_to_rgb

        source = np.full((8, 8, 3), 40, np.float32)
        source[..., 1] = 9
        source[:, :4, 1] = 1
        matrix = np.array([[1, 0, -0.5], [0, 1, 0]], np.float32)
        moved, field = warp_to_rgb(source, matrix, (8, 8))
        contributors = (
            warp_to_rgb(((source > 1) & (source < 254)).astype(np.float32), matrix, (8, 8))[0]
            > 0.999999
        )
        mixed = (moved[..., 1] > 1) & (moved[..., 1] < 9) & field
        self.assertTrue(mixed.any())
        self.assertFalse(contributors[..., 1][mixed].any())
        six = np.concatenate([np.full_like(moved, 50), moved], axis=2)
        channel_valid = np.concatenate([np.ones_like(moved, bool), contributors], axis=2)
        support = {key: field for key in ("rgb", "noir", "cross")}
        values = spatial_indices(six, support, channel_valid)["ND_rgb_R__noir_G"]
        self.assertTrue(np.isnan(values[mixed]).all())


if __name__ == "__main__":
    unittest.main()
