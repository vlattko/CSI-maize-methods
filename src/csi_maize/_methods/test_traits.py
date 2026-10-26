from bootstrap import BASE
import unittest
import numpy as np
from traits import measure_mask, segment, prune_skeleton


class TraitChecks(unittest.TestCase):
    def test_noir_blue_background_does_not_become_foreground(self):
        image = np.zeros((300, 300, 3), np.uint8)
        image[5:290, 15:285, 0] = 5
        image[5:290, 15:285, 2] = np.linspace(60, 20, 285).astype(np.uint8)[:, None]
        image[140:, 148:151] = [25, 1, 160]
        mask = segment(image, "noir")
        self.assertEqual(mask[:130].sum(), 0)
        self.assertGreater(mask[160:, 148:151].sum(), 300)

    def test_short_skeleton_burr_is_removed_without_deleting_long_branch(self):
        skeleton = np.zeros((100, 100), bool)
        skeleton[10:95, 50] = True
        skeleton[60, 50:85] = True
        skeleton[30, 50:56] = True
        pruned = prune_skeleton(skeleton, 12)
        self.assertEqual(pruned[30, 53:56].sum(), 0)
        self.assertTrue(pruned[60, 80])

    def test_visible_extent_area_and_chromaticity_on_known_shape(self):
        image = np.zeros((100, 100, 3), np.uint8)
        mask = np.zeros((100, 100), bool)
        mask[30:90, 40:60] = True
        image[mask] = [40, 120, 20]
        result, skeleton = measure_mask(image, mask, "rgb")
        self.assertEqual(result["silhouette_area_px"], 1200)
        self.assertEqual(result["visible_height_px"], 60)
        self.assertEqual(result["silhouette_width_px"], 20)
        self.assertAlmostEqual(result["g_chromaticity_median"], 2 / 3)
        self.assertEqual(result["bottom_censored"], 0)
        self.assertEqual(result["skeleton_endpoint_proxy"], 2)

    def test_frame_truncation_is_flagged(self):
        image = np.zeros((100, 100, 3), np.uint8)
        mask = np.zeros((100, 100), bool)
        mask[30:, 40:60] = True
        image[mask] = [30, 100, 20]
        values, _ = measure_mask(image, mask, "rgb")
        self.assertEqual(values["bottom_censored"], 1)

    def test_background_is_not_a_plant(self):
        image = np.zeros((100, 100, 3), np.uint8)
        with self.assertRaises(ValueError):
            segment(image, "rgb")

    def test_greenish_cloth_is_rejected_even_near_shoot(self):
        image = np.zeros((150, 150, 3), np.uint8)
        image[10:65, 25:100] = [50, 70, 50]
        image[60:95, 68:80] = [50, 70, 50]
        image[90:, 70:78] = [40, 120, 20]
        mask = segment(image, "rgb")
        self.assertLess(mask[:65].sum(), 10)
        self.assertGreater(mask[95:, 70:78].sum(), 300)


if __name__ == "__main__":
    unittest.main()
