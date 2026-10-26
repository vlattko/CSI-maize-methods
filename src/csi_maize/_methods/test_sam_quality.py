from bootstrap import BASE
import unittest
import numpy as np
from sam_traits import quality_checks


class SegmentationQualityChecks(unittest.TestCase):
    def test_empty_prediction_is_rejected(self):
        image = np.zeros((100, 100, 3), np.uint8)
        seed = np.zeros((100, 100), bool)
        seed[40:95, 48:52] = True
        result = quality_checks(image, seed, np.zeros_like(seed), 0.95)
        self.assertFalse(result["qc_pass"])
        self.assertIn("seed_not_covered", result["qc_issues"])

    def test_high_model_score_cannot_rescue_background_leakage(self):
        image = np.zeros((100, 100, 3), np.uint8)
        seed = np.zeros((100, 100), bool)
        seed[40:95, 48:52] = True
        image[seed] = [40, 120, 20]
        mask = seed.copy()
        mask[40:60, 52:60] = True
        result = quality_checks(image, seed, mask, 0.99)
        self.assertFalse(result["qc_pass"])
        self.assertIn("dark_background_leakage", result["qc_issues"])

    def test_clean_thin_shoot_passes_operational_checks(self):
        image = np.zeros((100, 100, 3), np.uint8)
        seed = np.zeros((100, 100), bool)
        seed[40:95, 48:52] = True
        image[seed] = [40, 120, 20]
        result = quality_checks(image, seed, seed, 0.96)
        self.assertTrue(result["qc_pass"])
        self.assertEqual(result["seed_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
