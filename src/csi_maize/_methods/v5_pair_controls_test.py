"""Known-answer behavioral tests for additive equal-effort camera controls."""

import unittest

try:
    import v5_pair_controls as target
except ModuleNotFoundError:
    target = None
from bootstrap import BASE
import numpy as np


class PairControlTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(target, "Equal-effort pair-control implementation is missing")
        return target

    def test_label_only_cross_assignment_is_bijective_different_genotype_same_round(self):
        api = self.api()
        visits = [
            dict(session=f"C{round_}_{gen}_{rep}", measurement=str(round_), genotype=gen)
            for round_ in [1, 2]
            for gen in ["A", "B", "C"]
            for rep in [1, 2]
        ]
        mapped = api.cross_assignments(visits)
        lookup = {v["session"]: v for v in visits}
        self.assertEqual(set(mapped), set(lookup))
        self.assertEqual(set(mapped.values()), set(lookup))
        for source, destination in mapped.items():
            self.assertEqual(lookup[source]["measurement"], lookup[destination]["measurement"])
            self.assertNotEqual(lookup[source]["genotype"], lookup[destination]["genotype"])

    def test_moment_registration_recovers_known_translation_without_borrowing_transform(self):
        api = self.api()
        a = np.zeros((80, 100), bool)
        a[20:45, 30:37] = True
        a[40:45, 37:55] = True
        b = np.zeros_like(a)
        b[16:41, 37:44] = True
        b[36:41, 44:62] = True
        matrix, iou, status = api.moment_registration(a, b)
        np.testing.assert_allclose(matrix, [[1, 0, 7], [0, 1, -4]], atol=1e-6)
        self.assertEqual(iou, 1.0)
        reverse, _, _ = api.moment_registration(b, a)
        np.testing.assert_allclose(reverse, [[1, 0, -7], [0, 1, 4]], atol=1e-6)

    def test_matching_preserves_unmatched_ids_and_common_support_counts(self):
        api = self.api()
        first = np.zeros((80, 100), bool)
        first[10:30, 20:40] = True
        second = np.zeros_like(first)
        second[45:65, 60:80] = True
        whole = first | second
        rgb = [(dict(detection_index=4), first), (dict(detection_index=8), second)]
        noir = [(dict(detection_index=7), first)]
        result = api.score_pair(
            whole, whole, rgb, noir, np.array([[1, 0, 0], [0, 1, 0]], np.float32), 1.0
        )
        self.assertEqual(result["geometry_candidates"], 1)
        self.assertEqual(result["pixel_hypotheses"], 1)
        self.assertEqual(result["unmatched_rgb_ids"], [8])
        self.assertEqual(result["unmatched_noir_ids"], [])
        self.assertEqual(result["proposals"][0]["common_interior_pixels"], 18 * 18)
        self.assertEqual(result["proposals"][0]["rgb_detection_index"], 4)
        self.assertEqual(result["proposals"][0]["noir_detection_index"], 7)

    def test_empty_foreground_and_absent_detections_remain_explicit(self):
        api = self.api()
        empty = np.zeros((80, 100), bool)
        matrix, iou, status = api.moment_registration(empty, empty)
        self.assertEqual(status, "empty_or_tiny_foreground")
        result = api.score_pair(empty, empty, [], [], matrix, iou)
        self.assertEqual(result["proposals"], [])
        self.assertEqual(result["geometry_candidates"], 0)
        self.assertEqual(result["pixel_hypotheses"], 0)


if __name__ == "__main__":
    unittest.main()
