import unittest
import numpy as np
from leaf_expanded_analyses import beam_budget, direction_weights, select_coverage, angular_sequence


class ExpandedTests(unittest.TestCase):
    def test_removal_budget_includes_stem_and_escape(self):
        before = np.array([[0, 0, 0, 1, 5, -1]])
        after = np.array([[1, 5, -1, 1, 5, -1]])
        r = beam_budget(before, after, [0], 0.1)
        self.assertAlmostEqual(r["removed_interception"], 0.3)
        self.assertAlmostEqual(r["retained_leaf_gain"], 0.1)
        self.assertAlmostEqual(r["stem_gain"], 0.1)
        self.assertAlmostEqual(r["escape_gain"], 0.1)
        self.assertAlmostEqual(r["budget_residual"], 0)
        self.assertLessEqual(r["after_total"], r["before_total"])

    def test_direction_weights_endpoints(self):
        np.testing.assert_array_equal(direction_weights(1), [1] + [0] * 8)
        np.testing.assert_allclose(direction_weights(1 / 9), np.ones(9) / 9)
        self.assertEqual(direction_weights(0)[0], 0)
        self.assertAlmostEqual(direction_weights(0.23).sum(), 1)

    def test_complementary_views_and_permanently_hidden_bins(self):
        support = np.zeros((4, 2, 3), bool)
        support[0, 0, 0] = 1
        support[1, 1, 0] = 1
        support[2, :, 1] = 1
        order, curve = select_coverage(support)
        self.assertEqual(order[0], 2)
        self.assertTrue(all(a <= b for a, b in zip(curve, curve[1:])))
        self.assertAlmostEqual(curve[-1], 4 / 6)

    def test_angular_baseline_is_nested_and_spread(self):
        seq = angular_sequence(40)
        self.assertEqual(seq[:4], [0, 20, 10, 30])
        self.assertEqual(sorted(seq), list(range(40)))


if __name__ == "__main__":
    unittest.main()
