"""Scientific regression tests for the scoped leaf-dynamics pilot."""

import unittest
import numpy as np
from leaf_dynamics_geometry import (
    train_indices,
    require_training,
    rank_matchings,
    light_counterfactual,
    frontmost_ownership,
)


class LeafDynamicsTests(unittest.TestCase):
    def test_holdout_cannot_be_used_for_seed_or_fit(self):
        self.assertEqual(len(train_indices()), 30)
        require_training([0, 20, 39])
        with self.assertRaises(ValueError):
            require_training([0, 2])

    def test_order_matching_enumerates_ambiguity_not_identity(self):
        matches = rank_matchings(3, 5)
        self.assertEqual(len(matches), 10)
        self.assertIn((0, 1, 2), matches)
        self.assertIn((2, 3, 4), matches)
        self.assertTrue(all(list(x) == sorted(x) for x in matches))

    def test_removal_reveals_rear_without_moving_shapes(self):
        # Two equal, parallel opaque sheets: front at z=1 shadows rear z=0.
        sheet = np.array([[-1, -0.5, 0], [1, -0.5, 0], [1, 0.5, 0], [-1, 0.5, 0.0]])
        vertices = np.r_[sheet, sheet + [0, 0, 1]]
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        result = light_counterfactual(
            vertices, faces, np.array([0, 0, 1, 1]), [0, 0, 1], removed=1, resolution=200
        )
        self.assertAlmostEqual(result["before"][0], 0, places=8)
        self.assertAlmostEqual(result["after"][0], 2, delta=0.05)
        self.assertAlmostEqual(result["gain"][0], 2, delta=0.05)
        self.assertEqual(result["after"][1], 0)

    def test_camera_ownership_uses_nearest_positive_depth(self):
        xy = np.array([[2, 2], [12, 2], [2, 12], [2, 2], [12, 2], [2, 12]], float)
        owner = frontmost_ownership(
            xy,
            np.array([8, 8, 8, 3, 3, 3]),
            np.array([[0, 1, 2], [3, 4, 5]]),
            np.array([0, 1]),
            (16, 16),
        )
        self.assertEqual(owner[5, 5], 1)
        owner = frontmost_ownership(
            xy,
            np.array([8, 8, 8, -3, -3, -3]),
            np.array([[0, 1, 2], [3, 4, 5]]),
            np.array([0, 1]),
            (16, 16),
        )
        self.assertEqual(owner[5, 5], 0)


if __name__ == "__main__":
    unittest.main()
