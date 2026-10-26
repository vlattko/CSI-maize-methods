import bootstrap
import unittest
import numpy as np

try:
    from v5_reposition import rigid_reposition, domain_anchors, raster_fixed, transition_budget
except ModuleNotFoundError:
    rigid_reposition = domain_anchors = raster_fixed = transition_budget = None


class RepositionTests(unittest.TestCase):
    def test_state_counts_follow_actual_family_and_deduplicate_only_zero_pose(self):
        import v5_reposition as module

        count = getattr(module, "reposition_state_counts", None)
        self.assertIsNotNone(count)
        modes = dict(yaw=list(range(-60, 61, 10)), elevation=list(range(-30, 31, 5)))
        self.assertEqual(count([0, 1, 3, 4, 6, 10, 14], range(9), [256, 512], modes), (3276, 3150))
        self.assertEqual(count([0, 5, 21], range(9), [256, 512], modes), (1404, 1350))
        self.assertEqual(count([0], ["beam"], [96], dict(yaw=[-30, 0, 30])), (3, 3))

    def require(self):
        self.assertIsNotNone(rigid_reposition, "V5 reposition operations are not implemented")

    def test_only_donor_moves_and_rigid_distances_are_preserved(self):
        self.require()
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 2]], float)
        mask = np.array([True, True, True, False])
        moved = rigid_reposition(v, mask, [0, 0, 0], [0, 0, 1], 90)
        np.testing.assert_array_equal(moved[~mask], v[~mask])
        np.testing.assert_allclose(moved[1], [0, 1, 0], atol=1e-14)
        np.testing.assert_allclose(
            np.linalg.norm(moved[:, None] - moved, axis=-1)[:3, :3],
            np.linalg.norm(v[:, None] - v, axis=-1)[:3, :3],
            atol=1e-14,
        )

    def test_zero_is_exact_and_negative_angle_reverses(self):
        self.require()
        v = np.array([[0.1, 0.2, 0.3], [1, 2, 3]], float)
        mask = np.array([True, False])
        np.testing.assert_array_equal(rigid_reposition(v, mask, [0.5, 0, 0], [1, 2, 3], 0), v)
        out = rigid_reposition(v, mask, [0.5, 0, 0], [1, 2, 3], 37)
        np.testing.assert_allclose(
            rigid_reposition(out, mask, [0.5, 0, 0], [1, 2, 3], -37), v, atol=1e-14
        )

    def test_invalid_axis_and_nonfinite_geometry_are_rejected(self):
        self.require()
        with self.assertRaises(ValueError):
            rigid_reposition(np.zeros((1, 3)), [True], [0, 0, 0], [0, 0, 0], 10)
        with self.assertRaises(ValueError):
            rigid_reposition(np.array([[np.nan, 0, 0]]), [True], [0, 0, 0], [0, 1, 0], 10)

    def test_fixed_projection_keeps_receiver_layer_and_pixel_area_exact(self):
        self.require()
        v = np.array(
            [
                [-1, 0, -1],
                [1, 0, -1],
                [1, 0, 1],
                [-1, 0, 1],
                [-1, 1, -1],
                [1, 1, -1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            float,
        )
        f = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        o = np.array([0, 0, 1, 1])
        m = v.copy()
        m[4:, 0] += 2
        a = domain_anchors([v, m], [0, 1, 0])
        lo, pa = raster_fixed(v, f, o, [0, 1, 0], 48, a, 2)
        hi, pb = raster_fixed(m, f, o, [0, 1, 0], 48, a, 2)
        self.assertEqual(pa, pb)
        np.testing.assert_array_equal(lo[0], hi[0])
        self.assertGreater(np.isfinite(lo[0]).sum(), 0)

    def test_oblique_extreme_pose_has_identical_grid(self):
        self.require()
        # Analytic receiver and raised blade replace the private fitted-plant fixture.
        # Retain the same extreme-pose schedule and oblique projection direction.
        v = np.array(
            [
                [-1, 0, -1],
                [1, 0, -1],
                [1, 0, 1],
                [-1, 0, 1],
                [-0.1, 0.2, 0],
                [0.1, 0.2, 0],
                [-0.1, 1, 1],
                [0.1, 1, 1],
            ],
            float,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 7], [4, 7, 6]])
        organs = np.array([0, 0, 0, 0, 3, 3, 3, 3])
        mask = organs == 3
        g = v[mask].reshape(2, 2, 3)
        pivot = g[0].mean(0)
        h = (g[-1].mean(0) - pivot) * [1, 0, 1]
        axis = np.cross(h, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        poses = [
            rigid_reposition(v, mask, pivot, a, angle)
            for a, angles in [([0, 1, 0], range(-60, 61, 10)), (axis, range(-30, 31, 5))]
            for angle in angles
        ]
        direction = [
            np.cos(np.radians(225)) / np.sqrt(2),
            1 / np.sqrt(2),
            np.sin(np.radians(225)) / np.sqrt(2),
        ]
        anchors = domain_anchors(poses, direction)
        fo = organs[faces[:, 0]]
        base, pa = raster_fixed(v, faces, fo, direction, 256, anchors)
        for pose in poses:
            got, pb = raster_fixed(pose, faces[fo == 3], fo[fo == 3], direction, 256, anchors)
            self.assertEqual(pa, pb)
            all_layers, pc = raster_fixed(pose, faces, fo, direction, 256, anchors)
            self.assertEqual(pa, pc)
            np.testing.assert_array_equal(base[0], all_layers[0])

    def test_transition_ledger_covers_all_pixels_and_signed_budget(self):
        self.require()
        before = np.array([[0, 0], [-1, 1]])
        after = np.array([[0, 1], [0, 0]])
        b = transition_budget(before, after, 2, 0.25)
        self.assertEqual(np.array(b["transition_pixels"]).sum(), 4)
        self.assertAlmostEqual(b["whole_shoot_change"], 0.25)
        self.assertAlmostEqual(b["organ_area_changes"][0], 0.25)
        self.assertAlmostEqual(b["escape_change"], -0.25)
        self.assertAlmostEqual(b["closure"], 0)


if __name__ == "__main__":
    unittest.main()
