import unittest
import numpy as np
from additive_geometry import (
    directional_raster,
    triangulate_tip,
    select_views,
    perturb_controls,
    accept_training,
    view_common_residuals,
)
from organ_camera import project


class GeometryExperimentTests(unittest.TestCase):
    def sheet(self, z=0):
        return np.array([[-1, -0.5, z], [1, -0.5, z], [1, 0.5, z], [-1, 0.5, z]], float), np.array(
            [[0, 1, 2], [0, 2, 3]]
        )

    def test_planar_projected_area(self):
        v, f = self.sheet()
        result = directional_raster(v, f, np.zeros(2, int), [0, 0, 1], resolution=240)
        self.assertAlmostEqual(result["intercepted_area"], 2.0, delta=0.035)
        oblique = directional_raster(
            v, f, np.zeros(2, int), [np.sqrt(0.75), 0, 0.5], resolution=240
        )
        self.assertAlmostEqual(oblique["intercepted_area"], 1.0, delta=0.035)

    def test_layered_occlusion(self):
        rear, f = self.sheet()
        front, _ = self.sheet(0.3)
        result = directional_raster(
            np.r_[rear, front], np.r_[f, f + 4], np.array([0, 0, 1, 1]), [0, 0, 1], resolution=200
        )
        self.assertAlmostEqual(result["organ_intercepted_area"][0], 0.0, delta=0.005)
        self.assertAlmostEqual(result["organ_intercepted_area"][1], 2.0, delta=0.04)

    def test_camera_tip_recovery(self):
        camera = dict(
            elevation_deg=-3.0,
            distance=850.0,
            roll_deg=3.0,
            principal_x=512.0,
            principal_y=384.5,
            axis_x=481.0,
            base_y=768.0,
        )
        theta = np.array([0, 0.8, 2.0, 3.5])
        point = np.array([65.0, 180.0, -40.0])
        xy, _ = project(np.tile(point, (4, 1)), theta, -3.0, 850.0, 3.0)
        fitted = triangulate_tip(theta, xy, camera)
        np.testing.assert_allclose(fitted["point"], point, atol=0.0001)
        self.assertEqual(fitted["status"], "fitted")
        self.assertEqual(triangulate_tip(theta[:1], xy[:1], camera)["status"], "insufficient_views")

    def test_view_selection_does_not_return_evaluation_views(self):
        pool = [0, 4, 8, 12, 16, 20, 24, 28]
        visibility = {i: np.array([True, i % 8 == 0]) for i in pool}
        selected = select_views(pool, 4, "preview_balanced", visibility)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(set(selected)), 4)
        self.assertTrue(set(selected) <= set(pool))

    def test_perturbation_preserves_tips_and_shared_stem(self):
        cp = np.array([[[0, 100, 0], [20, 150, 0], [80, 190, 0], [100, 200, 0]]], float)
        out, w, t = perturb_controls(
            cp, np.ones((1, 3)) * 10, np.zeros((1, 3)), np.zeros(4), 1.1, 0.2, 6, 8
        )
        np.testing.assert_array_equal(out[:, 3], cp[:, 3])
        np.testing.assert_allclose(out[:, 0], [[0, 106, 0]])
        self.assertTrue((w > 0).all())
        np.testing.assert_allclose(t, 0.2)

    def test_training_acceptance_has_no_test_score(self):
        self.assertTrue(accept_training(0.69, 0.70, 0.02))
        self.assertFalse(accept_training(0.67, 0.70, 0.02))

    def test_missing_tip_views_are_not_filled_with_nan_motion(self):
        self.assertEqual(view_common_residuals([]), {})
        result = view_common_residuals(
            [dict(view=0, dx_px=2.0, dy_px=3.0), dict(view=0, dx_px=4.0, dy_px=5.0)]
        )
        self.assertEqual(set(result), {0})
        np.testing.assert_allclose(result[0], [3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
