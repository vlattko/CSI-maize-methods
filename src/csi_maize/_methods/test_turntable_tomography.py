from bootstrap import BASE
import unittest
import numpy as np
from turntable_tomography import project_points, carve, render_points, mask_scores


class TomographyTests(unittest.TestCase):
    def test_quarter_turn_changes_depth_into_horizontal_position(self):
        camera = {"axis_x": 50.0, "base_y": 90.0, "elevation_deg": 0.0, "distance": 0.0}
        points = np.array([[10.0, 20.0, 0.0], [0.0, 20.0, 10.0]])
        u, v, valid = project_points(points, np.pi / 2, camera)
        np.testing.assert_allclose(u, [50, 60], atol=1e-5)
        np.testing.assert_allclose(v, [70, 70], atol=1e-5)
        self.assertTrue(valid.all())

    def test_outside_frame_is_censored_instead_of_background(self):
        camera = {"axis_x": 5.0, "base_y": 5.0, "elevation_deg": 0.0, "distance": 0.0}
        mask = np.ones((10, 10), bool)
        points = np.array([[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        counts, observable = carve(points, [mask], [0.0], camera, tolerance=0)
        np.testing.assert_array_equal(counts, [0, 0])
        np.testing.assert_array_equal(observable, [0, 1])

    def test_shared_camera_roll_changes_vertical_axis_projection(self):
        camera = {
            "axis_x": 50.0,
            "base_y": 90.0,
            "elevation_deg": 0.0,
            "distance": 0.0,
            "roll_deg": 90.0,
        }
        u, v, valid = project_points(np.array([[0.0, 20.0, 0.0]]), 0.0, camera)
        np.testing.assert_allclose(u, [70.0], atol=1e-5)
        np.testing.assert_allclose(v, [90.0], atol=1e-5)

    def test_synthetic_asymmetric_target_explains_unfitted_angle(self):
        camera = {"axis_x": 50.0, "base_y": 90.0, "elevation_deg": 0.0, "distance": 0.0}
        # Two distinct vertical organs, not a rotationally symmetric sphere.
        x, y, z = np.meshgrid(
            np.arange(-20, 21), np.arange(0, 61), np.arange(-20, 21), indexing="ij"
        )
        grid = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)
        target = ((grid[:, 0] ** 2 + grid[:, 2] ** 2 <= 9) & (grid[:, 1] < 45)) | (
            ((grid[:, 0] - 12) ** 2 + (grid[:, 2] - 7) ** 2 <= 16) & (grid[:, 1] > 30)
        )
        true = grid[target]
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        masks = [render_points(true, a, camera, (100, 100), radius=0) for a in angles]
        train = np.arange(24) != 5
        contradictions, observable = carve(
            grid, [masks[i] for i in np.flatnonzero(train)], angles[train], camera, tolerance=1
        )
        recovered = grid[(contradictions == 0) & (observable >= 20)]
        result = render_points(recovered, angles[5], camera, (100, 100), radius=0)
        score = mask_scores(result, masks[5])
        self.assertGreater(score["iou"], 0.75)
        self.assertGreater(score["recall"], 0.95)

    def test_empty_prediction_is_not_a_success(self):
        score = mask_scores(np.zeros((10, 10), bool), np.ones((10, 10), bool))
        self.assertEqual(score["iou"], 0.0)
        self.assertEqual(score["recall"], 0.0)

    def test_gpu_carving_matches_reference_with_roll_and_censoring(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        camera = {
            "axis_x": 12.3,
            "base_y": 20.2,
            "elevation_deg": 7.0,
            "distance": 0.0,
            "roll_deg": 3.0,
        }
        mask = np.zeros((24, 24), bool)
        mask[5:22, 8:17] = True
        points = np.array(
            [[0.0, 4.0, 0.0], [6.0, 10.0, 3.0], [-4.0, 14.0, -6.0], [200.0, 10.0, 0.0]], np.float32
        )
        expected = carve(points, [mask, mask], [0.2, 1.4], camera, 1, "cpu")
        actual = carve(points, [mask, mask], [0.2, 1.4], camera, 1, "cuda:0")
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
