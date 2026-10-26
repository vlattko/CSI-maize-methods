import bootstrap
import unittest
import torch
import numpy as np
from organ_inverse import project, soft_silhouette, triangle_silhouette


class InverseGraphicsTests(unittest.TestCase):
    def test_shared_rotation_changes_depth_into_horizontal(self):
        p = torch.tensor([[10.0, 20.0, 0.0], [0.0, 20.0, 10.0]])
        camera = {"axis_x": 50.0, "base_y": 90.0, "elevation_deg": 0.0, "roll_deg": 0.0}
        xy = project(p, torch.tensor([np.pi / 2]), camera)
        torch.testing.assert_close(
            xy[0], torch.tensor([[50.0, 70.0], [60.0, 70.0]]), atol=1e-4, rtol=1e-4
        )

    def test_soft_render_has_finite_position_gradient(self):
        xy = torch.tensor([[[7.3, 8.2], [8.4, 8.3]]], requires_grad=True)
        probability = soft_silhouette(xy, torch.ones((1, 2)), (20, 20))
        self.assertTrue(bool(((probability >= 0) & (probability <= 1)).all()))
        loss = (probability * torch.arange(20)[None, None, :]).sum()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(xy.grad).all()))
        self.assertGreater(float(xy.grad.abs().sum()), 0.0)

    def test_perspective_uses_optical_center_not_stem_base(self):
        camera = {
            "axis_x": 50.0,
            "base_y": 90.0,
            "elevation_deg": 0.0,
            "roll_deg": 0.0,
            "distance": 1000.0,
            "principal_x": 50.0,
            "principal_y": 50.0,
        }
        xy = project(
            torch.tensor([[0.0, 0.0, 0.0], [10.0, 20.0, 100.0]]), torch.tensor([0.0]), camera
        )
        torch.testing.assert_close(xy[0, 0], torch.tensor([50.0, 90.0]))
        torch.testing.assert_close(
            xy[0, 1],
            torch.tensor([50.0 + 1000 * 10 / 1100, 50.0 + 1000 * 20 / 1100]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_points_outside_frame_do_not_create_geometry_at_border(self):
        xy = torch.tensor([[[1000.0, 8.0]]])
        p = soft_silhouette(xy, torch.ones((1, 1)), (20, 20))
        self.assertEqual(float(p.sum()), 0.0)

    def test_triangle_renderer_covers_interior_and_position_gradient(self):
        vertices = torch.tensor([[[4.0, 4.0], [12.0, 4.0], [4.0, 12.0]]], requires_grad=True)
        p = triangle_silhouette(vertices, np.array([[0, 1, 2]]), (20, 20), sigma=0.15, radius=6)
        self.assertGreater(float(p[0, 6, 6].detach()), 0.99)
        self.assertLess(float(p[0, 1, 1].detach()), 0.001)
        (p * torch.arange(20)[None, None, :]).sum().backward()
        self.assertTrue(bool(torch.isfinite(vertices.grad).all()))
        self.assertGreater(float(vertices.grad.abs().sum()), 0)

    def test_triangle_renderer_rejects_interior_truncated_by_local_tile(self):
        vertices = torch.tensor([[[2.0, 2.0], [26.0, 2.0], [2.0, 26.0]]])
        with self.assertRaisesRegex(ValueError, "tile"):
            triangle_silhouette(vertices, np.array([[0, 1, 2]]), (32, 32), radius=5)

    def test_triangle_renderer_rejects_long_thin_triangle(self):
        vertices = torch.tensor([[[2.0, 4.0], [25.0, 4.0], [13.0, 5.0]]])
        with self.assertRaisesRegex(ValueError, "tile"):
            triangle_silhouette(vertices, np.array([[0, 1, 2]]), (32, 32), radius=5)

    def test_tile_guard_allows_fractional_vertices_and_offscreen_geometry(self):
        # Integer bbox [5,14] lies in tile [4,14], although 14.9 exceeds 14.
        vertices = torch.tensor([[[4.9, 4.9], [14.9, 4.9], [4.9, 14.9]]])
        probability = triangle_silhouette(vertices, np.array([[0, 1, 2]]), (24, 24), radius=5)
        self.assertGreater(float(probability[0, 7, 7]), 0.99)
        offscreen = torch.tensor([[[-40.0, -40.0], [-20.0, -40.0], [-40.0, -20.0]]])
        self.assertEqual(
            float(triangle_silhouette(offscreen, np.array([[0, 1, 2]]), (24, 24)).sum()), 0.0
        )
        degenerate = torch.tensor([[[2.0, 2.0], [26.0, 2.0], [14.0, 2.0]]])
        self.assertEqual(
            float(triangle_silhouette(degenerate, np.array([[0, 1, 2]]), (32, 32)).sum()), 0.0
        )

    def test_tile_guard_uses_only_in_frame_bbox(self):
        # Entire image intersection x=0..2,y=1..3 fits the tile about (-3,2).
        vertices = torch.tensor([[[-8.0, 1.0], [2.0, 1.0], [2.0, 3.0]]])
        probability = triangle_silhouette(vertices, np.array([[0, 1, 2]]), (8, 8), radius=5)
        self.assertGreater(float(probability[0, 2, 1]), 0.9)


if __name__ == "__main__":
    unittest.main()
