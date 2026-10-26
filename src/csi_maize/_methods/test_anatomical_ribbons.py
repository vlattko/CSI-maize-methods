"""Independent geometry invariants for the differentiable organ prior."""

import unittest
import bootstrap
import numpy as np
import torch
from anatomical_ribbons import cubic_ribbons, ribbon_faces


class RibbonTests(unittest.TestCase):
    def inputs(self):
        cp = torch.tensor(
            [
                [[2.0, 4.0, 3.0], [4.0, 15.0, 5.0], [12.0, 25.0, 8.0], [19.0, 29.0, 10.0]],
                [[-3.0, 9.0, 4.0], [-8.0, 20.0, 4.0], [-22.0, 24.0, 9.0], [-28.0, 18.0, 12.0]],
            ],
            dtype=torch.float64,
        )
        widths = torch.tensor([[2.0, 4.0, 1.0], [3.0, 5.0, 2.0]], dtype=torch.float64)
        twists = torch.tensor([[0.1, 0.4, -0.2], [-0.2, 0.1, 0.5]], dtype=torch.float64)
        return cp, widths, twists

    def test_exact_bezier_midline_and_endpoints(self):
        cp, w, tw = self.inputs()
        vertices, midribs, faces = cubic_ribbons(cp, w, tw)
        t = torch.linspace(0, 1, 48, dtype=cp.dtype)
        expected = (
            (1 - t)[None, :, None] ** 3 * cp[:, 0, None]
            + 3 * ((1 - t) ** 2 * t)[None, :, None] * cp[:, 1, None]
            + 3 * ((1 - t) * t * t)[None, :, None] * cp[:, 2, None]
            + t[None, :, None] ** 3 * cp[:, 3, None]
        )
        torch.testing.assert_close(midribs, expected)
        torch.testing.assert_close(vertices[:, :, 4], expected)
        torch.testing.assert_close(midribs[:, 0], cp[:, 0])
        torch.testing.assert_close(midribs[:, -1], cp[:, -1])
        torch.testing.assert_close(vertices[:, -1], cp[:, -1, None].expand(-1, 9, -1))
        self.assertTrue(
            bool(
                (
                    torch.linalg.vector_norm(vertices[:, :-1, -1] - vertices[:, :-1, 0], dim=-1) > 0
                ).all()
            )
        )

    def test_translation_and_y_rotation_equivariance(self):
        cp, w, tw = self.inputs()
        v, m, _ = cubic_ribbons(cp, w, tw)
        shift = torch.tensor([5.0, -7.0, 12.0], dtype=cp.dtype)
        shifted, _, _ = cubic_ribbons(cp + shift, w, tw)
        torch.testing.assert_close(shifted, v + shift, atol=1e-10, rtol=1e-10)
        angle = 0.83
        c = np.cos(angle)
        s = np.sin(angle)
        matrix = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=cp.dtype)
        rotated, _, _ = cubic_ribbons(cp @ matrix.T, w, tw)
        torch.testing.assert_close(rotated, v @ matrix.T, atol=1e-10, rtol=1e-10)

    def test_gradients_and_vertical_degenerate_frame_are_finite(self):
        cp, w, tw = self.inputs()
        cp.requires_grad_()
        w.requires_grad_()
        tw.requires_grad_()
        v, m, _ = cubic_ribbons(cp, w, tw)
        loss = (
            (v * torch.linspace(0.1, 1, v.numel(), dtype=v.dtype).reshape(v.shape)).square().mean()
        )
        loss.backward()
        for value in [cp, w, tw]:
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
        self.assertGreater(float(tw.grad.abs().sum()), 0)
        vertical = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 8.0, 0.0], [0.0, 10.0, 0.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        v, _, _ = cubic_ribbons(vertical, w[:1].detach(), tw[:1].detach())
        v.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(v).all()))
        self.assertTrue(bool(torch.isfinite(vertical.grad).all()))

    def test_faces_have_valid_leaf_local_indices(self):
        faces = ribbon_faces(48, 9)
        self.assertIsInstance(faces, np.ndarray)
        self.assertEqual(faces.shape, (47 * 8 * 2, 3))
        self.assertGreaterEqual(int(faces.min()), 0)
        self.assertLess(int(faces.max()), 48 * 9)
        self.assertTrue(np.all(np.diff(np.sort(faces, axis=1), axis=1) > 0))


if __name__ == "__main__":
    unittest.main()
