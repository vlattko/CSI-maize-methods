import importlib.util
import unittest
import numpy as np


class CrossViewTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("additive_learning"),
            "Cross-view experiment is not implemented",
        )
        import additive_learning

        return additive_learning

    def test_positive_pairs_never_pair_different_visits_or_self(self):
        m = self.module()
        labels = np.array([0, 0, 1, 1, 2, 2])
        mask = m.positive_mask(labels)
        self.assertEqual(int(mask.sum()), 6)
        self.assertFalse(np.diag(mask).any())
        self.assertFalse(mask[0, 2])
        self.assertTrue(mask[0, 1])

    def test_angular_amplitudes_are_invariant_to_rotation_origin(self):
        m = self.module()
        theta = np.arange(8) * 2 * np.pi / 8
        x = np.stack([2 + 3 * np.cos(theta), 4 + 2 * np.sin(2 * theta)], axis=1)[None]
        mean, amplitude = m.view_blocks(x)
        np.testing.assert_allclose(mean, [[2, 4]], atol=1e-10)
        np.testing.assert_allclose(amplitude[0, :2], [3, 0], atol=1e-10)
        np.testing.assert_allclose(amplitude[0, 2:], [0, 2], atol=1e-10)
        np.testing.assert_allclose(m.view_blocks(np.roll(x, 3, axis=1))[1], amplitude, atol=1e-10)

    def test_contrastive_head_uses_only_supplied_training_visits(self):
        m = self.module()
        rng = np.random.default_rng(5)
        x = rng.normal(size=(6, 4, 12)).astype("float32")
        x[:, :, :3] += np.arange(6)[:, None, None] * 2
        a = m.train_head(x[:4], epochs=25, dimensions=4, seed=20)
        changed = x.copy()
        changed[4:] += 10000
        b = m.train_head(changed[:4], epochs=25, dimensions=4, seed=20)
        np.testing.assert_allclose(a[0], b[0], atol=0)
        self.assertTrue(np.isfinite(a[1]).all())
        self.assertLess(a[1][-1], a[1][0])


if __name__ == "__main__":
    unittest.main()
