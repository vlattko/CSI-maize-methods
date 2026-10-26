"""Checks for additive distichous stress geometry and frozen fitting contract."""

import bootstrap
import unittest, importlib.util
import numpy as np


class OcclusionTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(
            importlib.util.find_spec("v5_recovery_occlusion"),
            "Distichous stress API not implemented",
        )
        import v5_recovery_occlusion as m

        return m

    def test_four_cases_and_unchanged_settings(self):
        m = self.api()
        d = m.stress_design()
        self.assertEqual(len(d["cases"]), 4)
        self.assertEqual(d["iterations"], m.core.design()["iterations"])
        self.assertEqual(d["train_views"], m.core.design()["train_views"])
        self.assertIn("after", d["chronology"])

    def test_rotation_preserves_leaf_shape(self):
        m = self.api()
        a = m.core.make_truth(5, "wavy")
        b = m.make_distichous("wavy")
        for k in range(5):
            np.testing.assert_allclose(
                np.linalg.norm(np.diff(a["midribs"][k], axis=0), axis=1),
                np.linalg.norm(np.diff(b["midribs"][k], axis=0), axis=1),
                atol=1e-9,
            )
            np.testing.assert_allclose(a["midribs"][k, 0], b["midribs"][k, 0], atol=1e-9)

    def test_alternating_azimuth_and_no_hidden_shape_initialization(self):
        m = self.api()
        g = m.make_distichous("cubic")
        delta = g["midribs"][:, -1] - g["midribs"][:, 0]
        np.testing.assert_allclose(delta[:, 2], 0.0, atol=1e-8)
        self.assertTrue(np.all(np.sign(delta[:, 0]) == np.array([1, -1, 1, -1, 1])))
        d = m.stress_design()
        obs, _ = m.core.observations_from_truth(g, d["cases"][0], d)
        init = m.core.initialize_from_observations(obs, d["camera"])
        self.assertTrue(np.all(init["widths"] == 3.0))

    def test_effect_relative_error_undefined_for_zero_truth(self):
        m = self.api()
        a = m.effect_errors(np.zeros((2, 2)), np.zeros((2, 2)))
        self.assertIsNone(a["magnitude_relative_l1_error"])
        b = m.effect_errors(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.0, 0.2], [0.0, 0.0]]))
        self.assertAlmostEqual(b["magnitude_relative_l1_error"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
