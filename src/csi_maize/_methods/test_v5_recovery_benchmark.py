"""Behavioral checks for synthetic recovery boundaries and geometry."""

import bootstrap  # Match the scientific runtime dependency order.
import unittest
import importlib.util
import numpy as np


class RecoveryTests(unittest.TestCase):
    def api(self):
        spec = importlib.util.find_spec("v5_recovery_benchmark")
        self.assertIsNotNone(spec, "Recovery benchmark API has not been implemented")
        import v5_recovery_benchmark as m

        return m

    def test_predeclared_matrix_and_disjoint_views(self):
        m = self.api()
        d = m.design()
        self.assertEqual(len(d["cases"]), 8)
        self.assertFalse(set(d["train_views"]) & set(d["test_views"]))
        self.assertEqual(set(d["train_views"]) | set(d["test_views"]), set(range(16)))
        self.assertEqual(sum(c["truth_family"] == "wavy" for c in d["cases"]), 4)

    def test_initialization_is_observation_only(self):
        m = self.api()
        c = m.design()["camera"]
        obs = m.tiny_observations(c)
        a = m.initialize_from_observations(obs, c)
        b = m.initialize_from_observations(
            dict(obs, truth_control_points=np.ones((1, 4, 3)) * 999), c
        )
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertTrue(np.allclose(a["widths"], 3.0))
        self.assertTrue(np.allclose(a["twists"], 0.0))

    def test_observation_initializer_recovers_known_tip(self):
        m = self.api()
        c = m.design()["camera"]
        obs = m.tiny_observations(c)
        a = m.initialize_from_observations(obs, c)
        np.testing.assert_allclose(a["control"][:, -1], np.array([[20.0, 60.0, 12.0]]), atol=0.02)

    def test_geometry_metrics_identity_and_known_scale(self):
        m = self.api()
        v, mid, faces = m.simple_test_surface()
        z = m.geometry_metrics(v, mid, v, mid, faces)
        self.assertAlmostEqual(z["surface_symmetric_distance"], 0.0, places=7)
        self.assertAlmostEqual(z["mean_area_relative_abs_error"], 0.0, places=7)
        q = m.geometry_metrics(2 * v, 2 * mid, v, mid, faces)
        self.assertAlmostEqual(q["mean_area_relative_abs_error"], 3.0, places=5)
        self.assertAlmostEqual(q["mean_midrib_length_relative_abs_error"], 1.0, places=5)

    def test_out_of_family_truth_is_not_identical_to_cubic(self):
        m = self.api()
        a = m.make_truth(3, "cubic", 0)
        b = m.make_truth(3, "wavy", 0)
        self.assertGreater(float(np.max(np.abs(a["vertices"] - b["vertices"]))), 0.5)
        # Nonzero fourth finite difference excludes an exactly cubic sampled midrib.
        self.assertGreater(float(np.max(np.abs(np.diff(b["midribs"], n=4, axis=1)))), 1e-4)

    def test_release_uses_triangle_depth_and_excludes_self(self):
        m = self.api()
        v, mid, faces = m.simple_test_surface()
        matrix, beams = m.release_matrix(v)
        self.assertEqual(matrix.shape, (1, 1))
        np.testing.assert_array_equal(matrix, 0.0)
        self.assertTrue(np.isfinite(beams).all())

    def test_withheld_values_do_not_change_fitting(self):
        m = self.api()
        d = m.design()
        g = m.make_truth(3, "cubic")
        obs, clean = m.observations_from_truth(g, d["cases"][0], d)
        init = m.initialize_from_observations(obs, d["camera"])
        a = dict(obs)
        b = dict(obs)
        b["angles"] = obs["angles"].copy()
        b["angles"][d["test_views"]] += 0.713
        b["hidden_evaluation_masks"] = 1 - clean[d["test_views"]]
        x, _ = m.fit_case(a, d["camera"], init, 2, "cpu")
        y, _ = m.fit_case(b, d["camera"], init, 2, "cpu")
        for key in x:
            np.testing.assert_array_equal(x[key], y[key])

    def test_release_front_leaf_removal_exposes_fixed_receiver(self):
        m = self.api()
        v, mid, faces = m.simple_test_surface()
        horizontal = v.copy()
        horizontal[..., 2] = v[..., 1]
        horizontal[..., 1] = 0.0
        top = horizontal.copy()
        top[..., 1] = 2.0
        matrix, beams = m.release_matrix(np.concatenate([horizontal, top]))
        self.assertAlmostEqual(float(beams[0, 1, 0]), 1.0)
        self.assertAlmostEqual(float(beams[0, 0, 1]), 0.0)

    def test_renderer_and_recovery_are_distinct_operations(self):
        m = self.api()
        d = m.design()
        self.assertTrue(d["known_camera"])
        self.assertEqual(d["initialization"], "2D observations only; no truth controls")
        self.assertNotIn(
            "truth", m.fit_case.__code__.co_varnames[: m.fit_case.__code__.co_argcount]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
