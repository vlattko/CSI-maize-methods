import unittest
import numpy as np
from circular_leaf_tracker import CircularTracker, harmonic_predict


class CircularTrackingTests(unittest.TestCase):
    def test_harmonic_predict_uses_nominal_turn_phase(self):
        views = np.arange(0, 25, 3)
        a = views * 2 * np.pi / 40
        xy = np.column_stack([100 + 20 * np.cos(a) + 10 * np.sin(a), 200 + 5 * np.cos(a)])
        out = harmonic_predict(views, xy, 30)
        np.testing.assert_allclose(out, [90, 200], atol=1e-8)

    def test_track_recovers_after_missing_view(self):
        t = CircularTracker("circular")

        def ob(x):
            m = np.zeros((80, 100), bool)
            m[20:40, x : x + 10] = 1
            return dict(mask=m, centroid=[x + 4.5, 29.5], area=200, confidence=0.9)

        a = t.step(0, [ob(10)])[0]
        t.step(1, [])
        b = t.step(2, [ob(12)])[0]
        self.assertEqual(a, b)

    def test_two_detections_cannot_share_one_track(self):
        t = CircularTracker("overlap")
        m = np.zeros((80, 100), bool)
        m[20:40, 10:20] = 1
        o = dict(mask=m, centroid=[15, 30], area=200, confidence=0.9)
        ids = t.step(0, [o, o])
        self.assertEqual(len(set(ids)), 2)


if __name__ == "__main__":
    unittest.main()
