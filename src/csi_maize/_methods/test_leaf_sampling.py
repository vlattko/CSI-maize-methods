import unittest
import numpy as np
from leaf_sampling_analysis import support_score, greedy_qualified


class SamplingTests(unittest.TestCase):
    def test_unobservable_organ_remains_in_denominator(self):
        self.assertEqual(support_score(np.array([40, 0]), 8), 0.5)

    def test_greedy_uses_complementarity_without_duplicates(self):
        q = np.array([[1, 0], [0, 1], [1, 1]], bool)
        order, curve = greedy_qualified(q, 2)
        self.assertEqual(order[0], 2)
        self.assertEqual(sorted(order), [0, 1, 2])
        self.assertEqual(curve, [0.5, 0.75, 1.0])


if __name__ == "__main__":
    unittest.main()
