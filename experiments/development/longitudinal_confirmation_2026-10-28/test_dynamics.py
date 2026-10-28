import unittest
import numpy as np


class DynamicsTests(unittest.TestCase):
    def test_first_observation_normalization_does_not_substitute_a_later_baseline(self):
        from dynamics import first_observation_change
        np.testing.assert_allclose(first_observation_change([2,4,6],True),[1,2,3])
        self.assertTrue(np.isnan(first_observation_change([np.nan,4,6],True)).all())
        np.testing.assert_allclose(first_observation_change([.8,.5,.75],False),[0,-.3,-.05])

    def test_daily_differences_use_declared_elapsed_days_and_keep_missing_intervals(self):
        from dynamics import interval_rates
        result=interval_rates([13,19,20,22,25],[2.,8.,np.nan,10.,16.])
        np.testing.assert_allclose([result[0],result[3]],[1.,2.])
        self.assertTrue(np.isnan(result[1]));self.assertTrue(np.isnan(result[2]))


if __name__=='__main__':unittest.main()
