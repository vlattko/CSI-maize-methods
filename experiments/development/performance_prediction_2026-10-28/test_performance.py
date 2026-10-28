import unittest
import numpy as np
from performance_core import partitions, trajectory, train_design, ridge_predict, cutoff_columns


class PerformanceTests(unittest.TestCase):
    def test_first_visit_has_no_history_or_duplicate_missingness_encoding(self):
        self.assertEqual(cutoff_columns(['area','ratio','slope'],13,13,['ratio','slope']),['area'])
        self.assertEqual(cutoff_columns(['area','ratio','slope'],19,13,['ratio','slope']),['area','ratio','slope'])
    def test_joint_transfer_excludes_either_heldout_group(self):
        g=np.repeat(['A','B','C'],6);t=np.tile(np.repeat(['C','P','S'],2),3)
        folds=partitions(g,t,'joint')
        self.assertEqual(len(folds),9)
        seen=[]
        for train,test,label in folds:
            self.assertFalse(set(g[train])&set(g[test]))
            self.assertFalse(set(t[train])&set(t[test]))
            self.assertEqual(len(train),8);self.assertEqual(len(test),2);seen.extend(test)
        self.assertEqual(sorted(seen),list(range(18)))

    def test_trajectory_cannot_read_future_values(self):
        before=trajectory([13,19,20,22,25],[1.,2.,3.,4.,5.],19)
        after=trajectory([13,19,20,22,25],[1.,2.,999.,-999.,777.],19)
        self.assertEqual(before,after)
        self.assertAlmostEqual(before['ratio_to_first'],2.)
        self.assertAlmostEqual(before['slope_per_day'],1/6)

    def test_missing_first_is_not_replaced_for_ratio(self):
        result=trajectory([13,19,20,22,25],[np.nan,2.,3.,4.,5.],22)
        self.assertTrue(np.isnan(result['ratio_to_first']))
        self.assertTrue(np.isfinite(result['slope_per_day']))

    def test_preprocessing_does_not_read_test_values_or_categories(self):
        rows=[dict(area=1.,treatment='C'),dict(area=3.,treatment='S'),dict(area=np.nan,treatment='P')]
        x,z,state=train_design(rows,np.array([0,1]),np.array([2]),['area'],['treatment'])
        self.assertEqual(state['medians'],[2.])
        self.assertEqual(x.shape,(2,3));self.assertEqual(z.shape,(1,3))
        np.testing.assert_array_equal(z,[[0.,0.,0.]])
        rows[2]['area']=10000.
        x2,z2,state2=train_design(rows,np.array([0,1]),np.array([2]),['area'],['treatment'])
        np.testing.assert_array_equal(x,x2);self.assertEqual(state,state2)

    def test_training_only_missing_indicator_and_empty_feature(self):
        rows=[dict(a=np.nan,b=2.),dict(a=np.nan,b=np.nan),dict(a=99.,b=8.)]
        x,z,state=train_design(rows,np.array([0,1]),np.array([2]),['a','b'])
        self.assertEqual(x.shape,(2,4));self.assertEqual(state['medians'],[0.,2.])
        self.assertEqual(state['indicator_columns'],[0,1])
        self.assertTrue(np.isfinite(z).all())

    def test_ols_is_exact_affine_single_area_reference(self):
        x=np.array([[1.],[2.],[3.]])
        pred=ridge_predict(x,np.array([[4.],[5.]]),2*x+1,0.)
        np.testing.assert_allclose(pred,[[9.],[11.]],atol=1e-12)


if __name__=='__main__':unittest.main()
