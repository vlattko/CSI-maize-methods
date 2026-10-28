import unittest
import numpy as np
import pandas as pd
from mass_validation import predict_fold


class MassValidationTests(unittest.TestCase):
    def test_outer_test_targets_cannot_change_tuning_or_prediction(self):
        frame=pd.DataFrame({'genotype':['A']*4+['B']*4+['C']*4,
                            'treatment':['C','S','P','C']*3,
                            'image':[1.,2.,np.nan,4.,2.,4.,6.,8.,7.,9.,11.,13.],
                            'mass':np.arange(12,dtype=float)+1})
        train=np.arange(8);test=np.arange(8,12)
        before=predict_fold(frame,'mass','genotype',train,test,['image','genotype','treatment'])
        frame.loc[test,'mass']=[10000,-20000,50000,90000]
        after=predict_fold(frame,'mass','genotype',train,test,['image','genotype','treatment'])
        np.testing.assert_array_equal(before[0],after[0])
        self.assertEqual(before[1:],after[1:])

    def test_training_mean_does_not_read_heldout_labels(self):
        frame=pd.DataFrame({'genotype':['A','A','B'], 'mass':[1.,3.,999.]})
        pred,alpha,_=predict_fold(frame,'mass','genotype',np.array([0,1]),np.array([2]),[])
        np.testing.assert_array_equal(pred,[2.])
        self.assertIsNone(alpha)


if __name__=='__main__':unittest.main()
