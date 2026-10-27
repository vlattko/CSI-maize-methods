import unittest
import numpy as np
from pairing_dn_controls import aggregate,frame_statistics,CHANNELS


class DNControlTests(unittest.TestCase):
    def test_guarded_common_support_and_forty_pixel_threshold(self):
        six=np.full((10,10,6),80,np.float32)
        common=np.ones((10,10),bool);valid=np.ones_like(six,bool)
        six[:6,:,4]=1
        rows=frame_statistics(six,valid,common)
        self.assertEqual(rows[4]['valid_pixels'],40)
        self.assertTrue(rows[4]['qualified'])
        self.assertEqual(rows[4]['median'],80)
        valid[6,0,4]=False
        rows=frame_statistics(six,valid,common)
        self.assertEqual(rows[4]['valid_pixels'],39)
        self.assertIsNone(rows[4]['median'])
        self.assertEqual(rows[0]['valid_pixels'],100)

    def test_channel_specific_session_missingness_and_technical_iqr(self):
        sessions=[dict(session='s',treatment='C',measurement='1',genotype='g',replicate='1'),
                  dict(session='empty',treatment='C',measurement='1',genotype='g',replicate='2')]
        rows=[dict(sessions[0],channel='rgb_R',view=v,qualified=True,valid_pixels=40,median=m)
              for v,m in [(0,20),(10,60)]]
        result=aggregate(sessions,rows)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['paired_dn__rgb_R__median'],40)
        self.assertEqual(result[0]['paired_dn__rgb_R__iqr'],20)
        self.assertEqual(result[0]['paired_dn__noir_G__n_views'],0)
        self.assertIsNone(result[0]['paired_dn__noir_G__median'])
        self.assertIsNone(result[1]['paired_dn__rgb_R__median'])


if __name__=='__main__':unittest.main()
