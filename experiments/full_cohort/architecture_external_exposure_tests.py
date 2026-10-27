import unittest
import numpy as np
from architecture_external_exposure import removal_exposure

class ExposureTests(unittest.TestCase):
    def test_fixed_receiver_denominator_and_redundant_blockers(self):
        layers=np.full((3,2,3),-np.inf,dtype='float32');layers[0]=0
        layers[1,0,:2]=1;layers[2,0,1:]=2
        states,edges=removal_exposure(layers,[1,2,3],[1,2,3])
        receiver=next(r for r in states if r['receiver_id']==1)
        self.assertEqual(receiver['potential_pixels'],6);self.assertEqual(receiver['baseline_visible_pixels'],3)
        gains={r['removed_id']:r['gain_pixels'] for r in edges if r['receiver_id']==1}
        self.assertEqual(gains,{2:1,3:1})
        self.assertEqual(receiver['redundantly_blocked_pixels'],1)
    def test_stem_is_retained_and_cannot_be_removed(self):
        layers=np.array([[[2,2]],[[0,0]],[[1,-np.inf]]],dtype='float32')
        states,edges=removal_exposure(layers,[1,2,3],[2,3])
        self.assertTrue(all(r['removed_id']!=1 for r in edges));self.assertTrue(all(r['gain_pixels']==0 for r in edges))
    def test_removal_never_changes_receiver_potential(self):
        rng=np.random.default_rng(5);layers=rng.normal(size=(5,10,10)).astype('float32');layers[rng.random(layers.shape)<.2]=-np.inf
        states,edges=removal_exposure(layers,list(range(1,6)),list(range(1,6)))
        self.assertTrue(all(r['gain_pixels']>=0 for r in edges))
        for r in states:
            self.assertLessEqual(sum(e['gain_pixels'] for e in edges if e['receiver_id']==r['receiver_id']),r['potential_pixels']-r['baseline_visible_pixels'])

if __name__=='__main__':unittest.main()
