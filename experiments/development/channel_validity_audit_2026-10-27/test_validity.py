import unittest
import numpy as np
import validity_audit as a

class ValidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        import csi_maize
        cls.ops=a.configure(Path(csi_maize.__file__).resolve().parents[2])

    def test_nonzero_corner_contributor_is_rejected(self):
        c,_,_=self.ops
        image=np.full((20,20,3),100,np.uint8);image[6,6]=254
        valid,_=c.warp_to_rgb(((image>1)&(image<254)).astype(np.float32),np.array([[1,0,1/32],[0,1,1/32]]),(20,20))
        self.assertEqual(float(valid[5,5,0]),1023/1024)
        self.assertTrue(valid[5,5,0]>.999);self.assertFalse(valid[5,5,0]>=a.STRICT_MINIMUM)

    def test_identity_does_not_reject_zero_weight_neighbour(self):
        c,_,_=self.ops
        valid=np.ones((20,20,3),np.float32);valid[6,6]=0
        moved,_=c.warp_to_rgb(valid,np.eye(2,3),(20,20))
        self.assertEqual(float(moved[5,5,0]),1)

    def test_only_noir_dependent_formula_counts_change(self):
        image=np.full((20,20,3),100,np.uint8);noir=image.copy();noir[6,6]=254;mask=np.ones((20,20),bool)
        rows,channels,_=a.compare_pair(image,mask,noir,mask,np.array([[1,0,1/32],[0,1,1/32]]),self.ops)
        self.assertTrue(any(r['lost_valid_pixels']>0 for r in rows if r['family']=='cross'))
        self.assertTrue(all(r['lost_valid_pixels']==0 for r in rows if r['family']=='rgb'))
        self.assertTrue(all(r['lost_valid_pixels']==0 for r in channels if r['channel'].startswith('rgb_')))

    def test_rejected_pair_keeps_empty_cross_support(self):
        image=np.full((20,20,3),100,np.uint8);mask=np.ones((20,20),bool)
        rows,_,_=a.compare_pair(image,mask,image,mask,np.eye(2,3),self.ops,usable=False)
        self.assertTrue(all(r['old_eligible_pixels']==r['strict_eligible_pixels']==0 for r in rows if r['family']=='cross'))
        self.assertTrue(all(r['old_valid_pixels']>0 for r in rows if r['family']=='rgb'))

if __name__=='__main__':unittest.main()
