import unittest
import numpy as np
from architecture_external_rendered import render_points

class ExternalRenderTests(unittest.TestCase):
    def test_zbuffer_ownership_and_hidden_points(self):
        # Native normalized xyz: native z is vertical, native y is depth.
        xyz=np.array([[0,-.1,.5],[0,.1,.5],[.2,0,.7]])
        labels=np.array([1,2,3]);mask,owner=render_points(xyz,labels,0,128,100,64,115)
        self.assertEqual(owner[65,64],2)
        self.assertTrue(mask[65,64]);self.assertTrue(np.any(owner==3))
    def test_annotation_values_do_not_change_silhouette(self):
        xyz=np.array([[0,0,.5],[.2,0,.7]])
        a,_=render_points(xyz,np.array([1,2]),3,128,100,64,115)
        b,_=render_points(xyz,np.array([9,7]),3,128,100,64,115)
        self.assertTrue(np.array_equal(a,b))

if __name__=='__main__':unittest.main()
