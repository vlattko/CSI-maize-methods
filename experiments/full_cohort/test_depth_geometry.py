import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('depth_geometry',Path(__file__).with_name('depth_geometry.py'))
m=importlib.util.module_from_spec(spec)
try:spec.loader.exec_module(m)
except FileNotFoundError:m=None


class GeometryTests(unittest.TestCase):
    def test_nominal_turntable_camera_centres_follow_known_quarter_circle(self):
        self.assertIsNotNone(m)
        e=m.turntable([0,10,20,30],4,1)
        c=np.array([-x[:3,:3].T@x[:3,3] for x in e])
        np.testing.assert_allclose(np.linalg.norm(c,axis=1),4,atol=1e-7)
        np.testing.assert_allclose(c.sum(0),0,atol=1e-7)

    def test_positive_depth_unprojection_and_projection_are_inverse(self):
        self.assertIsNotNone(m)
        k=np.array([[80,0,32],[0,80,24],[0,0,1]],float)
        ext=m.turntable([10],4,1)[0]
        mask=np.zeros((48,64),bool);mask[20:25,29:35]=True
        depth=np.full(mask.shape,3.5)
        world=m.unproject(depth,mask,k,ext,stride=1)
        predicted=m.project(world,k,ext,mask.shape,radius=0)
        np.testing.assert_array_equal(predicted,mask)

    def test_similarity_alignment_uses_only_supplied_training_centres(self):
        self.assertIsNotNone(m)
        a=np.array([[0,0,0],[1,0,0],[0,1,0]],float)
        r=np.array([[0,-1,0],[1,0,0],[0,0,1]],float)
        b=2*(a@r.T)+[3,4,5]
        scale,rot,shift=m.similarity(a,b)
        np.testing.assert_allclose(scale,2,atol=1e-8)
        np.testing.assert_allclose((a@rot.T)*scale+shift,b,atol=1e-8)


if __name__=='__main__':unittest.main()
