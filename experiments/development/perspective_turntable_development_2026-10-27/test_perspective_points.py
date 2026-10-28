import unittest
import numpy as np
from perspective_points import project_point,fit_point,fit_harmonic,select_families


class PointModelTests(unittest.TestCase):
    def test_known_perspective_heldout_and_occlusion(self):
        f=dict(kind='perspective',focal=1.2,elevation=5,roll=3);p=np.array([.2,.4,-.1]);views=[v for v in range(40) if v%4!=2 and v%7!=0]
        xy=project_point(p,views,f,offset=.03);model=fit_point(views,xy,f,offset=.03)
        self.assertEqual(model['status'],'fitted')
        self.assertLess(np.linalg.norm(np.array(model['point'])-p),1e-6)
        test=np.arange(2,40,4)
        self.assertLess(np.max(abs(project_point(model['point'],test,f,offset=.03)-project_point(p,test,f,offset=.03))),1e-6)

    def test_known_orthographic_point(self):
        f=dict(kind='orthographic',focal=1,elevation=-5,roll=3);p=[.3,.6,.1];v=[i for i in range(40) if i%4!=2]
        q=fit_point(v,project_point(p,v,f),f)
        self.assertLess(np.linalg.norm(np.array(q['point'])-p),1e-9)

    def test_outer_angles_rejected_by_fitter(self):
        with self.assertRaises(ValueError):fit_point([0,2,3],np.zeros((3,2)),dict(kind='perspective',focal=1.2,elevation=0,roll=0))

    def test_projection_changes_with_known_phase(self):
        f=dict(kind='perspective',focal=.8,elevation=0,roll=0);p=[.2,.4,.1]
        a=project_point(p,[0],f,phase_degrees=90);b=project_point(p,[10],f)
        np.testing.assert_allclose(a,b)

    def test_depth_singularity_is_flagged(self):
        f=dict(kind='perspective',focal=1,elevation=0,roll=0)
        with self.assertRaises(ValueError):project_point([1.2,.4,0],range(40),f,strict=True)

    def test_unrestricted_harmonic_is_image_baseline(self):
        v=np.array([i for i in range(40) if i%4!=2]);t=v*np.pi/20;xy=np.column_stack([.1+.2*np.cos(t),.4+.03*np.sin(t)])
        m=fit_harmonic(v,xy,vertical=True)
        self.assertLess(m['training_rmse'],1e-10)

    def test_camera_alternatives_retained_without_false_uniqueness(self):
        rows=[dict(family='a',inner_case_median=.030),dict(family='b',inner_case_median=.035),dict(family='c',inner_case_median=.050)]
        r=select_families(rows,tolerance=.01)
        self.assertEqual(r['selected'],'a');self.assertEqual(r['retained'],['a','b'])

    def test_orthographic_focal_scale_is_unidentifiable(self):
        from perspective_points import marginal_camera_information
        f=dict(kind='orthographic',focal=1,elevation=0,roll=0);v=[i for i in range(40) if i%4!=2]
        rows=[dict(point=[.3,.5,.1],views=v,offset=0.,scale=1.),dict(point=[-.2,.2,.2],views=v,offset=0.,scale=1.)]
        r=marginal_camera_information(rows,f)
        self.assertLess(r['information'][0][0],1e-8)
        self.assertLess(r['numerical_rank'],3)

    def test_perspective_camera_has_local_information_for_generic_points(self):
        from perspective_points import marginal_camera_information
        f=dict(kind='perspective',focal=1.2,elevation=0,roll=0);v=[i for i in range(40) if i%4!=2]
        rows=[dict(point=p,views=v,offset=0.,scale=1.) for p in [[.3,.5,.1],[-.2,.2,.2],[.2,.8,-.3]]]
        r=marginal_camera_information(rows,f)
        self.assertEqual(r['numerical_rank'],3)


if __name__=='__main__':unittest.main()
