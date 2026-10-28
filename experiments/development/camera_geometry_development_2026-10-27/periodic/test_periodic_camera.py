"""Known-answer tests for the fixed first-harmonic camera surrogate."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
import unittest
import numpy as np
import periodic_camera as p
import camera_geometry as g
from test_camera_geometry import rotation_masks
import cv2

def truth(index):
    t=index*2*np.pi/40
    return np.array([[1+.035*np.cos(t),.02*np.sin(t),5+4*np.sin(t)],
                     [.015*np.cos(t),1+.02*np.sin(t),-3+2*np.cos(t)]])

class PeriodicKnownAnswers(unittest.TestCase):
    def test_harmonic_recovers_unseen_angle_matrices_with_teacher_outliers(self):
        teachers=[]
        for i in range(40):
            if i%4==2:continue
            m=truth(i)
            if i in (5,21):m=m.copy();m[:,2]+=[24,-18]
            teachers.append(dict(view=i,matrix=m.tolist(),weight=1.))
        fit=p.fit_coefficients(teachers,(256,192))
        errors=[]
        for i in range(2,40,4):
            m=p.predict(fit,i)
            pts=np.array([[30,40,1],[210,160,1]])
            errors.extend(np.linalg.norm(pts@m.T-pts@truth(i).T,axis=1))
        self.assertLess(np.max(errors),1.2)

    def test_test_angle_teacher_is_rejected_before_regression(self):
        teachers=[dict(view=i,matrix=truth(i).tolist(),weight=1.) for i in range(8)]
        with self.assertRaisesRegex(ValueError,'heldout_teacher'):
            p.fit_coefficients(teachers,(256,192))

    def test_reflected_and_singular_matrices_abstain(self):
        self.assertFalse(p.matrix_guard([[-1,0,0],[0,1,0]],(256,192))['valid'])
        self.assertFalse(p.matrix_guard([[1,0,0],[0,.0001,0]],(256,192))['valid'])
        self.assertTrue(p.matrix_guard(truth(2),(256,192))['valid'])

    def test_inner_validation_requires_gain_and_retains_declines(self):
        self.assertFalse(p.select_periodic([.01,-.1,.02],30,.8,True)['accepted'])
        self.assertTrue(p.select_periodic([.06,.04,.07],30,.8,True)['accepted'])
        self.assertFalse(p.select_periodic([.06,.04,.07],12,.8,True)['accepted'])

    def test_mask_driven_teacher_and_inner_fits_ignore_outer_test_masks(self):
        rgb=rotation_masks()
        noir={i:cv2.warpAffine(m.astype(np.uint8),truth(i).astype(np.float32),(128,96),
                    flags=cv2.INTER_NEAREST)>0 for i,m in rgb.items()}
        shared=g.fit_session(rgb,noir)
        first=p.fit_periodic(rgb,noir,shared)
        self.assertIsNotNone(first['model'])
        for i in range(2,40,4):
            rgb[i][:]=False;noir[i][:]=True
        second=p.fit_periodic(rgb,noir,shared)
        self.assertEqual(first,second)

if __name__=='__main__':unittest.main(verbosity=2)
