"""Known-answer and information-boundary tests for the new surrogate layer."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import unittest
import numpy as np
import cv2
import camera_geometry as g

cv2.setNumThreads(2)

def rotation_masks():
    result={}
    for i in range(40):
        t=i*2*np.pi/40
        m=np.zeros((96,128),np.uint8)
        cv2.rectangle(m,(61,27),(65,89),1,-1)
        for j,(h,a,phi) in enumerate([(27,24,.2),(40,30,1.3),(55,21,3.1),(69,26,4.6)]):
            x=63+a*np.sin(t+phi)+4*np.sin(2*t+j)
            cv2.line(m,(63,h+10),(round(x),h),1,3)
            cv2.ellipse(m,(round(x),h),(max(3,round(7+4*np.cos(t+phi))),3),0,0,360,1,-1)
        result[i]=m.astype(bool)
    return result

class KnownAnswers(unittest.TestCase):
    def test_shared_affine_recovers_known_translation_on_unseen_angles(self):
        rgb=rotation_masks()
        truth=np.array([[1.05,.035,6],[-.02,.98,-4]],np.float32)
        noir={i:cv2.warpAffine(m.astype(np.uint8),truth,(128,96),flags=cv2.INTER_NEAREST)>0 for i,m in rgb.items()}
        fit=g.fit_shared([(rgb[i],noir[i],i) for i in range(40) if i%4!=2])
        scores=[g.iou_warp(rgb[i],noir[i],fit['matrix']) for i in range(2,40,4)]
        self.assertGreater(np.median(scores),.86)
        points=np.array([[30,25,1],[90,70,1],[60,88,1]])
        error=np.linalg.norm(points@np.asarray(fit['matrix']).T-points@truth.T,axis=1)
        self.assertLess(error.max(),2.5)

    def test_phase_and_sign_recover_permuted_rotation(self):
        rgb=rotation_masks()
        shift=np.array([[1,0,5],[0,1,-3]],np.float32)
        noir={(-i+8)%40:cv2.warpAffine(m.astype(np.uint8),shift,(128,96),flags=cv2.INTER_NEAREST)>0 for i,m in rgb.items()}
        fit=g.fit_session(rgb,noir)
        self.assertEqual((fit['selected']['sign'],fit['selected']['phase'] ),(-1,8))
        self.assertTrue(fit['phase_gate']['accepted_alternate'])

    def test_heldout_masks_cannot_change_training_fit_or_phase(self):
        rgb=rotation_masks(); noir={i:m.copy() for i,m in rgb.items()}
        first=g.fit_session(rgb,noir)
        for i in range(2,40,4):
            rgb[i][:]=False; noir[i][:]=True
        second=g.fit_session(rgb,noir)
        self.assertEqual(first,second)

    def test_ambiguous_rotation_keeps_nominal_and_reports_ambiguity(self):
        m=rotation_masks()[0]
        fit=g.fit_session({i:m for i in range(40)},{i:m for i in range(40)})
        self.assertEqual((fit['selected']['sign'],fit['selected']['phase']),(1,0))
        self.assertFalse(fit['phase_gate']['accepted_alternate'])

    def test_insufficient_training_support_has_no_lookup(self):
        fit=g.fit_session({}, {})
        self.assertEqual(fit['status'],'insufficient_training_pairs')
        self.assertIsNone(fit['selected'])

    def test_grid_conversion_preserves_affine_coordinates(self):
        m=np.array([[1.2,.1,4],[-.05,.9,-2]])
        converted=g.rescale_matrix(m,(128,96),(512,384),(128,96),(1024,768))
        self.assertTrue(np.allclose(converted,[[2.4,.2,32],[-.1,1.8,-16]]))

    def test_ecc_failure_keeps_valid_moment_and_records_reason(self):
        m=np.ones((96,128),bool)
        result=g.per_view_ecc(m,m)
        self.assertTrue(np.isfinite(result['matrix']).all())
        self.assertEqual(result['solver_status'],'opencv_error')
        self.assertEqual(result['iou'],1.)

    def test_deployment_abstains_on_training_quality_and_ignores_test_diagnostics(self):
        fit=dict(status='fitted',selected=dict(n_train=30,median_iou=.4,matrix=[[1,0,0],[0,1,0]]))
        self.assertTrue(g.deployment_decision(fit)['deployment_abstain'])
        fit['selected']['median_iou']=.8
        fit['heldout_iou']=0.
        self.assertFalse(g.deployment_decision(fit)['deployment_abstain'])

if __name__=='__main__':
    unittest.main(verbosity=2)
