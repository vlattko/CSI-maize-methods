import unittest
import numpy as np
import tracking_sensitivity as s

class SensitivityTests(unittest.TestCase):
    def test_heldout_genotype_scope_excludes_training_genotypes(self):
        self.assertTrue(s.selected_acquisition({'genotype':'Os7798/2'},'executed_with_hypotheses','Os7798/2'))
        self.assertFalse(s.selected_acquisition({'genotype':'B73'},'executed_with_hypotheses','Os7798/2'))
        self.assertFalse(s.selected_acquisition({'genotype':'Os7798/2'},'not_eligible','Os7798/2'))

    def test_empty_is_not_zero_disagreement(self):
        self.assertIsNone(s.set_distance([],[],1)['symmetric_mean'])
        self.assertEqual(s.set_distance([[1,2]],[],1)['state'],'student_empty')

    def test_symmetric_known_distance_and_scale(self):
        r=s.set_distance([[0,0]],[[3,4]],5);self.assertEqual(r['symmetric_mean'],1)
        self.assertEqual(s.set_distance([[0,0]],[[3,4]],5),s.set_distance([[3,4]],[[0,0]],5))

    def test_source_hash_and_identity_are_required(self):
        frame=dict(source_path='session/rgb/image_rgb_000.jpg',source_sha256='abc')
        identity=dict(treatment='C',genotype='B73',replicate='1')
        row=dict(identity,measurement='1',source=frame['source_path'],source_sha256='wrong',teacher_accepted='True')
        with self.assertRaisesRegex(ValueError,'source mismatch'):s.bind_frame(identity,1,0,frame,{('session','rgb',0):row})

    def test_rejected_views_cannot_enter_matched_support(self):
        frame=dict(source_path='session/rgb/image_rgb_000.jpg',source_sha256='abc');identity=dict(treatment='C',genotype='B73',replicate='1')
        row=dict(identity,measurement='1',source=frame['source_path'],source_sha256='abc',teacher_accepted='False')
        with self.assertRaisesRegex(ValueError,'view-support'):s.bind_frame(identity,1,0,frame,{('session','rgb',0):row})

if __name__=='__main__':unittest.main()
