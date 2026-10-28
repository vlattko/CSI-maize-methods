import unittest
import numpy as np
import torch
from topology_core import TinyUNet,soft_cldice,segmentation_loss,guard_record,select_validation,teacher_geometry,agreement_metrics


class TopologyTests(unittest.TestCase):
    def test_unet_preserves_odd_shape(self):
        model=TinyUNet(base=8);x=torch.zeros(2,3,97,129)
        self.assertEqual(tuple(model(x).shape),(2,1,97,129))
        self.assertLess(sum(p.numel() for p in model.parameters()),1000000)

    def test_real_gap_in_line_is_penalized(self):
        truth=torch.zeros(1,1,31,31);truth[:,:,15,3:28]=1
        gap=truth.clone();gap[:,:,15,14:17]=0
        self.assertLess(float(soft_cldice(truth,truth)),1e-5)
        self.assertGreater(float(soft_cldice(gap,truth)),.01)

    def test_cldice_gradient_is_finite_and_nonzero(self):
        target=torch.zeros(1,1,31,31);target[:,:,15,3:28]=1
        logits=torch.randn(1,1,31,31,requires_grad=True);loss=soft_cldice(logits.sigmoid(),target);loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all());self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_empty_masks_are_finite(self):
        empty=torch.zeros(2,1,31,31)
        self.assertLess(float(soft_cldice(empty,empty)),1e-5)
        logits=torch.zeros_like(empty,requires_grad=True);loss=segmentation_loss(logits,empty,topology=True);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(logits.grad).all())

    def test_forbid_test_and_wrong_genotype(self):
        with self.assertRaises(ValueError):guard_record(dict(split='test',genotype='Os7798/2',teacher_accepted='True'))
        with self.assertRaises(ValueError):guard_record(dict(split='train',genotype='Os6251/15',teacher_accepted='True'))
        self.assertTrue(guard_record(dict(split='val',genotype='Os6251/15',teacher_accepted='True')))

    def test_model_selection_never_accepts_test(self):
        with self.assertRaises(ValueError):select_validation([dict(split='test',score=.9,epoch=1,threshold=.5)])
        r=select_validation([dict(split='val',score=.8,epoch=1,threshold=.5),dict(split='val',score=.85,epoch=2,threshold=.35)])
        self.assertEqual(r['epoch'],2)

    def test_teacher_endpoint_and_thin_support_detect_gap(self):
        truth=np.zeros((40,40),bool);truth[20,5:35]=1;pred=truth.copy();pred[20,30:35]=False
        m=agreement_metrics(truth,pred,teacher_geometry(truth))
        self.assertLess(m['endpoint_recall_px3'],1.)
        self.assertLess(m['thin_skeleton_recall_px0'],1.)


if __name__=='__main__':unittest.main()
