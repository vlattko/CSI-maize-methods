import unittest
import numpy as np
from architecture_external_pheno4d import associate,sequence_associate,feature_cost
from architecture_external_uncertainty import margins

class ExternalCorrespondenceTests(unittest.TestCase):
    def test_geometry_matching_is_permutation_invariant(self):
        a=np.array([[0,.2,0,.2],[0,.5,0,.5],[0,.8,0,.8]])
        b=a[[2,0,1]]+.001
        links,_,_=associate(feature_cost(a,b),.75)
        self.assertEqual(set(links),{(0,1),(1,2),(2,0)})
    def test_known_birth_and_loss_use_unmatched_states(self):
        a=np.array([[0,.2,0,.2],[0,.5,0,.5]])
        b=np.array([[0,.5,0,.5],[0,.9,0,.9]])
        costs=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)*10
        links,unmatched_a,unmatched_b=associate(costs,.4)
        self.assertEqual(links,[(1,0)]);self.assertEqual(unmatched_a,[0]);self.assertEqual(unmatched_b,[1])
    def test_sequence_alignment_tracks_geometry_order_not_label_ids(self):
        links,a,b=sequence_associate(np.array([[.01,4,8],[4,.01,4]]),.5)
        self.assertEqual(links,[(0,0),(1,1)]);self.assertEqual(b,[2])
    def test_ambiguity_remains_when_geometry_identical(self):
        c=feature_cost(np.zeros((2,4)),np.zeros((2,4)))
        links,_,_=associate(c,.75);original=sum(c[i,j] for i,j in links)
        c2=c.copy();c2[links[0]]=1e6;other,_,_=associate(c2,.75)
        self.assertEqual(original,sum(c2[i,j] for i,j in other))
        self.assertTrue(all(m==0 for i,j,m in margins(c)))

if __name__=='__main__':unittest.main()
