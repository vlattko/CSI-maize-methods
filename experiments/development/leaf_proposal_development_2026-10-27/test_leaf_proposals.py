import unittest
import numpy as np
import leaf_proposals as p

class ProposalContracts(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((64,64),bool)
        for y in range(5,56):self.mask[max(0,y-2):y+3,28:33]=True
        for k in range(21):self.mask[3+k:8+k,8+k:13+k]=True
        self.path=[[30,55],[30,35],[30,25],[20,15],[10,5]]

    def test_shared_stem_cannot_become_a_positive_prompt(self):
        result=p.make_prompt(self.path,[10,5],28.284,self.mask,[[50,5],[30,10]])
        positives=[xy for xy,label in zip(result['points'],result['labels']) if label==1]
        self.assertEqual(len(positives),3)
        self.assertTrue(all(x<30 and y<25 for x,y in positives))
        self.assertTrue(all(self.mask[round(y),round(x)] for x,y in positives))

    def test_reversed_path_is_oriented_by_terminal(self):
        a=p.make_prompt(self.path,[10,5],28.284,self.mask,[])
        b=p.make_prompt(self.path[::-1],[10,5],28.284,self.mask,[])
        self.assertEqual(a['points'],b['points'])
        self.assertEqual(a['box'],b['box'])

    def test_negative_points_exclude_target_and_outside_image(self):
        result=p.make_prompt(self.path,[10,5],28.284,self.mask,[[10,5],[10,6],[-1,20],[50,50]])
        negatives=[xy for xy,label in zip(result['points'],result['labels']) if label==0]
        self.assertEqual(negatives,[[50.,50.]])

    def test_noir_terminal_matching_abstains_outside_gate(self):
        candidates=[dict(terminal_x_px=20,terminal_y_px=10,candidate_id=4)]
        self.assertIsNone(p.match_terminal([50,50],candidates,10)['candidate'])
        self.assertEqual(p.match_terminal([21,11],candidates,10)['candidate']['candidate_id'],4)

    def test_parent_clipping_accounting_keeps_raw_pixels(self):
        parent=np.zeros((4,4),bool);parent[1:3,1:3]=True
        raw=np.zeros((4,4),bool);raw[:3,:3]=True
        clipped,stats=p.clip_statistics(raw,parent)
        self.assertEqual(stats['raw_area_pixels'],9)
        self.assertEqual(stats['clipped_area_pixels'],4)
        self.assertEqual(stats['raw_outside_parent_pixels'],5)
        self.assertEqual(stats['parent_coverage_fraction'],1.)
        self.assertEqual(raw.sum(),9)

    def test_coco_rle_is_column_major_with_leading_zero_run(self):
        mask=np.array([[1,0,0],[1,1,0]],bool)
        self.assertEqual(p.coco_rle(mask),dict(size=[2,3],counts=[0,2,1,1,2]))

    def test_processing_points_use_pixel_centers_and_exact_foreground_support(self):
        parent=np.zeros((8,8),bool);parent[3,3]=True
        prompt=dict(box=[0,0,1,1],points=[[0,0]],labels=[1])
        result=p.processing_prompt(prompt,parent,grid_shape=(2,2))
        self.assertEqual(result['points_before_support_refinement'],[[1.5,1.5]])
        self.assertEqual(result['points'],[[3.,3.]])
        self.assertTrue(result['point_inside_exact_parent'][0])

if __name__=='__main__':unittest.main(verbosity=2)
