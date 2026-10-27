"""Known-answer checks for the conditional architecture experiments."""
import unittest
import json
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from architecture_analysis import harmonic, coverage, visual_hull, project_hull, mask_candidates, correspondence,load_mask,MaskIntegrityError


class ArchitectureTests(unittest.TestCase):
    def test_mask_source_hash_binding_is_a_hard_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'mask.png';Image.fromarray(np.ones((10,10),np.uint8)*255).save(path)
            with self.assertRaises(MaskIntegrityError):load_mask(dict(mask='mask.png',mask_sha256='wrong'),Path(folder))
    def test_harmonic_known_answer_and_sparse_alias(self):
        views=np.arange(40); y=4+2*np.cos(views*2*np.pi/40)+.7*np.sin(views*4*np.pi/40)
        fit=harmonic(views,y,2)
        self.assertLess(fit['rmse'],1e-10)
        self.assertAlmostEqual(fit['amplitude_1'],2)
        self.assertEqual(harmonic([0,10,20,30],[1,2,1,2],2)['status'],'insufficient_or_aliased')

    def test_coverage(self):
        self.assertFalse(coverage(range(20))['eligible'])
        self.assertTrue(coverage(range(0,40,2))['eligible'])
    def test_harmonic_requires_primary_angular_coverage(self):
        views=list(range(19));values=np.ones(19)
        self.assertEqual(harmonic(views,values)['status'],'insufficient_or_aliased')
    def test_coverage_is_json_serializable(self):
        result=coverage(range(40))
        json.dumps({'strict_eligible_count':sum([result['strict_eligible']])})

    def test_hull_heldout_cylinder(self):
        n=61; y,x=np.mgrid[:n,:n]; mask=(abs(x-30)<=12)&(y>=8)&(y<=55)
        masks={i:mask for i in range(0,40,4)}
        hull,grid=visual_hull(masks,axis=30,base=55,scale=47,resolution=48)
        pred=project_hull(hull,grid,2,mask.shape,30,55,47)
        self.assertGreater((pred&mask).sum()/(pred|mask).sum(),.75)
        self.assertGreater(hull.sum(),0)
        # Additional constraints cannot enlarge the occupied set.
        shrunk=dict(masks);shrunk[1]=np.zeros_like(mask)
        h2,_=visual_hull(shrunk,30,55,47,resolution=48)
        self.assertEqual(h2.sum(),0)
    def test_hull_mask_uncertainty_is_explicit_and_monotone(self):
        n=61;y,x=np.mgrid[:n,:n];mask=(abs(x-30)<=12)&(y>=8)&(y<=55);masks={i:mask for i in range(0,40,4)}
        strict,grid=visual_hull(masks,30,55,47,resolution=24)
        relaxed,_=visual_hull(masks,30,55,47,resolution=24,dilation_px=1,allowed_violation_fraction=.1)
        self.assertTrue(np.all(relaxed[strict]));self.assertGreaterEqual(relaxed.sum(),strict.sum())

    def test_candidate_endpoints_are_proxies(self):
        m=np.zeros((100,100),bool);m[10:95,48:53]=True;m[40:45,15:85]=True
        c=mask_candidates(m,min_length=8)
        self.assertGreaterEqual(len(c),2)
        self.assertTrue(all(q['status']=='visible_skeleton_path_hypothesis' for q in c))

    def test_correspondence_preserves_competing_maps(self):
        a=[{'terminal_y_fraction':.3,'root_geodesic_over_height':.5}]
        b=[dict(a[0]),dict(a[0])]
        r=correspondence(a,b)
        self.assertEqual(len(r),2)
        self.assertAlmostEqual(r[0]['conditional_weight'],r[1]['conditional_weight'])

if __name__=='__main__':unittest.main()
