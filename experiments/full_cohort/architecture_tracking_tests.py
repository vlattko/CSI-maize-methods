import unittest
import numpy as np
from architecture_tracking import fit_tracks,predict

class TrackingTests(unittest.TestCase):
    def test_recover_known_two_tracks_with_missing_views(self):
        candidates={}
        for v in range(40):
            if v%4==2:continue
            t=v*2*np.pi/40
            candidates[v]=[dict(x=.35*np.cos(t),y=.8,root_length=1),dict(x=.2*np.sin(t),y=.4,root_length=.7)]
            if v%7==0:candidates[v]=candidates[v][1:]
        tracks=fit_tracks(candidates)
        self.assertEqual(len(tracks),2)
        self.assertLess(abs(sorted(q['height'] for q in tracks)[0]-.4),1e-8)
        self.assertLess(abs(sorted(q['radius'] for q in tracks)[1]-.35),1e-8)
        p=predict(tracks[0],6);self.assertEqual(p.shape,(2,))
    def test_refuse_sparse_support(self):
        self.assertEqual(fit_tracks({0:[dict(x=0,y=.8,root_length=1)],9:[dict(x=.2,y=.8,root_length=1)]}),[])
    def test_heldout_leakage_guard(self):
        with self.assertRaises(ValueError):fit_tracks({2:[]})
    def test_birth_loss_and_rank_ambiguity(self):
        # Synthetic visit 1 contains a basal path and a tall path; visit 2 loses
        # the basal observation and acquires a new intermediate one. These are
        # known injected states; absent real-data paths do not establish death.
        def scene(heights):
            return {v:[dict(x=.25*np.cos(v*2*np.pi/40),y=h,root_length=h+.2) for h in heights] for v in range(40) if v%4!=2}
        first=fit_tracks(scene([.3,.9]));second=fit_tracks(scene([.6,.9]))
        self.assertEqual(len(first),2);self.assertEqual(len(second),2)
        self.assertAlmostEqual(max(q['height'] for q in first),max(q['height'] for q in second))
        # Two organs with exactly the same image trajectory cannot be resolved.
        # Multiple duplicate observations can generate duplicate hypotheses;
        # the experiment deliberately does not interpret their count as organs.
        ambiguous=fit_tracks(scene([.6,.6]))
        self.assertEqual(len(ambiguous),2)
        self.assertAlmostEqual(ambiguous[0]['height'],ambiguous[1]['height'])

if __name__=='__main__':unittest.main()
