import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('cohort_extraction',Path(__file__).with_name('extraction.py'))
module=importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
except FileNotFoundError:
    module=None


class CohortAggregationTests(unittest.TestCase):
    def test_partial_rotations_preserve_empty_samples_and_modality_eligibility(self):
        self.assertIsNotNone(module,'extraction module is not implemented')
        master=[dict(treatment='S',measurement='2',genotype='B73',replicate=str(i)) for i in [1,2]]
        frames=[]
        for camera,indices in [('rgb',list(range(0,40,2))),('noir',list(range(20)))]:
            frames += [dict(**master[0],session='IMAGES/sample',camera=camera,view=v,
                            qc_pass=True,silhouette_area_px=100+v,height_fraction=.8,
                            source=f'{camera}/{v}',mask='mask.png') for v in indices]
        result=module.aggregate(master,frames)
        self.assertEqual(len(result),2)
        self.assertTrue(result[0]['rgb_eligible'])
        self.assertFalse(result[0]['noir_eligible'])
        self.assertEqual(result[0]['rgb_quadrants'],4)
        self.assertEqual(result[0]['noir_quadrants'],2)
        self.assertFalse(result[0]['paired_eligible'])
        self.assertEqual(result[0]['rgb__silhouette_area_px__median'],119)
        self.assertEqual(result[1]['rgb_usable_views'],0)
        self.assertIsNone(result[1]['rgb__silhouette_area_px__median'])

    def test_rejects_duplicate_view_and_reference_identity(self):
        self.assertIsNotNone(module,'extraction module is not implemented')
        row=dict(treatment='C',measurement='1',genotype='B73',replicate='1')
        frame=dict(**row,camera='rgb',view=0,qc_pass=True,silhouette_area_px=12)
        with self.assertRaises(ValueError):module.aggregate([row,row],[])
        with self.assertRaises(ValueError):module.aggregate([row],[frame,frame])

    def test_failed_qc_cannot_enter_session_mean_or_become_zero(self):
        self.assertIsNotNone(module,'extraction module is not implemented')
        row=dict(treatment='P',measurement='4',genotype='Mo17',replicate='3')
        frames=[dict(**row,camera='rgb',view=v,qc_pass=False,silhouette_area_px=999) for v in range(40)]
        value=module.aggregate([row],frames)[0]
        self.assertEqual(value['rgb_usable_views'],0)
        self.assertIsNone(value['rgb__silhouette_area_px__median'])
        self.assertFalse(value['rgb_eligible'])


if __name__=='__main__':unittest.main()
