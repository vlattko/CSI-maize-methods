import unittest
from architecture_longitudinal import aggregate_reference

class LongitudinalTests(unittest.TestCase):
    def test_empty_reference_rows_are_retained_and_not_zero_phenotype(self):
        master=[dict(treatment='C',measurement='1',genotype='G',replicate='1')]
        rows=aggregate_reference(master,[],set(),'full_extraction')
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['verified_accepted_views'],0)
        self.assertIsNone(rows[0]['area_fraction_median'])
        self.assertFalse(rows[0]['primary_eligible'])
    def test_mask_hash_verification_gates_eligibility(self):
        master=[dict(treatment='C',measurement='1',genotype='G',replicate='1')]
        frames=[dict(**master[0],session='S',camera='rgb',view=v,qc_pass=True,area_fraction=.1) for v in range(40)]
        rows=aggregate_reference(master,frames,set(),'full_extraction')
        self.assertFalse(rows[0]['primary_eligible'])
        verified={('C','1','G','1','rgb',v) for v in range(40)}
        rows=aggregate_reference(master,frames,verified,'full_extraction')
        rgb=next(r for r in rows if r['camera']=='rgb');self.assertTrue(rgb['primary_eligible']);self.assertAlmostEqual(rgb['area_fraction_median'],.1)

if __name__=='__main__':unittest.main()
