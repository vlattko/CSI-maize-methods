import unittest


class RegistryTests(unittest.TestCase):
    def test_same_pot_across_visits_but_different_treatments_remain_distinct(self):
        import registry
        rows=[dict(treatment=t,genotype='B73',replicate='1',measurement=v) for t in ['C','S'] for v in ['1','2']]
        confirmation=dict(same_labels_refer_to_same_physical_pots=True,declared_das_by_visit={'1':13,'2':19},confirmation_id='human')
        records=registry.make_registry(rows,confirmation)
        self.assertEqual(records[0]['plant_uid'],records[1]['plant_uid'])
        self.assertNotEqual(records[0]['plant_uid'],records[2]['plant_uid'])
        self.assertEqual(records[1]['declared_das'],19)

    def test_duplicate_identity_visit_is_rejected(self):
        import registry
        row=dict(treatment='C',genotype='B73',replicate='1',measurement='1')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            registry.make_registry([row,row],dict(same_labels_refer_to_same_physical_pots=True,declared_das_by_visit={'1':13},confirmation_id='human'))


if __name__=='__main__':unittest.main()
