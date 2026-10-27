import unittest
from pairing_aggregate import aggregate, schema


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.sessions=[dict(session='s'+str(i),treatment='C',measurement='1',genotype='g',replicate=str(i))
                       for i in range(3)]
        self.definitions=[dict(id='i'+str(i)) for i in range(27)]

    def row(self,session='s0',view=0,index='i0',median=2,pixels=40,method='ECC',pair='nominal'):
        metadata=next(s for s in self.sessions if s['session']==session)
        return dict(metadata,view=str(view),index=index,median=str(median) if pixels>=40 else '',
                    valid_pixels=str(pixels),eligible_pixels='100',qualified=pixels>=40,method=method,pair_kind=pair)

    def test_two_quarters_per_feature_missingness_and_all_sessions(self):
        rows=[self.row(view=0,median=2),self.row(view=10,median=6),
              self.row(view=0,index='i1',median=99),self.row(view=10,index='i1',pixels=39)]
        result=aggregate(self.sessions,rows,[],self.definitions,'ECC')
        self.assertEqual(len(result),3)
        self.assertEqual(result[0]['paired_index__i0__median'],4)
        self.assertEqual(result[0]['paired_index__i0__iqr'],2)
        self.assertEqual(result[0]['paired_index__i0__n_views'],2)
        self.assertIsNone(result[0]['paired_index__i1__median'])
        self.assertEqual(result[0]['paired_index__i1__n_views'],1)
        self.assertEqual(result[1]['paired_index__i0__n_views'],0)
        self.assertIsNone(result[1]['paired_index__i0__median'])
        self.assertTrue(result[0]['paired_index_any_eligible'])
        self.assertFalse(result[0]['paired_index_all_27_eligible'])

    def test_controls_and_alternative_methods_cannot_enter_primary(self):
        rows=[self.row(median=999,pair='wrong_plant'),self.row(view=10,median=999,pair='wrong_view'),
              self.row(median=999,method='moment'),self.row(view=10,median=999,method='moment')]
        result=aggregate(self.sessions,rows,[],self.definitions,'ECC')
        self.assertIsNone(result[0]['paired_index__i0__median'])

    def test_duplicate_quarter_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate descriptor'):
            aggregate(self.sessions,[self.row(),self.row()],[],self.definitions,'ECC')

    def test_identity_mismatch_rejected(self):
        row=self.row();row['genotype']='different'
        with self.assertRaisesRegex(ValueError,'identity disagrees'):
            aggregate(self.sessions,[row],[],self.definitions,'ECC')

    def test_schema_feature_whitelist_excludes_availability(self):
        values=schema(self.definitions)
        self.assertEqual(len(values['feature_columns']),54)
        self.assertTrue(all(s.endswith(('__median','__iqr')) for s in values['feature_columns']))
        self.assertFalse(set(values['feature_columns'])&set(values['availability_columns']))


if __name__=='__main__':unittest.main()
