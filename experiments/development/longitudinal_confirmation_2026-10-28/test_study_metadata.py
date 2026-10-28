import datetime as dt
import unittest
from study_metadata import mass_records, timetable_records


class MetadataTests(unittest.TestCase):
    def rows(self):
        return [('L'+str(i), 'G'+str(i), 10, 1, 12, 2, 9, .5)
                for block in range(4) for i in ([1,2,3,4,5,6] if block==0 else [6,3,1,5,2,4])]

    def confirmation(self):
        return dict(replicate_blocks_confirmed=True, mass_units='g', tissue='whole aboveground plant')

    def codes(self):
        return {'G'+str(i):'L'+str(i) for i in range(1,7)}

    def test_written_genotype_survives_permuted_replicate_blocks(self):
        result=mass_records(self.rows(), self.codes(), self.confirmation())
        found=[r for r in result if r['genotype']=='G6' and r['replicate']=='2' and r['treatment']=='P']
        self.assertEqual(len(result),72)
        self.assertEqual(found[0]['source_fresh_cell'],'G8')
        self.assertEqual(found[0]['dry_matter_fraction'],.5/9)

    def test_duplicate_genotype_within_block_is_rejected(self):
        rows=self.rows(); rows[1]=rows[0]
        with self.assertRaises(ValueError):mass_records(rows,self.codes(),self.confirmation())

    def test_unconfirmed_mapping_or_impossible_mass_is_rejected(self):
        with self.assertRaises(ValueError):mass_records(self.rows(),self.codes(),{})
        rows=self.rows();rows[0]=('L1','G1',1,2,12,2,9,.5)
        with self.assertRaises(ValueError):mass_records(rows,self.codes(),self.confirmation())

    def test_timetable_rejects_date_age_disagreement(self):
        rows=[dict(row=5,date=dt.datetime(2025,3,13),das=0,action='sjetva',detail=None,day=25,night=18),
              dict(row=6,date=dt.datetime(2025,3,15),das=1,action='cold',detail=None,day=10,night=5)]
        with self.assertRaises(ValueError):timetable_records(rows)

    def test_schedule_ages_do_not_become_capture_timestamps(self):
        rows=[dict(row=5,date=dt.datetime(2025,3,13),das=0,action='sjetva',detail=None,day=25,night=18),
              dict(row=18,date=dt.datetime(2025,3,26),das=13,action='kontrola',detail='snimanje',day=25,night=18),
              dict(row=19,date=dt.datetime(2025,3,27),das=14,action='cold',detail=None,day=10,night=5)]
        result=timetable_records(rows)
        self.assertEqual(result[1]['treatment'],'S')
        self.assertEqual(result[1]['scheduled_date'],'2025-03-26')
        self.assertEqual(result[1]['measurement'],'1')
        self.assertNotIn('capture_timestamp',result[1])


if __name__=='__main__':unittest.main()
