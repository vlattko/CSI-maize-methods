import unittest
import strict_features as s


def formula(**changes):
    row=dict(session='s',treatment='C',measurement='1',genotype='g',replicate='1',view='0',
             pair_kind='nominal',method='ECC',index='ND_rgb_R__noir_R',family='cross',
             eligible_pixels='50',valid_pixels='40',qualified='True',valid_fraction='.8',
             median='.2',q10='.1',q25='.15',q75='.25',q90='.3',iqr='.1')
    return dict(row,**changes)


def audited(row,**changes):
    result={k:row[k] for k in (*s.IDENTITY,'session','view','index','family')}
    for field in s.FIELDS:
        result['old_'+field]=row[field];result['strict_'+field]=row[field]
    for field,value in changes.items():result['strict_'+field]=value
    return result


class StrictFeatureTests(unittest.TestCase):
    def test_exact_replay_and_qualification_loss_preserve_identity(self):
        row=formula();audit=audited(row,valid_pixels='39',qualified='False',valid_fraction='.78',
                                  median='',q10='',q25='',q75='',q90='',iqr='')
        result=s.replace_formulas([row],[audit])
        self.assertEqual(result[0]['valid_pixels'],'39')
        self.assertEqual(result[0]['qualified'],'False')
        self.assertEqual(result[0]['median'],'')
        self.assertEqual(result[0]['session'],'s')
        self.assertEqual(row['valid_pixels'],'40')

    def test_duplicate_and_missing_audit_are_errors(self):
        row=formula();audit=audited(row)
        with self.assertRaisesRegex(ValueError,'Duplicate'):s.replace_formulas([row],[audit,audit])
        with self.assertRaisesRegex(ValueError,'coverage'):s.replace_formulas([row],[])

    def test_wrong_identity_and_changed_old_stat_are_errors(self):
        row=formula();audit=audited(row)
        with self.assertRaises(ValueError):s.replace_formulas([row],[dict(audit,genotype='wrong')])
        with self.assertRaisesRegex(ValueError,'old statistic'):s.replace_formulas([row],[dict(audit,old_median='.3')])

    def test_no_control_row_can_be_replaced(self):
        row=formula(pair_kind='wrong_view')
        with self.assertRaisesRegex(ValueError,'nominal ECC'):s.replace_formulas([row],[audited(row)])

    def test_dn_uses_only_common_interior_and_retains_unavailable(self):
        row=dict(session='s',treatment='C',measurement='1',genotype='g',replicate='1',view='0',
                 channel='noir_R',eligible_pixels='50',valid_pixels='40',valid_fraction='.8',
                 qualified='True',median='100',iqr='20',failure_reason='')
        empty=dict(row,session='empty',valid_pixels='0',qualified='False',median='',iqr='',failure_reason='unavailable')
        audit={k:row[k] for k in (*s.IDENTITY,'session','view','channel','eligible_pixels')}
        audit.update(scope='common_interior',old_valid_pixels='40',strict_valid_pixels='39',
                     old_qualified='True',strict_qualified='False',old_median='100',strict_median='',old_iqr='20',strict_iqr='')
        result=s.replace_dns([row,empty],[audit,dict(audit,scope='camera_interior')])
        self.assertEqual(result[0]['valid_pixels'],'39');self.assertEqual(result[0]['median'],'')
        self.assertEqual(result[1],empty)

    def test_qualified_dn_without_audit_is_error(self):
        row=dict(session='s',treatment='C',measurement='1',genotype='g',replicate='1',view='0',
                 channel='rgb_R',eligible_pixels='50',valid_pixels='40',valid_fraction='.8',qualified='True',median='80',iqr='2')
        with self.assertRaisesRegex(ValueError,'Unaudited'):s.replace_dns([row],[])

    def test_session_eligibility_loss_is_missing_not_zero_and_separate_from_cohort(self):
        old=dict(session='s',treatment='C',measurement='1',genotype='g',replicate='1',
                 paired_index__X__median=.2,paired_index__X__iqr=.1,
                 paired_index__X__n_views=2,paired_index__X__eligible=True)
        new=dict(old,paired_index__X__median=None,paired_index__X__iqr=None,
                 paired_index__X__n_views=1,paired_index__X__eligible=False)
        changes,availability=s.feature_changes([old],[new],['X'],'paired_index',{('C','1','g','1')})
        self.assertEqual(availability[0]['primary329_eligibility_losses'],1)
        self.assertEqual(availability[0]['all360_strict_eligible'],0)
        median=next(r for r in changes if r['feature'].endswith('__median'))
        self.assertIsNone(median['strict']);self.assertIsNone(median['delta'])

    def test_session_replay_rejects_changed_feature_before_strict_substitution(self):
        old=dict(session='s',treatment='C',measurement='1',genotype='g',replicate='1',value='0.2')
        s.compare_tables([old],[dict(old,value=.2)])
        with self.assertRaisesRegex(ValueError,'replay'):s.compare_tables([old],[dict(old,value=.3)])

    def test_independent_multitarget_metric_known_answer(self):
        import numpy as np
        from strict_verify import metrics
        result=metrics(np.array([[0.,0.],[2.,4.]]),np.array([[1.,1.],[1.,3.]]))
        self.assertEqual(result['rmse'],1.)
        self.assertEqual(result['mae'],1.)
        self.assertEqual(result['mean_target_r2'],.375)


if __name__=='__main__':unittest.main(verbosity=2)
