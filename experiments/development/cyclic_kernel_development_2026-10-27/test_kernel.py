import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from kernel import ViewScaler,image_kernel,permutations,trajectory_permutation


class KernelTests(unittest.TestCase):
    def test_group_filenames_are_contained_even_for_slash_labels(self):
        import benchmark
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for label in ['Os6251/15','Os7798/2','../escape','C:device','B73']:
                marker=benchmark.group_marker(root,label)
                self.assertEqual(marker.parent,root)
                self.assertTrue(marker.name.endswith('.json'))
            self.assertNotEqual(benchmark.group_marker(root,'Os6251/15'),benchmark.group_marker(root,'Os6251_15'))

    def test_resume_rejects_changed_prediction_bytes(self):
        import benchmark
        with tempfile.TemporaryDirectory() as folder:
            marker=Path(folder)/'group.json';array=marker.with_suffix('.npz')
            np.savez(array,prediction=np.ones((2,3)),test_indices=np.array([2,4]))
            marker.write_text(json.dumps(dict(signature='test',parameters=[],npz_sha256=benchmark.sha(array))))
            np.savez(array,prediction=np.zeros((2,3)),test_indices=np.array([2,4]))
            with self.assertRaisesRegex(ValueError,'digest'):
                benchmark.load_fold(marker,'test',np.array([2,4]))

    def test_group_kernel_is_phase_and_direction_invariant(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(7,2,4,5));y=rng.normal(size=(4,2,4,5))
        reference=image_kernel(x,y,'cyclic',.3)
        for p in permutations():np.testing.assert_allclose(reference,image_kernel(x,y[:,:,p,:],'cyclic',.3),atol=1e-12)

    def test_group_gram_is_symmetric_positive_semidefinite(self):
        x=np.random.default_rng(11).normal(size=(12,2,4,6));k=image_kernel(x,x,'cyclic',.5)
        np.testing.assert_allclose(k,k.T,atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(k).min(),-1e-10)
        np.testing.assert_allclose(np.diag(k),1,atol=1e-12)

    def test_scaler_does_not_learn_from_heldout_rows_and_preserves_group_action(self):
        x=np.random.default_rng(9).normal(size=(6,2,4,5));x[0,0,1]=np.nan
        scaler=ViewScaler().fit(x[:4]);before=scaler.mean.copy();scaler.transform(x[4:]*100)
        np.testing.assert_equal(scaler.mean,before)
        p=permutations()[5]
        np.testing.assert_allclose(scaler.transform(x[:,:,p,:]),scaler.transform(x)[:,:,p,:],atol=1e-12)
        self.assertTrue(np.isfinite(scaler.transform(x)).all())

    def test_trajectory_shuffle_stays_inside_training_and_matches_visit(self):
        rows=[dict(treatment=t,genotype=g,replicate='1',measurement=str(v)) for t in 'CP' for g in 'ABCD' for v in [1,2,3]]
        train=np.array([i for i,r in enumerate(rows) if r['genotype']!='D'])
        donor=trajectory_permutation(rows,train,47)
        self.assertEqual(set(donor),set(train))
        for i,j in zip(train,donor):
            self.assertEqual(rows[i]['treatment'],rows[j]['treatment'])
            self.assertEqual(rows[i]['measurement'],rows[j]['measurement'])
            self.assertNotEqual(rows[j]['genotype'],'D')


if __name__=='__main__':unittest.main()
