"""Known-answer and bookkeeping tests, independent of cohort result values."""
import tempfile
import unittest
from pathlib import Path
from pairing_core import *
import pairing_run as run

REPO = Path(__file__).resolve().parents[2]
run.configure(REPO)
np = run.np


class PairingTests(unittest.TestCase):
    def test_duplicate_camera_key_rejected(self):
        row = dict(session='s',camera='rgb',view='0')
        with self.assertRaises(ValueError):
            frame_lookup([row,row])

    def test_control_excludes_same_candidate_across_visits(self):
        a = dict(session='a',treatment='C',measurement='1',genotype='g',replicate='1')
        repeated = dict(a,session='b',measurement='2')
        other = dict(a,session='c',replicate='2')
        selected, matching = wrong_plant([a,repeated,other],a)
        self.assertEqual(selected['session'],'c')
        self.assertEqual(matching,'treatment+measurement+genotype')

    def test_cache_signature_and_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cache.json'
            value=dict(signature='a',value=1)
            value['payload_sha256']=digest(value)
            atomic_json(path,value)
            self.assertEqual(cache_read(path,'a')['value'],1)
            self.assertIsNone(cache_read(path,'b'))
            value['value']=2
            atomic_json(path,value)
            with self.assertRaises(ValueError):cache_read(path,'a')

    def test_known_translation_and_public_initialization(self):
        a=np.zeros((80,100),bool);a[20:55,30:45]=True;a[25:30,20:60]=True
        b=np.zeros_like(a);b[25:60,37:52]=True;b[30:35,27:67]=True
        matrix=moment_affine(a,b,np)
        np.testing.assert_allclose(matrix,[[1,0,7],[0,1,5]],atol=1e-6)
        moved,_=run.warp_to_rgb(b.astype(np.uint8),matrix,a.shape,True)
        self.assertEqual(run.mask_iou(a,moved>0),1)
        _,_,status=run.affine_align(a,b)
        self.assertEqual(status,'moment_initialisation')

    def test_all_27_definitions_and_no_interpolated_clipping_repair(self):
        mask=np.ones((20,20),bool)
        a=np.full((20,20,3),80,np.uint8)
        b=np.full_like(a,120);b[:,10,0]=255
        matrix=np.array([[1,0,0.5],[0,1,0]],np.float32)
        ds,channels,maps,supports,wb,common=run.describe(a,mask,b,mask,matrix,True)
        self.assertEqual(len(ds),27)
        # The 255 contributor is invalid even though bilinear output is <254.
        invalid_column=(wb[...,0]>120)&(wb[...,0]<254)&supports['cross']
        self.assertTrue(invalid_column.any())
        self.assertTrue(np.isnan(maps['ND_rgb_R__noir_R'][invalid_column]).all())
        self.assertTrue(np.isfinite(maps['ExG'][invalid_column]).all())

    def test_rejected_geometry_retains_unimodal_descriptors(self):
        mask=np.ones((20,20),bool)
        a=np.full((20,20,3),80,np.uint8)
        matrix=np.array([[1,0,0],[0,1,0]],np.float32)
        ds,_,maps,supports,_,_=run.describe(a,mask,a,mask,matrix,False)
        cross=[d for d in ds if d['family']=='cross']
        self.assertTrue(all(d['eligible_pixels']==0 and d['median'] is None for d in cross))
        self.assertTrue(np.isfinite(maps['ExG']).any())
        exg=next(d for d in ds if d['index']=='ExG')
        self.assertEqual(exg['valid_pixels'],exg['eligible_pixels'])
        self.assertEqual(exg['median'],0)

    def test_smallest_bilinear_clipped_contributor_is_not_repaired(self):
        mask=np.ones((20,20),bool)
        rgb=np.full((20,20,3),80,np.uint8)
        noir=np.full_like(rgb,120);noir[6,6,0]=255
        matrix=np.array([[1,0,1/32],[0,1,1/32]],np.float32)
        _,_,maps,supports,warped,_=run.describe(rgb,mask,noir,mask,matrix,True)
        self.assertTrue(supports['cross'][5,5])
        # OpenCV's 1/32 grid gives the diagonal clipped contributor weight1/1024.
        self.assertEqual(float(warped[5,5,0]),120+(255-120)/1024)
        self.assertTrue(np.isnan(maps['ND_rgb_R__noir_R'][5,5]))
        self.assertTrue(np.isfinite(maps['ND_rgb_R__noir_B'][5,5]))
        self.assertTrue(np.isfinite(maps['ExG'][5,5]))

    def test_empty_guard(self):
        with self.assertRaises(ValueError):
            moment_affine(np.zeros((10,10)),np.ones((10,10)),np)

    def test_changed_mask_rejected_against_extraction_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'source.png';mask=root/'mask.png'
            run.Image.fromarray(np.full((20,20,3),80,np.uint8)).save(source)
            run.Image.fromarray(np.full((20,20),255,np.uint8)).save(mask)
            row=dict(qc_pass=True,source='source.png',source_sha256=sha(source),
                     mask='mask.png',mask_sha256='changed')
            decoded={'source.png':dict(decode_ok=True,sha256=sha(source))}
            with self.assertRaisesRegex(ValueError,'mask_hash_mismatch'):
                run.load(row,root,root,decoded)


if __name__=='__main__':
    unittest.main()
