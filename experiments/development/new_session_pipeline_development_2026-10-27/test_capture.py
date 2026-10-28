import json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from capture_core import validate_session,inventory,safe_output,provider_parameters,structural_usability,tracking_support


class CaptureTests(unittest.TestCase):
    def session(self,steps=40):return dict(sample_id='fresh-01',steps=steps)
    def test_missing_slots_do_not_shift_indices(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'rgb').mkdir();(p/'noir').mkdir();Image.new('RGB',(64,48)).save(p/'rgb/image_rgb_039.png')
            rows=inventory(p,validate_session(self.session()))
            self.assertEqual(len(rows),80);self.assertFalse(rows[0]['present'])
            self.assertEqual([r for r in rows if r['camera']=='rgb' and r['view']==39][0]['source'],'rgb/image_rgb_039.png')

    def test_duplicate_indices_are_errors(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'rgb').mkdir();(p/'noir').mkdir()
            for name in ['a_003.png','b_003.png']:Image.new('RGB',(64,48)).save(p/'rgb'/name)
            with self.assertRaises(ValueError):inventory(p,validate_session(self.session()))

    def test_safe_paths_prevent_raw_writes_and_escape(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);raw=p/'raw';raw.mkdir()
            with self.assertRaises(ValueError):safe_output(raw,raw/'results')
            with self.assertRaises(ValueError):validate_session(dict(sample_id='x',steps=40,cameras={'rgb':'../other','noir':'noir'}))

    def test_camera_directories_must_be_separate(self):
        with self.assertRaises(ValueError):validate_session(dict(sample_id='x',steps=40,cameras={'rgb':'one','noir':'one'}))
        with self.assertRaises(ValueError):validate_session(dict(sample_id='x',steps=40,cameras={'rgb':'one','noir':'one/child'}))

    def test_resolved_camera_alias_is_rejected(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);original=Path.resolve
            def alias(path,*args,**kwargs):
                if path==root/'rgb' or path==root/'noir':return root/'shared'
                return original(path,*args,**kwargs)
            with patch.object(Path,'resolve',alias):
                with self.assertRaises(ValueError):inventory(root,validate_session(self.session()))

    def test_provider_threshold_semantics_are_distinct(self):
        u=provider_parameters('unet',dict(threshold=.35,checkpoint='x',checkpoint_sha256='0'*64))
        self.assertEqual(u['threshold_kind'],'foreground_probability')
        y=provider_parameters('yolo',dict(confidence=.05,weights='x',weights_sha256='0'*64))
        self.assertEqual(y['threshold_kind'],'detection_confidence')
        with self.assertRaises(ValueError):provider_parameters('yolo',dict(confidence=.05),pixel_threshold=.35)
        with self.assertRaises(ValueError):provider_parameters('unet',dict(threshold=.35),detection_confidence=.05)

    def test_provider_card_binds_checkpoint_once(self):
        from capture_providers import UNetProvider
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'weights.pt').write_bytes(b'test checkpoint');(p/'topology_core.py').write_text('# fixture')
            module=p/'known_capture_provider_fixture.py';module.write_text('def predict_array(*args,**kwargs): return None\n')
            card=p/'card.json';card.write_text(json.dumps(dict(checkpoint='weights.pt',checkpoint_sha256=hashlib.sha256(b'test checkpoint').hexdigest(),threshold=.35)))
            provider=UNetProvider(card,module_path=module)
            self.assertEqual(provider.provenance['checkpoint'],str((p/'weights.pt').resolve()))

    def test_thirty_view_tracking_abstains(self):
        self.assertFalse(tracking_support(validate_session(self.session(30)))['supported'])
        self.assertTrue(tracking_support(validate_session(self.session()))['supported'])

    def test_nonuniform_angles_are_preserved_without_forcing_tracker(self):
        a=[float(i*9) for i in range(40)];a[3]+=.5;s=validate_session(dict(sample_id='x',steps=40,angles_degrees=a))
        self.assertEqual(s['angles_degrees'][3],27.5);self.assertFalse(tracking_support(s)['supported'])

    def test_no_physical_identity_is_inferred(self):
        s=validate_session(self.session());self.assertIsNone(s['plant_uid']);self.assertEqual(s['physical_identity_status'],'not_provided')

    def test_structural_check_uses_no_provider_score(self):
        mask=np.zeros((100,100),bool);mask[10:95,45:50]=True
        self.assertTrue(structural_usability(mask)['usable']);self.assertFalse(structural_usability(np.zeros_like(mask))['usable'])

    def test_fresh_single_session_runs_with_known_mask_fixture(self):
        from run_capture import run_pipeline
        class Fixture:
            provenance={'provider':'synthetic_known_mask_fixture','threshold_kind':'none','quality_score_interpretation':'Known-answer test only'}
            def predict(self,image,frame):return image[:,:,1]>100
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);raw=p/'raw';(raw/'rgb').mkdir(parents=True);(raw/'noir').mkdir()
            image=np.zeros((96,128,3),np.uint8);image[15:90,60:65,1]=180
            for camera in ['rgb','noir']:
                for view in [0,3]:Image.fromarray(image).save(raw/camera/f'frame_{view:03d}.png')
            result=run_pipeline(raw,self.session(4),p/'output',Fixture(),repo=Path(__file__).resolve().parents[3])
            self.assertEqual(result['planned_frames'],8);self.assertEqual(result['decoded_frames'],4)
            self.assertEqual(result['missing_frames'],4);self.assertEqual(result['trait_count'],39)
            self.assertTrue((p/'output/report.html').exists())
            self.assertEqual(result['assays'],{'status':'not_provided','predicted':False})
            self.assertTrue(result['duplicate_source_hash_warnings'])

    def test_report_preserves_but_does_not_execute_user_markup(self):
        from report_capture import make_report
        import base64,gzip,re
        bad='</script><script>window.USER_CODE=true</script>'
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);make_report(p,dict(session=dict(sample_id=bad,steps=30,plant_uid=None),assays=dict(user_text=bad)),[],{}, {})
            text=(p/'report.html').read_text(encoding='utf-8');self.assertNotIn(bad,text)
            encoded=re.search(r'<script id="payload" type="application/octet-stream">([^<]+)</script>',text).group(1)
            value=json.loads(gzip.decompress(base64.b64decode(encoded)));self.assertEqual(value['capture']['session']['sample_id'],bad)


if __name__=='__main__':unittest.main()
