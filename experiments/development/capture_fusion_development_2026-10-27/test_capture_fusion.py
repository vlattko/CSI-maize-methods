import tempfile,unittest,json,hashlib,sys
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image


class CaptureFusionTests(unittest.TestCase):
    def test_tiny_clipped_interpolation_contributor_invalidates_formula(self):
        import capture_fusion as f
        image=np.full((20,20,3),160,dtype=np.uint8);noir=image.copy();noir[6,6,0]=255;mask=np.ones((20,20),bool)
        matrix=np.array([[1,0,1/32],[0,1,1/32.]])
        maps,_,_=f.describe_pair(image,mask,noir,mask,matrix,True)
        self.assertTrue(np.isnan(maps['ND_rgb_G__noir_R'][5,5]))

    def test_frame_binding_rejects_camera_view_aliases(self):
        import capture_fusion as f
        row=dict(camera='noir',view=0,sample_id='same')
        with self.assertRaisesRegex(ValueError,'identity'):f.validate_frame_binding('rgb:0',row,'same',40)

    def test_manifest_must_bind_capture_identity(self):
        import capture_fusion as f
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'capture.json').write_text('{}')
            (root/'output_manifest.json').write_text(json.dumps(dict(status='completed',files=[])))
            with self.assertRaisesRegex(ValueError,'identity'):f.verify_capture(root)

    def test_complete_capture_cli_retains_all_index_rows(self):
        import capture_fusion as f
        import csi_maize
        repo=Path(csi_maize.__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';capture=root/'capture';raw.mkdir();capture.mkdir();(capture/'records').mkdir();(capture/'masks').mkdir()
            index={};mask=np.zeros((48,64),np.uint8);mask[8:40,8:50]=255
            for camera,color in [('rgb',(100,150,50)),('noir',(120,160,60))]:
                (raw/camera).mkdir()
                for view in range(40):
                    source=f'{camera}/{view:03d}.png';Image.new('RGB',(64,48),color).save(raw/source)
                    relative=f'masks/{camera}_{view:03d}.png';Image.fromarray(mask).save(capture/relative)
                    r=dict(status='inferred',sample_id='fixture',camera=camera,view=view,source=source,source_sha256=f.sha(raw/source),mask=relative,mask_sha256=f.sha(capture/relative),processing_width=64,processing_height=48,structural_usability=dict(usable=True))
                    name=f'records/{camera}_{view:03d}.json';(capture/name).write_text(json.dumps(r));index[f'{camera}:{view}']=name
            (capture/'capture.json').write_text(json.dumps(dict(session=dict(sample_id='fixture',steps=40,angles_degrees=[9*v for v in range(40)]),record_index=index)))
            (capture/'output_manifest.json').write_text(json.dumps(dict(status='completed',files=[dict(path=p.relative_to(capture).as_posix(),sha256=f.sha(p)) for p in capture.rglob('*') if p.is_file()])))
            run=subprocess.run([sys.executable,str(Path(f.__file__)), '--capture',str(capture),'--raw',str(raw),'--repo',str(repo),'--output',str(root/'output')],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr[-3000:])
            result=json.loads((root/'output/summary.json').read_text());self.assertEqual(result['descriptor_rows'],1080);self.assertEqual(result['planned_pairs'],40)

    def test_rgb_to_noir_matrix_samples_the_expected_noir_pixel(self):
        import capture_fusion as f
        rgb=np.full((48,64,3),80,dtype=np.uint8);noir=np.zeros_like(rgb)
        rgb[8:40,8:50]=[100,150,50];noir[11:43,13:55]=[120,160,60]
        rm=np.zeros((48,64),bool);rm[8:40,8:50]=True;nm=np.zeros_like(rm);nm[11:43,13:55]=True
        maps,stats,preview=f.describe_pair(rgb,rm,noir,nm,np.array([[1,0,5],[0,1,3.]]),True)
        nd=maps['ND_rgb_G__noir_R'];self.assertAlmostEqual(float(nd[20,20]),(150-120)/(150+120),places=6)
        self.assertEqual(len(stats),27);self.assertEqual(preview.shape,rgb.shape)

    def test_camera_abstention_withholds_cross_camera_formulas(self):
        import capture_fusion as f
        image=np.full((48,64,3),80,dtype=np.uint8);mask=np.zeros((48,64),bool);mask[8:40,8:50]=True
        maps,stats,_=f.describe_pair(image,mask,image,mask,np.eye(2,3),False)
        self.assertTrue(np.isnan(maps['ND_rgb_G__noir_R']).all())
        self.assertTrue(np.isfinite(maps['ND_rgb_R__rgb_G'][mask]).any())

    def test_manifest_rejects_mutated_bound_files(self):
        import capture_fusion as f
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'capture.json').write_text('{}')
            digest=hashlib.sha256((root/'capture.json').read_bytes()).hexdigest()
            (root/'output_manifest.json').write_text(json.dumps(dict(status='completed',files=[dict(path='capture.json',sha256=digest)])))
            (root/'capture.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'digest'):f.verify_capture(root)

    def test_camera_selection_does_not_receive_heldout_masks(self):
        import capture_fusion as f
        masks={v:np.ones((16,20),bool) for v in range(40)}
        train=f.training_masks(masks)
        self.assertEqual(set(train),set(range(40))-set(range(2,40,4)))
        changed=dict(masks);changed[2]=np.zeros((16,20),bool)
        for k,v in f.training_masks(changed).items():np.testing.assert_array_equal(v,train[k])


if __name__=='__main__':unittest.main()
