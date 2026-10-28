"""Known-answer checks for the distillation trust and measurement boundaries."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import foreground_core as core
import train_foreground as train
import cache_foreground as cache
import infer_foreground as infer
import evaluate_foreground as evaluate
import audit_foreground as audit


class ForegroundTests(unittest.TestCase):
    def test_cache_card_weights_are_relative_to_card_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'portable_model';folder.mkdir()
            weights=folder/'best.pt';weights.write_bytes(b'known checkpoint fixture')
            card_path=folder/'model_card.json'
            card=dict(weights='best.pt',weights_sha256=core.digest(weights))
            card_path.write_text(json.dumps(card))
            self.assertEqual(cache.model_card_weights(card_path,card),weights.resolve())
            self.assertEqual(cache.model_card_weights(card_path,dict(card,weights=str(weights.resolve()))),weights.resolve())
            weights.write_bytes(b'changed checkpoint')
            with self.assertRaises(ValueError):cache.model_card_weights(card_path,card)

    def test_prepared_source_cannot_change_identity_or_teacher_acceptance(self):
        row=dict(source='a',treatment='C',measurement='1',genotype='B73',replicate='1',session='s',camera='rgb',view='0',source_sha256='a',mask_sha256='b',qc_pass='True')
        prepared=dict(row,teacher_accepted='True')
        audit.check_source_bindings([row],[prepared])
        with self.assertRaises(ValueError):
            audit.check_source_bindings([row],[dict(prepared,genotype='Mo17')])
        with self.assertRaises(ValueError):
            audit.check_source_bindings([row],[dict(prepared,teacher_accepted='False')])

    def test_teacher_csv_accepts_utf8_bom_without_changing_field_names(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.csv'
            path.write_text('genotype,camera\nA,rgb\n',encoding='utf-8-sig')
            self.assertEqual(core.read_csv(path),[dict(genotype='A',camera='rgb')])

    def test_evaluation_rejects_changed_teacher_bytes_before_predicting(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'image.jpg').write_bytes(b'image'); (root/'mask.png').write_bytes(b'mask')
            row=dict(source='image.jpg',source_sha256=hashlib.sha256(b'image').hexdigest(),
                     mask='mask.png',mask_sha256=hashlib.sha256(b'mask').hexdigest())
            evaluate.verify_frame_inputs([row],root,root)
            (root/'mask.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                evaluate.verify_frame_inputs([row],root,root)

    def test_evaluation_cache_is_exact_and_rejects_corrupt_resume(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            image=np.full((12,16,3),17,np.uint8); mask=np.zeros((12,16),np.uint8); mask[3:10,8]=255
            Image.fromarray(image).save(root/'image.png'); Image.fromarray(mask).save(root/'mask.png')
            row=dict(id='abc',source='image.png',source_sha256=core.digest(root/'image.png'),
                     mask='mask.png',mask_sha256=core.digest(root/'mask.png'))
            entry=evaluate.prepare_cached_frame(row,root,root,root/'cache')
            rgb,truth,geometry=evaluate.load_cached_frame(entry)
            self.assertEqual(rgb.shape,(12,16,3)); self.assertEqual(int(rgb[0,0,0]),17)
            self.assertEqual(int(truth.sum()),7); self.assertEqual(int(geometry['endpoints'].sum()),2)
            del rgb  # release the read-only memory map before deliberately corrupting its file on Windows
            Path(entry['image']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                evaluate.prepare_cached_frame(row,root,root,root/'cache')

    def test_genotype_partition_keeps_all_candidate_visits_and_views(self):
        rows = [dict(genotype=g, treatment='C', replicate='1', measurement=str(v), camera=c)
                for g in ['A','B','C','D','E','F'] for v in range(1,6) for c in ['rgb','noir']]
        split = core.genotype_splits(rows)
        self.assertEqual([split[g] for g in 'ABCDEF'], ['train']*4+['val','test'])
        with self.assertRaises(ValueError):
            core.genotype_splits(rows[:10])

    def test_training_list_cannot_include_validation_frame_or_rejected_teacher(self):
        rows=[dict(image='a',split='train',camera='rgb',teacher_accepted='True'),
              dict(image='b',split='val',camera='rgb',teacher_accepted='True'),
              dict(image='c',split='train',camera='rgb',teacher_accepted='False')]
        train.check_membership(['a'],rows,'train','rgb')
        with self.assertRaises(ValueError):
            train.check_membership(['a','b'],rows,'train','rgb')
        with self.assertRaises(ValueError):
            train.check_membership(['a','c'],rows,'train','rgb')

    def test_cache_coverage_preserves_reference_without_any_available_image(self):
        refs=[dict(treatment='C',measurement='1',genotype=g,replicate='1') for g in ['A','B']]
        records=[dict(**refs[0],camera='rgb',teacher_accepted='False',nonempty=True)]
        coverage=cache.coverage_rows(refs,records,40)
        self.assertEqual(len(coverage),4)
        self.assertEqual(sum(r['available_student_frames'] for r in coverage),1)
        self.assertEqual(sum(r['nonempty_student_frames'] for r in coverage),1)
        self.assertEqual(sum(r['teacher_accepted_frames'] for r in coverage),0)
        self.assertEqual([r['available_student_frames'] for r in coverage if r['genotype']=='B'],[0,0])

    def test_preview_cache_cannot_open_heldout_images_before_protocol_freeze(self):
        rows=[dict(genotype='B73'),dict(genotype='Os7798/2')]
        with self.assertRaises(ValueError):
            cache.restrict_cache_rows(rows,False,False)
        self.assertEqual(cache.restrict_cache_rows(rows,False,True),[dict(genotype='B73')])
        self.assertEqual(len(cache.restrict_cache_rows(rows,True,False)),2)

    def test_test_scoring_refuses_models_changed_after_validation_selection(self):
        selected={'m':dict(sha256='abc',imgsz=640)}
        evaluate.validate_selected_models({'m':dict(sha256='abc',imgsz=640)},selected)
        with self.assertRaises(ValueError):
            evaluate.validate_selected_models({'m':dict(sha256='different',imgsz=640)},selected)
        with self.assertRaises(ValueError):
            evaluate.validate_selected_models({'m':dict(sha256='abc',imgsz=960)},selected)

    def test_empty_prediction_is_failure_not_missing_score(self):
        truth = np.zeros((8,8), bool); truth[2:6,2:6] = True
        result = core.mask_metrics(truth, np.zeros_like(truth))
        self.assertEqual(result['iou'], 0.)
        self.assertEqual(result['dice'], 0.)
        self.assertEqual(result['recall'], 0.)
        self.assertEqual(result['predicted_pixels'], 0)

    def test_inference_unions_masks_and_obeys_confidence(self):
        import torch
        masks=torch.zeros((2,8,8)); masks[0,1:3,1:3]=1; masks[1,5:7,5:7]=1
        result=SimpleNamespace(masks=SimpleNamespace(data=masks),
                               boxes=SimpleNamespace(conf=torch.tensor([.9,.2])))
        self.assertEqual(int(core.prediction_union(result,(8,8),.5).sum()),4)
        self.assertEqual(int(core.prediction_union(result,(8,8),.1).sum()),8)
        self.assertFalse(core.prediction_union(SimpleNamespace(masks=None,boxes=None),(8,8),.1).any())

    def test_inference_size_uses_card_and_explicit_override_without_hidden_resize(self):
        self.assertEqual(infer.inference_size({},None),640)
        self.assertEqual(infer.inference_size({'imgsz':960},None),960)
        self.assertEqual(infer.inference_size({'imgsz':640},960),960)
        with self.assertRaises(ValueError):
            infer.inference_size({'imgsz':777},None)

    def test_terminal_and_thin_support_detect_lost_tip_despite_remaining_stem(self):
        truth=np.zeros((24,24),bool); truth[3:18,10]=True
        prediction=np.zeros_like(truth); prediction[8:18,10]=True
        result=core.terminal_metrics(truth,prediction)
        self.assertEqual(result['teacher_interior_endpoints'],2)
        self.assertEqual(result['endpoint_recall_px0'],.5)
        self.assertEqual(result['endpoint_recall_px3'],.5)
        self.assertEqual(result['endpoint_recall_px5'],1.)
        self.assertAlmostEqual(result['thin_skeleton_recall_px0'],10/15)
        empty=core.terminal_metrics(truth,np.zeros_like(truth))
        self.assertEqual(empty['endpoint_recall_px5'],0.)

    def test_disconnected_polygon_preserves_both_foreground_components(self):
        truth = np.zeros((60,80), np.uint8)
        truth[3:20,5:15] = 1; truth[35:55,50:70] = 1
        polygon, info = core.mask_polygon(truth)
        restored = core.raster_polygon(polygon, truth.shape)
        self.assertEqual(info['external_components'], 2)
        self.assertTrue(restored[8,8]); self.assertTrue(restored[40,55])
        self.assertGreater(core.mask_metrics(truth,restored)['iou'], .90)

    def test_holes_are_explicitly_audited_in_polygon_roundtrip(self):
        truth = np.zeros((40,40), np.uint8); truth[3:35,3:35] = 1; truth[10:20,10:20] = 0
        polygon, info = core.mask_polygon(truth)
        self.assertEqual(info['holes'], 1)
        self.assertLess(core.mask_metrics(truth,core.raster_polygon(polygon,truth.shape))['iou'], 1.)

    def test_mask_metrics_reject_wrong_processing_geometry(self):
        with self.assertRaises(ValueError):
            core.mask_metrics(np.zeros((2,3)), np.zeros((3,2)))

    def test_threshold_selection_refuses_test_rows(self):
        records = [dict(split='test', confidence=.25, iou=.8)]
        with self.assertRaises(ValueError):
            core.select_threshold(records)
        records = [dict(split='val', confidence=.1, iou=.8),dict(split='val', confidence=.25, iou=.8)]
        self.assertEqual(core.select_threshold(records), .25)

    def test_frozen_topology_model_specs_reject_checkpoint_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'infer_topology.py').write_text('# inference')
            (root/'topology_core.py').write_text('# core')
            config=dict(source_sha256={name:core.digest(root/name) for name in ('infer_topology.py','topology_core.py')})
            (root/'configuration.json').write_text(json.dumps(config))
            (root/'frozen_model_selection.json').write_text('{}')
            models=[]
            for arm in ('baseline','cldice'):
                path=root/(arm+'.pt'); path.write_bytes(arm.encode())
                models.append(dict(arm=arm,checkpoint=str(path),checkpoint_sha256=core.digest(path),threshold=.35))
            gate=dict(status='frozen',models=models,inference_module=str(root/'infer_topology.py'),
                inference_module_sha256=core.digest(root/'infer_topology.py'),
                frozen_selection=str(root/'frozen_model_selection.json'),frozen_selection_sha256=core.digest(root/'frozen_model_selection.json'),
                configuration_sha256=core.digest(root/'configuration.json'))
            path=root/'gate.json'; path.write_text(json.dumps(gate))
            specs=evaluate.topology_specs(path)
            self.assertEqual(set(specs),{'unet_baseline','unet_cldice'})
            self.assertEqual(specs['unet_cldice']['threshold'],.35)
            (root/'cldice.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                evaluate.topology_specs(path)


if __name__ == '__main__':
    unittest.main()
