import tempfile,unittest
from pathlib import Path


class DigitizationTests(unittest.TestCase):
    def test_stage_plan_passes_explicit_identity_and_keeps_cpu_fusion_separate(self):
        import run_digitization as d
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            commands=d.stage_commands('python',root/'raw',root/'session.json',root/'new',root/'repo',root/'card.json','yolo','cpu',None,.05)
            self.assertEqual(len(commands),2)
            self.assertIn('--session',commands[0]);self.assertIn('--confidence',commands[0]);self.assertNotIn('--pixel-threshold',commands[0])
            self.assertNotIn('--device',commands[1]);self.assertIn(str(root/'new/capture'),commands[1])
            self.assertFalse((root/'new').exists())


if __name__=='__main__':unittest.main()
