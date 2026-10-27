"""Filesystem, DAG and cache-binding tests independent of private plant data."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
METHODS = REPO / "src" / "csi_maize" / "_methods"


class PortabilityTests(unittest.TestCase):
    def run_method_code(self, workspace, code):
        env = dict(os.environ, CSI_MAIZE_WORKSPACE=str(workspace), PYTHONUTF8="1")
        env["PYTHONPATH"] = str(METHODS)
        return subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True)

    def test_context_is_explicit_and_import_has_no_output_side_effect(self):
        with tempfile.TemporaryDirectory(prefix="maize space ") as directory:
            root = Path(directory)
            result = self.run_method_code(
                root, "import bootstrap; print(bootstrap.BASE); print(bootstrap.OUT)"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(root / "prototype" / "research_v2"), result.stdout)
            self.assertFalse((root / "prototype").exists())

    def test_lexical_traversal_and_absolute_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            code = """
from common import safe_source
for path in ['../escape', 'folder/../../escape', '/absolute', 'C:/absolute', 'C:relative', r'..\\escape']:
    try: safe_source(path)
    except ValueError: pass
    else: raise AssertionError(path)
"""
            result = self.run_method_code(Path(directory), code)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_registered_link_is_allowed_and_unregistered_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as raw:
            root, source = Path(directory), Path(raw)
            (source / "view.jpg").write_bytes(b"raw")
            try:
                (root / "images").symlink_to(source, target_is_directory=True)
                (root / "other").symlink_to(source, target_is_directory=True)
            except OSError:
                self.skipTest("Creating directory symlinks requires OS support/permission")
            (root / "input_bindings.json").write_text(
                json.dumps({"bindings": [{"logical_path": "images", "source_path": str(source)}]}),
                encoding="utf-8",
            )
            result = self.run_method_code(
                root,
                "from common import safe_source; assert safe_source('images/view.jpg').read_bytes()==b'raw'\ntry: safe_source('other/view.jpg')\nexcept ValueError: pass\nelse: raise AssertionError('unregistered link accepted')",
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_incomplete_and_duplicate_rotations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "prototype" / "research_v2"
            folder.mkdir(parents=True)
            (folder / "config.json").write_text('{"case":"new"}', encoding="utf-8")
            code = """
from bootstrap import validate_rotation_records
rows=[dict(session='one',camera=c,view=i) for c in ['rgb','noir'] for i in range(40)]
assert validate_rotation_records(rows)==1
for invalid in [rows[:-1], rows+[rows[0]]]:
    try:validate_rotation_records(invalid)
    except ValueError:pass
    else:raise AssertionError('invalid rotation accepted')
"""
            result = self.run_method_code(root, code)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_catalog_has_acyclic_dependencies_and_existing_commands(self):
        spec = importlib.util.spec_from_file_location(
            "stage_catalog", REPO / "src/csi_maize/stages.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        visited, active = set(), set()

        def visit(name):
            self.assertIn(name, module.STAGES)
            self.assertNotIn(name, active, "dependency cycle")
            if name in visited:
                return
            active.add(name)
            stage = module.STAGES[name]
            self.assertIn(stage["profile"], ["core", "learning"])
            for dependency in stage["dependencies"]:
                visit(dependency)
            for command in stage["commands"]:
                if not command[0].startswith("@"):
                    self.assertTrue((METHODS / command[0]).is_file(), command)
            self.assertTrue(
                all(
                    not Path(path).is_absolute() and ".." not in Path(path).parts
                    for path in stage["outputs"]
                )
            )
            active.remove(name)
            visited.add(name)

        for name in module.STAGES:
            visit(name)

    def test_index_requires_successful_decoding_status_not_only_nonempty_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            code="""
import csv,json
from bootstrap import BASE,ROOT
BASE.mkdir(parents=True)
(BASE/'config.json').write_text(json.dumps(dict(source_folders=['images'],reference_table='reference.csv',identity_mapping='prototype/mapping.csv',allowed_views_per_camera=[40])))
session=ROOT/'images/capture_session_C1_1-1'
for camera in ['rgb','noir']:
    (session/camera).mkdir(parents=True)
    for view in range(40):(session/camera/f'image_{view:03d}.jpg').write_bytes(b'nonempty')
def reference(status):
    with (ROOT/'reference.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['treatment','measurement','genotype','replicate','image_status'])
        writer.writerow(['C','1','Os024445','1',status])
    (ROOT/'research_audit').mkdir(exist_ok=True)
    with (ROOT/'research_audit/image_manifest.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['path','bytes','full_decode_ok','container_check_error'])
        for path in sorted(session.rglob('*.jpg')):
            writer.writerow([path.relative_to(ROOT).as_posix(),path.stat().st_size,
                             status=='complete_40_plus_40_container_checks',
                             '' if status=='complete_40_plus_40_container_checks' else 'OSError'])
from common import build_index
reference('incomplete_extracted_session')
assert build_index()[0]['eligible'] is False,'Nonempty but undecodable images became model-eligible'
reference('complete_40_plus_40_container_checks')
assert build_index()[0]['eligible'] is True
reference('')
assert build_index()[0]['eligible'] is False,'Missing decode evidence became model-eligible'
"""
            result=self.run_method_code(root,code)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_cache_fingerprint_tracks_masks_phase_helpers_and_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            code = """
import ast, hashlib, json, shutil
from pathlib import Path
from bootstrap import BASE, ROOT, OUT, method_source_hashes
from common import safe_source
BASE.mkdir(parents=True)
OUT.mkdir()
original=Path(METHODS)
for name in ['run_channel_fusion.py','channel_fusion_core.py','multimodal_leaf_core.py','bootstrap.py','common.py']:
    shutil.copy2(original/name,BASE/name)
(BASE/'config.json').write_text('{}',encoding='utf-8')
OLD=OUT/'weak';(OLD/'predictions').mkdir(parents=True)
(OUT/'sam_traits').mkdir()
for path in [OUT/'sam_traits/view_traits.csv',OLD/'paired_frames.csv',OLD/'phase_audit.csv']:
    path.write_text('declared-input',encoding='utf-8')
(ROOT/'image.raw').write_bytes(b'image')
(ROOT/'mask.raw').write_bytes(b'mask')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
POLICY={};CHANNELS=['rgb_R'];OFFSETS=[(0,0)]
tree=ast.parse((BASE/'run_channel_fusion.py').read_text(encoding='utf-8'))
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['input_fingerprint','cache_payload_hash','read_session_cache']]
exec(compile(ast.Module(body=functions,type_ignores=[]),'cache_functions','exec'))
rows=[dict(source='image.raw',source_sha256=sha(ROOT/'image.raw'),mask='mask.raw')]
_,a=input_fingerprint(rows)
(ROOT/'mask.raw').write_bytes(b'new mask');_,b=input_fingerprint(rows);assert a!=b
(OLD/'phase_audit.csv').write_text('new phase',encoding='utf-8');_,c=input_fingerprint(rows);assert b!=c
with (BASE/'multimodal_leaf_core.py').open('a',encoding='utf-8') as f:f.write('\\n# changed helper\\n')
_,d=input_fingerprint(rows);assert c!=d
payload=dict(signature=d,frames=[dict(session='one',view=i) for i in range(40)])
payload['payload_sha256']=cache_payload_hash(payload)
cache=OLD/'session.json';cache.write_text(json.dumps(payload),encoding='utf-8')
assert read_session_cache(cache,d,'one')['signature']==d
payload['frames'][0]['view']=99;cache.write_text(json.dumps(payload),encoding='utf-8')
try:read_session_cache(cache,d,'one')
except ValueError:pass
else:raise AssertionError('mutated cached payload accepted')
(ROOT/'image.raw').write_bytes(b'changed image')
try:input_fingerprint(rows)
except ValueError:pass
else:raise AssertionError('changed recorded source accepted')
"""
            result = self.run_method_code(
                Path(directory), "METHODS=" + repr(str(METHODS)) + "\n" + code
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
