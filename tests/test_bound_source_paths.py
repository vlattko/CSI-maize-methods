"""Logical provenance survives registered external raw symlinks and junctions."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


METHODS = Path(__file__).resolve().parents[1] / "src/csi_maize/_methods"


class BoundSourcePathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="maize bound paths ")
        self.root = Path(self.temporary.name) / "workspace"
        self.raw = Path(self.temporary.name) / "external raw"
        self.raw.mkdir()
        self.base = self.root / "prototype/research_v2"
        self.base.parent.mkdir(parents=True)
        self.base.mkdir()
        for source in METHODS.iterdir():
            if source.is_file() and source.suffix in {".py", ".json", ".html", ".js", ".css"}:
                shutil.copyfile(source, self.base / source.name)
        (self.base / "outputs").mkdir()
        (self.base / "config.json").write_text("{}", encoding="utf-8")
        (self.raw / "view.jpg").write_bytes(b"raw source fixture")
        self.bindings = []

    def tearDown(self):
        self.temporary.cleanup()

    def bind(self, logical="images", *, junction=False):
        target = self.root / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        if junction:
            result = subprocess.run(
                ["cmd.exe", "/c", "mklink", "/J", str(target), str(self.raw)],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            try:
                target.symlink_to(self.raw, target_is_directory=True)
            except OSError:
                if os.name != "nt":
                    raise
                return self.bind(logical, junction=True)
        self.bindings.append(dict(logical_path=logical, source_path=str(self.raw)))
        (self.root / "input_bindings.json").write_text(
            json.dumps(dict(bindings=self.bindings)), encoding="utf-8"
        )
        return target

    def execute(self, code):
        env = dict(
            os.environ,
            CSI_MAIZE_WORKSPACE=str(self.root),
            PYTHONPATH=str(self.base),
            PYTHONUTF8="1",
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=self.base, env=env, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)

    def check_helpers(self, logical):
        self.execute(
            """
from pathlib import Path
from bootstrap import ROOT
from common import safe_source, workspace_relative
from leaf_dynamics_geometry import rel
from additive_geometry import relative
from v5_pair_controls import relative as pair_relative
logical = """
            + repr(logical + "/view.jpg")
            + """
source=ROOT/logical
assert not source.resolve().is_relative_to(ROOT)
assert safe_source(logical)==source
for helper in [workspace_relative,rel,relative,pair_relative]:
    assert helper(source)==logical,(helper,helper(source))
assert workspace_relative(logical)==logical
assert workspace_relative(ROOT/'prototype/research_v2/outputs/future/result.json')=='prototype/research_v2/outputs/future/result.json'
for path in [source.resolve(),ROOT/'images/../escape.jpg','../escape.jpg']:
    try:workspace_relative(path)
    except ValueError:pass
    else:raise AssertionError('unsafe or ambiguous physical path accepted: '+str(path))
"""
        )

    def test_external_symlink_keeps_logical_identity_for_all_provenance_helpers(self):
        self.bind("CSI-maize_podaci/IMAGES")
        self.check_helpers("CSI-maize_podaci/IMAGES")

    @unittest.skipUnless(os.name == "nt", "Windows junction regression")
    def test_external_windows_junction_keeps_logical_identity(self):
        self.bind("CSI-maize_podaci/IMAGES", junction=True)
        self.check_helpers("CSI-maize_podaci/IMAGES")

    def test_registration_fingerprint_uses_safe_logical_paths_and_actual_source_bytes(self):
        self.bind()
        self.execute("""
import hashlib
from bootstrap import BASE,ROOT,OUT
from run_channel_fusion import OLD,input_fingerprint
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
(OUT/'sam_traits').mkdir()
(OLD/'predictions').mkdir(parents=True)
for path in [OUT/'sam_traits/view_traits.csv',OLD/'paired_frames.csv',OLD/'phase_audit.csv']:
    path.write_text('fixture',encoding='utf-8')
mask=OUT/'sam_traits/mask.png';mask.write_bytes(b'generated mask')
rows=[dict(source='images/view.jpg',source_sha256=sha(ROOT/'images/view.jpg'),mask='prototype/research_v2/outputs/sam_traits/mask.png')]
mapping,signature=input_fingerprint(rows)
assert mapping['images/view.jpg']==sha(ROOT/'images/view.jpg')
assert all(not __import__('pathlib').Path(key).is_absolute() for key in mapping)
assert 'input_bindings.json' in mapping
mask.write_bytes(b'mask changed')
assert input_fingerprint(rows)[1]!=signature
(ROOT/'images/view.jpg').write_bytes(b'raw changed')
try:input_fingerprint(rows)
except ValueError:pass
else:raise AssertionError('changed raw source accepted')
""")

    def test_unregistered_and_nested_escaping_links_are_rejected(self):
        self.bind()
        outside = Path(self.temporary.name) / "elsewhere"
        outside.mkdir()
        (outside / "view.jpg").write_bytes(b"unregistered")
        try:
            (self.root / "unregistered").symlink_to(self.raw, target_is_directory=True)
            (self.raw / "escape").symlink_to(outside, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                raise
            for target, source in [
                (self.root / "unregistered", self.raw),
                (self.raw / "escape", outside),
            ]:
                if not target.exists():
                    result = subprocess.run(
                        ["cmd.exe", "/c", "mklink", "/J", str(target), str(source)],
                        capture_output=True,
                    )
                    self.assertEqual(result.returncode, 0)
        self.execute("""
from bootstrap import ROOT
from common import safe_source,workspace_relative
for path in ['unregistered/view.jpg','images/escape/view.jpg']:
    for helper,arg in [(safe_source,path),(workspace_relative,ROOT/path)]:
        try:helper(arg)
        except ValueError:pass
        else:raise AssertionError('unregistered escape accepted: '+path)
""")


if __name__ == "__main__":
    unittest.main()
