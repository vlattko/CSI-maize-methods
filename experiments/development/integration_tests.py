"""Public-copy contracts; run separately from branch-local test modules."""
import os
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT = ROOT / "experiments" / "development"


class PublicCopyTests(unittest.TestCase):
    def run_python(self, code, *args):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code, str(ROOT), *args],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_tracking_import_uses_public_full_cohort_and_current_package(self):
        self.run_python("""
import pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / 'src'))
sys.path.insert(0, str(root / 'experiments/development/longitudinal_tracking_development_2026-10-27'))
import csi_maize, tracking_development as tracking, architecture_analysis
assert pathlib.Path(csi_maize.__file__).resolve().is_relative_to(root / 'src')
assert tracking.OLD.resolve() == (root / 'experiments/full_cohort').resolve()
assert pathlib.Path(architecture_analysis.__file__).resolve().parent == tracking.OLD.resolve()
""")

    def test_source_only_and_no_internal_plans(self):
        allowed = {".py", ".md", ".js", ".mjs", ".html", ".json"}
        for file in DEVELOPMENT.rglob("*"):
            if not file.is_file() or "__pycache__" in file.parts:
                continue
            self.assertNotIn(file.name, {"PLAN.md", "independent_audit.json"})
            self.assertTrue(file.name == "LICENSE" or file.suffix in allowed, file)
            self.assertLess(file.stat().st_size, 150_000, file)
            self.assertNotIn(b"\r", file.read_bytes(), file)
            self.assertNotIn("execution_sources", file.parts)
            self.assertNotIn("assets", file.parts)
            if file.suffix != ".json":
                text = file.read_text(encoding="utf-8")
                self.assertNotIn("D:" + "/CSI-maize-analysis", text, file)
                self.assertNotIn("C:" + "/Users/", text, file)
            if file.suffix == ".py":
                compile(file.read_text(encoding="utf-8"), str(file), "exec")

    def test_templates_have_placeholders_not_private_payloads(self):
        rotation = (DEVELOPMENT / "rotation_workbench_2026-10-27/workbench.html").read_text(encoding="utf-8")
        review = (DEVELOPMENT / "leaf_review_workbench_2026-10-27/workbench_shell.html").read_text(encoding="utf-8")
        self.assertIn("__WORKBENCH_PAYLOAD__", rotation)
        self.assertIn("<!-- DATA_AND_SCRIPTS -->", review)
        self.assertNotIn("data:image/jpeg;base64,", rotation)
        self.assertNotIn("data:image/jpeg;base64,", review)

    def test_public_source_hashes_match_provenance_and_exclude_internal_plan(self):
        from run_tests import BRANCHES
        manifest = json.loads((DEVELOPMENT / "source_provenance.json").read_text(encoding="utf-8"))
        seen = set()
        for entry in manifest["files"]:
            relative = entry["public_relative"]
            self.assertNotIn(relative, seen)
            seen.add(relative)
            path = (DEVELOPMENT / relative).resolve()
            self.assertTrue(path.is_relative_to(DEVELOPMENT.resolve()))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["public_sha256"], relative)
            self.assertNotEqual(path.name, "PLAN.md")
            if "original_sha256" in entry:
                self.assertEqual(entry["unchanged"], entry["original_sha256"] == entry["public_sha256"])
            if entry.get("unchanged") is False:
                self.assertTrue(entry["public_change"])
        actual = {path.relative_to(DEVELOPMENT).as_posix() for branch in BRANCHES
                  for path in (DEVELOPMENT / branch).rglob("*")
                  if path.is_file() and "__pycache__" not in path.parts}
        self.assertEqual(seen, actual)

    def test_missing_optional_is_declared_and_broken_loader_is_not_skipped(self):
        from run_tests import missing_optional
        branch = "topology_foreground_development_2026-10-27"
        self.assertEqual(missing_optional(branch, lambda name: None), ["torch"])
        self.assertEqual(missing_optional(branch, lambda name: object()), [])
        self.assertEqual(missing_optional("camera_geometry_development_2026-10-27", lambda name: None), [])
        def broken(name):
            raise RuntimeError("installed loader is broken")
        with self.assertRaisesRegex(RuntimeError, "installed loader"):
            missing_optional(branch, broken)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PublicCopyTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
