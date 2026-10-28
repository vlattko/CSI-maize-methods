"""Focused checks for portable reviewer paths; no project code is imported."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def load_core():
    spec = importlib.util.spec_from_file_location('portable_core_review', HERE/'verify_new_run_sklearn.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PortablePaths(unittest.TestCase):
    def test_rebases_windows_path_by_unique_basename_and_preserves_key(self):
        module = load_core()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/'nested'/'confirmed_mass.csv'
            target.parent.mkdir()
            target.write_text('mass\n1\n')
            original = r'Z:\old\confirmed_mass.csv'
            found = module.resolve_inputs({original:'digest'}, Path(temp))
            self.assertEqual(found, {original:target})

    def test_default_keeps_existing_original_path_and_key(self):
        module = load_core()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/'data.csv'
            target.write_text('a\n1\n')
            self.assertEqual(module.resolve_inputs({str(target):'digest'}), {str(target):target})

    def test_missing_basename_raises(self):
        module = load_core()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                module.resolve_inputs({r'Z:\old\missing.csv':'digest'}, Path(temp))

    def test_ambiguous_basename_raises_even_for_identical_bytes(self):
        module = load_core()
        with tempfile.TemporaryDirectory() as temp:
            for folder in ['one','two']:
                target=Path(temp)/folder/'data.csv'
                target.parent.mkdir()
                target.write_text('same')
            with self.assertRaises(ValueError):
                module.resolve_inputs({r'Z:\old\data.csv':'digest'}, Path(temp))

    def test_both_cli_help_interfaces(self):
        for filename,extra in [('verify_new_run_sklearn.py',[]),('verify_extensions_sklearn.py',['--final','--tables'])]:
            completed=subprocess.run([sys.executable,'-B',str(HERE/filename),'--help'],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            for flag in ['--run','--source','--output','--input-root']+extra:
                self.assertIn(flag,completed.stdout)

if __name__=='__main__':
    unittest.main()
