"""Run shipped numerical tests and expose their actual counts to CI."""

from pathlib import Path
import json
import os
import subprocess
import sys


CANONICAL_RUNNER = r"""
from importlib import metadata
from pathlib import Path
import json
import platform
import sys
import time
import unittest

class SummaryResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successful_tests = 0
        self.subtests_passed = 0
        self.subtests_failed = 0

    def addSuccess(self, test):
        self.successful_tests += 1
        super().addSuccess(test)

    def addSubTest(self, test, subtest, error):
        if error is None:
            self.subtests_passed += 1
        else:
            self.subtests_failed += 1
        super().addSubTest(test, subtest, error)


def flattened(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flattened(item)
        else:
            yield item

loader = unittest.TestLoader()
suite = loader.discover('.', pattern='test_*.py')
suite.addTests(loader.loadTestsFromName('v5_pair_controls_test'))
discovered = suite.countTestCases()
modules = sorted({test.id().split('.')[0] for test in flattened(suite)})
started = time.perf_counter()
result = unittest.TextTestRunner(verbosity=1, resultclass=SummaryResult).run(suite)
try:
    torch_version = metadata.version('torch')
except metadata.PackageNotFoundError:
    torch_version = None
cuda_available = sys.modules['torch'].cuda.is_available() if 'torch' in sys.modules else None
summary = {
    'schema_version': 1,
    'status': 'passed' if result.wasSuccessful() else 'failed',
    'discovery': ['test_*.py', 'v5_pair_controls_test'],
    'discovered_tests': discovered,
    'tests_run': result.testsRun,
    'successful_tests': result.successful_tests,
    'failures_count': len(result.failures),
    'failures': [test.id() for test, _ in result.failures],
    'errors_count': len(result.errors),
    'errors': [test.id() for test, _ in result.errors],
    'skipped_count': len(result.skipped),
    'skipped': [{'test': test.id(), 'reason': reason} for test, reason in result.skipped],
    'expected_failures_count': len(result.expectedFailures),
    'unexpected_successes_count': len(result.unexpectedSuccesses),
    'subtests': {'passed': result.subtests_passed, 'failed': result.subtests_failed},
    'test_modules': modules,
    'elapsed_seconds': round(time.perf_counter() - started, 6),
    'runtime': {'python': platform.python_version(), 'platform': platform.system(),
                'torch': torch_version, 'cuda_available': cuda_available},
}
output = Path(sys.argv[1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
sys.exit(0 if result.wasSuccessful() else 1)
"""


def run_canonical_suite(methods: Path, summary_path: Path):
    methods = Path(methods).absolute()
    summary_path = Path(summary_path).absolute()
    environment = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONPATH=str(methods),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        YOLO_OFFLINE="true",
        YOLO_AUTOINSTALL="false",
    )
    environment.pop("CSI_MAIZE_WORKSPACE", None)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", CANONICAL_RUNNER, str(summary_path)],
        cwd=methods,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert summary_path.is_file(), result.stdout + result.stderr
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return result, summary


def test_canonical_numerical_suite(tmp_path):
    methods = Path(__file__).resolve().parents[1] / "src/csi_maize/_methods"
    destination = Path(
        os.environ.get("CSI_MAIZE_CANONICAL_SUMMARY", str(tmp_path / "canonical_summary.json"))
    )
    result, summary = run_canonical_suite(methods, destination)
    assert result.returncode == 0, result.stdout + result.stderr
    assert summary["status"] == "passed"
    assert summary["tests_run"] == summary["discovered_tests"] > 0
    assert summary["failures_count"] == summary["errors_count"] == 0


def test_canonical_summary_reports_real_counts_and_skips(tmp_path):
    methods = tmp_path / "methods"
    methods.mkdir()
    (methods / "test_fixture.py").write_text(
        "import unittest\nclass Cases(unittest.TestCase):\n"
        " def test_pass(self): self.assertEqual(2+3,5)\n"
        " def test_skip(self): self.skipTest('fixture unavailable')\n"
        " def test_subtests(self):\n"
        "  for value in (1,2):\n"
        "   with self.subTest(value=value): self.assertGreater(value,0)\n"
    )
    (methods / "v5_pair_controls_test.py").write_text(
        "import unittest\nclass Additional(unittest.TestCase):\n"
        " def test_nonstandard_name(self): self.assertTrue(True)\n"
    )
    result, summary = run_canonical_suite(methods, tmp_path / "summary.json")
    assert result.returncode == 0
    assert summary["status"] == "passed"
    assert summary["discovered_tests"] == summary["tests_run"] == 4
    assert summary["successful_tests"] == 3
    assert summary["skipped"] == [
        {"test": "test_fixture.Cases.test_skip", "reason": "fixture unavailable"}
    ]
    assert summary["subtests"] == {"passed": 2, "failed": 0}
    assert summary["test_modules"] == ["test_fixture", "v5_pair_controls_test"]


def test_canonical_summary_records_failures_even_when_process_is_unsuccessful(tmp_path):
    methods = tmp_path / "methods"
    methods.mkdir()
    (methods / "test_fixture.py").write_text(
        "import unittest\nclass Cases(unittest.TestCase):\n"
        " def test_failed(self): self.assertEqual(2+3,99)\n"
    )
    (methods / "v5_pair_controls_test.py").write_text(
        "import unittest\nclass Additional(unittest.TestCase):\n"
        " def test_pass(self): self.assertTrue(True)\n"
    )
    result, summary = run_canonical_suite(methods, tmp_path / "summary.json")
    assert result.returncode == 1
    assert summary["status"] == "failed"
    assert summary["tests_run"] == 2
    assert summary["failures"] == ["test_fixture.Cases.test_failed"]
    assert summary["errors"] == []
