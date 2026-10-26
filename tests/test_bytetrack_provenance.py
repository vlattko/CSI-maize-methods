"""Tracker provenance follows the imported implementation, never a vendor path."""

from pathlib import Path
import hashlib
import os
import subprocess
import sys


METHODS = Path(__file__).resolve().parents[1] / "src/csi_maize/_methods"


def test_tracker_digest_reads_the_imported_python_class_without_vendor_folder(tmp_path):
    source = tmp_path / "installed_tracker.py"
    source.write_text(
        "class BYTETracker:\n    def update(self, observations):\n        return observations\n",
        encoding="utf-8",
    )
    program = (
        "from installed_tracker import BYTETracker\n"
        "from benchmark_leaf_bytetrack import tracker_source_sha256\n"
        "print(tracker_source_sha256(BYTETracker))\n"
    )
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", program],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(METHODS)])),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == hashlib.sha256(source.read_bytes()).hexdigest()
    assert not (METHODS / "yolo_deps").exists()


def test_tracker_provenance_rejects_an_uninspectable_implementation():
    program = (
        "from benchmark_leaf_bytetrack import tracker_source_sha256\n"
        "try: tracker_source_sha256(int)\n"
        "except RuntimeError as error: assert 'source' in str(error).lower()\n"
        "else: raise AssertionError('Missing Python source was accepted')\n"
    )
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", program],
        env=dict(os.environ, PYTHONPATH=str(METHODS)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
