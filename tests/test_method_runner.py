"""Analysis stages cannot silently fetch unregistered dependencies."""

from pathlib import Path
import os
import subprocess
import sys


def test_stage_outbound_connection_is_rejected(tmp_path):
    base = tmp_path / "prototype/research_v2"
    base.mkdir(parents=True)
    (base / "stage.py").write_text(
        "import socket\ntry:\n socket.create_connection(('example.com',443),timeout=.1)\n"
        "except OSError:\n print('NETWORK_DENIED')\nelse:\n raise RuntimeError('Unexpected network access')\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "csi_maize.method_runner", "stage.py"],
        env=dict(os.environ, CSI_MAIZE_WORKSPACE=str(tmp_path)),
        cwd=base,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "NETWORK_DENIED" in result.stdout


def test_runner_rejects_script_escape(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "csi_maize.method_runner", "../outside.py"],
        env=dict(os.environ, CSI_MAIZE_WORKSPACE=str(tmp_path)),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "contained" in result.stderr.lower()
