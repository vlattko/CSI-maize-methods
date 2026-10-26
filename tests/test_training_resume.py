"""Interrupted training must retain already completed ablation candidates."""

import json
from pathlib import Path
import subprocess
import sys
import textwrap


def test_cached_blue_runs_are_restored_to_rebuilt_selection_inventory(tmp_path):
    code = r"""
import json, sys, types
from pathlib import Path
import csi_maize.ultralytics_offline as offline
from bootstrap import OUT
offline.configure_ultralytics = lambda *a, **k: None
fake = types.ModuleType("ultralytics")
fake.YOLO = lambda *a, **k: (_ for _ in ()).throw(AssertionError("cached training executed"))
sys.modules["ultralytics"] = fake
import train_noir_blue_leaf as module
root = module.DEST
root.mkdir(parents=True)
(root/"training_summary.json").write_text(json.dumps({"runs":[{"run":f"{cam}_{scale}_weak_e30","camera":cam} for cam in ["rgb","noir"] for scale in ["n","s"]]}))
for scale in ["n","s"]:
    folder=root/"training"/f"noir_blue_{scale}_weak_e30"
    folder.mkdir(parents=True)
    (folder/"run_manifest.json").write_text(json.dumps({"run":folder.name,"camera":"noir","representation":"recorded_blue_channel_triplicated"}))
module.main()
records=json.loads((root/"training_summary.json").read_text())["runs"]
assert len(records)==6, records
assert len({r["run"] for r in records})==6
assert sum(r.get("representation")=="recorded_blue_channel_triplicated" for r in records)==2
"""
    import os

    env = dict(os.environ, CSI_MAIZE_WORKSPACE=str(tmp_path), PYTHONUTF8="1")
    import csi_maize

    env["PYTHONPATH"] = str(Path(csi_maize.__file__).parent / "_methods")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
