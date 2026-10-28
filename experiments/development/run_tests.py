"""Run source-only development checks in independent, offline subprocesses."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BRANCHES = (
    "camera_geometry_development_2026-10-27",
    "cyclic_kernel_development_2026-10-27",
    "depth_pose_development_2026-10-27",
    "foreground_distillation_2026-10-27",
    "leaf_proposal_development_2026-10-27",
    "leaf_review_workbench_2026-10-27",
    "longitudinal_tracking_development_2026-10-27",
    "new_session_pipeline_development_2026-10-27",
    "perspective_turntable_development_2026-10-27",
    "rotation_workbench_2026-10-27",
    "topology_foreground_development_2026-10-27",
    "capture_fusion_development_2026-10-27",
    "capture_fusion_review_2026-10-27",
    "channel_validity_audit_2026-10-27",
    "strict_index_sensitivity_2026-10-27",
    "foreground_tracking_sensitivity_2026-10-27",
    "longitudinal_confirmation_2026-10-28",
)
OPTIONAL_TEST_DEPENDENCIES = {
    "topology_foreground_development_2026-10-27": ("torch",),
    "foreground_distillation_2026-10-27": ("torch",),
}
# These commands exercise argument parsing and portable imports, never inference.
HELP_SCRIPTS = {
    "camera_geometry_development_2026-10-27": ("camera_geometry_run.py", "periodic/periodic_run.py"),
    "cyclic_kernel_development_2026-10-27": ("benchmark.py",),
    "depth_pose_development_2026-10-27": ("audit_pose.py",),
    "foreground_distillation_2026-10-27": ("train_foreground.py", "cache_foreground.py", "infer_foreground.py", "evaluate_foreground.py"),
    "leaf_proposal_development_2026-10-27": ("leaf_plan.py", "leaf_exports.py"),
    "leaf_review_workbench_2026-10-27": ("build_workbench.py", "verify_bundle.py"),
    "longitudinal_tracking_development_2026-10-27": ("tracking_development.py", "tracking_validation.py"),
    "new_session_pipeline_development_2026-10-27": ("run_capture.py", "audit_capture.py"),
    "perspective_turntable_development_2026-10-27": ("perspective_run.py", "perspective_export.py", "perspective_audit.py"),
    "rotation_workbench_2026-10-27": ("sprites.py", "build_workbench.py"),
    "topology_foreground_development_2026-10-27": ("train_topology.py", "infer_topology.py", "audit_topology.py"),
}


HELP_SCRIPTS.update({
    "capture_fusion_development_2026-10-27": ("capture_fusion.py", "run_digitization.py", "format_report.py"),
    "capture_fusion_review_2026-10-27": ("verify_fusion.py",),
    "channel_validity_audit_2026-10-27": ("validity_audit.py",),
    "strict_index_sensitivity_2026-10-27": ("strict_run.py", "strict_verify.py"),
    "foreground_tracking_sensitivity_2026-10-27": ("tracking_sensitivity.py", "heldout_unet.py"),
    "longitudinal_confirmation_2026-10-28": ("registry.py", "dynamics.py", "study_metadata.py", "mass_validation.py", "verify_mass.py"),
})

def missing_optional(branch, find_spec=importlib.util.find_spec):
    """Only a missing explicitly declared package permits skipping a suite.

    Do not catch import or loader errors: installed-but-broken dependencies must
    fail visibly in their worker, rather than being mistaken for absent extras.
    """
    return [name for name in OPTIONAL_TEST_DEPENDENCIES.get(branch, ()) if find_spec(name) is None]


def environment():
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false", "CUDA_VISIBLE_DEVICES": "",
    })
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "2"
    return env


def worker(branch):
    folder = HERE / branch
    sys.path[:0] = [str(ROOT / "src"), str(folder)]
    # Protect against a previously installed editable copy of the package.
    import csi_maize
    if not Path(csi_maize.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError("Development worker imported a different checkout")
    if "torch" in OPTIONAL_TEST_DEPENDENCIES.get(branch, ()):
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    tests = unittest.TestSuite()
    files = sorted(folder.rglob("test_*.py"))
    for file in files:
        sys.path.insert(1, str(file.parent))
        name = file.stem
        if name in sys.modules:
            module = sys.modules[name]
            if Path(module.__file__).resolve() != file.resolve():
                raise RuntimeError(f"Test module collision: {file}")
        else:
            spec = importlib.util.spec_from_file_location(name, file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        tests.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    print(json.dumps({"branch": branch, "tests": result.testsRun, "test_files": len(files),
                      "failures": len(result.failures), "errors": len(result.errors),
                      "skipped_tests": len(result.skipped)}))
    return 0 if result.wasSuccessful() else 1


def execute(command, label, timeout=180):
    result = subprocess.run(command, cwd=ROOT, env=environment(), text=True, encoding="utf-8",
                            capture_output=True, timeout=timeout)
    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", action="append", choices=BRANCHES,
                        help="Run only this branch; repeat for multiple branches")
    parser.add_argument("--require-optional", action="store_true",
                        help="Fail instead of skipping a suite whose declared extra is absent")
    parser.add_argument("--report", type=Path, help="Write a machine-readable test report")
    parser.add_argument("--worker", choices=BRANCHES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    selected = args.branch or list(BRANCHES)
    report = {"status": "running", "python": sys.version.split()[0], "branches": [],
              "node": [], "integration": None, "scope": "source-only synthetic and import tests; no model inference or private data"}
    try:
        result = execute([sys.executable, str(HERE / "integration_tests.py")], "Public-copy contracts")
        report["integration"] = json.loads(result.stdout.strip().splitlines()[-1])
        for branch in selected:
            missing = missing_optional(branch)
            if missing:
                if args.require_optional:
                    raise RuntimeError(f"{branch} requires missing optional dependencies: {', '.join(missing)}")
                report["branches"].append({"branch": branch, "status": "skipped_missing_optional",
                                            "missing": missing, "tests": 0})
                print(f"SKIP {branch}: missing declared optional dependencies {', '.join(missing)}", flush=True)
                continue
            result = execute([sys.executable, str(Path(__file__).resolve()), "--worker", branch], branch)
            item = json.loads(result.stdout.strip().splitlines()[-1])
            item["status"] = "passed"
            item["help_commands"] = 0
            for script in HELP_SCRIPTS.get(branch, ()):
                execute([sys.executable, str(HERE / branch / script), "--help"], f"{branch}/{script} --help")
                item["help_commands"] += 1
            report["branches"].append(item)
            print(f"PASS {branch}: {item['tests']} tests, {item['help_commands']} CLI imports", flush=True)
        node = shutil.which("node")
        node_branches = {"leaf_review_workbench_2026-10-27", "rotation_workbench_2026-10-27"} & set(selected)
        if node_branches and not node:
            raise RuntimeError("Node.js >=22 is required for the selected workbench checks")
        if "leaf_review_workbench_2026-10-27" in selected:
            folder = HERE / "leaf_review_workbench_2026-10-27"
            result = execute([node, "--test", "--test-reporter=tap", str(folder / "review_core.test.js"), str(folder / "review_app.test.js")], "Leaf review Node tests")
            match = re.search(r"# tests (\d+)", result.stdout)
            if not match or int(match.group(1)) != 17:
                raise RuntimeError("Expected all 17 leaf review Node behavior tests")
            report["node"].append({"suite": "leaf_review", "status": "passed", "tests": 17})
            print("PASS leaf review: 17 Node behavior tests", flush=True)
        if "rotation_workbench_2026-10-27" in selected:
            folder = HERE / "rotation_workbench_2026-10-27"
            result = execute([node, str(folder / "test_workbench.mjs"), str(folder / "workbench.html")], "Rotation synthetic template fixture")
            report["node"].append({"suite": "rotation_template_fixture", "status": "passed",
                                    "fixture_result": json.loads(result.stdout)})
            print("PASS rotation workbench: synthetic template fixture", flush=True)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        print(str(exc), file=sys.stderr)
    finally:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
