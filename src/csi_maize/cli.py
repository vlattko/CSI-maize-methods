"""Public commands for declared inputs, numerical runs and self-contained reports."""

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import json
import sys

from .config import RunConfig, DEFAULT_SOURCES
from .engine import execute, ordered_stages, verify_run
from .provenance import write_json
from .workspace import create_workspace

PUBLICATION_TARGETS = [
    "source-support",
    "view-sampling",
    "pair-controls",
    "spatial-fusion",
    "synthetic-stress",
    "reposition",
    "tracking-comparison",
    "channel-report",
]


def workflow(profile="publication"):
    from .stages import STAGES

    stages = deepcopy(STAGES)
    targets = list(PUBLICATION_TARGETS)
    if profile == "learning":
        targets += [
            "baseline-features",
            "phenotype-models",
            "spectral-models",
            "phenotype-controls",
        ]
    stages["paper"] = dict(
        dependencies=targets,
        commands=[["@report"]],
        outputs=["reports/CSI_maize_architecture.html"],
        profile="core",
        description="Build the architecture-first self-contained scientific companion",
    )
    return stages


def main(argv=None):
    parser = argparse.ArgumentParser(description="CSI-MAIZE reproducible scientific methods")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create an empty owned computational workspace")
    init.add_argument("--data-root", type=Path, required=True)
    init.add_argument("--work-dir", type=Path, required=True)
    init.add_argument("--models-root", type=Path)
    init.add_argument("--profile", choices=["publication", "learning"], default="publication")
    init.add_argument("--device", default="auto")
    init.add_argument("--expected-acquisitions", type=int, default=41)
    init.add_argument("--source-folder", action="append", help="Relative image source folder; repeat as needed")
    init.add_argument("--workbook-folder", action="append", default=[], help="Additional relative workbook folder; top-level workbooks are always included")
    for name in ("run", "plan", "doctor", "assets"):
        p = commands.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        if name in ("run", "plan"):
            p.add_argument(
                "--target", action="append", help="Target stage; repeat for multiple terminals"
            )
        if name == "run":
            p.add_argument("--resume", action="store_true")
        if name == "assets":
            p.add_argument(
                "--download", action="store_true", help="Allow pinned official asset downloads"
            )
    check = commands.add_parser("verify")
    check.add_argument("--run", type=Path, required=True)
    check.add_argument(
        "--outputs-only",
        action="store_true",
        help="Verify stored outputs without requiring original inputs or weights",
    )
    report = commands.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--output", type=Path)
    example = commands.add_parser("example", help="Run a no-download controlled CPU example")
    example.add_argument("--output", type=Path, required=True)
    example.add_argument("--steps", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            config = RunConfig(
                data_root=args.data_root,
                work_dir=args.work_dir,
                models_root=args.models_root,
                profile=args.profile,
                device=args.device,
                expected_acquisitions=args.expected_acquisitions or None,
                source_folders=tuple(args.source_folder) if args.source_folder else DEFAULT_SOURCES,
                workbook_folders=tuple(args.workbook_folder),
            )
            create_workspace(config)
            print(config.work_dir / "run_config.json")
        elif args.command == "example":
            from .example import run_example

            print(json.dumps(run_example(args.output, args.steps), indent=2, allow_nan=False))
        elif args.command == "verify":
            print(json.dumps(verify_run(args.run, verify_inputs=not args.outputs_only), indent=2))
        elif args.command == "report":
            from .report import build_report

            print(build_report(args.run, args.output))
        else:
            config = RunConfig.load(args.config)
            if args.command in ("run", "plan"):
                stages = workflow(config.profile)
                targets = args.target or ["paper"]
                order = ordered_stages(stages, targets)
                if args.command == "plan":
                    print(
                        json.dumps([dict(stage=name, **stages[name]) for name in order], indent=2)
                    )
                else:
                    result = execute(config, targets, stages=stages, resume=args.resume)
                    print(
                        json.dumps(
                            dict(
                                status=result["status"],
                                targets=result["targets"],
                                workspace=str(config.work_dir),
                            ),
                            indent=2,
                        )
                    )
            elif args.command == "assets":
                from .assets import prepare_models

                print(
                    json.dumps(
                        prepare_models(config, config.work_dir, allow_download=args.download),
                        indent=2,
                    )
                )
            else:
                modules = {
                    name: importlib.util.find_spec(name) is not None
                    for name in (
                        "numpy",
                        "scipy",
                        "PIL",
                        "openpyxl",
                        "cv2",
                        "torch",
                        "transformers",
                        "ultralytics",
                    )
                }
                findings = dict(
                    data_root_exists=config.data_root.is_dir(),
                    workspace_initialized=(config.work_dir / ".csi_workspace.json").exists(),
                    installed_modules=modules,
                    mode=config.mode,
                    profile=config.profile,
                    zenodo_record=config.zenodo_record_url,
                    note="A successful dependency check is not scientific validation or a completed run.",
                )
                print(json.dumps(findings, indent=2))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
