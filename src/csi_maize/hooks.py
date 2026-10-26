"""Small stage hooks executed from the immutable runtime snapshot."""

import argparse
from pathlib import Path
import json

from .config import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hook", choices=["metadata", "models", "report"])
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    config = RunConfig.load(args.workspace / "run_config.json")
    if args.hook == "metadata":
        from .metadata import build_reference_tables

        result = build_reference_tables(args.workspace)
    elif args.hook == "models":
        from .assets import prepare_models

        result = prepare_models(config, args.workspace, allow_download=False)
    else:
        from .report import build_report

        result = {"report": str(build_report(args.workspace))}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
