"""Portable entry points for identity indexing and the original fixed baselines."""

import argparse
import json
from common import OUT, build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["index", "features", "benchmark", "spectra"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "index":
        rows = build_index()
        result = {"sessions": len(rows), "eligible": sum(row["eligible"] for row in rows)}
    elif args.command == "features":
        from features import extract

        result = extract()
    else:
        from benchmark import run

        result = run(spectra=args.command == "spectra")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
