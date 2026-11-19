#!/usr/bin/env python3
"""Check detailed chunk measurements from a completed PD cohort."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidencestore.pd_chunk_report import build_chunk_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True, help="Replay label to include, repeat for more labels")
    parser.add_argument("--check-smoke", action="store_true", help="Check the controlled repeat and movement probes")
    args = parser.parse_args()
    try:
        summary = build_chunk_report(args.run_dir.resolve(), args.label, smoke=args.check_smoke)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Chunk review failed: {error}", file=sys.stderr)
        return 1
    print(f"{summary['status']}: {summary['requests']} requests, {summary['chunk_occurrences']} chunk occurrences")
    for issue in summary["issues"]:
        print(issue, file=sys.stderr)
    print(f"Chunk review: {args.run_dir / 'chunk-review'}")
    return int(summary["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
