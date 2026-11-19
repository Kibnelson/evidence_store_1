#!/usr/bin/env python3
"""Capture or summarize measurements for one PD replay dataset."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidencestore.pd_dataset_report import DATASETS, begin_dataset, finish_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Cohort directory such as 1p3d-native")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--begin", action="store_true", help="Capture dataset-window start counters before replay")
    phase.add_argument("--finish", action="store_true", help="Summarize a completed replay and capture end counters")
    parser.add_argument("--timeline-limit", type=int, default=8, help="Requests to include in the saved timeline")
    args = parser.parse_args()
    if args.timeline_limit < 1:
        parser.error("--timeline-limit must be positive")
    run_dir = args.run_dir.resolve()
    try:
        if args.begin:
            print(f"Started dataset measurement in {begin_dataset(run_dir, args.dataset)}")
            return 0
        summary = finish_dataset(run_dir, args.dataset, timeline_limit=args.timeline_limit)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Dataset report failed for {args.dataset} in {run_dir}: {exc}", file=sys.stderr)
        return 1
    print(
        f"{summary['cohort']} {summary['dataset']}: {summary['status']}, {summary['requests']} requests, "
        f"NIXL counter reconciliation {summary['counter_reconciliation']['status']}"
    )
    for issue in summary["issues"]:
        print(issue, file=sys.stderr)
    return int(summary["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
