#!/usr/bin/env python3
"""Capture native counters or generate the end-to-end PD trace report."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidencestore.pd_report import capture_metrics, report, verify_retrieval_smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--capture", choices=("before", "after"))
    parser.add_argument("--timeline-limit", type=int, default=32)
    parser.add_argument("--timeline-request", help="render just the supplied request ID")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--smoke-sample", type=Path, help="Verify the live MT-RAG smoke against this sample")
    args = parser.parse_args()
    if args.timeline_limit < 1:
        parser.error("--timeline-limit must be positive")
    if args.smoke_sample and args.capture:
        parser.error("--smoke-sample requires a trace report")
    run_dir = args.run_dir.resolve()
    if args.capture:
        capture_metrics(run_dir, args.capture)
        return 0
    result = report(
        run_dir, timeline_limit=args.timeline_limit, all_pairs=args.all_pairs or bool(args.smoke_sample),
        timeline_request=args.timeline_request,
    )
    print(json.dumps(result, indent=2))
    if args.smoke_sample:
        try:
            verify_retrieval_smoke(run_dir, args.smoke_sample, result)
        except ValueError as exc:
            print(f"Smoke verification failed: {exc}", file=sys.stderr)
            return 1
    return int(args.check and bool(result["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
