#!/usr/bin/env python3
"""Plan and verify raw-page captures for the six validated CLAPNQ conversations."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from vllm_pd_legacy_page_payloads import AVAILABLE, capture_records, load_payload


MIB = 1024 ** 2
REQUEST_KEY = re.compile(r"mtrag:([0-9a-f]{32})<::>([1-9][0-9]*)")
ARCHIVE_LINE = re.compile(r"([0-9a-f]{64})  (.+\.tar\.gz)")


def plan(selection, first_four):
    workload = json.loads(selection.read_text(encoding="utf-8"))
    if (workload["workload"], workload["dataset"], workload["domain"]) != (
        "mtrag-multi-turn-repeat", "mtrag", "clapnq"
    ):
        raise ValueError("Expected the validated CLAPNQ workload")
    turns = defaultdict(list)
    for request in workload["request_keys"]:
        match = REQUEST_KEY.fullmatch(request)
        if match is None:
            raise ValueError(f"Unexpected request key {request!r}")
        conversation, turn = match.groups()
        turns[conversation].append(int(turn))
    if len(turns) != 6 or sum(map(len, turns.values())) != 40:
        raise ValueError("Expected six conversations with 40 total turns")
    for conversation, sequence in turns.items():
        if sequence != list(range(1, len(sequence) + 1)):
            raise ValueError(f"Missing or out-of-order turns in {conversation}")
        print(f"{conversation}\t1:{min(4, len(sequence)) if first_four else len(sequence)}")


def _archive(capture_dir):
    matches = [
        match for line in (capture_dir / "launch.log").read_text(encoding="utf-8").splitlines()
        if (match := ARCHIVE_LINE.fullmatch(line)) is not None
        and Path(match[2]).parent == capture_dir
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one archive and SHA-256 receipt under {capture_dir}")
    digest, filename = matches[0].groups()
    path = Path(filename)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Missing capture archive {path}")
    return path, digest


def capture_receipt(report, conversation, window, *, budget_mib=None):
    """Check capture scope and return its replay-independent receipt fields."""
    first, last = map(int, window.split(":"))
    turns = [(row["conversation_id"], row["turn"]) for row in report["turns"]]
    if not turns or first < 1 or turns != [(conversation, turn) for turn in range(first, last + 1)]:
        raise ValueError(f"Archive does not cover {conversation} turns {window}")
    if report["status"] != "measured":
        raise ValueError(f"Page trace for {conversation} reports {report['status']}")
    if budget_mib is not None and report["trace_bounds"].get("max_capture_raw_bytes") != budget_mib * MIB:
        raise ValueError(f"Capture did not use its requested {budget_mib} MiB budget")

    for operation, coverage in report["native_movement_coverage"].items():
        fields = (
            "native_segments_with_unknown_bytes", "skipped_complete_raw_bytes",
            "skipped_events_with_unknown_bytes", "unaccounted_native_raw_bytes",
            "page_raw_bytes_above_native",
        )
        incomplete = {key: coverage[key] for key in fields if coverage[key] != 0}
        if incomplete:
            raise ValueError(f"Incomplete {operation} page coverage for {conversation}. {incomplete}")

    inventory = capture_records(report)
    if not inventory or any(
        row["status"] != "measured" or row.get("payload_capture_status") not in AVAILABLE
        or not row.get("native_transfer_matched") or row["tensor_sha256"] not in inventory
        for row in report["stores"]
    ):
        raise ValueError(f"Incomplete raw capture for {conversation}. {report['raw_page_capture']}")
    for row in report["restores"]:
        if (not row.get("native_transfer_matched") or not row.get("source_store_version_matched")
                or row.get("observed_sha256") not in inventory
                or row["observed_sha256"] != row.get("tensor_sha256")):
            raise ValueError(f"Unverified native page load for {conversation} on turn {row['turn']}")
    return {
        "conversation_id": conversation, "turn_window": window, "turns": len(turns),
        "stores": len(report["stores"]), "observed_loads": len(report["restores"]),
        "unique_raw_pages": len(inventory), "raw_capture_bytes": sum(inventory.values()),
    }


def record(repo, capture_dir, conversation, window, budget_mib):
    archive, digest = _archive(capture_dir)
    cohort = repo / "reports" / "runs" / archive.name.removesuffix(".tar.gz") / "1p3d" / "blend"
    report = json.loads((cohort / "legacy-kv-page-report.json").read_text(encoding="utf-8"))
    receipt = capture_receipt(report, conversation, window, budget_mib=budget_mib)
    for tensor_sha256, size in capture_records(report).items():
        load_payload(cohort, tensor_sha256, size)
    print(json.dumps({**receipt, "archive": str(archive), "sha256": digest}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("plan")
    selection.add_argument("workload", type=Path)
    selection.add_argument("first_four", type=int, choices=(0, 1))
    receipt = commands.add_parser("record")
    receipt.add_argument("runtime_repo", type=Path)
    receipt.add_argument("capture_dir", type=Path)
    receipt.add_argument("conversation")
    receipt.add_argument("turn_window")
    receipt.add_argument("budget_mib", type=int)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            plan(args.workload, args.first_four)
        else:
            record(args.runtime_repo, args.capture_dir, args.conversation, args.turn_window, args.budget_mib)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"MT-RAG campaign stopped. {error}\n")


if __name__ == "__main__":
    main()
