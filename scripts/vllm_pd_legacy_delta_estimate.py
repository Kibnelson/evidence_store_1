#!/usr/bin/env python3
"""Estimate changed-row NIXL savings from a traced Full Blend replay."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


SCHEMA_VERSION = 1
FILES = ("legacy-delta-estimate.json", "legacy-delta-estimate.jsonl")
POLICIES = {
    ("mtrag-multi-turn-repeat", "previous-turn-first"): "mtrag",
    ("musique-single-turn-40", "seen-first"): "musique",
}
DIFF_PERCENTAGES = (10, 50, 90, 100)


def _natural_rows(cohort, name, label):
    path = cohort / f"{name}.jsonl"
    if path.is_file():
        stream = path.open(encoding="utf-8")
    else:
        stream = gzip.open(path.with_suffix(".jsonl.gz"), "rt", encoding="utf-8")
    with stream:
        return [row for line in stream if line.strip() and (row := json.loads(line))["label"] == label]


def _full_read(request):
    transfer = request["nixl_transfer"]
    full_bytes = request["completed_nixl_bytes"]
    prompt_tokens = request["prompt_tokens"]
    padded_tokens = transfer["position_end"]
    if (transfer["status"] != "done" or transfer["worker_id"] != request["decode"]
            or transfer["completed_bytes"] != full_bytes or transfer["position_start"] != 0
            or transfer["valid_position_end"] != prompt_tokens or not 0 < prompt_tokens <= padded_tokens
            or full_bytes <= 0 or full_bytes % padded_tokens):
        raise ValueError(f"Invalid full NIXL READ for {request['request_key']}")
    timeline = [(event["event"], event["ts_ns"]) for event in request["timeline"]
                if event["event"] in ("nixl_read_submit", "nixl_read_complete")]
    if (tuple(event for event, _ in timeline) != ("nixl_read_submit", "nixl_read_complete")
            or not timeline[0][1] <= timeline[1][1] == transfer["ts_ns"]):
        raise ValueError(f"Missing or invalid observed NIXL READ timeline for {request['request_key']}")
    return full_bytes, full_bytes // padded_tokens, timeline[1][1] - timeline[0][1]


def _identity(chunk):
    return chunk["doc_id"], chunk["chunk_id"], chunk["chunk_version"]


def _candidate_chunks(request, chunks, previous_chunks, prior_requests, width):
    previous_lengths = {}
    for chunk in previous_chunks:
        key = _identity(chunk)
        previous_lengths[key] = max(previous_lengths.get(key, 0), chunk["end_token"] - chunk["start_token"])
    for chunk in chunks:
        start, end = chunk["start_token"], chunk["end_token"]
        for move in chunk["lmcache_movements"]:
            source = move.get("source_match")
            if move["direction"] != "h2d" or not source:
                continue
            prior = prior_requests.get(source["prior_request_id"])
            lo, hi = move["evidence_overlap"]
            move_start, move_end = move["start_token"], move["end_token"]
            if (prior is None or prior["request_key"] != source["prior_request_key"]
                    or prior["label"] != source["prior_label"] or prior["prefill"] != request["prefill"]
                    or move["worker_id"] != request["prefill"]
                    or (lo, hi) != (max(start, move_start), min(end, move_end))
                    or not 0 <= lo < hi <= request["prompt_tokens"]
                    or move["bytes"] != width * (move_end - move_start)):
                raise ValueError(f"Invalid verified H2D evidence span for {request['request_key']}")
            candidate = {field: chunk[field] for field in ("doc_id", "chunk_id", "chunk_version", "occurrence")}
            candidate.update(start_token=lo, end_token=hi, candidate_bytes=(hi - lo) * width,
                             restore_source_request_id=prior["request_id"],
                             restore_source_request_key=prior["request_key"], restore_source_decode=prior["decode"],
                             shifted=source["shifted"])
            if request["label"] == "mtrag":
                previous_stop = min(hi, start + previous_lengths.get(_identity(chunk), 0))
                candidate["previous_turn_candidate_bytes"] = max(0, previous_stop - lo) * width
            yield candidate


def _span_bytes(spans, width):
    """Count each covered target position once, including overlapping receipts."""
    count, end = 0, 0
    for start, stop in sorted(spans):
        count += max(0, stop - max(start, end))
        end = max(end, stop)
    return count * width


def _scenario_totals(full_bytes, logical_bytes, candidate_bytes, saved):
    return {
        "candidate_bytes": candidate_bytes,
        "assumed_base_coverage_of_logical_kv": candidate_bytes / logical_bytes,
        "scenarios": [
            {
                "assumed_diff_percent": diff,
                "estimated_bytes_saved": count,
                "estimated_nixl_bytes_sent": full_bytes - count,
                "estimated_nixl_fraction_saved": count / full_bytes,
                "ideal_bandwidth_only_speedup": full_bytes / (full_bytes - count),
            }
            for diff, count in zip(DIFF_PERCENTAGES, saved, strict=True)
        ],
    }


def _estimates(requests, chunks, scopes):
    by_request = defaultdict(list)
    for chunk in chunks:
        by_request[chunk["request_id"]].append(chunk)
    if set(by_request) != {request["request_id"] for request in requests}:
        raise ValueError("Natural requests and evidence rows disagree")
    prior_requests, by_turn = {}, {}
    for request in requests:
        full_bytes, width, observed_read_ns = _full_read(request)
        logical_bytes = width * request["prompt_tokens"]
        evidence = by_request[request["request_id"]]
        previous = ()
        if "previous_turn" in scopes:
            conversation, turn = request["conversation_id"], request["turn"]
            previous = by_turn.get((conversation, turn - 1), ())
            by_turn[conversation, turn] = evidence
        candidates = list(_candidate_chunks(request, evidence, previous, prior_requests, width))
        candidate_bytes = {
            "all_verified_restores": _span_bytes(((row["start_token"], row["end_token"]) for row in candidates), width),
        }
        if "previous_turn" in scopes:
            candidate_bytes["previous_turn"] = _span_bytes((
                (row["start_token"], row["start_token"] + row["previous_turn_candidate_bytes"] // width)
                for row in candidates
            ), width)
        if candidate_bytes["all_verified_restores"] > logical_bytes:
            raise ValueError(f"Candidate bytes exceed logical target for {request['request_key']}")
        prior_requests[request["request_id"]] = request
        estimate = {field: request[field] for field in (
            "request_key", "request_id", "label", "prefill", "decode", "prompt_sha256", "prompt_tokens",
        )}
        if "previous_turn" in scopes:
            estimate.update(conversation_id=conversation, turn=turn)
        estimate.update(full_nixl_bytes=full_bytes, logical_target_bytes=logical_bytes,
                        full_observed_nixl_read_ns=observed_read_ns,
                        full_native_transfer_duration_us=request["nixl_transfer"]["transfer_duration_us"],
                        candidate_chunks=candidates, scopes={
                            scope: _scenario_totals(full_bytes, logical_bytes, count,
                                                    [count * (100 - diff) // 100 for diff in DIFF_PERCENTAGES])
                            for scope, count in candidate_bytes.items()
                        })
        yield estimate


def write_estimate(cohort: Path, *, output: Path | None = None):
    """Write a summary and request log from a verified raw or compact replay."""
    target = output if output is not None else cohort
    if target.resolve() == cohort.resolve() and not (cohort / "legacy-requests.jsonl").is_file():
        raise ValueError("Use --output to keep the compact review archive unchanged")
    summary = json.loads((cohort / "legacy-blend-summary.json").read_text(encoding="utf-8"))
    pair = (summary.get("workload"), summary.get("prompt_policy"))
    label = POLICIES.get(pair)
    if (label is None or summary.get("status") != "passed" or summary.get("trace_enabled") is not True
            or summary.get("natural_label") != label):
        raise ValueError("Expected a passing traced MuSiQue seen-first or MT-RAG previous-turn-first replay")
    requests = _natural_rows(cohort, "legacy-requests", label)
    chunks = _natural_rows(cohort, "legacy-chunks", label)
    if (not requests or len(requests) != summary["natural_requests"]
            or len({row["request_id"] for row in requests}) != len(requests)
            or len(chunks) != summary["natural_evidence_occurrences"]):
        raise ValueError("Natural request or evidence counts differ from the measurement summary")
    scope_names = ("all_verified_restores", "previous_turn") if label == "mtrag" else ("all_verified_restores",)
    rows = list(_estimates(requests, chunks, scope_names))
    full_bytes = sum(row["full_nixl_bytes"] for row in rows)
    logical_bytes = sum(row["logical_target_bytes"] for row in rows)
    if full_bytes != summary["transfer_payload_bytes_by_label"][label]["nixl_p2d"]:
        raise ValueError("Natural full NIXL bytes differ from the measurement summary")
    estimate = {
        "schema_version": SCHEMA_VERSION,
        "status": "hypothetical",
        "workload": pair[0],
        "prompt_policy": pair[1],
        "natural_label": label,
        "natural_requests": len(rows),
        "full_nixl_bytes": full_bytes,
        "logical_target_bytes": logical_bytes,
        "assumptions": {
            "actual_transfer": "full native NIXL READ",
            "all_verified_restores": "D hypothetically has a compatible row for every verified P restore of evidence.",
            "previous_turn": ("D hypothetically retains the immediately preceding turn of the same conversation; "
                              "candidates must also appear in that turn with coverage of the same chunk offsets.")
            if label == "mtrag" else None,
            "model": ("Saved bytes = candidate bytes times (1 - assumed diff fraction), rounded down per request. "
                      "Full READ padding stays in the estimate, with zero incremental packing or metadata bytes."),
            "time": ("Speedup is the full to estimated payload ratio at constant bandwidth with no overhead. "
                     "Actual changed rows, delta transfer time, and TTFT speedup are unmeasured."),
        },
        "scopes": {},
    }
    for scope in scope_names:
        receipts = [row["scopes"][scope] for row in rows]
        saved = [sum(receipt["scenarios"][i]["estimated_bytes_saved"] for receipt in receipts)
                 for i, _ in enumerate(DIFF_PERCENTAGES)]
        estimate["scopes"][scope] = _scenario_totals(
            full_bytes, logical_bytes, sum(receipt["candidate_bytes"] for receipt in receipts), saved,
        )
    target.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        for name, contents in zip(FILES, (
            json.dumps(estimate, indent=2, sort_keys=True) + "\n",
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        ), strict=True):
            path = target / name
            with path.open("x", encoding="utf-8") as stream:
                written.append(path)
                stream.write(contents)
    except OSError:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path, help="Traced layout's blend directory")
    parser.add_argument("--output", type=Path, help="Write into a separate directory instead of the cohort")
    args = parser.parse_args()
    try:
        print(json.dumps(write_estimate(args.cohort, output=args.output), indent=2, sort_keys=True))
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Full Blend delta estimate failed: {error}\n")


if __name__ == "__main__":
    main()
