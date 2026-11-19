#!/usr/bin/env python3
"""Compare naturally recurring chunks across two serial live MuSiQue replays."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidencestore.retrieval import json_line, prompt_sha256, read_jsonl, sha256_file


COHORTS = ("2p2d-native", "2p2d-shared")
MUSIQUE_DEV_REQUESTS = 2417
IDENTITY_FIELDS = ("doc_id", "chunk_id", "chunk_version")
CHUNK_FIELDS = (*IDENTITY_FIELDS, "start_token", "end_token", "rank")
TRANSFER_PATHS = ("nixl_p2d", "lmcache_d2h", "lmcache_h2d")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _span(chunk):
    return tuple(chunk[name] for name in CHUNK_FIELDS)


def _measurements(row):
    work = row["token_work"]
    transfers = row["exclusive_transfer_bytes"]
    return {
        "p_scheduled_evidence_tokens": work.get("P", {}).get("scheduled_prompt_tokens", 0),
        "d_scheduled_evidence_tokens": work.get("D", {}).get("scheduled_prompt_tokens", 0),
        **{f"{path}_exclusive_bytes": transfers.get(path, 0) for path in TRANSFER_PATHS},
    }


def _load_cohort(run_root, cohort, query_keys, query_hash):
    run_dir = run_root / cohort
    replay = run_dir / "client" / "musique"
    review = run_dir / "chunk-review"
    manifest = _read_json(replay / "manifest.json")
    count = len(query_keys)
    if (manifest["mode"] != "live_hnsw" or manifest["trace_chunk_stages"] is not True
            or manifest["input_sha256"] != query_hash or manifest["concurrency"] != 1
            or (manifest["requested"], manifest["succeeded"], manifest["failed"], manifest["frozen_rows"])
            != (count, count, 0, count)):
        raise ValueError(f"{cohort} must complete the full serial live workload with chunk tracing")
    summary = _read_json(review / "summary.json")
    if (summary["status"] != "passed" or summary["requests"] != count or summary["labels"] != ["musique"]
            or summary["nixl_counter_reconciliation"]["status"] != "matched"
            or (cohort == "2p2d-shared"
                and (summary.get("lmcache_counter_reconciliation") or {}).get("status") != "matched")):
        raise ValueError(f"{cohort} does not have a complete, reconciled chunk review")

    prompts, routes, expected_chunks = [], {}, []
    previous_end = 0
    frozen = read_jsonl(replay / "frozen.jsonl")
    responses = read_jsonl(replay / "responses.jsonl")
    requests = read_jsonl(review / "requests.jsonl")
    for key, prompt, response, request in zip(query_keys, frozen, responses, requests, strict=True):
        rid = response["request_id"]
        digest = prompt["prompt_sha256"]
        milestones = request["milestones_ns"]
        start = milestones["client_request_start"]
        end = milestones["client_response_end"]
        if (prompt["request_key"] != key or response["request_key"] != key or response["success"] is not True
                or response["label"] != "musique" or prompt.get("pd_route") or response.get("pd_route")
                or request["request_id"] != rid or rid in routes or request["preparation_scope"] != "live"
                or digest != response["prompt_sha256"] or digest != request["prompt_sha256"]
                or digest != prompt_sha256(prompt["prompt_token_ids"]) or not previous_end <= start < end):
            raise ValueError(f"{cohort} differs from the serial live query order or prompt at {key}")
        previous_end = end
        routes[rid] = {"prefill": request["prefill"], "decode": request["decode"]}
        evidence = tuple(_span(chunk) for chunk in prompt["evidence"])
        if not evidence or evidence != tuple(_span(chunk) for chunk in request["evidence"]):
            raise ValueError(f"{cohort} has missing or inconsistent evidence for {key}")
        prompts.append((rid, digest, evidence))
        expected_chunks.extend((rid, occurrence, chunk) for occurrence, chunk in enumerate(evidence))
    if len(expected_chunks) != summary["chunk_occurrences"]:
        raise ValueError(f"{cohort} chunk count differs from its review summary")
    chunk_metrics = {}
    for (rid, occurrence, span), row in zip(expected_chunks, read_jsonl(review / "chunks.jsonl"), strict=True):
        if (row["request_id"], row["occurrence"], _span(row)) != (rid, occurrence, span):
            raise ValueError(f"{cohort} lacks the chunk review for {rid} occurrence {occurrence}")
        chunk_metrics[rid, occurrence] = _measurements(row)
    return {
        "prompts": prompts,
        "routes": routes,
        "chunks": chunk_metrics,
        "index_sha256": manifest["index"]["index_sha256"],
    }


def build_repeat_report(run_root, query_path):
    query_keys = [row["request_key"] for row in read_jsonl(query_path)]
    if len(query_keys) != MUSIQUE_DEV_REQUESTS or len(set(query_keys)) != len(query_keys):
        raise ValueError(f"Expected {MUSIQUE_DEV_REQUESTS} distinct MuSiQue dev question keys")
    query_hash = sha256_file(query_path)
    cohorts = {name: _load_cohort(run_root, name, query_keys, query_hash) for name in COHORTS}
    native, shared = (cohorts[name] for name in COHORTS)
    native_prompts = native["prompts"]
    if native["index_sha256"] != shared["index_sha256"]:
        raise ValueError("MuSiQue cohorts used different Faiss indexes")
    for key, native_prompt, shared_prompt in zip(query_keys, native_prompts, shared["prompts"], strict=True):
        if native_prompt != shared_prompt:
            raise ValueError(f"Native and shared prompt or chunk positions differ for {key}")

    previous, repeats = {}, []
    recurring, shifted, repeat_requests = set(), set(), set()
    counts = Counter()
    totals = {name: defaultdict(Counter) for name in COHORTS}
    for key, (rid, _, evidence) in zip(query_keys, native_prompts, strict=True):
        current = {}
        for number, span in enumerate(evidence):
            chunk = dict(zip(CHUNK_FIELDS, span, strict=True))
            identity = tuple(chunk[field] for field in IDENTITY_FIELDS)
            occurrence = {
                "request_key": key,
                "request_id": rid,
                "occurrence": number,
                **{field: chunk[field] for field in ("start_token", "end_token", "rank")},
                "cohorts": {
                    name: {**data["routes"][rid], **data["chunks"][rid, number]}
                    for name, data in cohorts.items()
                },
            }
            current[identity] = occurrence
            prior = previous.get(identity)
            group = "first_request"
            if prior is not None:
                shift = occurrence["start_token"] - prior["start_token"]
                group = "repeat_changed_start_token" if shift else "repeat_same_start_token"
                recurring.add(identity)
                repeat_requests.add(rid)
                if shift:
                    shifted.add(identity)
                repeats.append({
                    **{field: chunk[field] for field in IDENTITY_FIELDS},
                    "start_token_delta": shift,
                    "previous": prior,
                    "current": occurrence,
                })
            counts[group] += 1
            for name, data in cohorts.items():
                measured = data["chunks"][rid, number]
                totals[name]["all"].update(measured)
                totals[name][group].update(measured)
        # Track the most recent earlier question. Multiple occurrences inside
        # one question cannot turn into repeats of that same question.
        previous.update(current)

    summary = {
        "status": "passed",
        "dataset": "musique",
        "cohorts": list(COHORTS),
        "queries_sha256": query_hash,
        "index_sha256": native["index_sha256"],
        "requests_per_cohort": len(query_keys),
        "distinct_prompt_hashes": len({digest for _, digest, _ in native_prompts}),
        "chunk_occurrences_per_cohort": counts.total(),
        "distinct_chunks": len(previous),
        "recurring_chunks": len(recurring),
        "recurring_chunks_with_shifted_start": len(shifted),
        "first_request_occurrences": counts["first_request"],
        "repeat_occurrences": len(repeats),
        "repeat_same_start_token": counts["repeat_same_start_token"],
        "repeat_changed_start_token": counts["repeat_changed_start_token"],
        "requests_with_repeat_occurrences": len(repeat_requests),
        "measurements_by_cohort": totals,
    }
    with (run_root / "repeated-chunks.jsonl").open("w", encoding="utf-8") as stream:
        stream.writelines(json_line(row) for row in repeats)
    (run_root / "repeat-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = build_repeat_report(args.run_root, args.queries)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"MuSiQue repeat report failed: {error}\n")
    print(
        f"Verified {summary['requests_per_cohort']} paired MuSiQue requests. "
        f"{summary['repeat_occurrences']} of {summary['chunk_occurrences_per_cohort']} "
        f"chunk occurrences followed an earlier question with the same chunk, "
        f"including {summary['repeat_changed_start_token']} at a changed offset."
    )
    print(f"Repeat review: {args.run_root / 'repeat-summary.json'}")


if __name__ == "__main__":
    main()
