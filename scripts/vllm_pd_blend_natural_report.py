#!/usr/bin/env python3
"""Verify a live MuSiQue CacheBlend run and attribute restored KV to evidence."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from vllm_pd_blend_receipts import _deltas, _metric_samples
from vllm_pd_blend_trace_join import (
    _RANGES,
    _SOURCE_TRACE,
    _evidence_overlaps,
    _span,
    _verify_scatter_bytes,
)

from evidencestore.pd_chunk_report import _CHUNK_STAGES, _REQUEST_TIMELINE, _chunk_ops, _prep
from evidencestore.pd_report import (
    _OUTER_TIMELINE_EVENTS,
    _reconcile,
    load_events,
    request_joins,
    summarize_request,
)
from evidencestore.retrieval import json_line, prompt_sha256, read_jsonl, sha256_file


_LAYOUTS = {"1p3d": (1, 3), "2p2d": (2, 2)}
_LIVE_EVENTS = tuple(f"{stage}_{edge}" for stage in (
    "embedding", "hnsw_search", "chunk_materialize", "prompt_render",
) for edge in ("start", "end"))
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_rows(path, rows):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.writelines(json_line(row) for row in rows)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _keyed(path):
    rows = {}
    for row in read_jsonl(path):
        key = row["request_key"]
        if not isinstance(key, str) or not key or key in rows:
            raise ValueError(f"Missing or repeated request key in {path}: {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"No request rows in {path}")
    return rows


def _identity(chunk):
    return chunk["doc_id"], chunk["chunk_id"], chunk["chunk_version"]


def _load_replay(run_dir, queries, manifest, trace_enabled):
    inputs = _keyed(queries)
    replay = run_dir / "client" / "musique"
    replay_manifest = _read_json(replay / "manifest.json")
    frozen_path = replay / "frozen.jsonl"
    count = len(inputs)
    if (
        replay_manifest["mode"] != "live_hnsw"
        or replay_manifest["prompt_format"] != "legacy-blend"
        or replay_manifest["input_sha256"] != sha256_file(queries)
        or replay_manifest["frozen_sha256"] != sha256_file(frozen_path)
        or replay_manifest["model"] != manifest["model"]
        or replay_manifest["tokenizer_revision"] != manifest["revision"]
        or replay_manifest.get("trace_chunk_stages", False) is not trace_enabled
        or not _DIGEST.fullmatch(replay_manifest["index"]["index_sha256"])
        or replay_manifest["concurrency"] != 1
        or tuple(replay_manifest[field] for field in ("requested", "succeeded", "failed", "frozen_rows"))
        != (count, count, 0, count)
    ):
        raise ValueError("Expected a complete live HNSW replay with legacy-blend prompts and pinned inputs")
    frozen = _keyed(frozen_path)
    responses = _keyed(replay / "responses.jsonl")
    if list(inputs) != list(frozen) or inputs.keys() != responses.keys():
        raise ValueError("Query, frozen prompt, and completed request keys differ")
    request_ids = set()
    for key, item in inputs.items():
        prompt, response = frozen[key], responses[key]
        tokens = prompt["prompt_token_ids"]
        if (
            item["dataset"] != "musique" or item.get("pd_route") or item.get("case")
            or prompt.get("pd_route") or prompt.get("case")
            or response.get("pd_route") or response.get("case")
            or response["label"] != "musique" or response["dataset"] != "musique"
            or response["success"] is not True or response["request_id"] in request_ids
            or type(response["completion_tokens"]) is not int or response["completion_tokens"] < 1
            or not _DIGEST.fullmatch(response["output_token_ids_sha256"] or "")
            or any(prompt.get(field) != value for field, value in item.items())
            or prompt["prompt_format"] != "legacy-blend" or not prompt["blend_separator_token_spans"]
            or not tokens or not prompt["evidence"] or not prompt["retrieval_hits"]
            or response["prompt_sha256"] != prompt["prompt_sha256"]
            or prompt_sha256(tokens) != prompt["prompt_sha256"]
        ):
            raise ValueError(f"Invalid natural MuSiQue replay receipt for {key}")
        for chunk in prompt["evidence"]:
            _, stop = _span(chunk, "start_token", "end_token")
            if stop > len(tokens) or any(not isinstance(v, str) or not v for v in _identity(chunk)):
                raise ValueError(f"Invalid versioned evidence span for {key}")
        request_ids.add(response["request_id"])
    return replay_manifest, frozen, responses


def _one_event(rows, name):
    result = [row for row in rows if row["event"] == name]
    if len(result) != 1:
        raise ValueError(f"Expected one {name} event, got {len(result)}")
    return result[0]


def _add_preparation(request, rows):
    preparation, shared = _prep(rows)
    request["prompt_build_shared_stages"] = shared
    request["timeline_events"] = [
        {name: row[name] for name in ("event", "ts_ns", "worker_id", "role") if name in row}
        for row in sorted(rows, key=lambda item: item["ts_ns"]) if row["event"] in _REQUEST_TIMELINE
    ]
    gpu_events = [row for row in rows if row["event"] in (
        "gpu_forward_start", "gpu_forward_end", "gpu_forward_duration",
    )]
    request["gpu_batch_events"] = [
        {name: row[name] for name in ("event", "ts_ns", "worker_id", "role", "batch_id", "step_id", "duration_ns")
         if name in row} for row in gpu_events
    ]
    launches = {(row["worker_id"], row["step_id"]) for row in gpu_events if row["event"] == "gpu_forward_start"}
    for chunk in request["evidence"]:
        stages = preparation[_identity(chunk)]
        retained = [row for row in stages if row["stage"] == "corpus_row_read"
                    or (row["retained"] and row["accepted_attempt"])]
        mapped = next((row for row in retained if row["stage"] == "prompt_chunk_offset_map"), {})
        if (Counter(row["stage"] for row in retained) != Counter(_CHUNK_STAGES)
                or (mapped.get("start_token"), mapped.get("end_token")) != (chunk["start_token"], chunk["end_token"])):
            raise ValueError(f"{request['request_key']} chunk {chunk['occurrence']} has incomplete preparation stages")
        operations, prefix = _chunk_ops(chunk, rows)
        if any((row["worker_id"], row["step_id"]) not in launches for row in operations):
            raise ValueError(f"{request['request_key']} chunk {chunk['occurrence']} has scheduled work with no GPU batch")
        chunk.update(preparation_stages=stages, scheduled_operations=operations, prefix_observations=prefix)


def _measure_requests(events, prefill, decode, frozen, responses):
    groups, unjoined = request_joins(events)
    by_api = {key[0]: (key, rows) for key, rows in groups.items()}
    if len(by_api) != len(groups) or unjoined or set(by_api) != {r["request_id"] for r in responses.values()}:
        raise ValueError(f"Unexpected native PD request IDs, attempts, or unjoined events: {dict(unjoined)}")
    requests = []
    by_native_id = defaultdict(list)
    previous_end = 0
    for request_key, prompt in frozen.items():
        response = responses[request_key]
        key, rows = by_api[response["request_id"]]
        native = summarize_request(key, rows, lmcache_enabled=True)
        p, d = native["prefill"], native["decode"]
        dispatch = _one_event(rows, "client_dispatch")
        render = _one_event(rows, "prompt_render_end")
        end = _one_event(rows, "client_response_end")
        evidence = [{k: v for k, v in chunk.items() if k not in {"occurrence", "token_work"}}
                    for chunk in native["evidence"]]
        if (
            p not in prefill or d not in decode or native["dataset"] != "musique" or native["case"]
            or dispatch["request_key"] != request_key or dispatch["label"] != "musique"
            or not previous_end <= render["ts_ns"] <= dispatch["ts_ns"] < end["ts_ns"]
            or render["prompt_format"] != "legacy-blend"
            or render["blend_separator_token_spans"] != prompt["blend_separator_token_spans"]
            or render["prompt_sha256"] != native["prompt_sha256"]
            or native["prompt_sha256"] != prompt["prompt_sha256"] or evidence != prompt["evidence"]
            or native["prompt_tokens"] != len(prompt["prompt_token_ids"])
            or native["output_token_ids_sha256"] != response["output_token_ids_sha256"]
            or native["completion_tokens"] != response["completion_tokens"]
            or any(native["events"].get(event) != 1 for event in (*_OUTER_TIMELINE_EVENTS, *_LIVE_EVENTS))
            or not {p, d}.issubset(native["queued_workers"])
            or native["unknown_nixl_completions"] or native["nixl_failed_attempts"]
            or native["full_nixl_handoff"]["status"] != "full" or native["completed_nixl_bytes"] <= 0
        ):
            raise ValueError(f"Incomplete live PD trace or mismatched prompt/output/NIXL receipt for {request_key}")
        native.update(request_key=request_key, dispatch_ns=dispatch["ts_ns"], response_end_ns=end["ts_ns"])
        _add_preparation(native, rows)
        previous_end = end["ts_ns"]
        for worker, role in ((p, "P"), (d, "D")):
            identities = {r["native_request_id"] for r in rows
                          if r["event"] == "engine_request_id" and r["worker_id"] == worker}
            if len(identities) != 1:
                raise ValueError(f"Missing or ambiguous native {role} request for {request_key}")
            by_native_id[identities.pop()].append((native, worker, role))
        requests.append(native)
    return requests, by_native_id


def _prior_occurrences(requests):
    occurrences = defaultdict(list)
    for request in requests:
        for chunk in request["evidence"]:
            occurrences[_identity(chunk)].append((request, chunk))
    for rows in occurrences.values():
        rows.sort(key=lambda pair: pair[0]["response_end_ns"], reverse=True)
    return occurrences


def _repeat_opportunities(frozen):
    positions = defaultdict(set)
    repeat_chunks = set()
    total = repeat = shifted = 0
    for prompt in frozen.values():
        for chunk in prompt["evidence"]:
            total += 1
            identity = _identity(chunk)
            previous = positions.get(identity)
            if previous:
                repeat += 1
                repeat_chunks.add(identity)
                shifted += previous != {chunk["start_token"]}
        for chunk in prompt["evidence"]:
            positions[_identity(chunk)].add(chunk["start_token"])
    return {"occurrences": total, "distinct_versioned_chunks": len(positions),
            "recurring_versioned_chunks": len(repeat_chunks), "repeat_occurrences": repeat,
            "repeats_with_a_different_prior_start_token": shifted}


def _prior_matches(request, hits, old, new, occurrences, frozen):
    target = frozen[request["request_key"]]["prompt_token_ids"]
    if new[1] > len(target):
        raise ValueError("CacheBlend scatter lies beyond the joined prompt")
    expected_tokens = target[new[0]:new[1]]
    displacement = new[0] - old[0]
    matched = []
    for hit in hits:
        for earlier, chunk in occurrences[_identity(hit)]:
            if earlier["response_end_ns"] > request["dispatch_ns"]:
                continue
            lo = max(hit["start_token"], chunk["start_token"] + displacement)
            hi = min(hit["end_token"], chunk["end_token"] + displacement)
            if lo >= hi:
                continue
            tokens = frozen[earlier["request_key"]]["prompt_token_ids"]
            if old[1] <= len(tokens) and tokens[old[0]:old[1]] == expected_tokens:
                matched.append({
                    "prior_request_key": earlier["request_key"],
                    "prior_request_id": earlier["request_id"],
                    "prior_occurrence": chunk["occurrence"],
                    "target_occurrence": hit["occurrence"],
                    "prior_evidence_overlap": [lo - displacement, hi - displacement],
                    "target_evidence_overlap": [lo, hi],
                })
                break
    return matched


def _inspect_lmcache_trace(layout, server, by_native_id, requests, frozen):
    trace_root = layout / "lmcache-trace"
    occurrences = _prior_occurrences(requests)
    server_id = server["environment"]["EVIDENCESTORE_RUN_ID"]
    joined, counts, proof = [], Counter(), None
    proven_requests, proven_chunks = set(), set()
    for path in sorted(trace_root.glob("*.jsonl")):
        filename = _SOURCE_TRACE.fullmatch(path.name)
        if not filename or int(filename[1]) != server["pid"]:
            raise ValueError(f"Unexpected LMCache trace source in {trace_root}: {path.name}")
        for event in read_jsonl(path):
            if (
                event.get("schema_version") != 1 or event.get("component") != "lmcache"
                or event.get("pid") != server["pid"] or event.get("run_id") != server_id
                or type(event.get("ts_ns")) is not int or type(event.get("wall_ns")) is not int
            ):
                raise ValueError(f"Unexpected LMCache event identity in {path}")
            name = event["event"]
            matches = by_native_id.get(event["native_request_id"], ())
            association = {"status": "ambiguous" if matches else "unmatched"}
            if len(matches) == 1:
                request, worker, role = matches[0]
                association = {"status": "matched", "request_key": request["request_key"],
                               "request_id": request["request_id"], "worker_id": worker, "role": role}
                _verify_scatter_bytes(event)
                by_field = {}
                completed_scatter = name == "lmcache_blend_scatter_complete" and event["status"] == "done"
                if (name == "lmcache_transfer_complete" and role == "P" and event.get("direction") == "d2h"
                        and event.get("status") == "done" and (event.get("completed_bytes") or 0) > 0):
                    counts.update(completed_prefill_stores=1, completed_store_bytes=event["completed_bytes"])
                for field in _RANGES.get(name, ()):
                    annotations = []
                    for span in event.get(field, ()):
                        new = _span(span, "start_token", "end_token")
                        attribution, hits = _evidence_overlaps(request["evidence"], *new)
                        annotation = {"attribution": attribution, "evidence_overlaps": hits}
                        byte_count = span.get("h2d_bytes", span.get("bytes"))
                        if attribution == "exclusive" and type(byte_count) is int:
                            annotation["exclusive_evidence_bytes"] = byte_count
                        if "old_start_token" in span:
                            old = _span(span, "old_start_token", "old_end_token")
                            if old[1] - old[0] != new[1] - new[0]:
                                raise ValueError("LMCache changed a matched range's token length")
                            if completed_scatter and role == "P" and old[0] != new[0] and byte_count > 0:
                                priors = _prior_matches(request, hits, old, new, occurrences, frozen)
                                if priors:
                                    annotation["prior_versioned_evidence_matches"] = priors
                                    counts.update(confirmed_shifted_ranges=1, confirmed_h2d_payload_bytes=byte_count)
                                    proven_requests.add(request["request_key"])
                                    for prior in priors:
                                        chunk = request["evidence"][prior["target_occurrence"]]
                                        proven_chunks.add(_identity(chunk))
                                    if proof is None:
                                        proof = {"request_key": request["request_key"], "old_range": old,
                                                 "new_range": new, "completed_h2d_payload_bytes": byte_count,
                                                 "prior_evidence": priors[0]}
                        annotations.append(annotation)
                    by_field[field] = annotations
                association["ranges"] = by_field
                counts[f"matched_{name}"] += 1
            else:
                counts[association["status"]] += 1
            joined.append({**event, "source_trace": path.name, "request_join": association})
    if proof is None:
        raise ValueError("No completed shifted Blend restore overlaps a proven earlier occurrence of the same versioned MuSiQue chunk")
    if not counts["completed_prefill_stores"]:
        raise ValueError("No completed LMCache store joined to a MuSiQue prefiller request")
    return joined, {**counts, "confirmed_requests": len(proven_requests),
                    "confirmed_versioned_chunks": len(proven_chunks), "example": proof}


def verify_natural_replay(layout, *, trace_enabled=None, queries=None):
    """Verify a natural Blend layout and write the review files beside its traces."""
    layout = Path(layout)
    queries = Path(queries) if queries is not None else layout / "inputs" / "queries.jsonl"
    run_dir = layout / "blend"
    manifest = _read_json(run_dir / "run_manifest.json")
    topology = manifest["topology"]
    actual_trace = manifest["engine_trace_enabled"]
    if trace_enabled is None:
        trace_enabled = actual_trace
    endpoints = _read_json(run_dir / "endpoints.json")
    prefill, decode = ({worker["id"] for worker in endpoints[role]} for role in ("prefill", "decode"))
    cache = manifest["lmcache"]
    if (
        _LAYOUTS.get(topology) != (len(prefill), len(decode)) or prefill & decode
        or type(trace_enabled) is not bool or actual_trace is not trace_enabled
        or manifest["prefill_mode"] != "blend" or not manifest["full_nixl_handoff"]
        or manifest["prefix_cache_by_role"] != {"P": "off", "D": "off"}
        or cache["server_engine"] != "blend" or cache["server_ownership"] != "external"
        or cache["decode_workers"] != [] or cache["url"] != endpoints["lmcache"]["url"]
    ):
        raise ValueError("Expected the external MP Blend prefiller layout and bare NIXL decoders")
    server = _read_json(layout / "lmcache-server.json")
    expected_gate = "1" if trace_enabled else "0"
    if any(server["environment"].get(gate) != expected_gate for gate in (
        "EVIDENCESTORE_TRACE_LMCACHE_BLEND", "EVIDENCESTORE_TRACE_LMCACHE_TRANSFERS",
    )):
        raise ValueError("LMCache server and launcher traces used different gates")
    replay, frozen, responses = _load_replay(run_dir, queries, manifest, trace_enabled)
    events = list(load_events(run_dir))
    if not trace_enabled and (events or any((layout / "lmcache-trace").glob("*.jsonl"))):
        raise ValueError("Trace-off run emitted detailed traces")
    nixl = _reconcile(run_dir, events, prefill | decode, trace_enabled)
    if nixl["status"] != ("matched" if trace_enabled else "trace_disabled"):
        raise ValueError(f"Native NIXL counters {nixl['status']}")
    deltas = {worker: row["prometheus"] for worker, row in nixl["workers"].items()}
    total_bytes = sum(deltas[worker]["sum"] for worker in decode)
    transfer_count = sum(deltas[worker]["count"] for worker in decode)
    if any(any(deltas[worker].values()) for worker in prefill) or total_bytes <= 0 or transfer_count < len(frozen):
        raise ValueError("Expected positive D-side NIXL bytes and at least one completed read per request in aggregate")

    before, after = (_metric_samples((run_dir / f"lmcache-metrics-{name}.prom").read_bytes())
                     for name in ("before", "after"))
    lmcache = _deltas(before, after)
    if not all(lmcache[key] > 0 for key in (
        "fingerprints_registered", "non_prefix_hit_tokens", "scatter_shifted_chunks",
    )) or lmcache["retrieve_completed"] <= lmcache["retrieve_failures"]:
        raise ValueError("LMCache counters must show new fingerprints, non-prefix hits, and completed shifted restore activity")

    proof = None
    if trace_enabled:
        requests, by_native_id = _measure_requests(events, prefill, decode, frozen, responses)
        if len(requests) >= len(decode) and (
            {request["prefill"] for request in requests} != prefill
            or {request["decode"] for request in requests} != decode
        ):
            raise ValueError("Natural MuSiQue replay did not use every configured worker")
        if sum(request["completed_nixl_bytes"] for request in requests) != total_bytes:
            raise ValueError("Per-request NIXL payload bytes disagree with decoder counters")
        joined, proof = _inspect_lmcache_trace(layout, server, by_native_id, requests, frozen)
        _write_rows(layout / "lmcache-request-trace.jsonl", sorted(joined, key=lambda row: row["ts_ns"]))
    else:
        requests = [{key: response[key] for key in (
            "request_id", "request_key", "prompt_sha256", "output_token_ids_sha256", "completion_tokens",
        )} for response in responses.values()]
    summary = {
        "schema_version": 1, "status": "passed", "dataset": "musique", "topology": topology,
        "blend_layout": f"{len(prefill)}B/{len(decode)}D",
        "requests": len(requests), "concurrency": replay["concurrency"], "engine_trace_enabled": trace_enabled,
        "prompt_format": "legacy-blend", "queries_sha256": replay["input_sha256"],
        "index_sha256": replay["index"]["index_sha256"], "model": manifest["model"], "revision": manifest["revision"],
        "nixl": {**nixl, "completed_bytes": total_bytes, "completed_transfers": transfer_count,
                 "requests_with_verified_full_handoff": len(requests) if trace_enabled else None},
        "lmcache_counter_delta": lmcache, "natural_evidence_repeats": _repeat_opportunities(frozen),
        "shifted_repeated_evidence": proof,
        "shifted_repeated_evidence_status": "verified" if proof else "unavailable_trace_disabled",
        "retained_chunk_preparation_status": "verified" if trace_enabled else "unavailable_trace_disabled",
        "private_blend_repair_token_positions": None,
    }
    _write_rows(layout / "natural-blend-requests.jsonl", requests)
    _write_rows(layout / "natural-blend-summary.json", [summary])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--trace", choices=("on", "off"))
    args = parser.parse_args()
    try:
        summary = verify_natural_replay(args.layout_dir, queries=args.queries,
                                        trace_enabled=None if args.trace is None else args.trace == "on")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Natural MuSiQue Blend verification failed: {error}\n")
    print(f"Verified {summary['requests']} MuSiQue requests in {summary['topology']}.")
    print(f"Run summary: {args.layout_dir / 'natural-blend-summary.json'}")


if __name__ == "__main__":
    main()
