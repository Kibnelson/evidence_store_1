#!/usr/bin/env python3
"""Check live retrieval, in-process Blend reuse, and native NIXL handoffs."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidencestore.pd_chunk_report import _CHUNK_STAGES, _key, _prep
from evidencestore.pd_report import (
    _OUTER_TIMELINE_EVENTS, _attribute_bytes, _reconcile, load_events, request_joins,
)
from evidencestore.retrieval import json_line, prompt_sha256, read_jsonl, sha256_file


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _by_key(path):
    rows = list(read_jsonl(path))
    keyed = {row["request_key"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate request keys in {path}")
    return keyed


def _read_replay(run_dir, label, input_file, count, *, live, trace):
    output = run_dir / "client" / label
    inputs = _by_key(input_file)
    prompts = _by_key(output / "frozen.jsonl")
    responses = _by_key(output / "responses.jsonl")
    manifest = _json(output / "manifest.json")
    if (
        len(inputs) != count or list(inputs) != list(prompts) or prompts.keys() != responses.keys()
        or manifest["mode"] != ("live_hnsw" if live else "frozen")
        or manifest["concurrency"] != 1 or manifest["prompt_format"] != "legacy-blend"
        or manifest.get("trace_chunk_stages", False) is not (live and trace)
        or manifest["input_sha256"] != sha256_file(input_file)
        or manifest["frozen_sha256"] != sha256_file(output / "frozen.jsonl")
        or (manifest["requested"], manifest["succeeded"], manifest["failed"], manifest["frozen_rows"])
        != (count, count, 0, count)
    ):
        raise ValueError(f"Incomplete or mismatched {label} replay")
    for key, prompt in prompts.items():
        response = responses[key]
        tokens = prompt["prompt_token_ids"]
        spans = prompt["evidence"]
        routed = bool(prompt.get("pd_route"))
        if (
            response["success"] is not True or response["completion_tokens"] < 1
            or not response["output_token_ids_sha256"] or response["label"] != label
            or prompt_sha256(tokens) != prompt["prompt_sha256"]
            or prompt["prompt_sha256"] != response["prompt_sha256"]
            or not spans or not prompt["blend_separator_token_spans"] or routed is live
            or any(not 0 <= span["start_token"] < span["end_token"] <= len(tokens) for span in spans)
        ):
            raise ValueError(f"Invalid {label} prompt/response for {key}")
    return [(label, prompt, responses[key]) for key, prompt in prompts.items()]


def _one(events, name):
    rows = events[name]
    if len(rows) != 1:
        raise ValueError(f"Expected one {name}, got {len(rows)}")
    return rows[0]


def _full_transfer(events, prompt_tokens, prefill, decode):
    handoff = _one(events, "engine_kv_handoff_ready")
    p_response = _one(events, "gateway_prefill_response")
    submission = _one(events, "nixl_read_submit")
    completion = _one(events, "nixl_read_complete")
    block_size = completion["block_size"]
    blocks = handoff["remote_block_ids"]
    actual = completion["completed_bytes"]
    if (
        events["nixl_read_failed"] or handoff["worker_id"] != prefill
        or completion["worker_id"] != decode or p_response["remote_request_id"] != handoff["native_request_id"]
        or handoff["native_request_id"] != completion["remote_request_id"]
        or handoff["remote_token_count"] != prompt_tokens or completion["status"] != "done"
        or completion["transfer_id"] != submission["transfer_id"] or submission["ts_ns"] > completion["ts_ns"]
        or not (block_size > 0 and completion["remote_block_size"] == block_size)
        or completion["remote_physical_per_logical"] != 1 or len(blocks) != 1
        or completion["remote_block_ids"] != blocks
    ):
        raise ValueError("Native NIXL producer, reader, or block geometry differs")
    count = (prompt_tokens + block_size - 1) // block_size
    if (len(blocks[0]) != count or len(set(blocks[0])) != count
            or completion["local_block_counts"] != [count] or actual <= 0 or actual % count):
        raise ValueError("Completed native NIXL READ lacks full prompt block coverage")
    return {**completion, "position_start": 0, "position_end": count * block_size,
            "valid_position_end": prompt_tokens, "transferred_blocks": count}


def _overlap(lo, hi, start, end):
    return max(0, min(hi, end) - max(lo, start))


def _transfers(events, prompt_tokens, prefill, dispatch_ns, prefill_end_ns):
    movements = []
    for direction in ("d2h", "h2d"):
        for receipt in events[f"legacy_lmcache_{direction}_complete"]:
            if (receipt["worker_id"] != prefill or receipt["layers"] <= 0
                    or not dispatch_ns <= receipt["started_ns"] <= receipt["ts_ns"] <= prefill_end_ns
                    or direction == "h2d" and receipt["backend"] != "LocalCPUBackend"):
                raise ValueError("Invalid in-process LMCache movement binding")
            for span in receipt["mapped_ranges"]:
                if (not 0 <= span["start_token"] < span["end_token"] <= prompt_tokens
                        or span["bytes"] <= 0 or direction == "h2d" and (
                            type(span.get("old_start_token")) is not int or span["old_start_token"] < 0
                        )):
                    raise ValueError("In-process LMCache movement lacks a native position or byte count")
                movements.append({"direction": direction, "worker_id": prefill, "layers": receipt["layers"],
                                  "ts_ns": receipt["ts_ns"], "started_ns": receipt["started_ns"], **span})
    return movements


def _prior_restore(chunk, movement, tokens, prior):
    if movement["direction"] != "h2d":
        return None
    start, end, old = (movement[name] for name in ("start_token", "end_token", "old_start_token"))
    old_end = old + end - start
    for earlier, evidence in reversed(prior):
        if (
            tokens[start:end] == earlier["prompt_token_ids"][old:old_end]
            and _overlap(evidence["start_token"], evidence["end_token"], old, old_end)
            and any(row["direction"] == "d2h" and (row["start_token"], row["end_token"]) == (old, old_end)
                    for row in earlier["movements"])
        ):
            return {
                "prior_request_id": earlier["request_id"], "prior_request_key": earlier["request_key"],
                "prior_label": earlier["label"], "prior_start_token": old, "prior_end_token": old_end,
                "shifted": old != start and evidence["start_token"] != chunk["start_token"],
            }
    return None


def _chunk_record(chunk, occurrence, request, bytes_by_chunk, stages, events, by_worker, by_identity):
    identity = _key(chunk)
    start, end = chunk["start_token"], chunk["end_token"]
    if request["label"] == "musique":
        retained = [row for row in stages[identity] if row["stage"] == "corpus_row_read"
                    or (row["retained"] and row["accepted_attempt"])]
        mapped = next((row for row in retained if row["stage"] == "prompt_chunk_offset_map"), {})
        if (Counter(row["stage"] for row in retained) != Counter(_CHUNK_STAGES)
                or (mapped.get("start_token"), mapped.get("end_token")) != (start, end)):
            raise ValueError(f"Missing preparation stages for {request['request_key']}: {identity}")
    scheduled = Counter()
    for event in events["engine_token_work"]:
        scheduled[event["role"]] += _overlap(start, end, event["position_start"], event["position_end"])
    blend = [
        [sum(_overlap(start, end, lo, hi) for lo, hi in layer) for layer in event["computed_ranges_by_layer"]]
        for event in events["legacy_blend_complete"]
    ]
    prior = by_worker[(request["prefill"], identity)]
    movements = []
    for movement in request["movements"]:
        lo = max(start, movement["start_token"])
        hi = min(end, movement["end_token"])
        if lo < hi:
            previous = _prior_restore(chunk, movement, request["prompt_token_ids"], prior)
            movements.append({**movement, "evidence_overlap": [lo, hi], "source_match": previous})
    return {
        **chunk, "request_id": request["request_id"], "request_key": request["request_key"],
        "label": request["label"], "occurrence": occurrence, "prefill": request["prefill"],
        "decode": request["decode"], "previously_seen": bool(by_identity[identity]),
        "previously_seen_on_prefiller": bool(prior), "preparation_stages": stages[identity],
        "scheduled_tokens_by_role": dict(scheduled), "blend_computed_tokens_by_call_and_layer": blend,
        "native_nixl_exclusive_bytes": bytes_by_chunk.get(occurrence, 0), "lmcache_movements": movements,
    }


def _validate_layerwise(events, movements, request_tokens, prefill):
    calls = events["legacy_blend_complete"]
    restores = [move for move in movements if move["direction"] == "h2d"]
    if bool(calls) != bool(restores):
        raise ValueError("Each observed Blend call needs a completed H2D receipt")
    for call in calls:
        count = call["input_token_count"]
        layers = call["computed_ranges_by_layer"]
        if (call["worker_id"] != prefill or not 0 < count <= request_tokens
                or not layers or any(move["layers"] != len(layers) for move in restores)):
            raise ValueError("Completed Blend layer work differs from the H2D receipt")
        for spans in layers:
            end = 0
            if not spans:
                raise ValueError("A completed Blend layer has no computed positions")
            for lo, hi in spans:
                if not end <= lo < hi <= count:
                    raise ValueError("Blend layer position ranges overlap or exceed the input")
                end = hi


def _measure(replays, groups, prefill_workers, decode_workers):
    grouped = {key[0]: rows for key, rows in groups.items()}
    if len(grouped) != len(groups) or set(grouped) != {response["request_id"] for _, _, response in replays}:
        raise ValueError("Request attempts and response receipts differ")
    by_worker, by_identity = defaultdict(list), defaultdict(list)
    all_requests, all_chunks, previous_end = [], [], 0
    controlled_passed = set()
    for label, prompt, response in replays:
        rows = grouped[response["request_id"]]
        events = defaultdict(list)
        for row in rows:
            events[row["event"]].append(row)
        for name in _OUTER_TIMELINE_EVENTS:
            _one(events, name)
        dispatch, end = (_one(events, name) for name in ("client_dispatch", "client_response_end"))
        gateway = _one(events, "gateway_prompt_ready")
        p, d = (_one(events, f"gateway_route_{role}")["worker_id"] for role in ("prefill", "decode"))
        tokens = prompt["prompt_token_ids"]
        if (
            p not in prefill_workers or d not in decode_workers or dispatch["prompt_sha256"] != prompt["prompt_sha256"]
            or gateway["evidence"] != prompt["evidence"] or gateway["prompt_tokens"] != len(tokens)
            or end["output_token_ids_sha256"] != response["output_token_ids_sha256"]
            or not previous_end <= dispatch["ts_ns"] < end["ts_ns"]
            or any(not any(row["worker_id"] == worker for row in events["gpu_forward_duration"]) for worker in (p, d))
            or label == "legacy-probes" and prompt["pd_route"] != {"prefill": p, "decode": d}
        ):
            raise ValueError(f"Missing or mismatched request stages for {prompt['request_key']}")
        previous_end = end["ts_ns"]
        transfer = _full_transfer(events, len(tokens), p, d)
        evidence = [{**chunk, "occurrence": i} for i, chunk in enumerate(prompt["evidence"])]
        attribution = _attribute_bytes(evidence, [transfer])
        movements = _transfers(events, len(tokens), p, dispatch["ts_ns"],
                               _one(events, "gateway_prefill_response")["ts_ns"])
        _validate_layerwise(events, movements, len(tokens), p)
        preparation, shared = _prep(rows)
        request = {"label": label, "request_key": prompt["request_key"], "request_id": response["request_id"],
                   "prefill": p, "decode": d, "prompt_sha256": prompt["prompt_sha256"],
                   "prompt_token_ids": tokens, "movements": movements}
        chunks = [_chunk_record(chunk, i, request, attribution["by_chunk_occurrence"],
                                preparation, events, by_worker, by_identity)
                  for i, chunk in enumerate(prompt["evidence"])]
        for chunk in chunks:
            identity = _key(chunk)
            by_identity[identity].append(request["request_id"])
            by_worker[(p, identity)].append((request, chunk))
            if label == "legacy-probes" and prompt["case"] == "reordered" and any(
                hit["source_match"] and hit["source_match"]["shifted"]
                and hit["source_match"]["prior_label"] == "legacy-probes" for hit in chunk["lmcache_movements"]
            ):
                controlled_passed.add(p)
        all_chunks.extend(chunks)
        all_requests.append({
            **{key: value for key, value in request.items() if key != "prompt_token_ids"},
            "prompt_tokens": len(tokens), "completed_nixl_bytes": transfer["completed_bytes"],
            "nixl_transfer": transfer, "nixl_bytes_by_evidence": attribution,
            "shared_prompt_build_stages": shared,
            "timeline": [{key: row[key] for key in ("event", "ts_ns", "worker_id", "role", "batch_id", "duration_ns")
                          if key in row} for row in sorted(rows, key=lambda row: row["ts_ns"])],
        })
    if controlled_passed != prefill_workers:
        raise ValueError(f"Controlled probes did not prove moved chunk reuse on {sorted(prefill_workers - controlled_passed)}")
    return all_requests, all_chunks


def report(run_dir, *, trace):
    endings = _json(run_dir / "endpoints.json")
    prefill, decode = ({worker["id"] for worker in endings[side]} for side in ("prefill", "decode"))
    replays = _read_replay(run_dir, "musique", run_dir / "inputs/queries.jsonl", 10, live=True, trace=trace)
    probes = _read_replay(run_dir, "legacy-probes", run_dir / "inputs/controlled.jsonl",
                          len(prefill) * 2, live=False, trace=trace)
    cases = Counter((prompt["pd_route"]["prefill"], prompt["case"]) for _, prompt, _ in probes)
    if cases != Counter((worker, condition) for worker in prefill for condition in ("cold", "reordered")):
        raise ValueError("Expected one cold and one reordered probe on each Blend worker")
    replays.extend(probes)
    events = list(load_events(run_dir))
    if not trace and events:
        raise ValueError("Trace-off run wrote detailed trace records")
    reconciliation = _reconcile(run_dir, events, prefill | decode, trace)
    if reconciliation["status"] != ("matched" if trace else "trace_disabled"):
        raise ValueError(f"NIXL counter reconciliation: {reconciliation['status']}")
    counters = {worker: row["prometheus"] for worker, row in reconciliation["workers"].items()}
    total_reads = sum(row["count"] for row in counters.values())
    total_bytes = sum(row["sum"] for row in counters.values())
    if total_reads != len(replays) or total_bytes <= 0 or any(counters[p]["count"] for p in prefill):
        raise ValueError("Expected one native NIXL READ per completed request on the decode side")
    summary = {
        "status": "passed", "trace_enabled": trace, "cache_backend": "in_process_local_cpu_per_prefiller",
        "prefill_workers": sorted(prefill), "decode_workers": sorted(decode),
        "natural_requests": 10, "controlled_requests": len(replays) - 10,
        "completed_nixl_reads": total_reads, "completed_nixl_bytes": total_bytes,
        "nixl_reconciliation": reconciliation,
    }
    if trace:
        groups, unjoined = request_joins(events)
        if unjoined:
            raise ValueError(f"Unjoined native PD events: {dict(unjoined)}")
        requests, chunks = _measure(replays, groups, prefill, decode)
        natural = [row for row in chunks if row["label"] == "musique"]
        moved = defaultdict(Counter)
        for request in requests:
            moved[request["label"]]["nixl_p2d"] += request["completed_nixl_bytes"]
            for transfer in request["movements"]:
                moved[request["label"]][f"lmcache_{transfer['direction']}"] += transfer["bytes"]
        summary.update(
            natural_evidence_occurrences=len(natural),
            natural_repeat_occurrences=sum(row["previously_seen"] for row in natural),
            natural_same_worker_repeat_occurrences=sum(row["previously_seen_on_prefiller"] for row in natural),
            natural_shifted_restore_occurrences=sum(any(
                move["source_match"] and move["source_match"]["shifted"] for move in row["lmcache_movements"]
            ) for row in natural),
            controlled_reuse_verified_on=sorted(prefill),
            transfer_payload_bytes_by_label=dict(moved),
            measurement_scope={
                "model_gpu_duration": "CUDA event duration for each ordinary model forward batch",
                "blend_computation": (
                    "Token positions that reached each Blend layer's attention and MLP, mapped to evidence spans. "
                    "The check-layer projections and scoring can visit more positions."
                ),
                "lmcache_movement_duration": "Host interval through the native completion fence",
                "trace_only_gpu_synchronization": True,
            },
        )
        for name, rows in (("legacy-requests.jsonl", requests), ("legacy-chunks.jsonl", chunks)):
            (run_dir / name).write_text("".join(json_line(row) for row in rows), encoding="utf-8")
    (run_dir / "legacy-blend-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trace", choices=("on", "off"), required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.run_dir, trace=args.trace == "on"), indent=2))


if __name__ == "__main__":
    main()
