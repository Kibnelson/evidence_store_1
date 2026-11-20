#!/usr/bin/env python3
"""Check live retrieval, in-process Blend reuse, and native NIXL handoffs."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidencestore.pd_chunk_report import _CHUNK_STAGES, _key, _prep
from evidencestore.pd_dataset_report import _distribution
from evidencestore.pd_report import (
    _OUTER_TIMELINE_EVENTS, _attribute_bytes, _reconcile, load_events, request_joins,
)
from evidencestore.retrieval import json_line, prompt_sha256, read_jsonl, sha256_file


_PROMPT_POLICIES = {
    "seen-first": "legacy-blend",
    "previous-turn-first": "legacy-blend",
    "openclaw": "legacy-blend-openclaw",
}
_LATENCY_SPANS = {
    "prompt_render": ("prompt_render_start", "prompt_render_end"),
    "client_ttft": ("client_request_start", "client_first_token"),
    "dispatch_to_first_token": ("client_dispatch", "client_first_token"),
    "prefill_http": ("gateway_dispatch_prefill", "gateway_prefill_response"),
}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _by_key(path):
    rows = list(read_jsonl(path))
    keyed = {row["request_key"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate request keys in {path}")
    return keyed


def _read_replay(run_dir, label, input_file, count, *, live, trace,
                 prompt_format="legacy-blend", evidence_order=None):
    output = run_dir / "client" / label
    inputs = _by_key(input_file)
    prompts = _by_key(output / "frozen.jsonl")
    responses = _by_key(output / "responses.jsonl")
    manifest = _json(output / "manifest.json")
    if (
        len(inputs) != count or list(inputs) != list(prompts) or prompts.keys() != responses.keys()
        or manifest["mode"] != ("live_hnsw" if live else "frozen")
        or manifest["concurrency"] != 1 or manifest["prompt_format"] != prompt_format
        or manifest.get("evidence_order") != evidence_order
        or manifest.get("trace_chunk_stages", False) is not (live and trace)
        or manifest["input_sha256"] != sha256_file(input_file)
        or manifest["frozen_sha256"] != sha256_file(output / "frozen.jsonl")
        or (manifest["requested"], manifest["succeeded"], manifest["failed"], manifest["frozen_rows"])
        != (count, count, 0, count)
    ):
        raise ValueError(f"Incomplete or mismatched {label} replay")
    default_prompt_format = "legacy-blend" if evidence_order is None else None
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
            or prompt.get("prompt_format", default_prompt_format) != prompt_format
            or prompt.get("evidence_order") != evidence_order
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
    if request["label"] != "legacy-probes":
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


def _conversation_reuse(requests, chunks, live_replays):
    turns = {response["request_id"]: {"conversation_id": prompt["conversation_id"], "turn": prompt["turn"]}
             for _, prompt, response in live_replays}
    scopes = ("within_conversation", "across_conversations")
    counts = {scope: dict.fromkeys(("repeat_occurrences", "verified_restore_occurrences",
                                   "shifted_restore_occurrences"), 0) for scope in scopes}
    seen = defaultdict(set)
    for request in requests:
        if request["request_id"] in turns:
            request.update(turns[request["request_id"]])
    for chunk in chunks:
        request_id = chunk["request_id"]
        if request_id not in turns:
            continue
        chunk.update(turns[request_id])
        conversation = chunk["conversation_id"]
        prior = seen[_key(chunk)]
        chunk["previously_seen_in_conversation"] = conversation in prior
        chunk["previously_seen_across_conversations"] = bool(prior - {conversation})
        counts[scopes[0]]["repeat_occurrences"] += chunk["previously_seen_in_conversation"]
        counts[scopes[1]]["repeat_occurrences"] += chunk["previously_seen_across_conversations"]
        prior.add(conversation)
        restored, shifted = set(), set()
        for move in chunk["lmcache_movements"]:
            source = move["source_match"]
            if source:
                previous = turns[source["prior_request_id"]]["conversation_id"]
                scope = scopes[0 if previous == conversation else 1]
                source.update(prior_conversation_id=previous, conversation_scope=scope)
                restored.add(scope)
                if source["shifted"]:
                    shifted.add(scope)
        for scope in restored:
            counts[scope]["verified_restore_occurrences"] += 1
        for scope in shifted:
            counts[scope]["shifted_restore_occurrences"] += 1
    return counts


def _selection_agreement(replays, reference):
    fields = ("doc_id", "chunk_id", "chunk_version", "rank", "start_token", "end_token")
    expected = {row["request_key"]: row for row in reference}
    prompts = set()
    evidence = set()
    for _, prompt, _ in replays:
        key = prompt["request_key"]
        target = expected[key]
        if prompt["prompt_sha256"] == target["prompt_sha256"]:
            prompts.add(key)
        observed = [tuple(chunk[field] for field in fields) for chunk in prompt["evidence"]]
        ranked = [tuple(chunk[field] for field in fields) for chunk in target["evidence"]]
        if observed == ranked:
            evidence.add(key)
    return {"matching_prompt_hashes": len(prompts), "matching_ranked_evidence_and_spans": len(evidence),
            "mismatching_request_keys": sorted(expected.keys() - (prompts & evidence))}


def _retrieval_agreement(replays, reference):
    fields = ("doc_id", "chunk_id", "chunk_version", "rank")

    def ranked_chunks(chunks):
        return Counter(tuple(chunk[field] for field in fields) for chunk in chunks)

    expected = {row["request_key"]: ranked_chunks(row["evidence"]) for row in reference}
    retrieved, kept = set(), set()
    for _, prompt, _ in replays:
        key = prompt["request_key"]
        target = expected[key]
        if ranked_chunks(prompt["retrieval_hits"]) == target:
            retrieved.add(key)
        if ranked_chunks(prompt["evidence"]) == target:
            kept.add(key)
    return {"matching_retrieval_identity_and_ranks": len(retrieved),
            "matching_kept_identity_and_ranks": len(kept),
            "mismatching_request_keys": sorted(expected.keys() - (retrieved & kept))}


def _verify_previous_turn_order(replays):
    first_seen = {}
    previous_turns = {}
    for _, prompt, _ in replays:
        conversation = prompt.get("conversation_id")
        if not conversation:
            raise ValueError(f"Missing conversation for {prompt['request_key']}")
        previous = previous_turns.get(conversation, {})
        evidence = prompt["evidence"]

        def order(chunk):
            identity = _key(chunk)
            if identity in previous:
                return 0, previous[identity]
            if identity in first_seen:
                return 1, first_seen[identity]
            return 2, chunk["rank"]

        if evidence != sorted(evidence, key=order):
            raise ValueError(f"Previous-turn-first evidence order differs for {prompt['request_key']}")
        for chunk in evidence:
            first_seen.setdefault(_key(chunk), len(first_seen))
        previous_turns[conversation] = {_key(chunk): index for index, chunk in enumerate(evidence)}


def _prompt_settings(sample):
    if "prompt_policy" not in sample:
        if sample.get("prompt_format", "legacy-blend") != "legacy-blend" or "evidence_order" in sample:
            raise ValueError("Natural replay settings require a prompt policy")
        return {}

    policy = sample["prompt_policy"]
    prompt_format = _PROMPT_POLICIES.get(policy)
    if prompt_format is None or sample.get("prompt_format") != prompt_format:
        raise ValueError("Natural replay prompt format differs from its policy")
    settings = {"prompt_policy": policy, "prompt_format": prompt_format}
    if policy in ("seen-first", "previous-turn-first"):
        if sample.get("evidence_order") != policy:
            raise ValueError(f"The {policy} policy requires {policy} evidence order")
        settings["evidence_order"] = policy
    elif "evidence_order" in sample:
        raise ValueError("The openclaw policy requires retrieval-ranked evidence")
    return settings


def report(run_dir, *, trace):
    endings = _json(run_dir / "endpoints.json")
    prefill, decode = ({worker["id"] for worker in endings[side]} for side in ("prefill", "decode"))
    selection_path = run_dir / "inputs/manifest.json"
    sample = _json(selection_path) if selection_path.is_file() else {}
    settings = _prompt_settings(sample)
    selected = bool(sample.get("workload"))
    label = sample["dataset"] if selected else "musique"
    queries = run_dir / "inputs/queries.jsonl"
    expected = len(sample["query_keys"]) if selected else 10
    natural_replays = _read_replay(
        run_dir, label, queries, expected, live=True, trace=trace,
        prompt_format=settings.get("prompt_format", "legacy-blend"),
        evidence_order=settings.get("evidence_order"),
    )
    if selected and (sample["queries_sha256"] != sha256_file(queries)
                     or sample["query_keys"] != [prompt["request_key"] for _, prompt, _ in natural_replays]):
        raise ValueError("Natural replay differs from the selected workload")
    if settings.get("evidence_order") == "previous-turn-first":
        _verify_previous_turn_order(natural_replays)
    probes = _read_replay(run_dir, "legacy-probes", run_dir / "inputs/controlled.jsonl",
                          len(prefill) * 2, live=False, trace=trace)
    cases = Counter((prompt["pd_route"]["prefill"], prompt["case"]) for _, prompt, _ in probes)
    if cases != Counter((worker, condition) for worker in prefill for condition in ("cold", "reordered")):
        raise ValueError("Expected one cold and one reordered probe on each Blend worker")
    replays = natural_replays + probes
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
        "natural_requests": expected, "controlled_requests": len(probes),
        "completed_nixl_reads": total_reads, "completed_nixl_bytes": total_bytes,
        "nixl_reconciliation": reconciliation,
    }
    summary.update(settings)
    if selected:
        selection_agreement = _retrieval_agreement if settings else _selection_agreement
        summary.update(workload=sample["workload"], natural_label=label, domain=sample["domain"],
                       conversation_turn_counts=sample["conversation_turn_counts"],
                       selection_agreement=selection_agreement(natural_replays, sample["selection_metrics"]["per_request"]))
    if trace:
        groups, unjoined = request_joins(events)
        if unjoined:
            raise ValueError(f"Unjoined native PD events: {dict(unjoined)}")
        requests, chunks = _measure(replays, groups, prefill, decode)
        natural = [row for row in chunks if row["label"] == label]
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
        if settings:
            summary["natural_verified_restore_occurrences"] = sum(
                any(move["source_match"] for move in row["lmcache_movements"]) for row in natural
            )
        if label == "mtrag":
            summary["conversation_reuse"] = _conversation_reuse(requests, chunks, natural_replays)
        for name, rows in (("legacy-requests.jsonl", requests), ("legacy-chunks.jsonl", chunks)):
            (run_dir / name).write_text("".join(json_line(row) for row in rows), encoding="utf-8")
    (run_dir / "legacy-blend-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _comparison_rows(cohort, name):
    path = cohort / f"{name}.jsonl"
    if path.is_file():
        return list(read_jsonl(path))
    with gzip.open(path.with_suffix(".jsonl.gz"), "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _evidence_totals(chunks):
    verified = shifted = 0
    bytes_by_direction = Counter()
    for chunk in chunks:
        restores = []
        for move in chunk["lmcache_movements"]:
            span = [move["start_token"], move["end_token"]]
            if move["evidence_overlap"] != span or not (
                chunk["start_token"] <= span[0] < span[1] <= chunk["end_token"]
            ):
                raise ValueError(f"LMCache movement crosses evidence boundaries in {chunk['request_key']}")
            bytes_by_direction[move["direction"]] += move["bytes"]
            if move["direction"] == "h2d" and move["source_match"]:
                restores.append(move["source_match"])
        verified += bool(restores)
        shifted += any(source["shifted"] for source in restores)
    return {
        "occurrences": len(chunks),
        "verified_h2d_occurrences": verified,
        "shifted_h2d_occurrences": shifted,
        "ordinary_prefill_scheduled_tokens": sum(
            chunk["scheduled_tokens_by_role"].get("P", 0) for chunk in chunks
        ),
        "blend_attention_mlp_token_layer_positions": sum(
            sum(layers) for chunk in chunks for layers in chunk["blend_computed_tokens_by_call_and_layer"]
        ),
        "lmcache_completed_h2d_bytes": bytes_by_direction["h2d"],
        "lmcache_completed_d2h_bytes": bytes_by_direction["d2h"],
        "nixl_evidence_exclusive_bytes": sum(chunk["native_nixl_exclusive_bytes"] for chunk in chunks),
    }


def _request_latencies(requests):
    latencies = defaultdict(list)
    for request in requests:
        events = defaultdict(list)
        for row in request["timeline"]:
            events[row["event"]].append(row)
        for name, (start, end) in _LATENCY_SPANS.items():
            elapsed = _one(events, end)["ts_ns"] - _one(events, start)["ts_ns"]
            if elapsed < 0:
                raise ValueError(f"Reversed {name} timestamps for {request['request_key']}")
            latencies[name].append(elapsed / 1e6)
    return latencies


def _comparison_cohort(cohort):
    summary = _json(cohort / "legacy-blend-summary.json")
    if summary.get("status") != "passed" or summary.get("trace_enabled") is not True or not summary.get("workload"):
        raise ValueError(f"Expected a passing, trace-on selected workload in {cohort}")
    if summary["selection_agreement"]["mismatching_request_keys"]:
        raise ValueError(f"Evidence selection differs from the workload in {cohort}")

    provenance = cohort / "provenance.json"
    launch = _json(provenance if provenance.is_file() else cohort / "run_manifest.json")
    sample = launch["sample"] if provenance.is_file() else _json(cohort / "inputs/manifest.json")
    label = summary["natural_label"]
    requests = [row for row in _comparison_rows(cohort, "legacy-requests") if row["label"] == label]
    chunks = [row for row in _comparison_rows(cohort, "legacy-chunks") if row["label"] == label]
    request_keys = [row["request_key"] for row in requests]
    if (len(set(request_keys)) != len(request_keys) or len(requests) != summary["natural_requests"]
            or len(chunks) != summary["natural_evidence_occurrences"]
            or {row["request_key"] for row in chunks} != set(request_keys)):
        raise ValueError(f"Natural request and evidence receipts differ in {cohort}")

    fields = ("request_key", "doc_id", "chunk_id", "chunk_version", "rank")
    ranked_evidence = Counter(tuple(chunk[field] for field in fields) for chunk in chunks)
    workload = {field: summary[field] for field in (
        "workload", "domain", "prefill_workers", "decode_workers",
    )}
    workload.update(dataset=label, topology=launch["topology"], model=launch["model"], revision=launch["revision"],
                    corpus_sha256=sample["corpus_sha256"], index_sha256=sample["index_sha256"],
                    selection_sha256=sample["selection_sha256"])
    moves = Counter()
    for row in requests:
        for move in row["movements"]:
            moves[move["direction"]] += move["bytes"]
    latencies = _request_latencies(requests)
    return workload, (request_keys, ranked_evidence), latencies, {
        "prompt_policy": summary.get("prompt_policy", "default"),
        "latency_ms": {name: _distribution(values) for name, values in latencies.items()},
        "natural_requests": {
            "count": len(requests),
            "lmcache_completed_h2d_bytes": moves["h2d"],
            "lmcache_completed_d2h_bytes": moves["d2h"],
            "nixl_completed_bytes": sum(row["completed_nixl_bytes"] for row in requests),
        },
        "natural_evidence": _evidence_totals(chunks),
        "repeated_evidence": _evidence_totals([chunk for chunk in chunks if chunk["previously_seen"]]),
    }


def compare(default_cohort, gated_cohort):
    workload, default_selection, default_latencies, default = _comparison_cohort(default_cohort)
    gated_workload, gated_selection, gated_latencies, gated = _comparison_cohort(gated_cohort)
    if default["prompt_policy"] != "default" or gated["prompt_policy"] not in _PROMPT_POLICIES:
        raise ValueError("Compare a default cohort followed by a gated cohort")
    if workload != gated_workload:
        raise ValueError("Cohorts differ in workload, index, model, or worker layout")
    if default_selection != gated_selection:
        raise ValueError("Cohorts differ in natural request order or ranked evidence")
    return {
        **workload,
        "default": default,
        "gated": gated,
        "paired_gated_minus_default_latency_ms": {
            name: _distribution(gated_ms - default_ms for default_ms, gated_ms in
                                zip(values, gated_latencies[name], strict=True))
            for name, values in default_latencies.items()
        },
        "byte_scope": (
            "Request totals include the whole natural prompt. Evidence NIXL totals include only "
            "blocks attributed exclusively to the listed evidence and omit boundary or shared blocks."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=Path)
    mode.add_argument("--compare", type=Path, nargs=2, metavar=("DEFAULT_COHORT", "GATED_COHORT"),
                      help="compare two trace-on cohort directories with raw or published measurements")
    parser.add_argument("--trace", choices=("on", "off"))
    args = parser.parse_args()
    if args.compare:
        if args.trace is not None:
            parser.error("--compare reads the trace settings in the cohort reports")
        try:
            print(json.dumps(compare(*args.compare), indent=2))
        except (OSError, KeyError, ValueError) as error:
            parser.exit(1, f"Full Blend comparison stopped: {error}\n")
        return
    if args.trace is None:
        parser.error("--run-dir requires --trace")
    print(json.dumps(report(args.run_dir, trace=args.trace == "on"), indent=2))


if __name__ == "__main__":
    main()
