"""Join per-chunk input stages, scheduled work and measured KV transfers."""

from collections import Counter, defaultdict
import json

from .pd_dataset_report import _gpu_batches
from .pd_report import (
    _OUTER_TIMELINE_EVENTS,
    _contiguous_blocks,
    _lmcache_report,
    _reconcile,
    load_events,
    request_joins,
    summarize_request,
)
from .retrieval import json_line, read_jsonl


_CHUNK_STAGES = {"corpus_row_read", "prompt_chunk_append", "prompt_chunk_offset_map"}
_REQUEST_STAGES = {"prompt_template_prefix", "prompt_template_suffix", "prompt_join", "prompt_tokenize"}
_CACHE_EVENTS = {"lmcache_transfer_submit", "lmcache_transfer_complete"}
_MILESTONES = (
    "client_request_start", "client_dispatch", "client_first_token", "client_last_token", "client_response_end",
)
_REQUEST_TIMELINE = set(_MILESTONES) | {
    "embedding_start", "embedding_end", "hnsw_search_start", "hnsw_search_end",
    "gateway_received", "gateway_prompt_ready", "gateway_dispatch_prefill", "gateway_prefill_response",
    "gateway_dispatch_decode", "gateway_stream_end", "engine_queued", "engine_scheduled",
    "engine_remote_kv_wait_begin", "engine_remote_kv_wait_end",
}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _key(chunk):
    return tuple(chunk[name] for name in ("doc_id", "chunk_id", "chunk_version"))


def _prep(events):
    by_chunk, shared = defaultdict(list), []
    for event in events:
        for stage in event.get("chunk_stages", ()):
            if not 0 < stage["start_ns"] <= stage["end_ns"] <= event["ts_ns"]:
                raise ValueError(f"Invalid clock interval for {stage['stage']}")
            if stage["stage"] in _CHUNK_STAGES:
                by_chunk[_key(stage)].append(stage)
            elif stage["stage"] in _REQUEST_STAGES:
                shared.append(stage)
            else:
                raise ValueError(f"Unexpected preparation stage {stage['stage']}")
    return by_chunk, shared


def _chunk_ops(chunk, events):
    lo, hi = chunk["start_token"], chunk["end_token"]
    scheduled, matches = [], []
    for event in events:
        name = event["event"]
        if name == "engine_token_work":
            start = max(lo, event["position_start"])
            end = min(hi, event["position_end"])
            if start < end:
                scheduled.append({"ts_ns": event["ts_ns"], "role": event["role"], "worker_id": event["worker_id"],
                                  "step_id": event["step_id"], "start_token": start, "end_token": end})
        elif name == "engine_prefix_match":
            local, remote = event["local_token_count"] or 0, event["remote_token_count"] or 0
            observed = {"local_tokens": max(0, min(hi, local) - lo),
                        "remote_tokens": max(0, min(hi, local + remote) - max(lo, local))}
            if any(observed.values()):
                matches.append({"ts_ns": event["ts_ns"], "role": event["role"], "worker_id": event["worker_id"],
                                **observed})
    return scheduled, matches


def _transfer(event, evidence):
    lmcache = event["event"] == "lmcache_transfer_complete"
    measured = event["completed_bytes"]
    if lmcache:
        blocks = event["mapped_ranges"]
        transfer_id, direction = event["transfer_key"], event["direction"]
        unmapped = event["unmapped_bytes"]
    else:
        blocks = event.get("block_payloads") or (_contiguous_blocks(event, measured) if type(measured) is int else None)
        transfer_id, direction = event["transfer_id"], "p2d"
        unmapped = 0 if blocks else measured
    blocks = blocks or []
    if type(measured) is int and sum(block["bytes"] for block in blocks) + (unmapped or 0) != measured:
        raise ValueError(f"Transfer {transfer_id} byte ranges do not reconcile")
    classified = []
    for block in blocks:
        lo, hi = block["start_token"], block["end_token"]
        if type(block["bytes"]) is not int or block["bytes"] <= 0 or not 0 <= lo < hi:
            raise ValueError(f"Invalid block in transfer {transfer_id}")
        hits = [dict(occurrence=chunk["occurrence"],
                     start_token=max(lo, chunk["start_token"]),
                     end_token=min(hi, chunk["end_token"])) for chunk in evidence
                if lo < chunk["end_token"] and hi > chunk["start_token"]]
        exclusive = len(hits) == 1 and (hits[0]["start_token"], hits[0]["end_token"]) == (lo, hi)
        classified.append({**block, "evidence_overlaps": hits,
                           "attribution": "exclusive" if exclusive else "boundary" if hits else "outside"})
    extra = {name: event[name] for name in (
        "device", "device_index", "geometry_status", "staging_duration_ms", "kernel_bytes", "kernel_duration_ms",
        "transfer_start_wall_ns", "transfer_end_wall_ns",
    ) if name in event}
    return {"transfer_id": transfer_id, "worker_id": event["worker_id"],
            "path": "lmcache" if lmcache else "nixl", "direction": direction,
            "completion_ns": event["ts_ns"], "status": event["status"],
            "completed_bytes": measured, "unmapped_bytes": unmapped, "blocks": classified, **extra}


def _apply_lmcache_registration(run_dir, events):
    rows = [event for event in events if event["event"] in _CACHE_EVENTS]
    if not rows:
        return rows
    registrations = _json(run_dir / "lmcache-status.json")
    if registrations["verified"] is not True:
        raise ValueError("LMCache worker registrations were not verified")
    workers = {entry["instance_id"]: worker for worker, entry in registrations["worker_registrations"].items()}
    for row in rows:
        row["worker_id"] = workers[int(row["instance_id"])]
    return rows


def _cache_totals(run_dir, endpoint, events):
    receipt = _lmcache_report(run_dir, endpoint)
    observed = Counter()
    for event in events:
        if event["event"] == "lmcache_transfer_complete" and event["completed_bytes"]:
            observed[str(event["device_index"]), event["direction"]] += event["completed_bytes"]
    counters = Counter()
    if receipt and receipt["status"] == "captured":
        for row in receipt["counter_deltas"]:
            if (row["metric"] == "lmcache_mp_transfer_phase_bytes_total"
                    and row["labels"]["phase"] == "staging"):
                counters[row["labels"]["device_index"], row["labels"]["direction"]] += row["delta"]
    status = "matched" if receipt and receipt["status"] == "captured" and counters == observed else "mismatch"
    directions = lambda samples: {f"{device}/{direction}": value for (device, direction), value in sorted(samples.items())}
    return {"status": status, "staging_counter_bytes_by_device": directions(counters),
            "trace_staging_bytes_by_device": directions(observed)}


def _smoke_checks(probes, shared):
    keys = ("cold", "warm_exact", "cross_prefill_exact", "reordered")
    if set(probes) != set(keys):
        return ["Expected cold, warm_exact, cross_prefill_exact and reordered probe receipts"]
    cold, warm, cross, moved = (probes[key] for key in keys)
    if not (cold["prompt_sha256"] == warm["prompt_sha256"] == cross["prompt_sha256"] != moved["prompt_sha256"]):
        return ["Controlled requests did not preserve exact/reordered prompt identities"]
    chunks = lambda row: {_key(chunk): (chunk["start_token"], chunk["end_token"]) for chunk in row["evidence"]}
    anchor, reordered = chunks(cold), chunks(moved)
    issues = []
    if (anchor != chunks(warm) or anchor != chunks(cross) or anchor.keys() != reordered.keys()
            or anchor == reordered):
        issues.append("Reordered probe did not move the same evidence chunks")
    if not moved["evidence_p_scheduled_tokens"]:
        issues.append("Reordered probe did not show evidence computation")
    if not shared and any(row["full_nixl_handoff"]["status"] != "full" for row in probes.values()):
        issues.append("A native probe did not complete a full NIXL handoff")
    if shared:
        if warm["completed_nixl_bytes"] >= cold["completed_nixl_bytes"]:
            issues.append("Warm same-worker probe did not reduce NIXL movement")
        if warm["evidence_p_scheduled_tokens"] >= cold["evidence_p_scheduled_tokens"]:
            issues.append("Warm same-worker probe did not reduce scheduled evidence work")
        if (moved["evidence_p_scheduled_tokens"] <= warm["evidence_p_scheduled_tokens"]
                or moved["completed_nixl_bytes"] <= warm["completed_nixl_bytes"]):
            issues.append("Reordered evidence did not show the expected extra prefill work and NIXL movement")
        cache_selected = any(entry["connector_class"] == "LMCacheMPConnector" and entry["matched_tokens"] > 0
                             for entry in cross["selected_external_connectors_by_role"]["P"])
        if cross["prefill"] == cold["prefill"] or not cache_selected or not any(
            row["path"] == "lmcache" and row["direction"] == "h2d" and row["worker_id"] == cross["prefill"]
            and row["completed_bytes"] and row["unmapped_bytes"] == 0
            and any(block["evidence_overlaps"] for block in row["blocks"]) for row in cross["transfers"]
        ):
            issues.append("Cross-prefill repeat needs a measured LMCache restore with verified token positions")
    return issues


def build_chunk_report(run_dir, labels, *, smoke=False):
    endpoints = _json(run_dir / "endpoints.json")
    manifest = _json(run_dir / "run_manifest.json")
    if not manifest["engine_trace_enabled"]:
        raise ValueError("Chunk review requires native engine tracing")
    shared = bool(endpoints.get("lmcache"))
    cache_trace = (manifest.get("lmcache") or {}).get("transfer_trace_enabled", False)
    if smoke and shared and not cache_trace:
        raise ValueError("Shared smoke requires --trace-lmcache-transfers")
    receipts, live_labels = {}, set()
    for label in labels:
        directory = run_dir / "client" / label
        replay = _json(directory / "manifest.json")
        if replay["mode"] == "live_hnsw":
            live_labels.add(label)
        rows = list(read_jsonl(directory / "responses.jsonl"))
        if (len(rows) != replay["requested"] or len(rows) != replay["succeeded"]
                or any(row["label"] != label or not row["success"] for row in rows)):
            raise ValueError(f"Replay {label} did not complete successfully")
        for row in rows:
            if row["request_id"] in receipts:
                raise ValueError(f"Duplicate replay request {row['request_id']}")
            receipts[row["request_id"]] = {name: row[name] for name in (
                "label", "case", "prompt_sha256", "output_token_ids_sha256", "completion_tokens",
            )}
    if not receipts:
        raise ValueError("No completed requests were selected")
    events = list(load_events(run_dir))
    cache_events = _apply_lmcache_registration(run_dir, events)
    groups, _ = request_joins(events)
    assigned = {id(event) for rows in groups.values() for event in rows if event["event"] in _CACHE_EVENTS}
    if any(id(event) not in assigned for event in cache_events):
        raise ValueError("An LMCache transfer could not be joined to its native request")
    groups = {key: rows for key, rows in groups.items() if key[0] in receipts}
    if len(groups) != len(receipts) or {key[0] for key in groups} != receipts.keys():
        raise ValueError("Expected one joined engine attempt for every completed request")
    ordered = []
    for key, rows in groups.items():
        counts = Counter(event["event"] for event in rows)
        if any(counts[name] != 1 for name in _OUTER_TIMELINE_EVENTS):
            raise ValueError(f"Incomplete timeline for {key[0]}")
        dispatch = next(event for event in rows if event["event"] == "client_dispatch")
        ordered.append((dispatch["ts_ns"], key, dispatch, rows))
    ordered.sort(key=lambda item: item[:2])
    gpu, issues = _gpu_batches(run_dir, {}, groups, manifest["gpu_event_timing"])
    if smoke and not manifest["gpu_event_timing"]:
        issues.append("Smoke requires CUDA event timing")
    workers = {row["id"] for side in ("prefill", "decode") for row in endpoints[side]}
    nixl = _reconcile(run_dir, events, workers, True)
    if nixl["status"] != "matched":
        issues.append(f"NIXL trace and worker counters did not match: {nixl['status']}")
    cache = _cache_totals(run_dir, endpoints.get("lmcache"), cache_events) if cache_trace else None
    if cache and cache["status"] != "matched":
        issues.append("LMCache completed staging bytes did not match the endpoint counters")

    previous, probes = {}, {}
    request_rows, chunk_rows, transfer_rows = [], [], []
    for start, key, dispatch, timeline in ordered:
        timeline = sorted(timeline, key=lambda event: event["ts_ns"])
        rid = key[0]
        receipt = receipts[rid]
        report = summarize_request(key, timeline, lmcache_enabled=shared, shared_pd=shared)
        evidence = report["evidence"]
        expected = [{name: value for name, value in chunk.items() if name not in {"token_work", "occurrence"}}
                    for chunk in evidence]
        if (not evidence or expected != dispatch["evidence"] or report["prompt_sha256"] != receipt["prompt_sha256"]
                or report["output_token_ids_sha256"] != receipt["output_token_ids_sha256"]
                or report["completion_tokens"] != receipt["completion_tokens"] or report["nixl_failed_attempts"]):
            raise ValueError(f"Serving trace differs from the prompt or response receipt for {rid}")
        milestones = {event["event"]: event["ts_ns"] for event in timeline if event["event"] in _MILESTONES}
        builds, request_build = _prep(timeline)
        live = receipt["label"] in live_labels
        if live and not any(stage["stage"] == "prompt_tokenize" for stage in request_build):
            issues.append(f"{rid}: live prompt build has no tokenizer timing")
        transfers = [_transfer(event, evidence) for event in timeline
                     if event["event"] in {"lmcache_transfer_complete", "nixl_read_complete"}]
        submissions = {("nixl", event["transfer_id"]): event for event in timeline
                       if event["event"] == "nixl_read_submit"}
        cache_submissions = {("lmcache", event["transfer_key"]): event for event in timeline
                             if event["event"] == "lmcache_transfer_submit"}
        completed_cache = {("lmcache", transfer["transfer_id"]) for transfer in transfers if transfer["path"] == "lmcache"}
        if completed_cache != cache_submissions.keys():
            issues.append(f"{rid}: LMCache submit/completion IDs differ")
        submissions.update(cache_submissions)
        for transfer in transfers:
            if transfer["completed_bytes"] is None or transfer["unmapped_bytes"] or transfer["status"] != "done":
                issues.append(f"{rid}: {transfer['path']} transfer {transfer['transfer_id']} lacks a verified byte map")
            submitted = submissions.get((transfer["path"], transfer["transfer_id"]), {})
            transfer["submit_ns"] = submitted.get("ts_ns")
            if "submission_wall_ns" in submitted:
                transfer["submission_wall_ns"] = submitted["submission_wall_ns"]
            if transfer["submit_ns"] is None or transfer["submit_ns"] > transfer["completion_ns"]:
                issues.append(f"{rid}: {transfer['path']} completion has no matching submit timestamp")
            transfer_rows.append({"request_id": rid, **transfer})
        batch_events = [{name: event[name] for name in ("event", "ts_ns", "worker_id", "role", "batch_id", "step_id", "duration_ns")
                         if name in event} for event in timeline if event["event"] in {
                            "gpu_forward_start", "gpu_forward_end", "gpu_forward_duration"}]
        launched = {(row["worker_id"], row["step_id"]) for row in batch_events if row["event"] == "gpu_forward_start"}
        prompt_work = 0
        for chunk in evidence:
            identity = _key(chunk)
            occurrence = chunk["occurrence"]
            stages = builds[identity]
            retained = [row for row in stages if row["stage"] == "corpus_row_read"
                        or (row["retained"] and row["accepted_attempt"])]
            if live:
                retained_names = Counter(row["stage"] for row in retained)
                offsets = next((row for row in retained if row["stage"] == "prompt_chunk_offset_map"), {})
                if (retained_names != Counter(_CHUNK_STAGES)
                        or (offsets.get("start_token"), offsets.get("end_token")) !=
                        (chunk["start_token"], chunk["end_token"])):
                    issues.append(f"{rid} chunk {occurrence}: incomplete or changed preparation trace")
            work, prefix = _chunk_ops(chunk, timeline)
            if any((step["worker_id"], step["step_id"]) not in launched for step in work):
                issues.append(f"{rid} chunk {occurrence}: scheduled work missing a GPU forward batch")
            prompt_work += sum(step["end_token"] - step["start_token"] for step in work if step["role"] == "P")
            hits, byte_totals = [], Counter()
            for transfer in transfers:
                for block_index, block in enumerate(transfer["blocks"]):
                    for overlap in block["evidence_overlaps"]:
                        if overlap["occurrence"] != occurrence:
                            continue
                        hits.append({"path": transfer["path"], "direction": transfer["direction"],
                                     "transfer_id": transfer["transfer_id"], "block": block_index,
                                     "attribution": block["attribution"], **overlap})
                        if block["attribution"] == "exclusive":
                            byte_totals[f"{transfer['path']}_{transfer['direction']}"] += block["bytes"]
            if byte_totals["nixl_p2d"] != report["bytes"]["by_chunk_occurrence"][occurrence]:
                raise ValueError(f"{rid} chunk {occurrence}: NIXL totals disagree with the native report")
            earlier = previous.get(identity)
            chunk_rows.append({"request_id": rid, "label": receipt["label"], "case": receipt.get("case"), **chunk,
                               "previous_dispatch": earlier, "previous_finished_before_dispatch":
                                   earlier["response_end_ns"] <= start if earlier else None,
                               "preparation_stages": stages, "scheduled_operations": work, "prefix_observations": prefix,
                               "exclusive_transfer_bytes": dict(byte_totals), "transfer_block_overlaps": hits})
        for chunk in evidence:
            previous[_key(chunk)] = {"request_id": rid, "occurrence": chunk["occurrence"],
                                     "label": receipt["label"],
                                     "start_token": chunk["start_token"], "end_token": chunk["end_token"],
                                     "response_end_ns": milestones["client_response_end"]}
        stages = [{name: event[name] for name in ("event", "ts_ns", "worker_id", "role") if name in event}
                  for event in timeline if event["event"] in _REQUEST_TIMELINE]
        request_rows.append({**report, "label": receipt["label"], "milestones_ns": milestones,
                             "preparation_scope": "live" if live else "frozen",
                             "timeline_events": stages, "prompt_build_shared_stages": request_build,
                             "gpu_batch_events": batch_events})
        if receipt["label"] == "chunk-probes":
            probes[receipt["case"]] = {**report, "transfers": transfers, "evidence_p_scheduled_tokens": prompt_work}
    if smoke:
        issues += _smoke_checks(probes, shared)
        if not any(row["label"] in live_labels and row["previous_finished_before_dispatch"]
                   and row["previous_dispatch"]["label"] in live_labels
                   and row["start_token"] != row["previous_dispatch"]["start_token"] for row in chunk_rows):
            issues.append("Live replay did not show a chunk recurring at a different prompt position")
    cases = {}
    for name, row in probes.items():
        totals = Counter()
        for transfer in row["transfers"]:
            if transfer["path"] == "lmcache" and type(transfer["completed_bytes"]) is int:
                totals[transfer["direction"]] += transfer["completed_bytes"]
        cases[name] = {key: row[key] for key in (
            "request_id", "prefill", "decode", "evidence_p_scheduled_tokens", "completed_nixl_bytes",
        )}
        cases[name]["lmcache_staging_bytes"] = totals
    summary = {"status": "failed" if issues else "passed", "requests": len(request_rows), "labels": labels,
               "chunk_occurrences": len(chunk_rows), "repeat_occurrences": sum(row["previous_dispatch"] is not None for row in chunk_rows),
               "nixl_counter_reconciliation": nixl, "lmcache_counter_reconciliation": cache,
               "gpu_forward": gpu, "probe_cases": cases, "issues": issues}
    output = run_dir / "chunk-review"
    output.mkdir(exist_ok=True)
    for name, rows in (("requests", request_rows), ("chunks", chunk_rows), ("kv-transfers", transfer_rows)):
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            stream.writelines(map(json_line, rows))
    (output / "summary.json").write_text(json_line(summary), encoding="utf-8")
    return summary
