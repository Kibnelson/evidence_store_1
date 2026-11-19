"""Join native PD traces and produce request, transfer, and timeline reports."""

from collections import Counter, defaultdict, deque
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
import time
from urllib.request import urlopen

from .retrieval import read_jsonl


_ENGINE_HEADER = re.compile(r"^(?:cmpl-)?(es-[a-f0-9]{32}-[pd])(?:-|$)")
_METRIC = re.compile(r"^vllm:nixl_bytes_transferred_(sum|count)(?:\{[^}]*\})?\s+(\S+)")
_LMCACHE_COUNTERS = frozenset({
    "lmcache_mp_l1_read_chunks_total",
    "lmcache_mp_l1_write_chunks_total",
    "lmcache_mp_l1_evicted_chunks_total",
    "lmcache_mp_transfer_phase_bytes_total",
})
_PROMETHEUS_SAMPLE = re.compile(
    r'(?P<name>[a-zA-Z_][\w:]*)(?:\{(?P<labels>(?:[^"{}]|"(?:\\.|[^"\\])*")*)\})?'
    r'\s+(?P<value>\S+)(?:\s+\S+)?\s*$'
)
_PROMETHEUS_LABEL = re.compile(r'([a-zA-Z_]\w*)=("(?:\\.|[^"\\])*")')
_SPANS = {
    "client_request_start": ("client_response_end", "End-to-end"),
    "embedding_start": ("embedding_end", "Query embedding"),
    "hnsw_search_start": ("hnsw_search_end", "Faiss HNSW"),
    "chunk_materialize_start": ("chunk_materialize_end", "Evidence materialization"),
    "prompt_render_start": ("prompt_render_end", "Prompt rendering"),
    "gateway_received": ("gateway_prompt_ready", "Gateway parse and validation"),
    "gateway_dispatch_prefill": ("gateway_prefill_response", "P HTTP round trip"),
    "gateway_dispatch_decode": ("gateway_stream_end", "D HTTP stream"),
    "engine_queued": ("engine_scheduled", "Native scheduler wait"),
    "engine_remote_kv_wait_begin": ("engine_remote_kv_wait_end", "Native remote KV wait"),
    "gpu_forward_start": ("gpu_forward_end", "Model forward dispatch"),
    "nixl_read_submit": ("nixl_read_complete", "NIXL READ"),
}
_OUTER_TIMELINE_EVENTS = (
    "client_request_start", "client_dispatch", "client_first_token", "client_response_end",
    "gateway_received", "gateway_prompt_ready", "gateway_dispatch_prefill",
    "gateway_prefill_response", "gateway_dispatch_decode", "gateway_stream_end",
)


def _capture_native(workers):
    snapshot = {}
    raw = {}
    for worker in workers:
        with urlopen(worker["url"].rstrip("/") + "/metrics", timeout=10) as response:
            metrics = response.read().decode()
        counters = {"sum": 0, "count": 0, "present": False}
        for line in metrics.splitlines():
            if match := _METRIC.match(line):
                value = float(match[2])
                if not math.isfinite(value) or value < 0 or not value.is_integer():
                    raise ValueError(f"Unexpected NIXL counter in {worker['id']}: {line}")
                counters[match[1]] += int(value)
                counters["present"] = True
        snapshot[worker["id"]] = counters
        raw[worker["id"]] = metrics
    return snapshot, raw


def _nixl_totals(events):
    totals = defaultdict(Counter)
    for row in events:
        if (
            row["event"] == "nixl_read_complete" and row.get("status") in {"completed", "success", "done"}
            and isinstance(row.get("completed_bytes"), int)
        ):
            totals[row["worker_id"]].update({"sum": row["completed_bytes"], "count": 1})
    return totals


def capture_metrics(run_dir, label):
    endpoints = json.loads((run_dir / "endpoints.json").read_text())
    workers = endpoints["prefill"] + endpoints["decode"]
    before_path = run_dir / "metrics-before.json"
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    expected = {}
    if label == "after" and before_path.exists() and manifest.get("engine_trace_enabled", True):
        before = json.loads(before_path.read_text())
        totals = _nixl_totals(load_events(run_dir, event_name="nixl_read_complete"))
        expected = {
            worker: {key: before[worker][key] + totals[worker][key] for key in ("sum", "count")}
            for worker in totals if worker in before
        }

    deadline = time.monotonic() + 30
    while True:
        snapshot, raw = _capture_native(workers)
        if not any(
            worker in snapshot and any(snapshot[worker][key] < target[key] for key in target)
            for worker, target in expected.items()
        ) or time.monotonic() >= deadline:
            break
        time.sleep(0.5)

    for worker, metrics in raw.items():
        (run_dir / f"native-metrics-{label}-{worker}.prom").write_text(metrics)
    if lmcache := endpoints.get("lmcache"):
        with urlopen(lmcache["url"].rstrip("/") + "/metrics", timeout=10) as response:
            (run_dir / f"lmcache-metrics-{label}.prom").write_bytes(response.read())
    (run_dir / f"metrics-{label}.json").write_text(json.dumps(snapshot, indent=2) + "\n")


def _lmcache_counter_samples(snapshot):
    samples = {}
    for line in snapshot.splitlines():
        match = _PROMETHEUS_SAMPLE.fullmatch(line)
        if not match or match["name"] not in _LMCACHE_COUNTERS:
            continue
        labels = tuple(sorted(
            (name, json.loads(value))
            for name, value in _PROMETHEUS_LABEL.findall(match["labels"] or "")
        ))
        identity = (match["name"], labels)
        try:
            value = Decimal(match["value"])
            if not value.is_finite() or value < 0 or value != value.to_integral_value():
                raise InvalidOperation
        except InvalidOperation as exc:
            raise ValueError(f"Invalid integer LMCache counter: {line}") from exc
        if identity in samples:
            raise ValueError(f"Duplicate LMCache Prometheus counter series: {line}")
        samples[identity] = int(value)
    return samples


def _lmcache_report(run_dir, endpoint):
    if not endpoint:
        return None
    result = {"endpoint_id": endpoint["id"], "counter_deltas": []}
    paths = [run_dir / f"lmcache-metrics-{when}.prom" for when in ("before", "after")]
    if not all(path.exists() for path in paths):
        return {**result, "status": "missing_counter_snapshots"}
    before, after = (_lmcache_counter_samples(path.read_text()) for path in paths)
    if not after:
        return {**result, "status": "no_counter_samples"}
    status = "captured"
    for metric, labels in sorted(before.keys() | after.keys()):
        old, new = before.get((metric, labels), 0), after.get((metric, labels))
        delta = new - old if new is not None and new >= old else None
        if delta is None:
            status = "counter_reset_or_missing_series"
        result["counter_deltas"].append({
            "metric": metric, "labels": dict(labels), "before": old, "after": new, "delta": delta,
        })
    return {**result, "status": status}


def load_events(run_dir, *, event_name=None):
    event_text = f'"{event_name}"' if event_name else None
    for path in sorted((run_dir / "trace").glob("*.jsonl")):
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip() or (event_text is not None and event_text not in line):
                    continue
                try:
                    event = json.loads(line)
                    if event["schema_version"] != 1 or event["run_id"] != run_dir.name:
                        raise ValueError("event has a different schema or run ID")
                    if not isinstance(event["ts_ns"], int):
                        raise ValueError("event timestamp is not a monotonic nanosecond integer")
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if event_name is None or event["event"] == event_name:
                    yield event


def request_joins(events):
    headers, native = {}, {}
    for row in events:
        if row["event"] in ("gateway_dispatch_prefill", "gateway_dispatch_decode"):
            headers[row["engine_header_id"]] = (row["request_id"], row["attempt_id"])
    for row in events:
        if row["event"] == "engine_request_id":
            match = _ENGINE_HEADER.match(row.get("external_request_id") or "")
            if match and match[1] in headers:
                native[(row["worker_id"], row["native_request_id"])] = headers[match[1]]
    gateway_attempts = defaultdict(set)
    for request_id, attempt_id in headers.values():
        gateway_attempts[request_id].add(attempt_id)

    groups = defaultdict(list)
    unjoined = Counter()
    for row in events:
        name = row["event"]
        component = row["component"]
        if component == "gateway":
            matches = [(row["request_id"], row["attempt_id"])]
        elif component == "client":
            request_id = row.get("request_id")
            matches = [(request_id, attempt) for attempt in (gateway_attempts[request_id] or {None})] if request_id else []
        else:
            worker = row.get("worker_id")
            native_ids = set(row.get("request_tokens") or {})
            if native_id := row.get("native_request_id"):
                native_ids.add(native_id)
            matches = {native[(worker, item)] for item in native_ids if (worker, item) in native}
        for key in matches:
            groups[key].append(row)
        if not matches and (name.startswith("engine_") or name.startswith("nixl_")):
            unjoined[name] += 1
    return groups, unjoined


def _overlap(a, b, lo, hi):
    return max(0, min(b, hi) - max(a, lo))


def _attach_work(evidence, rows, engine_trace_enabled):
    result = []
    for index, chunk in enumerate(evidence):
        if not engine_trace_enabled:
            result.append({**chunk, "occurrence": index, "token_work": None})
            continue
        totals = defaultdict(Counter)
        for row in rows:
            phase = row.get("role")
            if row["event"] == "engine_token_work":
                lo = row.get("position_start")
                hi = row.get("position_end")
                if type(lo) is int and type(hi) is int:
                    totals[phase]["scheduled_prompt_tokens"] += _overlap(
                        chunk["start_token"], chunk["end_token"], lo, hi
                    )
            if row["event"] == "engine_prefix_match":
                local = row.get("local_token_count") or 0
                remote = row.get("remote_token_count") or 0
                totals[phase]["observed_local_prefix_tokens"] += _overlap(
                    chunk["start_token"], chunk["end_token"], 0, local
                )
                totals[phase]["observed_remote_prefix_tokens"] += _overlap(
                    chunk["start_token"], chunk["end_token"], local, local + remote
                )
        result.append({**chunk, "occurrence": index, "token_work": totals})
    return result


def _attribute_bytes(evidence, transfers):
    chunks = Counter()
    shared = outside = unmapped = 0
    for transfer in transfers:
        actual = transfer.get("completed_bytes")
        if not isinstance(actual, int):
            continue
        blocks = transfer.get("block_payloads") or _contiguous_blocks(transfer, actual)
        if not blocks or sum(block["bytes"] for block in blocks) != actual:
            unmapped += actual
            continue
        for block in blocks:
            lo, hi = block["start_token"], block["end_token"]
            hits = [item for item in evidence if _overlap(item["start_token"], item["end_token"], lo, hi)]
            if not hits:
                outside += block["bytes"]
            elif len(hits) == 1 and hits[0]["start_token"] <= lo < hi <= hits[0]["end_token"]:
                chunks[hits[0]["occurrence"]] += block["bytes"]
            else:
                shared += block["bytes"]
    return {
        "by_chunk_occurrence": chunks,
        "shared_or_boundary_bytes": shared,
        "outside_evidence_bytes": outside,
        "unmapped_bytes": unmapped,
    }


def _contiguous_geometry(transfer, actual):
    start = transfer.get("position_start")
    end = transfer.get("position_end")
    count = transfer.get("transferred_blocks")
    size = transfer.get("block_size")
    if (
        transfer.get("status") != "done"
        or not all(type(value) is int for value in (start, end, count, size))
        or count <= 0 or size <= 0 or end - start != count * size or actual % count
    ):
        return None
    return start, end, count, size


def _contiguous_blocks(transfer, actual):
    geometry = _contiguous_geometry(transfer, actual)
    if geometry is None:
        return []
    start, end, count, size = geometry
    return [
        {"start_token": position, "end_token": position + size, "bytes": actual // count}
        for position in range(start, end, size)
    ]


def _full_nixl_handoff(prompt_tokens, producer, transfers, prefiller, decoder):
    """Verify the native D reads cover P's assembled prompt, excluding block padding."""
    advertised = producer.get("remote_token_count")
    producer_id = producer.get("native_request_id")
    result = {
        "prompt_tokens": prompt_tokens,
        "p_advertised_tokens": advertised,
        "verified_prompt_tokens": None,
    }
    if (
        type(prompt_tokens) is not int or prompt_tokens <= 0 or type(advertised) is not int
        or not producer_id or not prefiller or producer.get("worker_id") != prefiller
    ):
        return {**result, "status": "missing_expected_prompt_or_producer"}
    if advertised != prompt_tokens:
        return {**result, "status": "producer_advertised_different_length"}
    completed = [
        row for row in transfers if row.get("status") == "done"
        and type(row.get("completed_bytes")) is int and row["completed_bytes"] > 0
    ]
    if not completed:
        return {**result, "status": "no_completed_read"}
    if any(row.get("worker_id") != decoder or row.get("remote_request_id") != producer_id for row in completed):
        return {**result, "status": "route_mismatch"}
    ids = [row.get("transfer_id") for row in completed]
    if not all(ids) or len(set(ids)) != len(ids):
        return {**result, "status": "missing_or_duplicate_transfer_id"}

    spans = []
    for row in completed:
        geometry = _contiguous_geometry(row, row["completed_bytes"])
        if geometry is None:
            return {**result, "status": "unverified_block_geometry"}
        start, physical_end, _, block_size = geometry
        end = row.get("valid_position_end")
        if (
            type(end) is not int or not 0 <= start < end <= min(prompt_tokens, physical_end)
            or physical_end - end >= block_size
        ):
            return {**result, "status": "unverified_block_geometry"}
        spans.append((start, end))
    mapped = position = 0
    for start, end in sorted(spans):
        mapped += max(0, end - max(start, position))
        position = max(position, end)
    return {
        **result,
        "verified_prompt_tokens": mapped,
        "status": "full" if mapped == prompt_tokens else "partial",
    }


def summarize_request(key, rows, *, engine_trace_enabled=True, lmcache_enabled=False, shared_pd=False):
    request_id, attempt_id = key
    by_name = defaultdict(list)
    for row in rows:
        by_name[row["event"]].append(row)

    def first(name):
        return min(by_name[name], key=lambda row: row["ts_ns"], default={})

    start = first("client_request_start") or first("gateway_received")
    dispatch = first("client_dispatch") or first("gateway_received")
    token = first("client_first_token")
    last = first("client_last_token")
    end = first("client_response_end")
    evidence = _attach_work(first("gateway_prompt_ready").get("evidence", []), rows, engine_trace_enabled)
    transfers = by_name["nixl_read_complete"] + by_name["nixl_read_skipped"]
    p_worker = first("gateway_route_prefill").get("worker_id")
    d_worker = first("gateway_route_decode").get("worker_id")
    producer = first("engine_kv_handoff_ready")
    prompt_tokens = first("gateway_prompt_ready").get("prompt_tokens") or producer.get("prompt_tokens")
    engine_measurements = {
        "local_prefix_tokens_by_worker": {
            worker: max(row.get("local_token_count", 0) for row in by_name["engine_prefix_match"] if row.get("worker_id") == worker)
            for worker in {row.get("worker_id") for row in by_name["engine_prefix_match"]}
        },
        "bytes": _attribute_bytes(evidence, transfers),
        "completed_nixl_bytes": sum(row["completed_bytes"] for row in transfers if isinstance(row.get("completed_bytes"), int)),
        "full_nixl_handoff": _full_nixl_handoff(prompt_tokens, producer, transfers, p_worker, d_worker),
        "unknown_nixl_completions": sum(row.get("completed_bytes") is None for row in transfers),
        "nixl_failed_attempts": len(by_name["nixl_read_failed"]) + sum(row.get("status") == "error" for row in transfers),
        "nixl_transfers": [
            {field: row.get(field) for field in (
                "worker_id", "transfer_id", "remote_request_id", "status", "completed_bytes",
                "position_start", "position_end", "valid_position_end", "transferred_blocks", "block_size",
            )}
            for row in transfers
        ],
        "queued_workers": sorted({row["worker_id"] for row in by_name["engine_queued"]}),
        "model_workers": sorted({row["worker_id"] for row in by_name["gpu_forward_start"]}),
    }
    if lmcache_enabled:
        engine_measurements.update({
            "prefill_lmcache_match_tokens": max((
                row["remote_token_count"] for row in by_name["engine_prefix_match"]
                if row.get("worker_id") == p_worker and type(row.get("remote_token_count")) is int
            ), default=None),
            "prefill_lmcache_usable_prefix_tokens": max((
                row["usable_prefix_token_count"] for row in by_name["engine_remote_kv_wait_end"]
                if row.get("worker_id") == p_worker and row.get("load_failed") is False
                and type(row.get("usable_prefix_token_count")) is int
            ), default=None),
        })
    if shared_pd:
        selected = {role: [] for role in ("P", "D")}
        workers = {p_worker: "P", d_worker: "D"}
        for row in sorted(by_name["engine_connector_selected"], key=lambda row: row["ts_ns"]):
            role = workers.get(row.get("worker_id"))
            if role is not None:
                selected[role].append({
                    field: row.get(field)
                    for field in ("connector_class", "connector_index", "matched_tokens")
                })
        prefixes = engine_measurements["local_prefix_tokens_by_worker"]
        engine_measurements.update({
            "prefill_local_prefix_tokens": prefixes.get(p_worker),
            "decode_local_prefix_tokens": prefixes.get(d_worker),
            "selected_external_connectors_by_role": selected,
        })
        for role, prefix in (("P", "prefill"), ("D", "decode")):
            engine_measurements[f"{prefix}_lmcache_match_tokens"] = max((
                entry["matched_tokens"] for entry in selected[role]
                if entry["connector_class"] == "LMCacheMPConnector"
                and type(entry["matched_tokens"]) is int
            ), default=None)
        if engine_measurements["prefill_lmcache_match_tokens"] is None:
            engine_measurements["prefill_lmcache_usable_prefix_tokens"] = None
    if not engine_trace_enabled:
        engine_measurements = dict.fromkeys(engine_measurements)
    return {
        "request_id": request_id,
        "attempt_id": attempt_id,
        "dataset": start.get("dataset"),
        "phase": start.get("phase", "measured"),
        "case": start.get("case") or start.get("condition"),
        "prefill": p_worker,
        "decode": d_worker,
        "ttft_ms": (token["ts_ns"] - start["ts_ns"]) / 1e6 if start and token else None,
        "dispatch_to_first_token_ms": (token["ts_ns"] - dispatch["ts_ns"]) / 1e6 if dispatch and token else None,
        "dispatch_to_last_token_ms": (last["ts_ns"] - dispatch["ts_ns"]) / 1e6 if dispatch and last else None,
        "end_to_end_ms": (end["ts_ns"] - start["ts_ns"]) / 1e6 if start and end else None,
        "completion_tokens": end.get("completion_tokens"),
        "prompt_sha256": first("client_dispatch").get("prompt_sha256"),
        "output_token_ids_sha256": end.get("output_token_ids_sha256"),
        "http_request_body_bytes": first("client_dispatch").get("http_request_body_bytes"),
        "gateway_prefill_request_bytes": first("gateway_dispatch_prefill").get("http_request_body_bytes"),
        "gateway_prefill_response_bytes": first("gateway_prefill_response").get("http_response_body_bytes"),
        "gateway_decode_request_bytes": first("gateway_dispatch_decode").get("http_request_body_bytes"),
        "gateway_sse_payload_bytes": first("gateway_stream_end").get("sse_payload_bytes"),
        "prompt_tokens": prompt_tokens,
        "evidence": evidence,
        "events": {name: len(items) for name, items in by_name.items()},
        **engine_measurements,
    }


def trace_view(groups, limit, request_id, *, lmcache_enabled=False, shared_pd=False):
    output = []
    endings = {end: start for start, (end, _) in _SPANS.items()}
    chosen = [
        (key, rows) for key, rows in groups.items()
        if (request_id is None or key[0] == request_id)
        and any(row["event"] == "gateway_received" for row in rows)
    ][:limit]
    if request_id is not None and not chosen:
        raise ValueError(f"No gateway timeline for request {request_id}")
    for process, ((label, _), rows) in enumerate(chosen, start=1):
        origin = min(row["ts_ns"] for row in rows)
        output.append({"name": "process_name", "ph": "M", "pid": process, "tid": 0, "args": {"name": label}})
        opened = defaultdict(deque)
        lanes = {}
        selected = {}
        for row in sorted(rows, key=lambda item: item["ts_ns"]):
            event = row["event"]
            worker = row.get("worker_id", "")
            if shared_pd and event == "engine_connector_selected":
                selected[worker] = row.get("connector_class")
            stage = "NIXL" if event.startswith("nixl_") else "GPU" if event.startswith("gpu_") else row["component"]
            lane = f"{stage} {worker}".strip()
            if lane not in lanes:
                lanes[lane] = len(lanes) + 1
                output.append({"name": "thread_name", "ph": "M", "pid": process, "tid": lanes[lane], "args": {"name": lane}})
            common = {"pid": process, "tid": lanes[lane], "ts": (row["ts_ns"] - origin) / 1e3}
            args = {key: val for key, val in row.items() if key not in {"event", "ts_ns", "wall_ns"}}
            if event in _SPANS:
                opened[(event, worker, row.get("transfer_id"), row.get("batch_id"))].append(
                    (row, selected.get(worker))
                )
            elif event in endings:
                start_event = endings[event]
                pending = opened[(start_event, worker, row.get("transfer_id"), row.get("batch_id"))]
                if pending:
                    start, connector = pending.popleft()
                    span_name = _SPANS[start_event][1]
                    if start_event == "engine_remote_kv_wait_begin":
                        if shared_pd:
                            source = {"NixlPullConnector": "NIXL", "LMCacheMPConnector": "LMCache"}.get(connector)
                            if source:
                                span_name = f"{row.get('role', 'Worker')} scheduler waiting for {source} KV"
                        elif lmcache_enabled and row.get("role") == "P":
                            span_name = "Prefill scheduler waiting for LMCache KV"
                    output.append({
                        **common,
                        "name": span_name,
                        "ph": "X",
                        "ts": (start["ts_ns"] - origin) / 1e3,
                        "dur": (row["ts_ns"] - start["ts_ns"]) / 1e3,
                        "args": args,
                    })
            output.append({**common, "name": event, "ph": "i", "s": "t", "args": args})
    return {"displayTimeUnit": "ms", "traceEvents": output}


def verify_retrieval_smoke(run_dir, sample_dir, summary):
    """Check the real requests in the MT-RAG smoke after the transport report passes."""
    receipt = run_dir / "smoke-check.json"
    receipt.unlink(missing_ok=True)
    if summary["issues"] or not summary["engine_trace_enabled"] or not summary["gpu_event_timing"]:
        raise ValueError("Smoke requires a clean trace report with GPU timing enabled")
    sample = json.loads((sample_dir / "manifest.json").read_text())
    output = run_dir / "client" / "mtrag-smoke"
    replay = json.loads((output / "manifest.json").read_text())
    expected = sample["isolated_requests"] + [
        key for conversation in sample["conversations"] for key in conversation["request_keys"]
    ]
    if (replay["mode"] != "live_hnsw" or replay["input_sha256"] != sample["queries_sha256"]
            or replay["index"]["corpus_sha256"] != sample["corpus_sha256"]
            or replay["requested"] != sample["requests"] or replay["succeeded"] != sample["requests"]):
        raise ValueError("Smoke replay did not complete the selected live HNSW workload")

    responses = list(read_jsonl(output / "responses.jsonl"))
    frozen = {row["request_key"]: row for row in read_jsonl(output / "frozen.jsonl")}
    if Counter(row["request_key"] for row in responses) != Counter(expected) or set(frozen) != set(expected):
        raise ValueError("Smoke receipts do not match the selected requests")
    groups, _ = request_joins(list(load_events(run_dir)))
    reports = {row["request_id"]: row for row in read_jsonl(run_dir / "requests.jsonl")}
    stages = (
        "client_request_start", "embedding_start", "embedding_end", "hnsw_search_start", "hnsw_search_end",
        "chunk_materialize_start", "chunk_materialize_end", "prompt_render_start", "prompt_render_end",
        "client_dispatch", "gateway_received", "gateway_prompt_ready", "gateway_route_prefill",
        "gateway_dispatch_prefill", "gateway_prefill_response", "gateway_route_decode", "gateway_dispatch_decode",
        "gateway_decode_headers", "gateway_first_decode_bytes", "client_first_token", "client_last_token",
        "client_response_end",
    )
    checked = {}
    for response in responses:
        key, request_id = response["request_key"], response["request_id"]
        row = reports[request_id]
        joined = groups[(request_id, row["attempt_id"])]
        events = {}
        for event in sorted(joined, key=lambda event: event["ts_ns"]):
            events.setdefault(event["event"], event)
        if missing := {name for name in (*stages, "gateway_stream_end", "engine_kv_handoff_ready",
                                        "engine_token_work") if name not in events}:
            raise ValueError(f"{key}: missing joined trace events {sorted(missing)}")
        times = [events[name]["ts_ns"] for name in stages]
        if times != sorted(times) or not events["gateway_stream_end"]["completed"]:
            raise ValueError(f"{key}: request stages are out of order or the gateway stream is incomplete")
        forwards = {name: {(event["worker_id"], event["batch_id"]) for event in joined if event["event"] == name}
                    for name in ("gpu_forward_start", "gpu_forward_end")}
        gpu_workers = {worker for worker, _ in forwards["gpu_forward_start"] & forwards["gpu_forward_end"]}
        if not {row["prefill"], row["decode"]} <= gpu_workers:
            raise ValueError(f"{key}: missing prefill or decode GPU forward span")
        if row["completed_nixl_bytes"] > 0:
            decoder_events = [event for event in joined if event.get("worker_id") == row["decode"]]
            submissions = {event["transfer_id"]: event["ts_ns"] for event in decoder_events
                           if event["event"] == "nixl_read_submit"}
            completions = [event for event in decoder_events if event["event"] == "nixl_read_complete"
                           and event["completed_bytes"] > 0]
            waits = {event["event"] for event in decoder_events}
            if (not completions or any(submissions.get(event["transfer_id"], float("inf")) > event["ts_ns"]
                                       for event in completions)
                    or not {"engine_remote_kv_wait_begin", "engine_remote_kv_wait_end"} <= waits):
                raise ValueError(f"{key}: missing decode wait or NIXL READ span")

        prompt = frozen[key]
        evidence = row["evidence"]
        if (not response["success"] or row["prompt_sha256"] != prompt["prompt_sha256"]
                or response["prompt_sha256"] != prompt["prompt_sha256"]
                or not evidence or len(evidence) != len(prompt["evidence"])
                or events["chunk_materialize_end"]["hits"] != prompt["retrieval_hits"]
                or any(not all(actual.get(field) == value for field, value in chunk.items())
                       for actual, chunk in zip(evidence, prompt["evidence"]))):
            raise ValueError(f"{key}: retrieved evidence or prompt IDs differ across trace stages")
        if summary["full_nixl_handoff_required"]:
            payload = row["bytes"]
            if (row["full_nixl_handoff"]["status"] != "full" or payload["unmapped_bytes"]
                    or not (sum(payload["by_chunk_occurrence"].values()) + payload["shared_or_boundary_bytes"])):
                raise ValueError(f"{key}: NIXL payload does not map to the retrieved evidence")
        checked[key] = (request_id, times[0], times[-1])

    for conversation in sample["conversations"]:
        turns = [checked[key] for key in conversation["request_keys"]]
        if any(previous[2] > following[1] for previous, following in zip(turns, turns[1:])):
            raise ValueError(f"{conversation['id']}: conversation turns overlapped or replayed out of order")
    receipt.write_text(json.dumps({
        "status": "passed", "sample": sample, "request_ids": {key: checked[key][0] for key in expected},
    }, indent=2) + "\n")


def _shared_pd_probes(requests, expected_routes, all_pairs):
    """Compare actual D reuse and NIXL reads for controlled prompts on each route."""
    case_names = ("cold", "warm_exact", "reordered")
    observed = defaultdict(lambda: defaultdict(list))
    for row in requests:
        route = (row["prefill"], row["decode"])
        if row["case"] in case_names and route in expected_routes:
            observed[route][row["case"]].append(row)

    routes = {}
    for p_worker, d_worker in sorted(expected_routes if all_pairs else observed):
        cases = observed[(p_worker, d_worker)]
        problems = []
        if all_pairs:
            problems.extend(f"missing {case} probe" for case in case_names if not cases[case])
        cold_by_prompt = {}
        for cold in cases["cold"]:
            if (
                cold["full_nixl_handoff"]["status"] != "full"
                or cold["decode_local_prefix_tokens"] != 0
                or not _selected_nixl_on_decode(cold)
            ):
                problems.append(f"{cold['request_id']}: cold D lacks full native NIXL proof")
            if cold["prompt_sha256"]:
                cold_by_prompt[cold["prompt_sha256"]] = cold

        for warm in cases["warm_exact"]:
            cold = cold_by_prompt.get(warm["prompt_sha256"])
            if cold is None:
                problems.append(f"{warm['request_id']}: warm prompt has no matching cold run on this route")
                continue
            warm_bytes = warm["completed_nixl_bytes"]
            expected_read = "partial" if warm_bytes else "no_completed_read"
            valid_read = (
                warm["full_nixl_handoff"]["status"] == expected_read
                and (not warm_bytes or _selected_nixl_on_decode(warm))
            )
            if (
                not warm["decode_local_prefix_tokens"]
                or warm_bytes >= cold["completed_nixl_bytes"]
                or not valid_read
            ):
                problems.append(f"{warm['request_id']}: warm run does not prove D local reuse and fewer NIXL bytes")
            if not warm["output_token_ids_sha256"] or warm["output_token_ids_sha256"] != cold["output_token_ids_sha256"]:
                problems.append(f"{warm['request_id']}: warm output tokens differ or are unavailable")

        for reordered in cases["reordered"]:
            if not cold_by_prompt or not reordered["prompt_sha256"] or reordered["prompt_sha256"] in cold_by_prompt:
                problems.append(f"{reordered['request_id']}: reordered prompt must differ from the cold prompt")

        routes[f"{p_worker}→{d_worker}"] = {
            "issues": problems,
            "probes": {case: [
                {field: row[field] for field in (
                    "request_id", "prompt_sha256", "output_token_ids_sha256", "prefill_local_prefix_tokens", "decode_local_prefix_tokens",
                    "completed_nixl_bytes", "full_nixl_handoff", "selected_external_connectors_by_role",
                )} for row in cases[case]
            ] for case in case_names},
        }
    return routes


def _selected_nixl_on_decode(row):
    return any(
        connector["connector_class"] == "NixlPullConnector"
        for connector in row["selected_external_connectors_by_role"]["D"]
    )


def _reconcile(run_dir, events, worker_ids, engine_trace_enabled):
    before_path, after_path = (run_dir / f"metrics-{name}.json" for name in ("before", "after"))
    if not before_path.exists() or not after_path.exists():
        return {"status": "missing_counter_snapshots"}
    before, after = (json.loads(path.read_text()) for path in (before_path, after_path))
    if before.keys() != worker_ids or after.keys() != worker_ids:
        return {"status": "worker_set_changed"}
    totals = _nixl_totals(events)
    workers = {}
    for worker, latest in after.items():
        delta = {key: latest[key] - before[worker][key] for key in ("sum", "count")}
        measured = {key: totals[worker][key] for key in ("sum", "count")} if engine_trace_enabled else None
        workers[worker] = {
            "prometheus": delta,
            "trace": measured,
            "matched": delta == measured if engine_trace_enabled else None,
        }
    has_counters = any(value["present"] for value in after.values())
    if engine_trace_enabled:
        status = "matched" if has_counters and all(item["matched"] for item in workers.values()) else "mismatch"
    elif not has_counters:
        status = "missing_native_counters"
    elif any(value < 0 for item in workers.values() for value in item["prometheus"].values()):
        status = "native_counter_reset"
    else:
        status = "trace_disabled"
    return {"status": status, "workers": workers}


def report(run_dir, *, timeline_limit, all_pairs, timeline_request=None):
    endpoints = json.loads((run_dir / "endpoints.json").read_text())
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    engine_trace_enabled = manifest.get("engine_trace_enabled", True)
    gpu_event_timing = manifest.get("gpu_event_timing", True)
    full_nixl_handoff_required = manifest.get("full_nixl_handoff", False)
    lmcache_endpoint = endpoints.get("lmcache")
    shared_pd = (manifest.get("lmcache") or {}).get("mode") == "shared_pd"
    decode_prefix_cache = manifest.get("prefix_cache_by_role", {}).get("D", manifest.get("prefix_cache"))
    if type(engine_trace_enabled) is not bool or type(gpu_event_timing) is not bool:
        raise ValueError("Engine trace and GPU timing settings in the run manifest must be booleans")
    gpu_event_timing = engine_trace_enabled and gpu_event_timing
    events = list(load_events(run_dir))
    unexpected_engine_events = 0
    if not engine_trace_enabled:
        outer_events = [row for row in events if row["component"] in {"client", "gateway"}]
        unexpected_engine_events = len(events) - len(outer_events)
        events = outer_events
    groups, unjoined = request_joins(events)
    requests = [summarize_request(
        key, rows, engine_trace_enabled=engine_trace_enabled, lmcache_enabled=bool(lmcache_endpoint),
        shared_pd=shared_pd,
    ) for key, rows in groups.items()]
    expected_routes = {(p["id"], d["id"]) for p in endpoints["prefill"] for d in endpoints["decode"]}
    worker_ids = {worker["id"] for worker in endpoints["prefill"] + endpoints["decode"]}
    actual_routes = Counter((row["prefill"], row["decode"]) for row in requests if row["prefill"] and row["decode"])
    issues = []
    if not requests:
        issues.append("No requests found")
    if unexpected_engine_events:
        issues.append(f"Engine or transport trace emitted with tracing off: {unexpected_engine_events} events")
    if shared_pd and not lmcache_endpoint:
        issues.append("Shared LMCache PD run has no cache metrics endpoint")
    incomplete = []
    for row in requests:
        workers = [row["prefill"], row["decode"]]
        if not all(workers) or row["ttft_ms"] is None or row["end_to_end_ms"] is None:
            incomplete.append(row["request_id"])
            continue
        if not engine_trace_enabled:
            if not all(row["events"].get(event) for event in _OUTER_TIMELINE_EVENTS):
                incomplete.append(row["request_id"])
            continue
        cold = row["case"] == "cold"
        warm = not shared_pd and manifest.get("prefix_cache") == "on" and row["case"] == "warm_exact"
        completed_bytes = row["completed_nixl_bytes"]
        if (
            any(worker not in row["queued_workers"] for worker in workers)
            or (gpu_event_timing and any(worker not in row["model_workers"] for worker in workers))
            or row["unknown_nixl_completions"]
            or row["nixl_failed_attempts"]
            or (cold and completed_bytes <= 0)
            or (cold and row["evidence"] and row["bytes"]["unmapped_bytes"] == completed_bytes)
            or (warm and not any(row["local_prefix_tokens_by_worker"].values()))
        ):
            incomplete.append(row["request_id"])
    if incomplete:
        issues.append(f"Incomplete or failed request traces: {len(incomplete)}")
    if unjoined:
        issues.append(f"Unjoined engine or transport events: {sum(unjoined.values())}")
    if all_pairs and expected_routes - actual_routes.keys():
        issues.append("Some configured P→D pairs have no request trace")
    shared_pd_probes = None
    if shared_pd and engine_trace_enabled and decode_prefix_cache == "on":
        shared_pd_probes = _shared_pd_probes(requests, expected_routes, all_pairs)
        failed_routes = [route for route, probes in shared_pd_probes.items() if probes["issues"]]
        if failed_routes:
            issues.append(f"Unverified shared LMCache PD reuse probes: {', '.join(failed_routes)}")
    handoff_statuses = None
    if engine_trace_enabled:
        handoff_statuses = Counter(row["full_nixl_handoff"]["status"] for row in requests)
        if full_nixl_handoff_required and handoff_statuses["full"] != len(requests):
            issues.append(f"Unverified full NIXL prompt handoffs: {len(requests) - handoff_statuses['full']}")
    reconciliation = _reconcile(run_dir, events, worker_ids, engine_trace_enabled)
    expected_status = "matched" if engine_trace_enabled else "trace_disabled"
    if reconciliation["status"] != expected_status:
        issues.append(f"NIXL reconciliation: {reconciliation['status']}")
    lmcache = _lmcache_report(run_dir, lmcache_endpoint)
    if lmcache is not None and lmcache["status"] != "captured":
        issues.append(f"LMCache counters: {lmcache['status']}")
    gpu_by_worker = defaultdict(Counter)
    for row in events:
        if row["event"] == "gpu_forward_duration" and isinstance(row.get("duration_ns"), int):
            gpu_by_worker[row["worker_id"]].update({"timed_batches": 1, "batch_gpu_time_ns": row["duration_ns"]})
        elif row["event"] == "gpu_forward_timing_unavailable":
            gpu_by_worker[row["worker_id"]]["unavailable_batches"] += 1
    if gpu_event_timing:
        untimed = [worker for worker in sorted(worker_ids) if not gpu_by_worker[worker]["timed_batches"]]
        if untimed:
            issues.append(f"No GPU batch timing from {', '.join(untimed)}")
    summary = {
        "engine_trace_enabled": engine_trace_enabled,
        "gpu_event_timing": gpu_event_timing,
        "unexpected_engine_events": unexpected_engine_events,
        "requests": len(requests),
        "requests_by_phase": Counter(row["phase"] for row in requests),
        "datasets": Counter(row["dataset"] for row in requests if row["dataset"]),
        "routes": {f"{p}→{d}": count for (p, d), count in sorted(actual_routes.items())},
        "missing_routes": [f"{p}→{d}" for p, d in sorted(expected_routes - actual_routes.keys())],
        "incomplete_requests": incomplete,
        "unjoined": unjoined,
        "gpu_by_worker": gpu_by_worker if engine_trace_enabled else None,
        "full_nixl_handoff_required": full_nixl_handoff_required,
        "nixl_handoff_statuses": handoff_statuses,
        "nixl_reconciliation": reconciliation,
        "lmcache": lmcache,
        "issues": issues,
    }
    if shared_pd:
        summary["shared_pd_probes"] = shared_pd_probes
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (run_dir / "requests.jsonl").open("w") as stream:
        for row in requests:
            stream.write(json.dumps(row) + "\n")
    timeline = trace_view(
        groups, timeline_limit, timeline_request,
        lmcache_enabled=bool(lmcache_endpoint), shared_pd=shared_pd,
    )
    (run_dir / "timeline.json").write_text(json.dumps(timeline) + "\n")
    return summary
