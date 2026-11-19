"""Measure one completed replay while preserving its PD cohort and trace IDs."""

from collections import Counter, defaultdict, deque
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import time
from tempfile import NamedTemporaryFile

from .pd_report import (
    _lmcache_report,
    _nixl_totals,
    _OUTER_TIMELINE_EVENTS,
    _reconcile,
    _SPANS,
    capture_metrics,
    request_joins,
    summarize_request,
    trace_view,
)
from .retrieval import DOMAINS, json_line, read_jsonl


DATASETS = ("musique", "2wiki", *(f"mtrag/{domain}" for domain in DOMAINS))
_GPU_RESULTS = ("gpu_forward_duration", "gpu_forward_timing_unavailable")
_GPU_BATCH_EVENTS = (*_GPU_RESULTS, "gpu_forward_start", "gpu_forward_end")
_REQUEST_SPANS = {start: end for start, (end, _) in _SPANS.items() if start != "gpu_forward_start"}
_REQUEST_SPAN_ENDS = {end: start for start, end in _REQUEST_SPANS.items()}
_HTTP_BYTES = (
    "http_request_body_bytes", "gateway_prefill_request_bytes",
    "gateway_prefill_response_bytes", "gateway_decode_request_bytes", "gateway_sse_payload_bytes",
)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, value):
    _write_jsonl(path, (value,))


def _write_jsonl(path, rows):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.writelines(json_line(row) for row in rows)
            stream.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


@contextmanager
def _stage(run_dir, dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset}")
    label = dataset.replace("/", "-")
    output = run_dir / "datasets" / label
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".measurement.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield output, label


def begin_dataset(run_dir, dataset):
    """Snapshot live counters and trace file positions before the replay starts."""
    with _stage(run_dir, dataset) as (output, label):
        if (run_dir / "client" / label).exists() or (output / "begin.json").exists():
            raise ValueError(f"Measurement or replay already started for {dataset} in {run_dir}")
        capture_metrics(run_dir, "before", output_dir=output)
        offsets = {path.name: path.stat().st_size for path in (run_dir / "trace").glob("*.jsonl")}
        _write_json(output / "begin.json", {
            "dataset": dataset, "run_id": run_dir.name, "wall_ns": time.time_ns(), "offsets": offsets,
        })
        return output


def _read_trace(run_dir, offsets, files="*.jsonl"):
    """Read through the file size observed at open, leaving any partial line for the next read."""
    for path in sorted((run_dir / "trace").glob(files)):
        with path.open("rb") as stream:
            size = path.stat().st_size
            start = offsets.get(path.name, 0)
            if start > size:
                raise ValueError(f"Trace was truncated since the dataset checkpoint: {path}")
            stream.seek(start)
            while stream.tell() < size:
                offset = stream.tell()
                line = stream.readline(size - offset)
                if not line.endswith(b"\n"):
                    break
                offsets[path.name] = stream.tell()
                row = json.loads(line)
                if row["schema_version"] != 1 or row["run_id"] != run_dir.name or type(row["ts_ns"]) is not int:
                    raise ValueError(f"Unexpected trace schema, run ID, or timestamp in {path} at byte {offset}")
                if row["event"] != "client_stream_chunk":
                    yield row


def _read_replay(run_dir, dataset, label):
    receipt = run_dir / "client" / label
    manifest = _read_json(receipt / "manifest.json")
    count = manifest["requested"]
    if count <= 0 or not count == manifest["succeeded"] == manifest["frozen_rows"] or manifest["failed"] != 0:
        raise ValueError(f"Incomplete replay receipt for {dataset} in {run_dir}")
    planned = run_dir.parent / "dataset-counts.json"
    if planned.exists() and count != _read_json(planned)[dataset]["requests"]:
        raise ValueError(f"Completed replay count for {dataset} disagrees with the campaign plan")
    source, _, domain = dataset.partition("/")
    rows = {}
    keys = (
        "request_id", "request_key", "conversation_id", "turn", "prompt_sha256",
        "output_token_ids_sha256", "completion_tokens",
    )
    for row in read_jsonl(receipt / "responses.jsonl"):
        if (
            row["label"] != label or row["dataset"] != source or row.get("domain") != (domain or None)
            or row["success"] is not True or row["request_id"] in rows
        ):
            raise ValueError(f"Unexpected or duplicate completion for {dataset}: {row['request_id']}")
        rows[row["request_id"]] = {key: row.get(key) for key in keys}
    if count != len(rows):
        raise ValueError(f"Replay manifest and response count disagree for {dataset}: {count} != {len(rows)}")
    return manifest, rows


def _scoped_groups(run_dir, offsets, receipts, live):
    events = _read_trace(run_dir, offsets)
    if not live:
        # The earlier campaign has no trace offsets. Keep only the target replay's
        # clock window in memory, then scope by request/attempt identity.
        by_name = defaultdict(list)
        for row in _read_trace(run_dir, {}, "client-*.jsonl"):
            if row["event"] in {"client_request_start", "client_response_end"} and row["request_id"] in receipts:
                by_name[row["event"]].append(row["ts_ns"])
        if not all(by_name[name] for name in ("client_request_start", "client_response_end")):
            raise ValueError("No client timing window found for the completed replay")
        first = min(by_name["client_request_start"])
        last = max(by_name["client_response_end"]) + 5_000_000_000
        events = (row for row in events if first <= row["ts_ns"] <= last)
    groups, _ = request_joins(list(events))
    return {key: rows for key, rows in groups.items() if key[0] in receipts}


def _gpu_key(row):
    return row["worker_id"], row["pid"], row["batch_id"]


def _gpu_batches(run_dir, offsets, groups, enabled):
    batches = defaultdict(dict)
    batch_requests = defaultdict(set)
    for request, rows in groups.items():
        for row in rows:
            if row["event"] in _GPU_BATCH_EVENTS:
                key = _gpu_key(row)
                batches[key][row["event"]] = row
                batch_requests[key].add(request)
    awaiting = {key for key, batch in batches.items() if not any(name in batch for name in _GPU_RESULTS)}
    if enabled and awaiting:
        deadline = time.monotonic() + 3
        while awaiting and time.monotonic() < deadline:
            for row in _read_trace(run_dir, offsets, "vllm-*.jsonl"):
                if row["event"] in _GPU_RESULTS and (key := _gpu_key(row)) in awaiting:
                    batches[key][row["event"]] = row
                    for request in batch_requests[key]:
                        groups[request].append(row)
                    awaiting.remove(key)
            if awaiting:
                time.sleep(0.1)
    by_worker = defaultdict(Counter)
    by_role = defaultdict(Counter)
    dispatch_ms = defaultdict(list)
    issues = []
    for batch in batches.values():
        row = next(iter(batch.values()))
        measure = Counter({"forward_batches": 1})
        begin, end = batch.get("gpu_forward_start"), batch.get("gpu_forward_end")
        if begin and end:
            dispatch_ms[row["role"]].append((end["ts_ns"] - begin["ts_ns"]) / 1e6)
        timing = batch.get("gpu_forward_duration", {}).get("duration_ns")
        if type(timing) is int and timing >= 0:
            measure.update({"timed_batches": 1, "batch_gpu_time_ns": timing})
        elif "gpu_forward_timing_unavailable" in batch:
            measure["unavailable_batches"] += 1
        elif enabled:
            measure["missing_timing_batches"] += 1
        by_worker[row["worker_id"]].update(measure)
        by_role[row["role"]].update(measure)
    if enabled:
        untimed = [worker for worker, values in by_worker.items() if not values["timed_batches"]]
        if not by_worker:
            issues.append("No CUDA forward batches found")
        elif untimed:
            issues.append(f"No CUDA event timing for active workers: {', '.join(sorted(untimed))}")
        missing = sum(values["missing_timing_batches"] for values in by_role.values())
        if missing:
            issues.append(f"No usable CUDA event result for {missing} forward batches")
    return {
        "by_worker": by_worker,
        "by_role": by_role,
        "forward_dispatch_ms_by_role": {role: _distribution(values) for role, values in dispatch_ms.items()},
    }, issues


def _distribution(values):
    ordered = sorted(values)
    count = len(ordered)
    if not count:
        return None
    return {
        "count": count, "mean": sum(ordered) / count, "max": ordered[-1],
        "p50": ordered[math.ceil(0.50 * count) - 1], "p95": ordered[math.ceil(0.95 * count) - 1],
    }


def _collect_intervals(rows, intervals):
    start_events = defaultdict(deque)
    request_intervals = defaultdict(list)
    for row in sorted(rows, key=lambda item: item["ts_ns"]):
        event = row["event"]
        if event not in _REQUEST_SPANS and event not in _REQUEST_SPAN_ENDS:
            continue
        worker_key = (row.get("worker_id"), row.get("native_request_id"), row.get("transfer_id"))
        if event in _REQUEST_SPANS:
            start_events[(event, worker_key)].append(row["ts_ns"])
        elif (start := _REQUEST_SPAN_ENDS[event]) and start_events[(start, worker_key)]:
            elapsed_ms = (row["ts_ns"] - start_events[(start, worker_key)].popleft()) / 1e6
            role = row.get("role") or row["component"]
            name = f"{role}/{start}"
            request_intervals[name].append(elapsed_ms)
            intervals[name].append(elapsed_ms)
    return request_intervals


def _measure(groups, receipts, *, lmcache, shared_pd, full_handoff):
    total_http = Counter()
    nixl_bytes = Counter()
    prompt_work = defaultdict(Counter)
    evidence_work = defaultdict(Counter)
    by_connector = defaultdict(lambda: defaultdict(Counter))
    documents = {}
    routes = Counter()
    handoffs = Counter()
    intervals = defaultdict(list)
    latency_names = ("ttft_ms", "dispatch_to_first_token_ms", "dispatch_to_last_token_ms", "end_to_end_ms")
    latencies = defaultdict(list)
    started, ended, incomplete = [], [], []
    requests = []
    transfer_events = []
    seen_ids = set()
    conversations = Counter()
    total_prompt_tokens = total_completion_tokens = requests_with_evidence = 0
    for key, events in groups.items():
        request_id = key[0]
        identity = receipts[request_id]
        row = summarize_request(key, events, lmcache_enabled=lmcache, shared_pd=shared_pd)
        anchors = {e["event"]: e for e in events if e["event"] in {
            "client_request_start", "client_dispatch", "client_response_end", "gateway_prompt_ready",
            "gateway_stream_end",
        }}
        request_events = row["events"]
        handoff = row["full_nixl_handoff"]["status"]
        if (
            request_id in seen_ids or not all(request_events.get(name) == 1 for name in _OUTER_TIMELINE_EVENTS)
            or not row["prefill"] or not row["decode"]
            or any(worker not in row["queued_workers"] for worker in (row["prefill"], row["decode"]))
            or any(row[name] is None or row[name] < 0 for name in latency_names)
            or row["unknown_nixl_completions"] or row["nixl_failed_attempts"]
            or (full_handoff and handoff != "full")
            or not all(identity[field] == row[field] for field in (
                "prompt_sha256", "output_token_ids_sha256", "completion_tokens",
            ))
            or anchors["gateway_stream_end"].get("completed") is not True
            or anchors["client_dispatch"].get("evidence") != anchors["gateway_prompt_ready"].get("evidence")
            or anchors["client_dispatch"].get("prompt_token_count") != row["prompt_tokens"]
        ):
            incomplete.append(request_id)
        seen_ids.add(request_id)
        if start := anchors.get("client_request_start"):
            started.append(start["ts_ns"])
        if end := anchors.get("client_response_end"):
            ended.append(end["ts_ns"])
        if conversation := identity["conversation_id"]:
            conversations[conversation] += 1
        request_work = defaultdict(Counter)
        for event in events:
            name = event["event"]
            if name == "engine_token_work":
                lo, hi = event["position_start"], event["position_end"]
                scheduled_prompt = max(0, min(hi, row["prompt_tokens"] or 0) - lo)
                request_work[event["role"]].update({
                    "scheduled_prompt_tokens": scheduled_prompt, "scheduled_decode_tokens": hi - lo - scheduled_prompt,
                })
            elif name == "engine_prefix_match":
                request_work[event["role"]].update({
                    "observed_local_prefix_tokens": event["local_token_count"],
                    "observed_remote_prefix_tokens": event["remote_token_count"],
                })
            elif name == "nixl_read_complete":
                transfer_events.append(event)
        for role, counts in request_work.items():
            prompt_work[role].update(counts)
        for role, selections in row.get("selected_external_connectors_by_role", {}).items():
            for selected in selections:
                by_connector[role][selected["connector_class"]].update({
                    "selections": 1, "matched_tokens": selected["matched_tokens"],
                })
        request_intervals = _collect_intervals(events, intervals)
        for name in latency_names:
            if row[name] is not None:
                latencies[name].append(row[name])
        total_http.update({name: row[name] for name in _HTTP_BYTES if row.get(name) is not None})
        total_prompt_tokens += row["prompt_tokens"] or 0
        total_completion_tokens += row["completion_tokens"] or 0
        routes[f"{row['prefill']}→{row['decode']}"] += 1
        handoffs[handoff] += 1
        requests_with_evidence += bool(row["evidence"])
        bytes_by_occurrence = row["bytes"]["by_chunk_occurrence"]
        attributed = sum(bytes_by_occurrence.values())
        nixl_bytes.update({
            "completed_bytes": row["completed_nixl_bytes"], "exclusive_evidence_bytes": attributed,
            **{name: row["bytes"][name] for name in (
                "shared_or_boundary_bytes", "outside_evidence_bytes", "unmapped_bytes",
            )},
        })
        for chunk in row["evidence"]:
            doc_id = chunk["doc_id"]
            document = documents.setdefault(doc_id, {
                "doc_id": doc_id, "occurrences": 0, "request_count": 0, "last_request_id": None,
                "chunk_ids": set(), "exclusive_nixl_bytes": 0, "token_work_by_role": defaultdict(Counter),
            })
            document["occurrences"] += 1
            if document["last_request_id"] != request_id:
                document["request_count"] += 1
                document["last_request_id"] = request_id
            document["chunk_ids"].add(chunk["chunk_id"])
            chunk_bytes = bytes_by_occurrence[chunk["occurrence"]]
            document["exclusive_nixl_bytes"] += chunk_bytes
            if chunk["token_work"]:
                for role, counts in chunk["token_work"].items():
                    evidence_work[role].update(counts)
                    document["token_work_by_role"][role].update(counts)
        requests.append({
            **row, "request_key": identity["request_key"], "conversation_id": identity["conversation_id"],
            "turn": identity["turn"], "token_work_by_role": request_work, "stage_spans_ms": request_intervals,
        })

    missing = receipts.keys() - seen_ids
    incomplete = sorted(set(incomplete) | missing)
    stage_ms = (max(ended) - min(started)) / 1e6 if started and ended else None
    aggregates = {
        "latency_ms": {name: _distribution(latencies[name]) for name in latency_names},
        "stage_duration_ms": stage_ms,
        "requests_per_second": len(receipts) * 1000 / stage_ms if stage_ms and stage_ms > 0 else None,
        "stage_spans_ms": {name: _distribution(values) for name, values in sorted(intervals.items())},
        "routes": routes,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "conversations": len(conversations),
        "multi_turn_conversations": sum(turns > 1 for turns in conversations.values()),
        "nixl_bytes": nixl_bytes,
        "nixl_handoff_statuses": handoffs,
        "http_payload_bytes": total_http,
        "token_work_by_role": prompt_work,
        "evidence": {
            "requests_with_evidence": requests_with_evidence,
            "distinct_documents": len(documents),
            "occurrences": sum(row["occurrences"] for row in documents.values()),
            "token_work_by_role": evidence_work,
        },
        "connector_selections_by_role": by_connector,
    }
    doc_rows = []
    for doc in documents.values():
        doc.pop("last_request_id")
        doc["chunk_ids"] = sorted(doc["chunk_ids"])
        doc_rows.append(doc)
    return aggregates, requests, doc_rows, transfer_events, incomplete


def _provenance(run_dir, manifest):
    campaign_revision = run_dir.parent / "full-campaign-source.txt"
    cache = manifest.get("lmcache")
    return {
        "campaign_commit": campaign_revision.read_text(encoding="utf-8").strip() if campaign_revision.is_file() else None,
        "topology": manifest["topology"],
        "model": manifest["model"],
        "model_revision": manifest["revision"],
        "prefix_cache_by_role": manifest["prefix_cache_by_role"],
        "vllm_commit": manifest["vllm_source"]["commit"],
        "lmcache": {"mode": cache["mode"], "commit": cache["source"]["commit"]} if cache else None,
    }


def finish_dataset(run_dir, dataset, *, timeline_limit=8):
    """Summarize a complete replay and reconcile live dataset-window counters if available."""
    with _stage(run_dir, dataset) as (output, label):
        summary_path = output / "summary.json"
        if summary_path.exists():
            summary = _read_json(summary_path)
            if summary["status"] == "passed":
                return summary
        replay, receipts = _read_replay(run_dir, dataset, label)
        endpoints = _read_json(run_dir / "endpoints.json")
        run = _read_json(run_dir / "run_manifest.json")
        if run.get("engine_trace_enabled") is not True:
            raise ValueError("Dataset measurement requires native vLLM engine tracing")
        checkpoint = output / "begin.json"
        live = checkpoint.is_file()
        offsets = {}
        if live:
            begin = _read_json(checkpoint)
            if (begin["dataset"], begin["run_id"]) != (dataset, run_dir.name):
                raise ValueError("Dataset measurement checkpoint belongs to a different replay")
            offsets = begin["offsets"]
        groups = _scoped_groups(run_dir, offsets, receipts, live)
        gpu, issues = _gpu_batches(run_dir, offsets, groups, run.get("gpu_event_timing", True))
        lmcache = endpoints.get("lmcache")
        shared_pd = (run.get("lmcache") or {}).get("mode") == "shared_pd"
        aggregates, requests, docs, transfers, incomplete = _measure(
            groups, receipts, lmcache=bool(lmcache), shared_pd=shared_pd, full_handoff=run["full_nixl_handoff"],
        )
        workers = {worker["id"] for worker in endpoints["prefill"] + endpoints["decode"]}
        if live:
            if not (output / "metrics-after.json").exists():
                capture_metrics(run_dir, "after", output_dir=output, nixl_events=transfers)
            reconciliation = _reconcile(output, transfers, workers, True)
            if reconciliation["status"] != "matched":
                issues.append(f"NIXL trace and dataset counter deltas disagree: {reconciliation['status']}")
            lmcache_counters = _lmcache_report(output, lmcache)
            if lmcache_counters and lmcache_counters["status"] != "captured":
                issues.append(f"LMCache dataset counter snapshots: {lmcache_counters['status']}")
        else:
            traced = _nixl_totals(transfers)
            reconciliation = {
                "status": "trace_only",
                "workers": {worker: {"trace": {key: traced[worker][key] for key in ("sum", "count")}}
                            for worker in sorted(workers)},
            }
            lmcache_counters = {"status": "no_dataset_window_snapshots"} if lmcache else None
        if incomplete:
            issues.append(f"Incomplete or failed request traces: {len(incomplete)}")
        if aggregates["nixl_bytes"]["unmapped_bytes"]:
            issues.append("Completed NIXL bytes have no verified token position")
        routes = {
            f"{p['id']}→{d['id']}" for p in endpoints["prefill"] for d in endpoints["decode"]
        }
        if aggregates["routes"].keys() - routes:
            issues.append("Completed requests used routes missing from the cohort manifest")
        _write_jsonl(output / "requests.jsonl", requests)
        _write_jsonl(output / "documents.jsonl", sorted(docs, key=lambda doc: doc["doc_id"]))
        _write_json(output / "timeline.json", trace_view(
            groups, timeline_limit, None, lmcache_enabled=bool(lmcache), shared_pd=shared_pd,
        ))
        summary = {
            "schema_version": 1,
            "campaign": run_dir.parent.name,
            "cohort": run_dir.name,
            "dataset": dataset,
            "label": label,
            "status": "failed" if issues else "passed",
            "requests": len(receipts),
            "provenance": _provenance(run_dir, run),
            "replay_mode": replay["mode"],
            "concurrency": replay.get("concurrency"),
            "input_sha256": replay["input_sha256"],
            "frozen_sha256": replay["frozen_sha256"],
            "counter_reconciliation": reconciliation,
            "lmcache": lmcache_counters,
            "gpu_forward": gpu,
            "incomplete_requests": incomplete,
            "issues": issues,
            **aggregates,
        }
        _write_json(summary_path, summary)
        return summary
