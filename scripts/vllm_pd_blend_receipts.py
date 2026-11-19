"""Check the serial CacheBlend pilot against server counters and native PD traces."""

from collections import Counter
from contextlib import suppress
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidencestore.pd_report import (
    _OUTER_TIMELINE_EVENTS,
    _PROMETHEUS_LABEL,
    _PROMETHEUS_SAMPLE,
    _reconcile,
    load_events,
    request_joins,
    summarize_request,
    trace_view,
)
from evidencestore.retrieval import prompt_sha256, read_jsonl, sha256_file


METRIC_NAMES = {
    "fingerprints_registered": ("lmcache_blend_fingerprints_registered_total",),
    "prefix_hit_tokens": (
        "lmcache_blend_lookup_prefix_hit_tokens_total",
        "lmcache_blend_lookup_prefix_hit_tokens_tokens_total",
    ),
    "non_prefix_hit_tokens": (
        "lmcache_blend_lookup_non_prefix_hit_tokens_total",
        "lmcache_blend_lookup_non_prefix_hit_tokens_tokens_total",
    ),
    "scatter_prefix_chunks": ("lmcache_blend_scatter_prefix_chunks_total",),
    "scatter_shifted_chunks": ("lmcache_blend_scatter_shifted_chunks_total",),
    "retrieve_requests": ("lmcache_blend_retrieve_requests_total",),
    "retrieve_completed": (
        "lmcache_blend_retrieve_duration_milliseconds_count",
        "lmcache_blend_retrieve_duration_ms_count",
        "lmcache_blend_retrieve_duration_count",
    ),
    "retrieve_failures": ("lmcache_blend_retrieve_failures_total",),
    "mp_chunks_loaded": ("lmcache_mp_num_chunks_loaded_total",),
    "mp_h2d_staging_bytes": ("lmcache_mp_transfer_phase_bytes_total", "lmcache_mp_transfer_phase_bytes_bytes_total"),
}
METRIC_FIELDS = {name: field for field, names in METRIC_NAMES.items() for name in names}
CASES = ("seed-exact", "seed-shifted", "exact", "shifted")


def _save_json(path, row):
    path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_samples(raw):
    samples = {}
    for line in raw.decode().splitlines():
        match = _PROMETHEUS_SAMPLE.fullmatch(line)
        if not match or (field := METRIC_FIELDS.get(match["name"])) is None:
            continue
        labels = {label: json.loads(value) for label, value in _PROMETHEUS_LABEL.findall(match["labels"] or "")}
        if field == "mp_h2d_staging_bytes" and (labels.get("direction"), labels.get("phase")) != ("h2d", "staging"):
            continue
        sample = (field, tuple(sorted(labels.items())))
        try:
            value = Decimal(match["value"])
        except InvalidOperation as exc:
            raise ValueError(f"Invalid LMCache sample: {line}") from exc
        if sample in samples or not value.is_finite() or value < 0 or value != value.to_integral_value():
            raise ValueError(f"Invalid or duplicate LMCache sample: {line}")
        samples[sample] = int(value)
    return samples


def _deltas(before, after):
    totals = Counter()
    for (field, labels), old in before.items():
        if after.get((field, labels), -1) < old:
            raise ValueError(f"LMCache metric reset or disappeared: {field} {dict(labels)}")
    for sample, value in after.items():
        totals[sample[0]] += value - before.get(sample, 0)
    return totals


def _missing_evidence(case, delta):
    if case.startswith("seed-"):
        return [] if delta["fingerprints_registered"] > 0 else ["new fingerprints"]
    missing = []
    if delta["retrieve_failures"]:
        missing.append("zero CB retrieve failures")
    complete = delta["retrieve_completed"] > 0 and delta["retrieve_completed"] == delta["retrieve_requests"]
    if case == "exact":
        if not delta["prefix_hit_tokens"]:
            missing.append("CB prefix hit tokens")
        if not (delta["mp_h2d_staging_bytes"] or delta["mp_chunks_loaded"]
                or (delta["scatter_prefix_chunks"] and complete)):
            missing.append("a completed prefix load")
    else:
        if not delta["non_prefix_hit_tokens"]:
            missing.append("CB non-prefix hit tokens")
        if not delta["scatter_shifted_chunks"]:
            missing.append("shifted scatter chunks")
        if not complete:
            missing.append("completed CB retrieve observations")
    return missing


def _scrape(url):
    with urlopen(url.rstrip("/") + "/metrics", timeout=5) as response:
        return response.read()


def capture_case(url, run_dir, case, command):
    """Replay one case between snapshots from an otherwise idle LMCache server."""
    if case not in CASES:
        raise ValueError(f"Unknown Blend case {case}")
    if urlsplit(url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("LMCache metrics must use a loopback endpoint")
    target = run_dir / "blend-receipts" / case
    target.mkdir(parents=True)
    before = _scrape(url)
    (target / "before.prom").write_bytes(before)
    baseline = _metric_samples(before)
    after_path = target / "after.prom"
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        with suppress(OSError):
            after_path.write_bytes(_scrape(url))
        raise

    deadline = time.monotonic() + 30
    previous = None
    missing = ["post-request server metrics"]
    delta = Counter()
    while time.monotonic() < deadline:
        try:
            after = _scrape(url)
        except OSError:
            time.sleep(0.2)
            continue
        after_path.write_bytes(after)
        delta = _deltas(baseline, _metric_samples(after))
        now = time.monotonic()
        if delta != previous:
            previous, stable_since = delta, now
        missing = _missing_evidence(case, delta)
        if delta["retrieve_failures"] and not case.startswith("seed-"):
            break
        # Stream callbacks and fingerprint insertion can lag the HTTP response.
        if not missing and now - stable_since >= 0.5:
            _save_json(target / "lmcache.json", {"status": "passed", "case": case, "delta": delta})
            print(f"{case} LMCache counter checks passed: {dict(delta)}", flush=True)
            return
        time.sleep(0.2)

    _save_json(target / "lmcache.json", {"status": "failed", "case": case, "delta": delta, "missing": missing})
    raise ValueError(f"{case} missing stable LMCache evidence ({', '.join(missing) or 'counter window did not settle'}); inspect {target}")


def _one_row(path):
    rows = list(read_jsonl(path))
    if len(rows) != 1:
        raise ValueError(f"Expected one request in {path}")
    return rows[0]


def _case_metrics(run_dir, case):
    directory = run_dir / "blend-receipts" / case
    receipt = _read_json(directory / "lmcache.json")
    before, after = (_metric_samples((directory / f"{name}.prom").read_bytes()) for name in ("before", "after"))
    delta = _deltas(before, after)
    if receipt != {"status": "passed", "case": case, "delta": delta} or _missing_evidence(case, delta):
        raise ValueError(f"{case} LMCache receipt differs from its raw counters")
    return delta


def _verify_replay(run_dir, fixture_path, case):
    fixture = _one_row(fixture_path)
    replay = run_dir / "client" / case
    manifest = _read_json(replay / "manifest.json")
    frozen_path = replay / "frozen.jsonl"
    response = _one_row(replay / "responses.jsonl")
    digest = prompt_sha256(fixture["prompt_token_ids"])
    if (manifest["mode"] != "frozen" or manifest["concurrency"] != 1
            or (manifest["requested"], manifest["succeeded"], manifest["failed"], manifest["frozen_rows"])
            != (1, 1, 0, 1)
            or manifest["input_sha256"] != sha256_file(fixture_path)
            or manifest["frozen_sha256"] != sha256_file(frozen_path)
            or _one_row(frozen_path) != fixture
            or response["success"] is not True or response["label"] != case or response["case"] != case
            or fixture["case"] != case or response["pd_route"] != fixture["pd_route"]
            or response["request_key"] != fixture["request_key"]
            or response["prompt_sha256"] != digest or fixture["prompt_sha256"] != digest
            or not response["output_token_ids_sha256"] or not response["completion_tokens"]):
        raise ValueError(f"{case} replay failed or used an unexpected request")
    return fixture, response


def _request_without_trace(fixture, response):
    return {
        "request_id": response["request_id"],
        "case": response["case"],
        **fixture["pd_route"],
        "route_source": "requested",
        "request_level_trace_status": "trace_disabled",
        "prompt_sha256": response["prompt_sha256"],
        "prompt_tokens": len(fixture["prompt_token_ids"]),
        "output_token_ids_sha256": response["output_token_ids_sha256"],
        "completion_tokens": response["completion_tokens"],
        "evidence": [
            {**chunk, "occurrence": index, "token_work": None}
            for index, chunk in enumerate(fixture["evidence"])
        ],
        "ttft_ms": None,
        "dispatch_to_first_token_ms": None,
        "dispatch_to_last_token_ms": None,
        "end_to_end_ms": None,
        "p_scheduler_prompt_tokens": None,
        "p_scheduler_evidence_tokens": None,
        "completed_nixl_bytes": None,
        "full_nixl_handoff": None,
        "nixl_transfers": None,
        "bytes": None,
    }


def _verify_traced_request(fixture, response, case, groups):
    matches = [(key, rows) for key, rows in groups.items() if key[0] == response["request_id"]]
    if len(matches) != 1:
        raise ValueError(f"{case} has {len(matches)} joined request attempts")
    key, rows = matches[0]
    native = summarize_request(key, rows, lmcache_enabled=True, shared_pd=True)
    route = fixture["pd_route"]
    prefill, decode = native["prefill"], native["decode"]
    native_evidence = [
        {name: value for name, value in chunk.items() if name not in {"occurrence", "token_work"}}
        for chunk in native["evidence"]
    ]
    if ({"prefill": prefill, "decode": decode} != route
            or native["case"] != case or native["prompt_sha256"] != fixture["prompt_sha256"]
            or native["prompt_tokens"] != len(fixture["prompt_token_ids"])
            or native_evidence != fixture["evidence"]
            or native["output_token_ids_sha256"] != response["output_token_ids_sha256"]
            or not native["output_token_ids_sha256"] or native["completion_tokens"] != response["completion_tokens"]
            or any(native["events"].get(event) != 1 for event in _OUTER_TIMELINE_EVENTS)
            or any(worker not in native["queued_workers"] for worker in (prefill, decode))
            or native["unknown_nixl_completions"] or native["nixl_failed_attempts"]):
        raise ValueError(f"{case} has an incomplete or mismatched native PD request trace")
    if native["full_nixl_handoff"]["status"] != "full" or native["completed_nixl_bytes"] <= 0:
        raise ValueError(f"{case} missing full native NIXL prompt transfer: {native['full_nixl_handoff']}")

    prompt_tokens = native["prompt_tokens"]
    steps = [row for row in rows if row["event"] == "engine_token_work" and row["worker_id"] == prefill]
    native["p_scheduler_prompt_tokens"] = sum(
        max(0, min(step["position_end"], prompt_tokens) - step["position_start"]) for step in steps
    ) if steps else None
    native["p_scheduler_evidence_tokens"] = sum(
        chunk["token_work"].get("P", {}).get("scheduled_prompt_tokens", 0) for chunk in native["evidence"]
    ) if steps else None
    native["route_source"] = "gateway"
    native["request_level_trace_status"] = "captured"
    return native


def verify_topology(directory, topology, cases, *, trace_enabled=True):
    """Validate every completed native PD phase independently in this topology."""
    requests, reconciliations = {}, {}
    server_endpoint = None
    for mode, names in cases.items():
        run_dir = directory / mode
        manifest = _read_json(run_dir / "run_manifest.json")
        endpoints = _read_json(run_dir / "endpoints.json")
        cache = manifest["lmcache"]
        if (manifest["topology"] != topology or manifest["engine_trace_enabled"] is not trace_enabled
                or manifest["prefill_mode"] != mode
                or manifest["prefix_cache_by_role"]["D"] != "off"
                or cache["decode_workers"] != [] or cache["server_ownership"] != "external"
                or cache["server_engine"] != "blend"):
            raise ValueError(f"{topology} {mode} does not match the trace mode and bare NIXL decoder configuration")
        endpoint = {"http_url": cache["url"], "rpc_port": cache["port"]}
        if (endpoint["http_url"] != endpoints["lmcache"]["url"]
                or (server_endpoint is not None and endpoint != server_endpoint)):
            raise ValueError(f"{topology} {mode} used a different LMCache server endpoint")
        server_endpoint = endpoint
        events = []
        if trace_enabled:
            events = list(load_events(run_dir))
            groups, unjoined = request_joins(events)
            if len(groups) != len(names) or unjoined:
                raise ValueError(f"{topology} {mode} has unexpected requests or unjoined events: {dict(unjoined)}")
        elif any((run_dir / "trace").glob("*.jsonl")):
            raise ValueError(f"{topology} {mode} emitted detailed JSONL with tracing disabled")
        prefill_workers, decode_workers = ({worker["id"] for worker in endpoints[side]} for side in ("prefill", "decode"))
        reconciliation = _reconcile(run_dir, events, prefill_workers | decode_workers, trace_enabled)
        expected_status = "matched" if trace_enabled else "trace_disabled"
        if reconciliation["status"] != expected_status:
            raise ValueError(f"{topology} {mode} NIXL counter reconciliation: {reconciliation['status']}")
        by_decoder = Counter()
        for case in names:
            fixture, response = _verify_replay(run_dir, directory / "inputs" / f"{case}.jsonl", case)
            native = (_verify_traced_request(fixture, response, case, groups) if trace_enabled
                      else _request_without_trace(fixture, response))
            if native["prefill"] not in prefill_workers or native["decode"] not in decode_workers:
                raise ValueError(f"{topology} {mode}/{case} used an unknown worker")
            native["lmcache_counter_delta"] = _case_metrics(run_dir, case)
            requests[case] = native
            by_decoder[native["decode"]] += 1
        deltas = {worker: row["prometheus"] for worker, row in reconciliation["workers"].items()}
        for worker, count in by_decoder.items():
            if deltas[worker]["sum"] <= 0 or deltas[worker]["count"] < count:
                raise ValueError(f"{topology} {mode} needs positive NIXL bytes and at least {count} transfers on {worker}")
        reconciliation["aggregate"] = {
            "completed_bytes": sum(delta["sum"] for delta in deltas.values()),
            "completed_transfers": sum(delta["count"] for delta in deltas.values()),
        }
        reconciliations[mode] = reconciliation
        if trace_enabled:
            _save_json(run_dir / "blend-receipts" / "native-timeline.json",
                       trace_view(groups, len(names), None, lmcache_enabled=True))

    if requests["seed-exact"]["prompt_sha256"] != requests["exact"]["prompt_sha256"]:
        raise ValueError("The exact seed and repeat have different prompt hashes")
    summary = {
        "status": "passed",
        "topology": topology,
        "engine_trace_enabled": trace_enabled,
        "lmcache_server_endpoint": server_endpoint,
        "cases": requests,
        "nixl_reconciliation_by_phase": reconciliations,
        "exact_output_hashes_match": (
            requests["seed-exact"]["output_token_ids_sha256"] == requests["exact"]["output_token_ids_sha256"]
        ),
        "measurement_scope": {
            "p_scheduler_prompt_tokens": "Native vLLM scheduler token positions." if trace_enabled else None,
            "request_level_trace": "captured" if trace_enabled else "unavailable_trace_disabled",
            "nixl_aggregate": "Native counters per worker and phase.",
            "lmcache_counter_delta": "Server-wide counters bracketed around one serial request.",
            "private_blend_repair_token_positions": None,
            "private_blend_scatter_bytes": None,
        },
    }
    _save_json(directory / "blend-summary.json", summary)
    return summary
