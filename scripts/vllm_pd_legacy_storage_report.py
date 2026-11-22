#!/usr/bin/env python3
"""Report shadow LMCache host-storage measurements and native lookup proof."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidencestore.pd_report import load_events, request_joins
from evidencestore.retrieval import read_jsonl
from vllm_pd_legacy_storage_smoke import VERIFICATION_LABEL


REPORT_FILENAME = "legacy-kv-storage-report.json"
PUT = "legacy_kv_storage_put"
RESTORE = "legacy_kv_storage_restore"
PUT_STATUSES = {"measured", "oversized", "unsupported", "not_resident", "error"}
RESTORE_STATUSES = {"verified", "missing_base", "base_mismatch", "target_mismatch", "reconstruction_mismatch", "error"}
SAFE_FIELDS = (
    "event", "worker_id", "pid", "native_request_id", "backend", "ts_ns", "duration_ns", "decode_duration_ns", "layer_id",
    "start_token", "end_token", "key_sha256", "target_sha256", "raw_bytes", "encoding", "encoded_bytes",
    "candidate_count", "base_key_sha256", "base_sha256", "base_raw_bytes", "base_start_token", "base_end_token",
    "manifest_bytes", "patch_bytes", "tile_bytes", "tile_count", "changed_tiles", "common_prefix_tokens",
    "status", "observed_sha256", "reconstructed_sha256", "tensor_shape", "tensor_dtype", "row_bytes",
)


def _responses(run_dir, label, *, optional=False):
    path = run_dir / "client" / label / "responses.jsonl"
    if optional and not path.exists():
        return {}
    rows = list(read_jsonl(path))
    responses = {row["request_id"]: row for row in rows}
    if (not rows or len(rows) != len(responses) or len({row["request_key"] for row in rows}) != len(rows)
            or any(row["label"] != label or row.get("success") is not True for row in rows)):
        raise ValueError(f"Invalid {label} response receipts")
    return responses


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _nonnegative(value):
    return type(value) is int and value >= 0


def _measurement(row):
    status = row["status"]
    put = row["event"] == PUT
    if (status not in (PUT_STATUSES if put else RESTORE_STATUSES)
            or row["component"] != "lmcache" or row["role"] != "P" or row["backend"] != "LocalCPUBackend"
            or not isinstance(row["native_request_id"], str) or not row["native_request_id"]
            or not _nonnegative(row["pid"]) or not _nonnegative(row["duration_ns"])):
        raise ValueError("Invalid LMCache storage trace receipt")
    decode_ns = row.get("decode_duration_ns")
    if decode_ns is not None and (put or not _nonnegative(decode_ns) or decode_ns > row["duration_ns"]):
        raise ValueError("Invalid delta codec duration")
    if status == "measured" or not put:
        integers = ("layer_id", "start_token", "end_token", "raw_bytes", "encoded_bytes", "manifest_bytes", "patch_bytes")
        if (any(not _nonnegative(row[field]) for field in integers)
                or not 0 <= row["start_token"] < row["end_token"] or row["raw_bytes"] <= 0
                or not _sha(row["key_sha256"]) or not _sha(row["target_sha256"])):
            raise ValueError("Invalid LMCache measured target or byte count")
        encoding = row["encoding"]
        if encoding == "delta":
            if (row["encoded_bytes"] != row["manifest_bytes"] + row["patch_bytes"]
                    or not 0 < row["manifest_bytes"] <= row["encoded_bytes"] < row["raw_bytes"]
                    or not _nonnegative(row["base_raw_bytes"]) or row["base_raw_bytes"] == 0
                    or not _sha(row["base_key_sha256"]) or not _sha(row["base_sha256"])
                    or put and not (_nonnegative(row["candidate_count"]) and row["candidate_count"] > 0)):
                raise ValueError("Invalid LMCache delta accounting")
        elif (encoding != "raw" or not put or row["encoded_bytes"] != row["raw_bytes"]
                or row["manifest_bytes"] or row["patch_bytes"]
                or not _nonnegative(row["candidate_count"])):
            raise ValueError("Invalid LMCache raw accounting")
    return {field: row[field] for field in SAFE_FIELDS if field in row}


def _cache_key(row, field="key_sha256"):
    return row["worker_id"], row["pid"], row["layer_id"], row[field]


def _native_transfer(row, events):
    name = "legacy_lmcache_d2h_complete" if row["event"] == PUT else "legacy_lmcache_h2d_complete"
    return any(
        event["event"] == name and event["worker_id"] == row["worker_id"] and event["pid"] == row["pid"]
        and event["native_request_id"] == row["native_request_id"] and event["layers"] > row["layer_id"]
        and event["started_ns"] <= row["ts_ns"] <= event["ts_ns"]
        and any((span["start_token"], span["end_token"]) == (row["start_token"], row["end_token"])
                for span in event["mapped_ranges"])
        for event in events
    )


def _request_trace(events):
    request_routes = {}
    bindings = {}
    groups, _ = request_joins(events)
    for (request_id, _), grouped in groups.items():
        measurements = [event for event in grouped if event["event"] in (PUT, RESTORE)]
        routes = {name: [event for event in grouped if event["event"] == name]
                  for name in ("gateway_dispatch_prefill", "gateway_prefill_response", "gateway_route_decode")}
        if all(len(rows) == 1 for rows in routes.values()):
            if request_id in request_routes:
                raise ValueError(f"Multiple native request attempts for {request_id}")
            start = routes["gateway_dispatch_prefill"][0]
            end = routes["gateway_prefill_response"][0]
            request_routes[request_id] = (start["worker_id"], routes["gateway_route_decode"][0]["worker_id"])
            for event in measurements:
                if (id(event) in bindings or event["worker_id"] != start["worker_id"]
                        or event["native_request_id"] != end["remote_request_id"]
                        or not start["ts_ns"] <= event["ts_ns"] <= end["ts_ns"]):
                    raise ValueError(f"LMCache storage event differs from native prefill attempt for {request_id}")
                bindings[id(event)] = (request_id, grouped)
    return request_routes, bindings


def _verification_sources(natural, verification, routes):
    sources = {}
    for request_id, row in verification.items():
        prefix = VERIFICATION_LABEL + ":"
        source = natural.get(row["request_key"].removeprefix(prefix))
        if (not row["request_key"].startswith(prefix) or source is None
                or not _sha(row.get("prompt_sha256")) or row["prompt_sha256"] != source.get("prompt_sha256")
                or request_id not in routes or routes[request_id] != routes.get(source["request_id"])
                or row.get("pd_route") != dict(zip(("prefill", "decode"), routes[request_id], strict=True))):
            raise ValueError(f"Verification request {request_id} differs from its original prompt or route")
        sources[request_id] = source["request_id"]
    return sources


def _duration(rows, field="duration_ns"):
    times = sorted(row[field] for row in rows if row.get(field) is not None)
    return {"events": len(times), "sum_ns": sum(times), "median_ns": median(times) if times else None,
            "max_ns": times[-1] if times else None}


def _byte_totals(rows):
    raw = sum(row["raw_bytes"] for row in rows)
    selected = sum(row["encoded_bytes"] for row in rows)
    return {"raw_bytes": raw, "selected_bytes": selected, "gross_saved_bytes": raw - selected,
            "gross_saved_fraction": (raw - selected) / raw if raw else None}


def build_report(run_dir: Path, *, replay_label="mtrag"):
    """Validate trace measurements without touching the source replay."""
    if replay_label != "mtrag":
        raise ValueError("LMCache storage reporting requires the MT-RAG replay label")
    run_dir = run_dir.expanduser().resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("legacy_full_blend", {}).get("kv_storage_trace_enabled") is not True:
        raise ValueError("LMCache storage tracing was not enabled for this replay")
    natural = _responses(run_dir, replay_label)
    verification = _responses(run_dir, VERIFICATION_LABEL, optional=True)
    events = list(load_events(run_dir))
    routes, bindings = _request_trace(events)
    verification_sources = _verification_sources(natural, verification, routes)
    scoped_stores, scoped_restores = [], []
    latest, verified_store_events, verified_exact_replays = {}, set(), set()
    unjoined = 0
    storage_events = sorted((row for row in events if row["event"] in (PUT, RESTORE)), key=lambda row: row["ts_ns"])
    for event in storage_events:
        row = _measurement(event)
        request_id, grouped = bindings.get(id(event), (None, ()))
        if request_id is None:
            unjoined += 1
        if row["event"] == PUT:
            if row["status"] == "measured":
                raw_base = latest.get(_cache_key(row, "base_key_sha256")) if row["encoding"] == "delta" else None
                row["matched_observed_raw_base"] = bool(raw_base and raw_base["encoding"] == "raw"
                                                       and raw_base["raw_bytes"] == row["base_raw_bytes"]
                                                       and raw_base["target_sha256"] == row["base_sha256"]
                                                       and raw_base["native_transfer_matched"])
                row["native_transfer_matched"] = _native_transfer(row, grouped)
                latest[_cache_key(row)] = row
            row["request_id"] = request_id
            if request_id in natural:
                replay = natural[request_id]
                row.update(request_key=replay["request_key"], conversation_id=replay.get("conversation_id"),
                           turn=replay.get("turn"))
                scoped_stores.append(row)
        elif request_id in natural or request_id in verification:
            replay = natural.get(request_id) or verification[request_id]
            stored = latest.get(_cache_key(row))
            matched_store = bool(stored and stored["encoding"] == "delta" and stored["ts_ns"] < row["ts_ns"]
                                 and all(stored[field] == row[field] for field in (
                                     "target_sha256", "raw_bytes", "encoded_bytes", "base_key_sha256", "base_sha256",
                                     "base_raw_bytes", "manifest_bytes", "patch_bytes",
                                 )))
            matched_source = matched_store and stored["request_id"] == verification_sources.get(request_id)
            equality = (row["status"] == "verified" and row["target_sha256"] == row.get("observed_sha256")
                        == row.get("reconstructed_sha256"))
            native = _native_transfer(row, grouped)
            base = latest.get(_cache_key(row, "base_key_sha256"))
            matched_base = bool(base and base["encoding"] == "raw" and base["target_sha256"] == row["base_sha256"]
                                and base["raw_bytes"] == row["base_raw_bytes"] and base["native_transfer_matched"])
            verified = bool(matched_store and matched_base and equality and native
                            and stored["native_transfer_matched"] and stored["matched_observed_raw_base"])
            row.update(request_id=request_id, request_key=replay["request_key"], label=replay["label"],
                       conversation_id=replay.get("conversation_id"), turn=replay.get("turn"),
                       store_request_id=stored["request_id"] if matched_store else None,
                       requested_store_source_matched=matched_source if request_id in verification_sources else None,
                       matched_observed_raw_base=matched_base, native_transfer_matched=native,
                       reconstruction_proof_verified=verified)
            if verified:
                verified_store_events.add(id(stored))
                if matched_source:
                    verified_exact_replays.add(request_id)
            scoped_restores.append(row)

    measured = [row for row in scoped_stores if row["status"] == "measured"]
    delta = [row for row in measured if row["encoding"] == "delta"]
    nonidentical = [row for row in delta if row["base_sha256"] != row["target_sha256"]]
    verified = [row for row in delta if id(row) in verified_store_events]
    verified_loads = [row for row in scoped_restores if row["reconstruction_proof_verified"]]
    failed = [row for row in scoped_restores if not row["reconstruction_proof_verified"]]
    diagnostics = []
    if not measured:
        diagnostics.append("No resident storage samples joined to the MT-RAG replay.")
    if measured and not delta:
        diagnostics.append("Measured MT-RAG samples selected raw storage; gross savings are zero.")
    if delta and not verified:
        diagnostics.append("No selected MT-RAG target could be verified on a later native lookup.")
    if set(verification) - verified_exact_replays:
        diagnostics.append("Some exact replays did not verify the storage target selected from their source request.")
    if failed:
        diagnostics.append("Some native lookups failed reconstruction, digest, or request/transfer linkage checks.")
    if unjoined:
        diagnostics.append("Some storage events could not be joined to a native P request.")
    return {
        "schema_version": 1,
        "status": "proof_failed" if failed else "measured" if measured else "insufficient_evidence",
        "replay_label": replay_label,
        "natural_requests": len(natural),
        "exact_replay_requests": len(verification),
        "scope": {
            "sample": "Sampled MT-RAG layer storage writes only; no extrapolation to all layers or requests.",
            "bytes": "Gross per-store raw vs selected representation, including manifest and patch bytes.",
            "retention": "Native raw tensors remain authoritative. Additional base pin lifetimes and net occupancy are unmodeled.",
            "duration": "Per-event diagnostic encoding or reconstruction time; sums are not request latency.",
        },
        "storage": {
            "put_event_statuses": dict(Counter(row["status"] for row in scoped_stores)),
            "measured_store_samples": len(measured),
            "requests_with_measured_samples": len({row["request_id"] for row in measured}),
            "sampled_layers": sorted({row["layer_id"] for row in measured}),
            "resident_candidates_examined": sum(row["candidate_count"] for row in measured),
            "stores_with_resident_candidates": sum(row["candidate_count"] > 0 for row in measured),
            "raw_selected_samples": len(measured) - len(delta),
            "delta_selected_samples": len(delta),
            "identical_delta_samples": len(delta) - len(nonidentical),
            "nonidentical_delta_samples": len(nonidentical),
            "nonidentical_delta_byte_totals": _byte_totals(nonidentical),
            **_byte_totals(measured),
            "delta_manifest_bytes": sum(row["manifest_bytes"] for row in delta),
            "delta_patch_bytes": sum(row["patch_bytes"] for row in delta),
            "store_diagnostic_duration": _duration(scoped_stores),
        },
        "native_lookup_verification": {
            "restore_event_statuses": dict(Counter(row["status"] for row in scoped_restores)),
            "delta_store_samples_with_observed_raw_base_write": sum(row["matched_observed_raw_base"] for row in delta),
            "delta_store_samples_with_verified_load": len(verified),
            "nonidentical_delta_store_samples_with_verified_load": sum(
                row["base_sha256"] != row["target_sha256"] for row in verified
            ),
            "delta_store_samples_without_verified_load": len(delta) - len(verified),
            "gross_saved_bytes_with_verified_load": sum(row["raw_bytes"] - row["encoded_bytes"] for row in verified),
            "verified_restore_base_raw_bytes": sum(row["base_raw_bytes"] for row in verified_loads),
            "verified_restore_encoded_bytes_including_manifest": sum(row["encoded_bytes"] for row in verified_loads),
            "exact_replay_requests_with_verified_source_target": len(verified_exact_replays),
            "exact_replay_requests_without_verified_source_target": len(verification) - len(verified_exact_replays),
            "failed_proof_events": len(failed),
            "unjoined_storage_events": unjoined,
            "restore_diagnostic_duration": _duration(scoped_restores),
            "restore_codec_duration": _duration(scoped_restores, "decode_duration_ns"),
        },
        "diagnostics": diagnostics,
        "stores": scoped_stores,
        "restores": scoped_restores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--replay-label", default="mtrag")
    args = parser.parse_args()
    try:
        report = build_report(args.run_dir, replay_label=args.replay_label)
        path = args.run_dir / REPORT_FILENAME
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(f"LMCache storage report ({report['status']}): {path}")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"LMCache storage report failed: {error}\n")


if __name__ == "__main__":
    main()
