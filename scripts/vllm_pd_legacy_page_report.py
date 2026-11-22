#!/usr/bin/env python3
"""Report complete LMCache page versions and shadow native-lookup verification."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidencestore.pd_report import load_events, request_joins
from evidencestore.retrieval import read_jsonl


REPORT_FILENAME = "legacy-kv-page-report.json"
PUT = "legacy_kv_page_put"
RESTORE = "legacy_kv_page_restore"
FRAGMENT = "legacy_kv_page_fragment"
EVENTS = {PUT, RESTORE, FRAGMENT}
SELECTED = {"reference", "patch"}
PROOF_FAILURES = {"base_mismatch", "target_mismatch", "reconstruction_mismatch", "error"}
LIMIT_FIELDS = ("max_pages_per_request", "max_shadow_encoded_bytes", "max_tracked_pages", "max_tracked_objects",
                "max_candidates")
FIELDS = (
    "event", "worker_id", "pid", "native_request_id", "backend", "ts_ns", "duration_ns", "decode_duration_ns",
    "operation", "status", "fragment_reason", "worker_block_id", "source_worker_block_id", "destination_block_ids",
    "block_size", "layer_count", "row_bytes", "tensor_dtype", "native_kv_format", "object_key_sha256",
    "logical_page_sha256", "version_sha256", "tensor_sha256",
    "start_token", "end_token", "object_start_token", "object_end_token", "relative_start_token",
    "old_start_token", "old_end_token", "raw_bytes", "encoding", "encoded_bytes", "manifest_bytes", "patch_bytes",
    "candidate_status", "candidate_count", "base_available", "base_object_key_sha256", "base_logical_page_sha256",
    "base_version_sha256", "base_tensor_sha256", "base_start_token", "base_end_token", "base_raw_bytes",
    "base_read_bytes", "common_prefix_tokens", "common_in_page_prefix_tokens", "observed_sha256",
    "reconstructed_sha256", "compared_base_version_sha256", *LIMIT_FIELDS,
)


def _responses(run_dir, label):
    rows = list(read_jsonl(run_dir / "client" / label / "responses.jsonl"))
    responses = {row["request_id"]: row for row in rows}
    if (not rows or len(rows) != len(responses) or len({row["request_key"] for row in rows}) != len(rows)
            or any(row["label"] != label or row.get("success") is not True for row in rows)):
        raise ValueError(f"Invalid {label} response receipts")
    selection = json.loads((run_dir / "inputs" / "manifest.json").read_text(encoding="utf-8"))
    scope = selection["kv_page_trace"]
    summary = json.loads((run_dir / "legacy-blend-summary.json").read_text(encoding="utf-8"))
    agreement = summary["selection_agreement"]
    if (scope["schema_version"] != 1 or scope["turns"] != [1, 2, 3, 4]
            or [row["turn"] for row in rows] != scope["turns"]
            or any(row["conversation_id"] != scope["conversation_id"] for row in rows)
            or [row["request_key"] for row in rows] != selection["query_keys"]
            or summary["status"] != "passed" or summary["natural_requests"] != len(rows)
            or agreement["matching_retrieval_identity_and_ranks"] != len(rows)
            or agreement["matching_kept_identity_and_ranks"] != len(rows)
            or agreement["mismatching_request_keys"]):
        raise ValueError("Page trace requests do not match the four-turn smoke and fixed retrieval ranking")
    return responses


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _int(value):
    return type(value) is int and value >= 0


def _page_version(row, *, base=False):
    prefix = "base_" if base else ""
    logical = row.get(prefix + "logical_page_sha256")
    version = row.get(prefix + "version_sha256")
    if not (_sha(logical) and _sha(version)):
        return None
    return row["worker_id"], row["pid"], logical, version


def _page_id(version):
    return ":".join(str(field) for field in version) if version is not None else None


def _lookup_location(row):
    if not (_sha(row.get("object_key_sha256")) and _int(row.get("relative_start_token"))):
        return None
    return row["worker_id"], row["pid"], row["object_key_sha256"], row["relative_start_token"]


def _layer_rows(row, field):
    layers = [
        {key: layer[key] for key in ("layer_id", "k_changed_rows", "v_changed_rows")}
        for layer in row[field]
    ]
    if (sorted(layer["layer_id"] for layer in layers) != list(range(row["layer_count"]))
            or any(not _int(layer[side]) or layer[side] > row["block_size"]
                   for layer in layers for side in ("k_changed_rows", "v_changed_rows"))):
        raise ValueError("Page comparison needs K and V counts at each model layer")
    return sorted(layers, key=lambda layer: layer["layer_id"])


def _receipt(row):
    if (row["component"] != "lmcache" or row["role"] != "P" or row["backend"] != "LocalCPUBackend"
            or not _int(row["pid"]) or not isinstance(row["native_request_id"], str) or not row["native_request_id"]):
        raise ValueError("Invalid native KV page receipt identity")
    result = {field: row[field] for field in FIELDS if field in row}
    if row["status"] == "error" and "start_token" not in row:
        return result
    if not _int(row["start_token"]) or not _int(row["end_token"]) or row["start_token"] >= row["end_token"]:
        raise ValueError("Invalid native KV page receipt range")
    for field in ("raw_bytes", "duration_ns", "decode_duration_ns", "candidate_count", "base_read_bytes", *LIMIT_FIELDS):
        if result.get(field) is not None and not _int(result[field]):
            raise ValueError(f"Invalid page receipt field {field}")
    if row["event"] == FRAGMENT:
        if row["operation"] not in {"put", "restore"} or not isinstance(row["fragment_reason"], str):
            raise ValueError("Invalid page fragment disposition")
        return result
    if not isinstance(row["status"], str) or not row["status"]:
        raise ValueError("Page receipt has no status")
    encoding = row.get("encoding")
    if encoding not in {"raw", *SELECTED}:
        if row["status"] in {"measured", "verified"}:
            raise ValueError("Measured page receipt has no encoding")
        return result
    counts = ("block_size", "layer_count", "row_bytes", "raw_bytes", "encoded_bytes", "manifest_bytes", "patch_bytes")
    if (any(not _int(row[field]) for field in counts) or row["block_size"] != 16 or row["layer_count"] == 0
            or row["row_bytes"] == 0 or row["end_token"] - row["start_token"] != row["block_size"]
            or row["raw_bytes"] != 2 * row["block_size"] * row["layer_count"] * row["row_bytes"]
            or not _sha(row["tensor_sha256"]) or _page_version(row) is None):
        raise ValueError("Invalid complete native KV page geometry or identity")
    blocks = row["destination_block_ids"]
    if (not isinstance(blocks, list) or not 1 <= len(blocks) <= 2 or any(not _int(block) for block in blocks)
            or len(set(blocks)) != len(blocks) or (len(blocks) == 1 and row["worker_block_id"] != blocks[0])
            or (len(blocks) > 1 and row["worker_block_id"] is not None)
            or (row["event"] == PUT and (len(blocks) != 1 or row["start_token"] % row["block_size"]))):
        raise ValueError("Complete page receipt disagrees with native block mapping")
    if encoding == "raw":
        if row["encoded_bytes"] != row["raw_bytes"] or row["manifest_bytes"] or row["patch_bytes"]:
            raise ValueError("Raw page receipt does not reconcile")
        if row.get("compared_layers"):
            result["compared_layers"] = _layer_rows(row, "compared_layers")
        return result
    layers = _layer_rows(row, "layers")
    changed_rows = sum(layer["k_changed_rows"] + layer["v_changed_rows"] for layer in layers)
    if (row["encoded_bytes"] != row["manifest_bytes"] + row["patch_bytes"]
            or not 0 < row["manifest_bytes"] <= row["encoded_bytes"] < row["raw_bytes"]
            or row["patch_bytes"] != changed_rows * row["row_bytes"]
            or (encoding == "reference") != (changed_rows == 0)
            or _page_version(row, base=True) is None or not _sha(row["base_tensor_sha256"])):
        raise ValueError("Selected page delta does not reconcile with its all-layer row comparison")
    result["layers"] = layers
    return result


def _bound_events(events, responses):
    groups, _ = request_joins(events)
    bound = {}
    attempts = set()
    joined = set()
    transfers = {"put": [], "restore": []}
    for (request_id, _), group in groups.items():
        page_events = [row for row in group if row["event"] in EVENTS]
        joined.update(id(row) for row in page_events)
        if request_id not in responses:
            continue
        for row in group:
            if row["event"] == "legacy_lmcache_d2h_complete":
                transfers["put"].extend(span.get("bytes") for span in row["mapped_ranges"])
            elif row["event"] == "legacy_lmcache_h2d_complete":
                transfers["restore"].extend(span.get("bytes") for span in row["mapped_ranges"])
        if not page_events:
            continue
        dispatch = [row for row in group if row["event"] == "gateway_dispatch_prefill"]
        response = [row for row in group if row["event"] == "gateway_prefill_response"]
        if len(dispatch) != 1 or len(response) != 1 or request_id in attempts:
            raise ValueError(f"Unexpected prefill attempt for page trace request {request_id}")
        attempts.add(request_id)
        start, end = dispatch[0], response[0]
        for row in page_events:
            if (id(row) in bound or row["worker_id"] != start["worker_id"]
                    or row["native_request_id"] != end["remote_request_id"]
                    or not start["ts_ns"] <= row["ts_ns"] <= end["ts_ns"]):
                raise ValueError(f"Page event differs from its native prefill attempt for {request_id}")
            bound[id(row)] = (responses[request_id], group)
    return bound, sum(row["event"] in EVENTS and id(row) not in joined for row in events), transfers


def _native_transfer(row, group):
    name = "legacy_lmcache_d2h_complete" if row["event"] == PUT else "legacy_lmcache_h2d_complete"
    for event in group:
        if (event["event"] != name or event["worker_id"] != row["worker_id"] or event["pid"] != row["pid"]
                or event["native_request_id"] != row["native_request_id"] or event["layers"] != row["layer_count"]
                or not event["started_ns"] <= row["ts_ns"] <= event["ts_ns"]):
            continue
        for span in event["mapped_ranges"]:
            if span["start_token"] <= row["start_token"] < row["end_token"] <= span["end_token"]:
                if (row["event"] == RESTORE and span.get("old_start_token") is not None
                        and row.get("old_start_token") is not None and row["old_start_token"]
                        != span["old_start_token"] + row["start_token"] - span["start_token"]):
                    continue
                return True
    return False


def _duration(rows, field="duration_ns"):
    times = sorted(row[field] for row in rows if row.get(field) is not None)
    return {"events": len(times), "sum_ns": sum(times), "median_ns": median(times) if times else None,
            "max_ns": times[-1] if times else None}


def _bytes(rows):
    raw = sum(row["raw_bytes"] for row in rows)
    encoded = sum(row["encoded_bytes"] for row in rows)
    return {"pages": len(rows), "raw_bytes": raw, "encoded_bytes": encoded, "gross_saved_bytes": raw - encoded,
            "manifest_bytes": sum(row["manifest_bytes"] for row in rows),
            "patch_bytes": sum(row["patch_bytes"] for row in rows),
            "gross_saved_fraction": (raw - encoded) / raw if raw else None}


def _raw_page_bytes(row):
    if _int(row.get("raw_bytes")):
        return row["raw_bytes"]
    if all(_int(row.get(field)) for field in ("block_size", "layer_count", "row_bytes")):
        return 2 * row["block_size"] * row["layer_count"] * row["row_bytes"]
    return None


def _raw_coverage(native, complete, skipped, fragments):
    measured_bytes = sum(_raw_page_bytes(row) or 0 for row in complete)
    skipped_counts = [_raw_page_bytes(row) for row in skipped]
    skipped_bytes = sum(count or 0 for count in skipped_counts)
    fragment_bytes = sum(row.get("raw_bytes") or 0 for row in fragments)
    native_bytes = sum(native) if all(_int(count) for count in native) else None
    accounted = measured_bytes + skipped_bytes + fragment_bytes
    return {
        "completed_native_movement_bytes": native_bytes,
        "native_segments_with_unknown_bytes": sum(not _int(count) for count in native),
        "measured_complete_raw_bytes": measured_bytes,
        "skipped_complete_raw_bytes": skipped_bytes,
        "skipped_events_with_unknown_bytes": skipped_counts.count(None),
        "fragment_raw_bytes": fragment_bytes,
        "accounted_raw_bytes": accounted,
        "unaccounted_native_raw_bytes": max(0, native_bytes - accounted) if native_bytes is not None else None,
        "page_raw_bytes_above_native": max(0, accounted - native_bytes) if native_bytes is not None else None,
    }


def build_report(run_dir: Path, *, replay_label="mtrag"):
    """Reconcile a gated MT-RAG replay without modifying its trace receipts."""
    if replay_label != "mtrag":
        raise ValueError("KV page reporting requires the MT-RAG replay label")
    run_dir = run_dir.expanduser().resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("legacy_full_blend", {}).get("kv_page_trace_enabled") is not True:
        raise ValueError("KV page tracing was not enabled for this replay")
    responses = _responses(run_dir, replay_label)
    events = list(load_events(run_dir))
    bound, unjoined, native_bytes = _bound_events(events, responses)
    pages, restores, fragments = [], [], []
    nodes, latest, latest_by_location, edges = {}, {}, {}, []
    observed, verified = set(), set()
    for event in sorted((row for row in events if id(row) in bound), key=lambda row: row["ts_ns"]):
        row = _receipt(event)
        response, group = bound[id(event)]
        row.update({field: response.get(field) for field in ("request_id", "request_key", "conversation_id", "turn")})
        if row["event"] == FRAGMENT:
            fragments.append(row)
            continue
        target = _page_version(row)
        base = _page_version(row, base=True)
        location = _lookup_location(row)
        row.update(target_page_id=_page_id(target), base_page_id=_page_id(base))
        if row["event"] == PUT:
            pages.append(row)
            if row["status"] != "measured":
                if location is not None:
                    latest_by_location.pop(location, None)
                continue
            node = nodes.setdefault(target, {
                "page_id": row["target_page_id"], "tensor_sha256": row["tensor_sha256"],
                "raw_bytes": row["raw_bytes"], "first_store_ns": row["ts_ns"], "stores": 0,
            })
            if (node["tensor_sha256"], node["raw_bytes"]) != (row["tensor_sha256"], row["raw_bytes"]):
                raise ValueError("Repeated page version has conflicting target contents")
            node["last_store_ns"] = row["ts_ns"]
            node["stores"] += 1
            row["native_transfer_matched"] = _native_transfer(row, group)
            if row["encoding"] in SELECTED:
                earlier = latest.get(base)
                row["observed_raw_base_store"] = bool(
                    earlier and earlier["encoding"] == "raw" and earlier["tensor_sha256"] == row["base_tensor_sha256"]
                    and earlier["native_transfer_matched"]
                )
                if row["observed_raw_base_store"]:
                    nodes[base].setdefault("first_dependent_store_ns", row["ts_ns"])
                    nodes[base]["last_dependent_store_ns"] = row["ts_ns"]
                edges.append(row)
            latest[target] = row
            if location is not None:
                latest_by_location[location] = row
        else:
            restores.append(row)
            previous = latest_by_location.get(location)
            if previous is None and location is None:
                previous = latest.get(target)
            matched_version = bool(previous and target is not None and previous["target_page_id"] == _page_id(target))
            row["native_transfer_matched"] = bool(row.get("layer_count") and _native_transfer(row, group))
            if previous is not None and row["native_transfer_matched"] and (matched_version or target is None):
                observed.add(id(previous))
                row["last_observed_source_store_request_id"] = previous["request_id"]
                row["source_store_version_matched"] = matched_version
            if row["status"] != "verified" or row.get("encoding") not in SELECTED:
                continue
            current_base = latest.get(base)
            proof = bool(
                matched_version and previous["encoding"] == row["encoding"] and previous["native_transfer_matched"]
                and previous["base_page_id"] == row["base_page_id"] and previous["observed_raw_base_store"]
                and previous["encoded_bytes"] == row["encoded_bytes"] and previous["raw_bytes"] == row["raw_bytes"]
                and previous["start_token"] == row.get("old_start_token")
                and previous["end_token"] == row.get("old_end_token")
                and previous["worker_block_id"] == row.get("source_worker_block_id")
                and previous["layers"] == row["layers"] and previous["ts_ns"] < row["ts_ns"]
                and current_base and current_base["encoding"] == "raw" and current_base["native_transfer_matched"]
                and current_base["tensor_sha256"] == row["base_tensor_sha256"] and row.get("base_available") is True
                and row.get("base_read_bytes", current_base["raw_bytes"]) == current_base["raw_bytes"]
                and row.get("observed_sha256") == row.get("reconstructed_sha256") == row["tensor_sha256"]
                == previous["tensor_sha256"] and row["native_transfer_matched"]
            )
            row["all_layer_reconstruction_verified"] = proof
            if proof:
                verified.add(id(previous))
                nodes[base]["last_verified_available_ns"] = row["ts_ns"]
                nodes[target]["last_verified_lookup_ns"] = row["ts_ns"]

    complete = [row for row in pages if row["status"] == "measured"]
    limits = {}
    for row in (*pages, *restores, *fragments):
        for field in LIMIT_FIELDS:
            if field in row:
                if limits.setdefault(field, row[field]) != row[field]:
                    raise ValueError(f"Page trace uses conflicting {field} bounds")
    for row in edges:
        row["later_demand_observed"] = id(row) in observed
        row["later_reconstruction_verified"] = id(row) in verified
    failed = [row for row in restores if row["status"] in PROOF_FAILURES
              or row.get("all_layer_reconstruction_verified") is False]
    verified_reads = [row for row in restores if row.get("all_layer_reconstruction_verified")]
    baselines = {row["base_page_id"] for row in edges if row["observed_raw_base_store"]}
    put_fragments = [row for row in fragments if row["operation"] == "put"]
    restore_fragments = [row for row in fragments if row["operation"] == "restore"]
    coverage = {
        "put": _raw_coverage(native_bytes["put"], complete,
                             [row for row in pages if row["status"] != "measured"], put_fragments),
        "restore": _raw_coverage(native_bytes["restore"],
                                 [row for row in restores if row.get("encoding") in {"raw", *SELECTED}],
                                 [row for row in restores if row.get("encoding") not in {"raw", *SELECTED}],
                                 restore_fragments),
    }
    diagnostics = []
    if not (pages or restores or fragments):
        diagnostics.append("No gated page events joined to the natural MT-RAG replay.")
    elif not complete:
        diagnostics.append("Page instrumentation ran without a measured complete stored page.")
    elif not edges:
        diagnostics.append("Measured complete pages selected full raw representations.")
    if edges and not verified:
        diagnostics.append("Selected page storage savings have no subsequent all-layer reconstruction proof.")
    if failed:
        diagnostics.append("Some selected page restore attempts failed digest, source version, or native transfer checks.")
    hook_errors = sum(row["status"] == "error" for row in pages)
    if hook_errors:
        diagnostics.append("Some native page put requests emitted hook errors.")
    if unjoined:
        diagnostics.append("Some page events could not be joined to a native request in the cohort.")
    for operation, counts in coverage.items():
        if counts["unaccounted_native_raw_bytes"] or counts["page_raw_bytes_above_native"]:
            diagnostics.append(f"Page trace {operation} raw bytes do not fully reconcile with native receipts.")
    return {
        "schema_version": 1,
        "status": ("proof_failed" if failed else "instrumentation_error" if hook_errors else "measured" if complete else
                   "no_complete_pages" if pages or restores or fragments else "instrumentation_missing"),
        "replay_label": replay_label,
        "natural_requests": len(responses),
        "trace_bounds": limits,
        "unjoined_page_events": unjoined,
        "turns": [{field: row.get(field) for field in ("request_id", "request_key", "conversation_id", "turn")}
                  for row in responses.values()],
        "scope": {
            "sample": "Complete P-native 16-token pages whose full K/V rows are observed across layers.",
            "codec": "Shadow raw/reference/patch representations of stored source pages. Real LMCache consumes raw objects.",
            "bytes": "Gross modeled representation bytes including metadata. Fragment and cap skips are recorded separately.",
            "transfer": "NIXL behavior is unchanged. Shifted restores prove stored source bytes before Blend repair.",
            "retention": "Base presence is observed at write or verified lookup instants. Pin lifetimes and net occupancy are unmodeled.",
            "demand": "A completed matching native lookup establishes observed demand. An absent receipt does not prove a page was never reused.",
        },
        "storage": {
            "put_statuses": dict(Counter(row["status"] for row in pages)),
            "put_candidate_statuses": dict(Counter(row.get("candidate_status") or "unspecified" for row in pages)),
            "candidate_count": sum(row.get("candidate_count") or 0 for row in pages),
            "all_complete_pages": _bytes(complete),
            "representations": {encoding: _bytes([row for row in complete if row["encoding"] == encoding])
                                for encoding in ("raw", "reference", "patch")},
            "unique_observed_raw_base_versions": len(baselines),
            "unique_observed_raw_base_bytes": sum(node["raw_bytes"] for node in nodes.values()
                                                  if node["page_id"] in baselines),
            "put_diagnostic_duration": _duration(pages),
        },
        "native_lookup_verification": {
            "restore_statuses": dict(Counter(row["status"] for row in restores)),
            "selected_stores_with_observed_demand": sum(row["later_demand_observed"] for row in edges),
            "selected_stores_without_observed_demand": sum(not row["later_demand_observed"] for row in edges),
            "selected_stores_with_verified_lookup": len(verified),
            "measured_stores_with_observed_demand_by_encoding": dict(
                Counter(row["encoding"] for row in complete if id(row) in observed)),
            "selected_stores_with_demand_without_verified_lookup": sum(
                row["later_demand_observed"] and not row["later_reconstruction_verified"] for row in edges),
            "lookup_events_with_missing_base": sum(row["status"] == "missing_base" for row in restores),
            "lookup_events_with_available_base": sum(row.get("base_available") is True for row in restores),
            "lookup_base_bytes_read": sum(row.get("base_read_bytes") or 0 for row in restores),
            "verified_lookup_base_bytes_read": sum(row.get("base_read_bytes") or 0 for row in verified_reads),
            "verified_lookup_encoded_bytes": sum(row["encoded_bytes"] for row in verified_reads),
            "verified_lookup_base_plus_encoded_bytes": sum((row.get("base_read_bytes") or 0) + row["encoded_bytes"]
                                                           for row in verified_reads),
            "verified_lookup_native_raw_bytes": sum(row["raw_bytes"] for row in verified_reads),
            "all_layer_verified_events": len(verified_reads),
            "failed_proof_events": len(failed),
            "gross_bytes_of_verified_stores": _bytes([row for row in edges if row["later_reconstruction_verified"]]),
            "restore_diagnostic_duration": _duration(restores),
            "restore_codec_duration": _duration(restores, "decode_duration_ns"),
        },
        "fragment_statuses": {
            operation: dict(Counter(row["fragment_reason"] for row in fragments if row["operation"] == operation))
            for operation in ("put", "restore")
        },
        "fragment_raw_bytes": {operation: sum(row.get("raw_bytes") or 0 for row in fragments
                                            if row["operation"] == operation)
                               for operation in ("put", "restore")},
        "native_movement_coverage": coverage,
        "version_graph": {
            "nodes": list(nodes.values()),
            "edges": [{field: row[field] for field in (
                "base_page_id", "target_page_id", "request_id", "ts_ns", "encoding", "raw_bytes", "encoded_bytes",
                "manifest_bytes", "patch_bytes", "observed_raw_base_store", "native_transfer_matched",
                "later_demand_observed", "later_reconstruction_verified", "layers",
            )} for row in edges],
        },
        "diagnostics": diagnostics,
        "stores": pages,
        "restores": restores,
        "fragments": fragments,
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
        print(f"LMCache page report ({report['status']}): {path}")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"LMCache page report failed: {error}\n")


if __name__ == "__main__":
    main()
