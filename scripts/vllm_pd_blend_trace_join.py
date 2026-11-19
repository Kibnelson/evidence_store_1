"""Attach native PD request and evidence IDs to public LMCache smoke events."""

from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidencestore.pd_report import load_events, request_joins
from evidencestore.retrieval import json_line, prompt_sha256, read_jsonl


_SOURCE_TRACE = re.compile(r"lmcache-(?:blend-)?([0-9]+)\.jsonl")
_RANGES = {
    "lmcache_transfer_complete": ("mapped_ranges",),
    "lmcache_blend_lookup_complete": ("non_prefix_ranges", "segmented_prefix_ranges"),
    "lmcache_blend_scatter_complete": ("ranges",),
}


def _native_requests(directory, cases):
    case_ids = {row["request_id"]: (case, row) for case, row in cases.items()}
    by_native_id = defaultdict(list)
    by_case = {}
    for mode in ("prefix", "blend"):
        groups, _ = request_joins(list(load_events(directory / mode)))
        for (request_id, attempt_id), rows in groups.items():
            if request_id not in case_ids:
                continue
            case, receipt = case_ids[request_id]
            starts = [row["ts_ns"] for row in rows if row["event"] == "client_dispatch"]
            ends = [row["ts_ns"] for row in rows if row["event"] == "client_response_end"]
            if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0] or case in by_case:
                raise ValueError(f"{case} has no unique completed native request for the LMCache join")
            request = {"case": case, "mode": mode, "request_id": request_id, "attempt_id": attempt_id,
                       "dispatch_ns": starts[0], "response_end_ns": ends[0], "evidence": receipt["evidence"]}
            by_case[case] = request
            roles = {receipt["prefill"]: "P", receipt["decode"]: "D"}
            identities = {(row["native_request_id"], row["worker_id"]) for row in rows
                          if row["event"] == "engine_request_id" and row["worker_id"] in roles}
            for native_id, worker in identities:
                by_native_id[native_id].append((request, worker, roles[worker]))
    if by_case.keys() != cases.keys():
        raise ValueError("The LMCache join cannot find every completed smoke case in the native trace")
    return by_native_id, by_case


def _token_fixtures(directory, cases):
    fixtures = {}
    for case, receipt in cases.items():
        path = directory / "inputs" / f"{case}.jsonl"
        if path.is_file():
            rows = list(read_jsonl(path))
            if len(rows) != 1 or prompt_sha256(rows[0]["prompt_token_ids"]) != receipt["prompt_sha256"]:
                raise ValueError(f"{case} fixture differs from its verified prompt")
            fixtures[case] = rows[0]["prompt_token_ids"]
    return fixtures


def _span(row, start_key, end_key):
    lo, hi = row[start_key], row[end_key]
    if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi:
        raise ValueError(f"Invalid LMCache token range {lo}, {hi}")
    return lo, hi


def _evidence_overlaps(evidence, lo, hi):
    hits = [
        {**{key: chunk[key] for key in ("occurrence", "doc_id", "chunk_id", "chunk_version")},
         "start_token": max(lo, chunk["start_token"]), "end_token": min(hi, chunk["end_token"])}
        for chunk in evidence if lo < chunk["end_token"] and hi > chunk["start_token"]
    ]
    if len(hits) == 1 and (hits[0]["start_token"], hits[0]["end_token"]) == (lo, hi):
        return "exclusive", hits
    return ("boundary" if hits else "outside"), hits


def _prior_fixture_match(request, by_case, fixtures, old, new):
    target = fixtures.get(request["case"])
    if target is None or new[1] > len(target):
        return None
    tokens = target[new[0]:new[1]]
    matches = [
        reference for case, reference in by_case.items()
        if reference["response_end_ns"] <= request["dispatch_ns"] and case in fixtures
        and old[1] <= len(fixtures[case]) and fixtures[case][old[0]:old[1]] == tokens
    ]
    if len(matches) != 1:
        return None
    earlier = matches[0]
    attribution, evidence = _evidence_overlaps(earlier["evidence"], *old)
    # Equal tokens at the reported old position establish a fixture match.
    # LMCache's physical source request is independent of this comparison.
    return {"case": earlier["case"], "request_id": earlier["request_id"],
            "attribution": attribution, "evidence_overlaps": evidence}


def _ranges_for(event, request, by_case, fixtures):
    result = {}
    shifted = False
    for field in _RANGES.get(event["event"], ()):
        annotations = []
        for row in event.get(field, ()):
            new = _span(row, "start_token", "end_token")
            attribution, evidence = _evidence_overlaps(request["evidence"], *new)
            byte_count = row.get("h2d_bytes", row.get("bytes"))
            annotation = {"attribution": attribution, "evidence_overlaps": evidence}
            if type(byte_count) is int and attribution == "exclusive":
                annotation["exclusive_evidence_bytes"] = byte_count
            if "old_start_token" in row:
                old = _span(row, "old_start_token", "old_end_token")
                if old[1] - old[0] != new[1] - new[0]:
                    raise ValueError("An LMCache Blend range changed length between old and new positions")
                annotation["prior_fixture_token_match"] = _prior_fixture_match(request, by_case, fixtures, old, new)
                shifted |= old[0] != new[0] and bool(evidence) and type(byte_count) is int and byte_count > 0
            annotations.append(annotation)
        result[field] = annotations
    return result, shifted


def _verify_scatter_bytes(event):
    if event["event"] != "lmcache_blend_scatter_complete" or event["status"] != "done":
        return
    counts = [row.get("h2d_bytes") for row in event["ranges"]]
    moved = [row["h2d_bytes"] for row in event["ranges"] if row["old_start_token"] != row["start_token"]]
    if (any(type(count) is not int or count < 0 for count in counts)
            or sum(counts) != event["completed_h2d_bytes"] or sum(moved) != event["shifted_h2d_bytes"]):
        raise ValueError("LMCache Blend scatter ranges disagree with measured H2D totals")


def join_lmcache_trace(directory):
    """Write a sidecar after the server stops and require measured shifted scatter."""
    directory = Path(directory)
    summary = json.loads((directory / "blend-summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "passed":
        raise ValueError("A passed native Blend summary is required for the LMCache join")
    cases = summary["cases"]
    by_native_id, by_case = _native_requests(directory, cases)
    fixtures = _token_fixtures(directory, cases)
    server_path = directory / "lmcache-server.json"
    server = json.loads(server_path.read_text(encoding="utf-8")) if server_path.is_file() else {}
    server_run_id = server.get("environment", {}).get("EVIDENCESTORE_RUN_ID")
    joined = []
    shifted_scatter = False
    for path in sorted((directory / "lmcache-trace").glob("*.jsonl")):
        filename = _SOURCE_TRACE.fullmatch(path.name)
        if not filename:
            continue
        for event in read_jsonl(path):
            if (event.get("schema_version") != 1 or event.get("component") != "lmcache"
                    or event.get("pid") != int(filename[1]) or not isinstance(event.get("event"), str)
                    or (server_run_id and event.get("run_id") != server_run_id)
                    or type(event.get("ts_ns")) is not int or type(event.get("wall_ns")) is not int):
                raise ValueError(f"Unexpected LMCache trace schema in {path}")
            native_id = event.get("native_request_id")
            matches = by_native_id.get(native_id, ()) if isinstance(native_id, str) else ()
            association = {"status": "ambiguous" if matches else "unmatched"}
            if len(matches) == 1:
                request, worker, role = matches[0]
                association = {key: request[key] for key in (
                    "case", "mode", "request_id", "attempt_id", "dispatch_ns", "response_end_ns",
                )}
                _verify_scatter_bytes(event)
                ranges, shifted = _ranges_for(event, request, by_case, fixtures)
                association.update(status="matched", worker_id=worker, role=role, ranges=ranges)
                shifted_scatter |= (
                    request["case"] == "shifted" and request["mode"] == "blend" and role == "P"
                    and event["event"] == "lmcache_blend_scatter_complete"
                    and event["status"] == "done" and shifted
                )
            joined.append({**event, "source_trace": path.name, "request_join": association})
    if not shifted_scatter:
        raise ValueError("No measured shifted LMCache scatter joined to the shifted prefill request")
    output = directory / "lmcache-request-trace.jsonl"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as stream:
        staging = Path(stream.name)
        try:
            stream.writelines(json_line(row) for row in sorted(joined, key=lambda item: item["ts_ns"]))
            stream.close()
            staging.replace(output)
        finally:
            staging.unlink(missing_ok=True)
    return output
