"""Stage and verify review artifacts for a completed PD campaign."""

from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import re


DATASETS = ("musique", "2wiki", "mtrag/clapnq", "mtrag/cloud", "mtrag/fiqa", "mtrag/govt")
COHORTS = ("1p3d-native", "1p3d-shared", "2p2d-native", "2p2d-shared")
MANIFEST = "archive-manifest.json"
DOCUMENT_GZIP_BYTES = 8 * 1024 * 1024
MAX_GIT_FILE_BYTES = 100 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTITY = (
    "request_key", "conversation_id", "turn", "prompt_sha256",
    "output_token_ids_sha256", "completion_tokens",
)
_RESPONSE_FIELDS = (
    "request_id", *_IDENTITY, "dataset", "domain", "label", "success", "stream_chunks", "finish_reason",
)
_REQUEST_FIELDS = (
    "request_id", *_IDENTITY, "dataset", "prefill", "decode", "prompt_tokens",
    "ttft_ms", "dispatch_to_first_token_ms", "dispatch_to_last_token_ms", "end_to_end_ms",
    "http_request_body_bytes", "gateway_prefill_request_bytes", "gateway_prefill_response_bytes",
    "gateway_decode_request_bytes", "gateway_sse_payload_bytes", "completed_nixl_bytes",
    "prefill_local_prefix_tokens", "decode_local_prefix_tokens", "prefill_lmcache_match_tokens",
    "prefill_lmcache_usable_prefix_tokens", "decode_lmcache_match_tokens",
)
_CHUNK_FIELDS = ("doc_id", "chunk_id", "chunk_version", "start_token", "end_token", "occurrence", "rank", "score")
_WORK_FIELDS = (
    "scheduled_prompt_tokens", "scheduled_decode_tokens", "observed_local_prefix_tokens",
    "observed_remote_prefix_tokens",
)
_NIXL_BYTES = ("shared_or_boundary_bytes", "outside_evidence_bytes", "unmapped_bytes")
_TRANSFER_FIELDS = (
    "worker_id", "status", "completed_bytes", "position_start", "position_end", "valid_position_end",
    "transferred_blocks", "block_size",
)
_TIMELINE_DETAIL = (
    "worker_id", "role", "status", "duration_ns", "completed_bytes", "transferred_blocks", "block_size",
    "scheduled_tokens", "prompt_tokens", "prompt_token_count", "matched_tokens", "connector_class",
    "local_token_count", "remote_token_count", "usable_prefix_token_count",
    "http_request_body_bytes", "http_response_body_bytes", "sse_payload_bytes",
)


def _select(row, fields):
    return {field: row[field] for field in fields if field in row}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            yield row


def _encode(row):
    return (json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def _output(path):
    with path.open("xb") as stream:
        if path.suffix == ".gz":
            with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed:
                yield compressed
        else:
            yield stream


def _work(row):
    return {role: _select(row[role], _WORK_FIELDS) for role in ("P", "D") if role in row}


def _published_checkpoint(campaign_reports, dataset, label, cohort):
    path = campaign_reports / label / f"{cohort}.json"
    report = _read_json(path)
    expected = (campaign_reports.name, dataset, label, cohort)
    observed = tuple(report[field] for field in ("campaign", "dataset", "label", "cohort"))
    if (
        observed != expected or type(report["schema_version"]) is not int or report["schema_version"] != 1
        or report["status"] != "passed"
        or report["counter_reconciliation"]["status"] != "matched"
        or type(report["requests"]) is not int or report["requests"] <= 0
    ):
        raise ValueError(f"Expected a passed, reconciled dataset checkpoint at {path}")
    return report, _sha256(path)


def _responses(source, destination, dataset, label):
    receipts = {}
    base, _, domain = dataset.partition("/")
    with _output(destination) as stream:
        for row in _read_rows(source):
            request_id = row["request_id"]
            if (
                not isinstance(request_id, str) or not request_id or request_id in receipts
                or row["success"] is not True or row["label"] != label
                or row["dataset"] != base or row.get("domain") != (domain or None)
                or type(row["completion_tokens"]) is not int or row["completion_tokens"] <= 0
                or type(row["stream_chunks"]) is not int or row["stream_chunks"] <= 0
                or not re.fullmatch(r"[a-z_]{1,32}", row["finish_reason"])
                or not all(_SHA256.fullmatch(row.get(field) or "") for field in (
                    "prompt_sha256", "output_token_ids_sha256",
                ))
            ):
                raise ValueError(f"Invalid or duplicate response receipt: {request_id!r}")
            receipts[request_id] = tuple(row[field] for field in _IDENTITY)
            stream.write(_encode(_select(row, _RESPONSE_FIELDS)))
    return receipts


def _request_measurement(row):
    result = _select(row, _REQUEST_FIELDS)
    result["evidence"] = [
        {**_select(chunk, _CHUNK_FIELDS), "token_work": _work(chunk["token_work"])}
        for chunk in row["evidence"]
    ]
    result["token_work_by_role"] = _work(row["token_work_by_role"])
    result["stage_spans_ms"] = row["stage_spans_ms"]
    result["bytes"] = {
        **_select(row["bytes"], _NIXL_BYTES),
        "by_chunk_occurrence": row["bytes"]["by_chunk_occurrence"],
    }
    result["full_nixl_handoff"] = _select(
        row["full_nixl_handoff"], ("prompt_tokens", "p_advertised_tokens", "verified_prompt_tokens", "status"),
    )
    result["nixl_transfers"] = [_select(transfer, _TRANSFER_FIELDS) for transfer in row["nixl_transfers"]]
    if "selected_external_connectors_by_role" in row:
        choices = row["selected_external_connectors_by_role"]
        result["selected_external_connectors_by_role"] = {
            role: [_select(connector, ("connector_class", "connector_index", "matched_tokens"))
                   for connector in choices[role]]
            for role in ("P", "D") if role in choices
        }
    return result


def _requests(source, destination, receipts, report):
    totals = Counter()
    with _output(destination) as stream:
        for row in _read_rows(source):
            request_id = row["request_id"]
            identity = tuple(row[field] for field in _IDENTITY)
            if receipts.pop(request_id, None) != identity or row["dataset"] != report["dataset"].partition("/")[0]:
                raise ValueError(f"Measurement has a missing, duplicate, or different response receipt: {request_id!r}")
            totals.update({
                "requests": 1,
                "completion_tokens": row["completion_tokens"],
                "prompt_tokens": row["prompt_tokens"],
                "nixl_bytes": row["completed_nixl_bytes"],
                "occurrences": len(row["evidence"]),
            })
            stream.write(_encode(_request_measurement(row)))
    expected = {
        "requests": report["requests"],
        "completion_tokens": report["completion_tokens"],
        "prompt_tokens": report["prompt_tokens"],
        "nixl_bytes": report["nixl_bytes"]["completed_bytes"],
        "occurrences": report["evidence"]["occurrences"],
    }
    if receipts or dict(totals) != expected:
        raise ValueError(f"Request measurement totals differ from the published report: {source}")
    return totals["requests"]


def _documents(source, destination, report):
    seen = set()
    occurrences = total_bytes = 0
    fields = ("doc_id", "occurrences", "request_count", "chunk_ids", "exclusive_nixl_bytes")
    with _output(destination) as stream:
        for row in _read_rows(source):
            doc_id = row["doc_id"]
            if not isinstance(doc_id, str) or not doc_id or doc_id in seen:
                raise ValueError(f"Invalid or duplicate document ID in {source}: {doc_id!r}")
            seen.add(doc_id)
            occurrences += row["occurrences"]
            total_bytes += row["exclusive_nixl_bytes"]
            stream.write(_encode({**_select(row, fields), "token_work_by_role": _work(row["token_work_by_role"])}))
    if (
        len(seen) != report["evidence"]["distinct_documents"]
        or occurrences != report["evidence"]["occurrences"]
        or total_bytes != report["nixl_bytes"]["exclusive_evidence_bytes"]
    ):
        raise ValueError(f"Document measurement totals differ from the published report: {source}")
    return len(seen)


def _timeline(source, destination, request_ids):
    timeline = _read_json(source)
    events = timeline["traceEvents"]
    process_ids, sample = {}, set()
    for event in events:
        if event["ph"] == "M" and event["name"] == "process_name":
            request_id = event["args"]["name"]
            if request_id not in request_ids or request_id in sample or event["pid"] in process_ids:
                raise ValueError(f"Timeline has a duplicate or unknown request: {request_id!r}")
            sample.add(request_id)
            process_ids[event["pid"]] = len(process_ids) + 1
    if not sample:
        raise ValueError(f"Timeline has no sampled requests: {source}")

    output = []
    for event in events:
        phase = event["ph"]
        row = _select(event, ("name", "ph", "tid", "ts", "dur", "s"))
        # Chrome trace viewers require a process key. Publish only artificial IDs
        # assigned to the sampled requests and omit all process IDs in event details.
        row["pid"] = process_ids[event["pid"]]
        if phase == "M":
            lane = event["args"]["name"]
            if event["name"] not in ("process_name", "thread_name") or (
                event["name"] == "thread_name"
                and not re.fullmatch(r"(?:client|gateway|(?:gateway|vllm|GPU|NIXL) [pd][0-9]+)", lane)
            ):
                raise ValueError(f"Unexpected timeline metadata in {source}")
            row["args"] = {"name": lane}
        elif phase in ("X", "i"):
            row["args"] = _select(event["args"], _TIMELINE_DETAIL)
        else:
            raise ValueError(f"Unexpected timeline phase in {source}: {phase!r}")
        output.append(row)
    destination.write_bytes(_encode({"displayTimeUnit": "ms", "traceEvents": output}))
    return len(sample)


def _file_record(path, rows):
    return {"bytes": _check_size(path), "sha256": _sha256(path), "rows": rows}


def _check_size(path):
    size = path.stat().st_size
    if size > MAX_GIT_FILE_BYTES:
        raise ValueError(f"Archive file is larger than GitHub's 100 MiB limit: {path}")
    return size


def _stage_checkpoint(run_root, target, dataset, label, cohort, report, digest):
    cohort_dir = run_root / cohort
    measurement = cohort_dir / "datasets" / label
    if _read_json(measurement / "summary.json") != report:
        raise ValueError(f"Scratch and published summaries differ for {cohort} {dataset}")
    output = target / label / cohort
    output.mkdir(parents=True)
    responses_path = output / "responses.jsonl.gz"
    responses = _responses(
        cohort_dir / "client" / label / "responses.jsonl", responses_path, dataset, label,
    )
    if len(responses) != report["requests"]:
        raise ValueError(f"Response receipt count differs from the published report for {cohort} {dataset}")
    request_ids = set(responses)
    requests_path = output / "requests.jsonl.gz"
    count = _requests(measurement / "requests.jsonl", requests_path, responses, report)
    document_source = measurement / "documents.jsonl"
    document_name = "documents.jsonl.gz" if document_source.stat().st_size > DOCUMENT_GZIP_BYTES else "documents.jsonl"
    document_path = output / document_name
    documents = _documents(document_source, document_path, report)
    timeline_path = output / "timeline.json"
    sampled = _timeline(measurement / "timeline.json", timeline_path, request_ids)
    files = {
        path.name: _file_record(path, rows) for path, rows in (
            (responses_path, count), (requests_path, count), (document_path, documents), (timeline_path, sampled),
        )
    }
    return {
        "dataset": dataset, "requests": count, "distinct_documents": documents, "timeline_requests": sampled,
        "summary_sha256": digest, "files": files,
    }


def build_archive(run_root: Path, campaign_reports: Path, target: Path) -> dict:
    """Write a review archive to a fresh directory using passed campaign artifacts."""
    target.mkdir()
    checkpoints, request_count = {}, 0
    for dataset in DATASETS:
        label = dataset.replace("/", "-")
        checkpoints[label] = {}
        for cohort in COHORTS:
            try:
                report, digest = _published_checkpoint(campaign_reports, dataset, label, cohort)
                checkpoint = _stage_checkpoint(run_root, target, dataset, label, cohort, report, digest)
            except (KeyError, TypeError) as error:
                raise ValueError(f"Malformed source artifact for {cohort} {dataset}: {error}") from error
            checkpoints[label][cohort] = checkpoint
            request_count += checkpoint["requests"]
    manifest = {
        "schema_version": 1, "campaign": campaign_reports.name, "checkpoint_count": len(DATASETS) * len(COHORTS),
        "request_count": request_count, "checkpoints": checkpoints,
    }
    (target / MANIFEST).write_bytes(_encode(manifest))
    _check_size(target / MANIFEST)
    return manifest


def verify_archive(target: Path, campaign_reports: Path) -> dict:
    """Verify an archive and its published checkpoints without scratch."""
    if target.is_symlink() or (target / MANIFEST).is_symlink():
        raise ValueError(f"Archive and manifest must be local paths: {target}")
    _check_size(target / MANIFEST)
    manifest = _read_json(target / MANIFEST)
    labels = {dataset.replace("/", "-"): dataset for dataset in DATASETS}
    try:
        checkpoints = manifest["checkpoints"]
        if (
            manifest["schema_version"] != 1 or manifest["campaign"] != campaign_reports.name
            or manifest["checkpoint_count"] != len(DATASETS) * len(COHORTS) or set(checkpoints) != set(labels)
        ):
            raise ValueError("Archive manifest does not describe the published campaign")
        expected_paths = {target / MANIFEST}
        request_count = 0
        for label, dataset in labels.items():
            if set(checkpoints[label]) != set(COHORTS):
                raise ValueError(f"Archive has an unexpected cohort set for {dataset}")
            for cohort, checkpoint in checkpoints[label].items():
                report, digest = _published_checkpoint(campaign_reports, dataset, label, cohort)
                docs = report["evidence"]["distinct_documents"]
                sampled = checkpoint["timeline_requests"]
                if (
                    checkpoint["dataset"] != dataset or checkpoint["summary_sha256"] != digest
                    or checkpoint["requests"] != report["requests"] or checkpoint["distinct_documents"] != docs
                    or type(sampled) is not int or not 0 < sampled <= report["requests"]
                ):
                    raise ValueError(f"Archive checkpoint differs from the published report for {cohort} {dataset}")
                files = checkpoint["files"]
                document_name = "documents.jsonl.gz" if "documents.jsonl.gz" in files else "documents.jsonl"
                rows_by_name = {
                    "responses.jsonl.gz": report["requests"], "requests.jsonl.gz": report["requests"],
                    document_name: docs, "timeline.json": sampled,
                }
                if set(files) != set(rows_by_name):
                    raise ValueError(f"Archive has an unexpected file set for {cohort} {dataset}")
                for name, rows in rows_by_name.items():
                    path = target / label / cohort / name
                    expected_paths.add(path)
                    if path.is_symlink() or _file_record(path, rows) != files[name]:
                        raise ValueError(f"Archive artifact failed hash, size, or count verification: {path}")
                request_count += report["requests"]
        actual_paths = {path for path in target.rglob("*") if path.is_file() or path.is_symlink()}
        if request_count != manifest["request_count"] or expected_paths != actual_paths:
            raise ValueError("Archive request count or file set differs from the manifest")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed archive manifest at {target / MANIFEST}: {error}") from error
    return manifest
