#!/usr/bin/env python3
"""Verify a Full Blend tarball and stage its measurement receipts for Git."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
from tempfile import TemporaryDirectory

from vllm_pd_smoke_archive import MANIFEST, ROOT, fingerprint, read_json, sha256_file
from vllm_pd_detail_archive import _RESPONSE_FIELDS, _encode, _output, _read_rows, _select
from vllm_pd_legacy_blend_report import report
from vllm_pd_legacy_delta_estimate import FILES as ESTIMATE_FILES, SCHEMA_VERSION, write_estimate
from vllm_pd_legacy_page_report import REPORT_FILENAME as PAGE_REPORT, build_report as build_page_report
from vllm_pd_legacy_storage_report import REPORT_FILENAME as STORAGE_REPORT, build_report
from vllm_pd_legacy_storage_smoke import VERIFICATION_INPUT, VERIFICATION_LABEL
from vllm_pd_legacy_workloads import DATASETS


LAYOUTS = ("1p3d", "2p2d")
RUN_NAME = re.compile(r"(musique|musique-single-turn-40|mtrag-multi-turn-repeat)-legacy-full-blend-(\d+)-\d{8}T\d{6}Z")
TRACE_PATH = re.compile(r"trace/[A-Za-z0-9_.-]+\.jsonl")
CORE_FILES = ("legacy-blend-summary.json", "run_manifest.json", "endpoints.json", "cleanup.json",
              "metrics-before.json", "metrics-after.json", "inputs/manifest.json",
              "inputs/queries.jsonl", "inputs/controlled.jsonl")
CLIENT_FILES = ("manifest.json", "frozen.jsonl", "responses.jsonl")
RUNTIME_FIELDS = ("vllm", "lmcache", "torch", "nixl", "faiss", "torch_cuda", "vllm_wheel_base",
                  "vllm_wheel_sha256", "ucx_memory_types", "standalone_patch_sha256",
                  "trace_overlay_sha256", "lane_support_patch_sha256", "worker_runner_module")
NIXL_FIELDS = ("worker_id", "status", "completed_bytes", "transferred_blocks", "block_size",
               "position_start", "position_end", "valid_position_end", "ts_ns", "started_ns",
               "transfer_duration_us", "post_duration_us")


def read_member(package, name):
    member = package.getmember(name)
    if not member.isfile():
        raise ValueError(f"Expected a regular archive member at {name}")
    return package.extractfile(member)


def write_json(path, contents):
    path.write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")


def copy_member(package, prefix, relative, directory):
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with read_member(package, prefix + relative) as source, path.open("xb") as destination:
        shutil.copyfileobj(source, destination)


def source_receipt(package, names, run_name):
    with read_member(package, "check/runtime.json") as stream:
        runtime = _select(json.load(stream), RUNTIME_FIELDS)
    with read_member(package, "check/experiment-source.txt") as stream:
        runtime["build_commit"] = stream.read().decode().strip()
    retry_commit = f"check/{run_name}-replay-source.txt"
    runtime["replay_commit"] = runtime["build_commit"]
    if retry_commit in names:
        with read_member(package, retry_commit) as stream:
            runtime["replay_commit"] = stream.read().decode().strip()
        with read_member(package, f"check/{run_name}-lane-support.sha256") as stream:
            digest = stream.read().decode().split()[0]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid lane support patch digest")
        runtime["lane_support_patch_sha256"] = digest
    report_source = f"check/{run_name}-report-source.txt"
    if report_source in names:
        with read_member(package, report_source) as stream:
            runtime["report_commit"] = stream.read().decode().strip()
    if any(not re.fullmatch(r"[0-9a-f]{40}", runtime[field])
           for field in ("build_commit", "replay_commit", "report_commit") if field in runtime):
        raise ValueError("Invalid Full Blend source commit")
    return runtime


def write_rows(path, rows):
    with _output(path) as stream:
        for row in rows:
            stream.write(_encode(row))


def public_request(row):
    transfer = row["nixl_transfer"]
    return {**row, "nixl_transfer": _select(transfer, NIXL_FIELDS)}


def stage_delta_estimate(package, names, prefix, work, destination, sample, trace):
    schema = sample.get("delta_estimate_schema")
    present = {name for name in ESTIMATE_FILES if prefix + name in names}
    if schema is None and not present:
        return
    if schema != SCHEMA_VERSION or not trace or present != set(ESTIMATE_FILES):
        raise ValueError(f"Unexpected or incomplete delta estimate in {destination}")
    write_estimate(work)
    for name in ESTIMATE_FILES:
        with read_member(package, prefix + name) as source:
            if source.read() != (work / name).read_bytes():
                raise ValueError(f"Recomputed {name} differs in {destination}")
        if name.endswith(".jsonl"):
            write_rows(destination / f"{name}.gz", _read_rows(work / name))
        else:
            shutil.copyfile(work / name, destination / name)


def stage_cohort(package, names, prefix, work, destination, layout, job_id, workload):
    for name in CORE_FILES:
        copy_member(package, prefix, name, work)
    sample = read_json(work / "inputs/manifest.json")
    dataset = DATASETS.get(workload, "musique")
    if sample.get("dataset", "musique") != dataset or sample.get("workload", "musique") != workload:
        raise ValueError(f"Unexpected dataset selection in {layout}")
    launch = read_json(work / "run_manifest.json")
    original = read_json(work / "legacy-blend-summary.json")
    storage_trace = launch["legacy_full_blend"].get("kv_storage_trace_enabled", False)
    labels = (dataset, "legacy-probes")
    if original.get("storage_verification_requests"):
        if not storage_trace:
            raise ValueError(f"Unexpected KV storage verification in {layout}")
        copy_member(package, prefix, VERIFICATION_INPUT.as_posix(), work)
        labels += (VERIFICATION_LABEL,)
    files = [f"client/{label}/{name}" for label in labels for name in CLIENT_FILES]
    files.extend(sorted(name[len(prefix):] for name in names
                        if name.startswith(prefix) and TRACE_PATH.fullmatch(name[len(prefix):])))
    for name in files:
        copy_member(package, prefix, name, work)
    cleanup = read_json(work / "cleanup.json")
    if (launch["slurm_job_id"] != job_id or launch["topology"] != layout
            or launch["legacy_full_blend"]["mode"] != "in_process"
            or launch["prefix_cache_by_role"] != {"P": "off", "D": "off"}
            or launch["full_nixl_handoff"] is not True
            or cleanup["remaining_process_groups"] or cleanup["unverified_process_groups"]):
        raise ValueError(f"Unexpected Full Blend launch or incomplete cleanup in {layout}")
    trace = launch["engine_trace_enabled"]
    if report(work, trace=trace) != original:
        raise ValueError(f"Recomputed measurement summary differs in {layout}")
    destination.mkdir(parents=True)
    for name in ("legacy-blend-summary.json", "metrics-before.json", "metrics-after.json", "cleanup.json"):
        shutil.copyfile(work / name, destination / name)
    provenance = _select(launch, ("topology", "slurm_job_id", "model", "revision", "prefix_cache_by_role",
                                  "full_nixl_handoff", "engine_trace_enabled", "gpu_event_timing"))
    provenance["cache_per_producer_gib"] = launch["legacy_full_blend"]["cache_per_producer_gib"]
    provenance["sample"] = sample
    write_json(destination / "provenance.json", provenance)
    write_rows(destination / "responses.jsonl.gz", (
        _select(row, (*_RESPONSE_FIELDS, "case")) for label in labels
        for row in _read_rows(work / "client" / label / "responses.jsonl")
    ))
    if trace:
        for name, transform in (("legacy-requests.jsonl", public_request), ("legacy-chunks.jsonl", dict)):
            with read_member(package, prefix + name) as stream:
                previous = [json.loads(line) for line in stream]
            measured = list(_read_rows(work / name))
            if measured != previous:
                raise ValueError(f"Recomputed {name} differs in {layout}")
            write_rows(destination / f"{name}.gz", map(transform, measured))
    stage_delta_estimate(package, names, prefix, work, destination, sample, trace)
    if storage_trace:
        with read_member(package, prefix + STORAGE_REPORT) as stream:
            storage = json.load(stream)
        if build_report(work, replay_label=dataset) != storage:
            raise ValueError(f"Recomputed KV storage report differs in {layout}")
        write_json(destination / STORAGE_REPORT, storage)
    if launch["legacy_full_blend"].get("kv_page_trace_enabled", False):
        with read_member(package, prefix + PAGE_REPORT) as stream:
            pages = json.load(stream)
        if build_page_report(work, replay_label=dataset) != pages:
            raise ValueError(f"Recomputed KV page report differs in {layout}")
        write_json(destination / PAGE_REPORT, pages)


def gzip_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def check_layout(path):
    summary = read_json(path / "legacy-blend-summary.json")
    label = summary.get("natural_label", "musique")
    expected = {label: summary["natural_requests"], "legacy-probes": summary["controlled_requests"]}
    if summary.get("storage_verification_requests"):
        expected[VERIFICATION_LABEL] = summary["storage_verification_requests"]
    responses = gzip_rows(path / "responses.jsonl.gz")
    identities = {row["request_id"] for row in responses}
    if (summary["status"] != "passed" or Counter(row["label"] for row in responses) != expected
            or len(identities) != len(responses) or not all(row["success"] is True for row in responses)
            or len(responses) != summary["completed_nixl_reads"]):
        raise ValueError(f"Response receipts disagree with the summary in {path}")
    if summary["trace_enabled"]:
        requests = gzip_rows(path / "legacy-requests.jsonl.gz")
        chunks = gzip_rows(path / "legacy-chunks.jsonl.gz")
        natural = [row for row in chunks if row["label"] == label]
        shifted = sum(any(move["source_match"] and move["source_match"]["shifted"]
                          for move in row["lmcache_movements"]) for row in natural)
        if (len(requests) != len(responses) or {row["request_id"] for row in requests} != identities
                or sum(row["completed_nixl_bytes"] for row in requests) != summary["completed_nixl_bytes"]
                or len(natural) != summary["natural_evidence_occurrences"]
                or sum(row["previously_seen"] for row in natural) != summary["natural_repeat_occurrences"]
                or shifted != summary["natural_shifted_restore_occurrences"]):
            raise ValueError(f"Request and chunk measurements disagree with the summary in {path}")


def verify(directory):
    manifest = read_json(directory / MANIFEST)
    if (manifest["run_id"] != directory.name or manifest["schema_version"] != 1
            or not manifest["layouts"] or any(item not in LAYOUTS for item in manifest["layouts"])
            or fingerprint(directory) != manifest["files"]):
        raise ValueError(f"Invalid Full Blend archive manifest in {directory}")
    for layout in manifest["layouts"]:
        check_layout(directory / layout / "blend")


def archive(source, digest):
    run_name = source.name.removesuffix(".tar.gz")
    match = RUN_NAME.fullmatch(run_name)
    if not match or source.name == run_name or sha256_file(source) != digest:
        raise ValueError("The Full Blend archive filename or SHA-256 differs from the run receipt")
    target = ROOT / "reports/smoke" / run_name
    if target.exists():
        verify(target)
        if read_json(target / MANIFEST)["source_archive_sha256"] != digest:
            raise ValueError(f"A different archive is already published at {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as scratch, tarfile.open(source, "r:gz") as package:
        names = set(package.getnames())
        staged = Path(scratch) / run_name
        staged.mkdir()
        layouts = []
        for layout in LAYOUTS:
            prefix = f"{run_name}/{layout}/blend/"
            if prefix + "legacy-blend-summary.json" not in names:
                continue
            work = Path(scratch) / "private" / layout / "blend"
            stage_cohort(package, names, prefix, work, staged / layout / "blend", layout, match[2], match[1])
            layouts.append(layout)
        if layouts == list(LAYOUTS):
            label = DATASETS.get(match[1], "musique")
            copy_member(package, f"{run_name}/", f"{label}-pair.json", staged)
        write_json(staged / MANIFEST, {
            "schema_version": 1, "run_id": run_name, "layouts": layouts,
            "source_archive_sha256": digest, "runtime": source_receipt(package, names, run_name),
            "files": fingerprint(staged),
        })
        verify(staged)
        staged.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    try:
        if args.verify:
            verify(args.verify)
            print(f"Verified {args.verify}")
        else:
            if not re.fullmatch(r"[0-9a-f]{64}", args.sha256 or ""):
                parser.error("--source requires the SHA-256 printed by the smoke runner")
            print(archive(args.source, args.sha256).relative_to(ROOT))
    except (KeyError, OSError, TypeError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"Full Blend archive failed: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
