#!/usr/bin/env python3
"""Compare standalone and exponent-delta KV caches on archived conversations."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
from random import Random
import re
from statistics import median
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import process_time_ns

from vllm_pd_legacy_blend_archive import PAGE_REPORT, RUN_NAME, _payload_specs, archive, sha256_file, verify
from vllm_pd_legacy_cache_replay import MIB, _prepare
from vllm_pd_legacy_exponent_codec import ExponentCodec
from vllm_pd_legacy_packed_replay import Replay
from vllm_pd_legacy_page_campaign import capture_receipt
from vllm_pd_legacy_page_payloads import capture_records, load_payload, payload_path


POLICIES = ("independent", "forward_pinned")
DEFAULT_CAPACITIES = (96, 128, 160, 192, 224, 256, 384, 512, 768, 1024, 1536, 2048, 3072)
MIN_SAVING_PCT = 5
TIMING_REPEATS = 7
SCHEMA_VERSION = 1


class ExponentReplay(Replay):
    def __init__(self, pages, codec, policy, capacity, verify_hits):
        super().__init__(pages, {}, codec.decode_independent, codec.decode_delta,
                         policy, capacity, MIN_SAVING_PCT, verify_hits)
        self.codec = codec

    def _restore(self, entries, key):
        entry = entries[key]
        if entry.base is None:
            return self.codec.decode_independent(entry.blob)
        base = entries[entry.base]
        if base.base is not None:
            raise ValueError("Exponent replay exceeded its one-hop dependency limit")
        return self.codec.restore_delta(entry.kind, entry.blob, base.blob)

    def load(self, key):
        entry = self.resident.get(key)
        base = self.resident[entry.base] if entry is not None and entry.base is not None else None
        hit = super().load(key)
        if base is not None:
            # The inherited replay charges the full independent base frame.
            self.stats["cold_base_reads_bytes"] -= base.input_bytes
            self.stats["cold_base_reads_bytes"] += self.codec.encoded_base_read_bytes(base.blob, entry.kind)
        return hit


def simulate(pages, events, codec, policy, budget_mib, *, verify_hits=False):
    cache = ExponentReplay(pages, codec, policy, budget_mib * MIB, verify_hits)
    for _, op, key, _ in events:
        if op == "store" or not cache.load(key):
            cache.admit(key)
    stats = cache.stats
    result = {name: stats[name] for name in (
        "hits", "misses", "delta_hits", "target_reads_bytes", "cold_base_reads_bytes",
        "rewrites", "rewritten_bytes", "rejected_admissions", "verified_hits",
    )}
    result.update(peak_resident_bytes=cache.peak_bytes,
                  resident_bytes_at_end=sum(entry.resident_bytes for entry in cache.resident.values()),
                  admission_formats=dict(cache.admission_formats))
    if verify_hits and result["hits"] != result["verified_hits"]:
        raise ValueError(f"Hit verification differs for {policy} at {budget_mib} MiB")
    return result


def best_exponent_pairs(pages):
    pairs = []
    for page in pages.values():
        eligible = next((entry for entry in page.deltas if entry.kind == "exponent_xor"
                         and 100 * entry.resident_bytes <= (100 - MIN_SAVING_PCT) * page.independent.resident_bytes), None)
        if eligible is not None:
            pairs.append((page, eligible))
    return pairs


def measure_pair(page, delta, base, codec, rng):
    base_frame = base.independent.blob
    ready_exponent = codec.decode_base_exponent(base_frame)
    operations = {
        "standalone_encode": (lambda: codec.encode_independent(page.raw), page.independent.blob),
        "ready_base_encode": (lambda: codec.encode_delta_from_exponent(page.raw, ready_exponent), delta.blob),
        "encoded_base_encode": (
            lambda: codec.encode_delta_from_exponent(page.raw, codec.decode_base_exponent(base_frame)), delta.blob),
        "standalone_restore": (lambda: codec.decode_independent(page.independent.blob), page.raw),
        "ready_base_restore": (lambda: codec.restore_from_exponent(delta.blob, ready_exponent), page.raw),
        "encoded_base_restore": (lambda: codec.restore_delta(delta.kind, delta.blob, base_frame), page.raw),
    }
    order = list(operations)
    times = defaultdict(list)
    for _ in range(TIMING_REPEATS):
        rng.shuffle(order)
        for name in order:
            operation, expected = operations[name]
            start = process_time_ns()
            observed = operation()
            elapsed = process_time_ns() - start
            if observed != expected:
                raise ValueError(f"{name} differed from the reference for {page.page_id}")
            times[name].append(elapsed)
    return {"target_page_id": page.page_id, "base_page_id": base.page_id,
            "standalone_bytes": page.independent.resident_bytes, "delta_bytes": delta.resident_bytes,
            "cold_base_read_bytes": codec.encoded_base_read_bytes(base_frame, delta.kind),
            "median_cpu_ns": {key: median(values) for key, values in times.items()}}


def quantify_timings(pairs):
    values = defaultdict(list)
    for pair in pairs:
        for operation, duration in pair["median_cpu_ns"].items():
            values[operation].append(duration)
    return {operation: {"p50_ms": median(times) / 1e6,
                        "p95_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1] / 1e6}
            for operation, times in values.items()}


def checked_capture(cohort, *, conversation=None, window=None, expected=None):
    if cohort.parts[-2:] != ("1p3d", "blend"):
        raise ValueError(f"Expected a 1p3d/blend cohort, got {cohort}")
    report = json.loads((cohort / PAGE_REPORT).read_text(encoding="utf-8"))
    turns = report["turns"]
    if not turns:
        raise ValueError(f"No natural turns in {cohort}")
    conversation = conversation or turns[0]["conversation_id"]
    window = window or f"{turns[0]['turn']}:{turns[-1]['turn']}"
    if not re.fullmatch(r"[0-9a-f]{32}", conversation):
        raise ValueError(f"Invalid conversation ID {conversation!r}")
    receipt = capture_receipt(report, conversation, window)
    if expected is not None:
        for key, value in receipt.items():
            if expected.get(key) != value:
                raise ValueError(f"Capture field {key} differs from the receipt for {cohort}")
    return report, receipt


def replay_capture(cohort, report, receipt, capacities, timing_pairs, library_path, basis):
    stores = report["stores"]
    if not stores or {row["raw_bytes"] for row in stores if row["status"] == "measured"} != {stores[0]["raw_bytes"]}:
        raise ValueError(f"Expected one complete KV page size in {cohort}")
    codec = ExponentCodec(stores[0]["raw_bytes"], library_path)
    pages, events, scope = _prepare(cohort, encode_page=codec.encode_independent,
                                    decode_page=codec.decode_independent,
                                    encode_patch=codec.encode_delta, decode_patch=codec.decode_delta,
                                    allow_no_loads=True)
    for field, expected in (("natural_requests", receipt["turns"]), ("captured_stores", receipt["stores"]),
                            ("native_loads", receipt["observed_loads"])):
        if scope[field] != expected:
            raise ValueError(f"{field} differs from the campaign receipt for {cohort}")
    print(f"  {len(pages)} pages, {scope['native_loads']} loads, {scope['candidate_pairs_verified']} verified base pairs", flush=True)
    results = {}

    def sweep(capacity):
        if capacity not in results:
            results[capacity] = {policy: simulate(pages, events, codec, policy, capacity) for policy in POLICIES}

    for capacity in capacities:
        sweep(capacity)
    first_full = dict.fromkeys(POLICIES)
    if scope["native_loads"]:
        for policy in POLICIES:
            index = next((i for i, c in enumerate(capacities) if results[c][policy]["misses"] == 0), None)
            if index is not None:
                lower = capacities[index - 1] if index else 0
                for capacity in range(lower + 4, capacities[index], 4):
                    sweep(capacity)
        for policy in POLICIES:
            first = next((c for c in sorted(results) if results[c][policy]["misses"] == 0), None)
            first_full[policy] = first
            audit_mib = first if first is not None else capacities[-1]
            results[audit_mib][policy] = simulate(pages, events, codec, policy, audit_mib, verify_hits=True)

    eligible = best_exponent_pairs(pages)
    count = min(timing_pairs, len(eligible))
    rng = Random(30568)
    sampled = [eligible[i * len(eligible) // count] for i in range(count)]
    timings = [measure_pair(page, entry, pages[entry.base], codec, rng) for page, entry in sampled]
    return {
        "input": basis, "scope": scope, "zstd_version": codec.zstd_version,
        "standalone_frame_bytes": sum(page.independent.resident_bytes for page in pages.values()),
        "eligible_exponent_targets": len(eligible), "first_full_capacity_mib": first_full,
        "capacity_results": {str(capacity): row for capacity, row in sorted(results.items())}, "matched_pairs": timings,
        "matched_cpu_times": quantify_timings(timings),
    }


def write_report(path, report):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        staged = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)


def source_fingerprint():
    files = (Path(__file__), *(Path(__file__).with_name(name) for name in (
        "vllm_pd_legacy_exponent_codec.py", "vllm_pd_legacy_exponent_codec.c",
        "vllm_pd_legacy_cache_replay.py", "vllm_pd_legacy_packed_replay.py",
        "vllm_pd_legacy_blend_archive.py", "vllm_pd_legacy_page_campaign.py",
        "vllm_pd_legacy_page_payloads.py",
    )))
    digest = sha256()
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def replay_config(args, **scope):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": SCHEMA_VERSION, "capacities_mib": sorted(set(args.capacity_mib)),
        "minimum_saving_pct": MIN_SAVING_PCT, "timing_pairs": args.timing_pairs,
        "implementation_sha256": source_fingerprint(), "zstd": ExponentCodec(2, args.zstd_library).zstd_version,
        "host": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
        "zstd_library": args.zstd_library, **scope,
    }


def replay_one(args, cohort, page_report, receipt, config, **identity):
    key = receipt["conversation_id"]
    basis = {"configuration": config, "capture": receipt, **identity}
    path = args.output_dir / f"{key}.json"
    if path.exists():
        report = json.loads(path.read_text())
        if report["input"] != basis:
            raise ValueError(f"Existing replay {path} has different inputs. Use another output directory.")
        print(f"Reusing {path}", flush=True)
    else:
        print(f"Replaying {key}, {receipt['turns']} turns", flush=True)
        report = replay_capture(cohort, page_report, receipt, config["capacities_mib"],
                                args.timing_pairs, args.zstd_library, basis)
        write_report(path, report)
    print(f"  First tested full-hit budgets {report['first_full_capacity_mib']}", flush=True)
    return report


def published_packet(cohort, digest):
    manifest_bytes = (cohort.parents[1] / "archive-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["source_archive_sha256"] != digest or manifest["include_raw_kv_pages"] is not True:
        raise ValueError(f"Archive manifest differs from the expected raw capture in {cohort}")
    verify(cohort.parents[1])
    return sha256(manifest_bytes).hexdigest()


def run_campaign(args):
    index_path = args.campaign.resolve(strict=True)
    if index_path.parent.name != "campaigns" or index_path.parent.parent.name != "reports":
        raise ValueError("Use the campaign index under reports/campaigns in the results checkout")
    root = index_path.parents[2]
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    receipts = {row["conversation_id"]: row for row in index["captures"]}
    if len(receipts) != len(index["captures"]) or not receipts:
        raise ValueError("Campaign contains duplicate IDs or no captures")
    requested = set(args.conversation)
    if len(requested) != len(args.conversation) or not requested <= receipts.keys():
        raise ValueError("Unknown or duplicate requested conversation IDs")
    selection = [row for key, row in receipts.items() if not requested or key in requested]
    config = replay_config(args, index_sha256=sha256(index_bytes).hexdigest())
    captures = {}
    for row in selection:
        key = row["conversation_id"]
        relative = Path(row["cohort"])
        cohort = (root / relative).resolve()
        if (relative.is_absolute() or not cohort.is_relative_to(root / "reports/smoke")
                or cohort.parts[-2:] != ("1p3d", "blend")):
            raise ValueError(f"Unexpected cohort path in index: {relative}")
        manifest_digest = published_packet(cohort, row["sha256"])
        page_report, _ = checked_capture(cohort, conversation=key, window=row["turn_window"], expected=row)
        captures[key] = replay_one(args, cohort, page_report, row, config, manifest_sha256=manifest_digest)

    if len(captures) == len(receipts):
        summarized = {}
        summed_fields = ("hits", "misses", "delta_hits", "target_reads_bytes", "cold_base_reads_bytes", "rewrites")
        for capacity in config["capacities_mib"]:
            summarized[capacity] = {policy: {
                name: sum(c["capacity_results"][str(capacity)][policy][name] for c in captures.values())
                for name in summed_fields} for policy in POLICIES}
        summary = {"campaign": index["slurm_job_id"], "configuration": config,
                   "captures": {key: c["first_full_capacity_mib"] for key, c in captures.items()},
                   "observed_loads": sum(c["scope"]["native_loads"] for c in captures.values()),
                   "aggregate_capacity_results": summarized,
                   "matched_cpu_times": quantify_timings([p for c in captures.values() for p in c["matched_pairs"]])}
        path = args.output_dir / "campaign-summary.json"
        write_report(path, summary)
        print(f"Completed {len(captures)} conversations. Summary: {path}", flush=True)


def verify_live_capture(cohort, source, digest):
    run_name = source.name.removesuffix(".tar.gz")
    if (not source.name.endswith(".tar.gz") or RUN_NAME.fullmatch(run_name) is None
            or cohort.parts[-3:] != (run_name, "1p3d", "blend") or sha256_file(source) != digest):
        raise ValueError("Live cohort, capture archive filename, or SHA-256 differs from the receipt")
    prefix = f"{run_name}/1p3d/blend/"
    report_bytes = (cohort / PAGE_REPORT).read_bytes()
    payloads = _payload_specs(json.loads(report_bytes))
    expected = {prefix + PAGE_REPORT: (Path(PAGE_REPORT), len(report_bytes))}
    for tensor_digest, (_, size) in payloads.items():
        path = payload_path(tensor_digest)
        expected[prefix + path.as_posix()] = (path, size)
    remaining = set(expected)
    with tarfile.open(source, "r|gz") as package:
        for member in package:
            if member.name not in expected:
                continue
            relative, size = expected[member.name]
            if member.name not in remaining or not member.isfile() or member.size != size:
                raise ValueError(f"Unexpected duplicate, type, or size for archive member {member.name}")
            local = cohort / relative
            if local.is_symlink() or local.parent.is_symlink() or not local.is_file() or local.stat().st_size != size:
                raise ValueError(f"Live capture differs from the verified archive at {local}")
            with package.extractfile(member) as archived, local.open("rb") as original:
                while chunk := archived.read(1024 ** 2):
                    if chunk != original.read(len(chunk)):
                        raise ValueError(f"Live capture differs from the verified archive at {local}")
            remaining.remove(member.name)
    if remaining:
        raise ValueError(f"Missing {len(remaining)} expected page capture members in {source}")


def run_single(args, cohort, digest, *, expected=None):
    conversation = args.conversation[0] if args.conversation else None
    window = args.turn_window
    if expected is not None:
        for value, name in ((conversation, "conversation_id"), (window, "turn_window")):
            if value is not None and value != expected[name]:
                raise ValueError(f"Requested {name} differs from the capture receipt")
        conversation, window = expected["conversation_id"], expected["turn_window"]
    page_report, receipt = checked_capture(cohort, conversation=conversation, window=window, expected=expected)
    receipt.update(sha256=digest, run_name=cohort.parents[1].name)
    if args.verify_only:
        for page_digest, size in capture_records(page_report).items():
            load_payload(cohort, page_digest, size)
        print(json.dumps(receipt), flush=True)
        return
    report_digest = sha256(json.dumps(page_report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    replay_one(args, cohort, page_report, receipt, replay_config(args), page_report_sha256=report_digest)


def run(args):
    if args.campaign:
        run_campaign(args)
    elif args.archive:
        with TemporaryDirectory(prefix="es-exponent-replay-", dir=args.scratch_dir) as scratch:
            packet = archive(args.archive, args.sha256, include_raw_kv_pages=True, output_root=Path(scratch))
            run_single(args, packet / "1p3d/blend", args.sha256)
    else:
        cohort = args.cohort.resolve(strict=True)
        if args.capture_receipt:
            expected = json.loads(args.capture_receipt.read_text(encoding="utf-8"))
            source, digest = Path(expected["archive"]), expected["sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Capture receipt requires a SHA-256 digest")
            verify_live_capture(cohort, source, digest)
            run_single(args, cohort, digest, expected=expected)
        else:
            published_packet(cohort, args.sha256)
            run_single(args, cohort, args.sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--campaign", type=Path, help="Published campaign index JSON in the results checkout")
    source.add_argument("--archive", type=Path, help="Original captured conversation tar.gz")
    source.add_argument("--cohort", type=Path, help="Published or live 1p3d/blend cohort directory")
    parser.add_argument("--capture-receipt", type=Path, help="Capture receipt JSON for a live cohort")
    parser.add_argument("--sha256", help="Expected original archive digest for --archive or a published --cohort")
    parser.add_argument("--turn-window", help="Expected FIRST:LAST captured window for a single conversation")
    parser.add_argument("--scratch-dir", type=Path, help="Temporary extraction parent for --archive")
    parser.add_argument("--verify-only", action="store_true", help="Verify a single capture and print its receipt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--conversation", action="append", default=[], help="Replay only this conversation ID, repeatable")
    parser.add_argument("--capacity-mib", nargs="+", type=int, default=DEFAULT_CAPACITIES)
    parser.add_argument("--timing-pairs", type=int, default=9, help="Maximum exact matched timing pairs per conversation")
    parser.add_argument("--zstd-library", help="Path to libzstd if it is outside the system library search path")
    args = parser.parse_args()
    if min(args.capacity_mib) <= 0 or args.timing_pairs < 0:
        parser.error("Capacities must be positive and timing-pairs must be nonnegative")
    if not args.verify_only and args.output_dir is None:
        parser.error("--output-dir is required for a replay")
    if args.campaign and (args.verify_only or args.sha256 or args.turn_window):
        parser.error("--verify-only, --sha256, and --turn-window apply to single captures")
    if args.capture_receipt is not None and (not args.cohort or args.sha256):
        parser.error("--capture-receipt requires --cohort and supplies its own archive digest")
    if not args.campaign and len(args.conversation) > 1:
        parser.error("A single capture accepts one --conversation")
    if not args.campaign and not args.capture_receipt and not re.fullmatch(r"[0-9a-f]{64}", args.sha256 or ""):
        parser.error("Provide the expected archive digest with --sha256")
    if args.scratch_dir is not None and not args.archive:
        parser.error("--scratch-dir applies to --archive")
    try:
        run(args)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError, tarfile.TarError) as error:
        parser.exit(1, f"Exponent replay stopped: {error}\n")


if __name__ == "__main__":
    main()
