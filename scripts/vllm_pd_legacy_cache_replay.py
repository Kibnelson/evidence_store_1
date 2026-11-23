#!/usr/bin/env python3
"""Replay complete P-side KV page stores and loads from a verified MT-RAG archive."""

import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path

from vllm_pd_legacy_page_payloads import AVAILABLE, capture_records, load_payload
from vllm_pd_legacy_replay_codec import (
    decode_delta,
    decode_independent,
    encode_delta,
    encode_independent,
)


MIB = 1024 ** 2
VERSION_BYTES = 32
BASE_ID_BYTES = 32
POLICIES = ("raw", "independent", "differential")
COMPATIBILITY_KEYS = (
    "worker_id", "pid", "block_size", "layer_count", "row_bytes", "tensor_dtype", "native_kv_format",
)


@dataclass(frozen=True)
class Entry:
    kind: str
    blob: bytes
    base: str | None = None

    @property
    def input_bytes(self):
        return len(self.blob) + (BASE_ID_BYTES if self.base is not None else 0)

    @property
    def resident_bytes(self):
        return self.input_bytes + VERSION_BYTES


@dataclass
class Page:
    store: dict
    raw: bytes
    independent: Entry
    deltas: list[Entry] = field(default_factory=list)

    @property
    def page_id(self):
        return self.store["target_page_id"]

    def delta_args(self):
        return self.store["block_size"], self.store["row_bytes"]


def _choose_bases(target, earlier):
    fields = ("logical_page_sha256", "start_token", "relative_start_token")
    store = target.store
    compatible = [
        page for page in earlier
        if page.store["ts_ns"] < store["ts_ns"] and page.store["request_id"] != store["request_id"]
        and all(page.store[key] == store[key] for key in COMPATIBILITY_KEYS)
    ]
    chosen = {}
    for name in fields:
        page = next((page for page in reversed(compatible) if page.store[name] == store[name]), None)
        if page is not None:
            chosen[page.page_id] = page
    traced = {store.get("base_version_sha256"), store.get("compared_base_version_sha256")} - {None}
    chosen.update((page.page_id, page) for page in compatible if page.store["version_sha256"] in traced)
    return chosen.values()


def _prepare(cohort, *, encode_page=encode_independent, decode_page=decode_independent,
             encode_patch=encode_delta, decode_patch=decode_delta, allow_no_loads=False):
    report = json.loads((cohort / "legacy-kv-page-report.json").read_text(encoding="utf-8"))
    inventory = capture_records(report)
    natural = {row["request_id"]: row["request_key"] for row in report["turns"]}
    stores = sorted(
        (row for row in report["stores"] if row["status"] == "measured"
         and natural.get(row.get("request_id")) == row.get("request_key")),
        key=lambda row: row["ts_ns"],
    )
    if report["status"] != "measured" or not stores:
        raise ValueError("Replay requires a measured, complete native page trace")
    pages = {}
    puts = defaultdict(list)
    events = []
    pairs = 0
    payloads = {}
    for row in stores:
        page_id, digest = row["target_page_id"], row["tensor_sha256"]
        if (row.get("payload_capture_status") not in AVAILABLE or not row["native_transfer_matched"]
                or digest not in inventory or row["tensor_dtype"] != "torch.bfloat16"):
            raise ValueError(f"Missing matched BF16 tensor for {page_id}")
        puts[page_id].append(row)
        events.append((row["ts_ns"], "store", page_id, row["turn"]))
        if page_id in pages:
            previous = pages[page_id].store
            if (previous["tensor_sha256"] != digest
                    or any(previous[k] != row[k] for k in COMPATIBILITY_KEYS)):
                raise ValueError(f"Conflicting stores for immutable page version {page_id}")
            continue
        if digest not in payloads:
            payloads[digest] = load_payload(cohort, digest, inventory[digest])
        raw = payloads[digest]
        independent = Entry("independent", encode_page(raw))
        if decode_page(independent.blob) != raw:
            raise ValueError(f"Independent codec failed for {page_id}")
        page = Page(row, raw, independent)
        for base in _choose_bases(page, pages.values()):
            pairs += 1
            kind, blob = encode_patch(raw, base.raw, *page.delta_args())
            if decode_patch(kind, blob, base.raw, *page.delta_args()) != raw:
                raise ValueError(f"Differential codec failed for {page_id} against {base.page_id}")
            entry = Entry(kind, blob, base.page_id)
            if entry.resident_bytes < independent.resident_bytes:
                page.deltas.append(entry)
        page.deltas.sort(key=lambda entry: entry.resident_bytes)
        pages[page_id] = page

    loads = 0
    for row in report["restores"]:
        if row.get("status") not in ("raw", "verified", "missing_base"):
            continue
        page_id = row["target_page_id"]
        source = next((put for put in reversed(puts[page_id]) if put["ts_ns"] < row["ts_ns"]), None)
        if (source is None or not row.get("source_store_version_matched")
                or not row["native_transfer_matched"]
                or row["last_observed_source_store_request_id"] != source["request_id"]
                or row["tensor_sha256"] != source["tensor_sha256"]
                or row.get("observed_sha256") != source["tensor_sha256"]):
            raise ValueError(f"Could not match native load to its captured store for {page_id}")
        events.append((row["ts_ns"], "load", page_id, row["turn"]))
        loads += 1
    if not loads and not allow_no_loads:
        raise ValueError("Replay requires observed native page loads")
    events.sort()
    return pages, events, {
        "natural_requests": report["natural_requests"],
        "captured_stores": len(stores),
        "captured_raw_bytes": sum(row["raw_bytes"] for row in stores),
        "native_loads": loads,
        "versions_with_loads": len({page_id for _, op, page_id, _ in events if op == "load"}),
        "candidate_pairs_verified": pairs,
        "candidate_pairs_smaller_than_independent": sum(len(page.deltas) for page in pages.values()),
    }


class Cache:
    def __init__(self, capacity, min_delta_saving_pct):
        self.capacity = capacity
        self.min_delta_saving_pct = min_delta_saving_pct
        self.resident = OrderedDict()
        self.pins = Counter()
        self.bytes = 0
        self.peak_bytes = 0
        self.peak_pinned_base_bytes = 0
        self.stats = Counter()
        self.formats = Counter()

    def _eviction_plan(self, entry):
        if entry.resident_bytes > self.capacity:
            return None
        needed = self.bytes + entry.resident_bytes - self.capacity
        victims = []
        if needed <= 0:
            return victims
        pins = self.pins.copy()
        if entry.base is not None:
            pins[entry.base] += 1
        remaining = self.resident.copy()
        while needed > 0:
            victim = next((key for key in remaining if pins[key] == 0), None)
            if victim is None:
                return None
            removed = remaining.pop(victim)
            if removed.base is not None:
                pins[removed.base] -= 1
            needed -= removed.resident_bytes
            victims.append(victim)
        return victims

    def admit(self, page, policy):
        if page.page_id in self.resident:
            self.resident.move_to_end(page.page_id)
            self.stats["store_refreshes"] += 1
            return
        independent = Entry("raw", page.raw) if policy == "raw" else page.independent
        options = []
        if policy == "differential":
            options = [entry for entry in page.deltas
                       if entry.base in self.resident and self.resident[entry.base].base is None
                       and entry.resident_bytes * 100 <= independent.resident_bytes * (
                           100 - self.min_delta_saving_pct)]
        for entry in (*options, independent):
            victims = self._eviction_plan(entry)
            if victims is not None:
                break
        else:
            self.stats["rejected_stores"] += 1
            return

        for victim in victims:
            removed = self.resident.pop(victim)
            self.bytes -= removed.resident_bytes
            if removed.base is not None:
                self.pins[removed.base] -= 1
            self.stats["evictions"] += 1
        if entry.base is not None:
            self.pins[entry.base] += 1
            self.stats["encoded_bytes_saved_on_delta_admissions"] += (
                independent.resident_bytes - entry.resident_bytes
            )
        self.resident[page.page_id] = entry
        self.bytes += entry.resident_bytes
        self.stats["admissions"] += 1
        self.stats["admission_bytes"] += entry.resident_bytes
        self.formats[entry.kind] += 1
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.peak_pinned_base_bytes = max(
            self.peak_pinned_base_bytes,
            sum(self.resident[key].resident_bytes for key, count in self.pins.items() if count),
        )

    def load(self, page, verify):
        entry = self.resident.get(page.page_id)
        if entry is None:
            self.stats["misses"] += 1
            return False
        self.stats["hits"] += 1
        self.stats["target_read_bytes"] += entry.input_bytes
        if entry.base is not None:
            base = self.resident[entry.base]
            self.stats["delta_hits"] += 1
            self.stats["cold_base_read_bytes"] += base.input_bytes
            self.resident.move_to_end(entry.base)
        self.resident.move_to_end(page.page_id)
        if verify:
            if entry.kind == "raw":
                restored = entry.blob
            elif entry.base is None:
                restored = decode_independent(entry.blob)
            else:
                restored = decode_delta(entry.kind, entry.blob, decode_independent(base.blob),
                                        *page.delta_args())
            if restored != page.raw:
                raise ValueError(f"Incorrect bytes at replay load of {page.page_id}")
            self.stats["byte_verified_loads"] += 1
        return True


def _simulate(pages, events, policy, capacity_mib, refill, min_delta_saving_pct, verify=False):
    cache = Cache(capacity_mib * MIB, min_delta_saving_pct)
    hit_versions = set()
    hits_by_turn = Counter()
    misses_by_turn = Counter()
    resident_byte_ns = 0
    last = events[0][0]
    for ts, op, page_id, turn in events:
        resident_byte_ns += cache.bytes * (ts - last)
        last = ts
        page = pages[page_id]
        if op == "store":
            cache.admit(page, policy)
        elif cache.load(page, verify):
            hit_versions.add(page_id)
            hits_by_turn[turn] += 1
        else:
            misses_by_turn[turn] += 1
            if refill:
                cache.stats["modeled_miss_refills"] += 1
                cache.admit(page, policy)

    stats = cache.stats
    return {
        **dict(stats),
        "hits": stats["hits"],
        "misses": stats["misses"],
        "hits_by_turn": dict(sorted(hits_by_turn.items())),
        "misses_by_turn": dict(sorted(misses_by_turn.items())),
        "delta_hits": stats["delta_hits"],
        "distinct_versions_served": len(hit_versions),
        "peak_resident_bytes": cache.peak_bytes,
        "peak_pinned_base_bytes": cache.peak_pinned_base_bytes,
        "resident_bytes_at_end": cache.bytes,
        "resident_byte_seconds": resident_byte_ns / 1e9,
        "encoded_read_bytes_cold_base": stats["target_read_bytes"] + stats["cold_base_read_bytes"],
        "encoded_read_bytes_warm_base": stats["target_read_bytes"],
        "admissions_by_representation": dict(cache.formats),
    }


def replay(cohort, capacities, refill, min_delta_saving_pct, verify_at):
    pages, events, scope = _prepare(cohort)
    results = []
    for capacity in capacities:
        row = {"capacity_mib": capacity}
        for policy in POLICIES:
            row[policy] = _simulate(pages, events, policy, capacity, refill, min_delta_saving_pct)
        results.append(row)
    if verify_at is None:
        verification_point = max(
            results,
            key=lambda row: (row["differential"]["delta_hits"] > 0,
                             row["differential"]["hits"] - row["independent"]["hits"],
                             row["differential"]["delta_hits"],
                             -row["capacity_mib"]),
        )
        verify_at = verification_point["capacity_mib"]
    for row in results:
        if row["capacity_mib"] == verify_at:
            for policy in POLICIES:
                verified = _simulate(pages, events, policy, verify_at, refill, min_delta_saving_pct, verify=True)
                if verified["hits"] != row[policy]["hits"]:
                    raise ValueError("Hit count changed in byte-verifying replay")
                row[policy] = verified
    return {
        "schema_version": 1,
        "cohort": str(cohort),
        "scope": scope,
        "model": {
            "capacity": "One P cache, complete captured pages only",
            "eviction": "LRU by version with independent bases pinned while a stored delta references them",
            "delta_admission": "Smallest exact encoding among eligible resident independent bases, with independent fallback",
            "min_delta_saving_pct": min_delta_saving_pct,
            "miss_refill": refill,
            "miss_refill_scope": "An enabled refill assumes the original captured version is recreated and inserted after counting a miss",
            "version_id_bytes": VERSION_BYTES,
            "base_id_bytes": BASE_ID_BYTES,
            "read_cost": "Encoded target and base inputs on resident hits. Warm estimate assumes decoded base already available",
            "verify_at_mib": verify_at,
        },
        "capacities": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path, help="Published Full Blend cohort with raw page tensors")
    parser.add_argument("--capacity-mib", type=int, nargs="+", default=range(64, 165),
                        help="Logical complete-page cache capacities in MiB")
    parser.add_argument("--verify-at-mib", type=int, help="Capacity to verify every hit byte for byte")
    parser.add_argument("--min-delta-saving-pct", type=int, default=0,
                        help="Minimum target size saving before admitting a delta")
    parser.add_argument("--no-refill", action="store_true", help="Replay only the original recorded stores")
    args = parser.parse_args()
    capacities = sorted(set(args.capacity_mib))
    if capacities[0] <= 0 or (args.verify_at_mib is not None and args.verify_at_mib not in capacities):
        parser.error("Provide positive capacities and select the verification capacity from the sweep")
    if not 0 <= args.min_delta_saving_pct <= 100:
        parser.error("Minimum delta saving must be between 0 and 100 percent")
    try:
        result = replay(args.cohort, capacities, not args.no_refill, args.min_delta_saving_pct, args.verify_at_mib)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"KV cache replay stopped. {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
