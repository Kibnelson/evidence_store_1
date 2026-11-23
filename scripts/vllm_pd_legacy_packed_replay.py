#!/usr/bin/env python3
"""Compare byte-exact packed KV frames and anchor policies across captured conversations."""

import argparse
from collections import Counter, OrderedDict
from hashlib import sha256
import json
import math
from pathlib import Path

from vllm_pd_legacy_cache_replay import Entry, MIB, _choose_bases, _prepare
from vllm_pd_legacy_packed_codec import decode_packed_delta, encode_packed_delta
from vllm_pd_legacy_packed_independent import decode_packed_independent, encode_packed_independent
from vllm_pd_legacy_replay_codec import decode_delta, decode_independent, encode_delta, encode_independent


POLICIES = ("independent", "forward_pinned", "demand_pinned", "forward_repack")


def _reverse_candidates(pages, encode_patch, decode_patch):
    earlier = []
    candidates = {}
    for page in pages.values():
        reverse = {}
        for base in _choose_bases(page, earlier):
            kind, blob = encode_patch(base.raw, page.raw, *base.delta_args())
            if decode_patch(kind, blob, page.raw, *base.delta_args()) != base.raw:
                raise ValueError(f"Reverse delta does not reconstruct {base.page_id}")
            entry = Entry(kind, blob, page.page_id)
            if entry.resident_bytes < base.independent.resident_bytes:
                reverse[base.page_id] = entry
        candidates[page.page_id] = reverse
        earlier.append(page)
    return candidates


def _dependencies(entries):
    return Counter(entry.base for entry in entries.values() if entry.base is not None)


def _beneficial(entry, full, minimum):
    return entry.resident_bytes * 100 <= full.resident_bytes * (100 - minimum)


class Replay:
    def __init__(self, pages, reverse, decode_page, decode_patch, policy, capacity, cutoff, verify):
        self.pages = pages
        self.reverse = reverse
        self.decode_page = decode_page
        self.decode_patch = decode_patch
        self.policy = policy
        self.capacity = capacity
        self.cutoff = cutoff
        self.verify = verify
        self.resident = OrderedDict()
        self.requests = Counter()
        self.stats = Counter()
        self.admission_formats = Counter()
        self.peak_bytes = 0

    def _restore(self, entries, key):
        frame = entries[key]
        if frame.base is None:
            return self.decode_page(frame.blob)
        base = entries[frame.base]
        if base.base is not None:
            raise ValueError("Replay exceeded its one-hop dependency limit")
        return self.decode_patch(frame.kind, frame.blob, self.decode_page(base.blob),
                                 *self.pages[key].delta_args())

    def _verify(self, entries, key):
        data = self._restore(entries, key)
        if sha256(data).hexdigest() != self.pages[key].store["tensor_sha256"]:
            raise ValueError(f"Incorrect replay reconstruction of {key}")

    def _proposals(self, key):
        page = self.pages[key]
        if self.policy != "independent":
            forward = [
                entry for entry in page.deltas
                if entry.base in self.resident and self.resident[entry.base].base is None
                and _beneficial(entry, page.independent, self.cutoff)
            ]
            if self.policy == "demand_pinned":
                # Compare only demand observed before this admission.
                forward = sorted(
                    (entry for entry in forward if self.requests[entry.base] > self.requests[key]),
                    key=lambda entry: (-self.requests[entry.base], entry.resident_bytes),
                )
            for entry in forward:
                yield entry, ()

            if self.policy == "demand_pinned":
                pins = _dependencies(self.resident)
                reverse = [
                    (base, entry) for base, entry in self.reverse[key].items()
                    if base in self.resident and self.resident[base].base is None and not pins[base]
                    and self.requests[base] <= self.requests[key]
                    and _beneficial(entry, self.pages[base].independent, self.cutoff)
                ]
                if reverse:
                    yield page.independent, reverse
        yield page.independent, ()

    def _plan(self, key, entry, edits):
        proposed = self.resident.copy()
        proposed.update(edits)
        proposed[key] = entry
        used = sum(item.resident_bytes for item in proposed.values())
        refs = _dependencies(proposed)
        while used > self.capacity:
            victim = next((
                page for page in proposed
                if page != key and (self.policy == "forward_repack" or not refs[page])
            ), None)
            if victim is None:
                return None
            old = proposed.pop(victim)
            used -= old.resident_bytes
            if old.base is not None:
                refs[old.base] -= 1
            if refs[victim]:
                for page, item in proposed.items():
                    if item.base == victim:
                        full = self.pages[page].independent
                        used += full.resident_bytes - item.resident_bytes
                        proposed[page] = full
                refs[victim] = 0
        return proposed, used

    def admit(self, key):
        if key in self.resident:
            self.resident.move_to_end(key)
            self.stats["store_refreshes"] += 1
            return
        for entry, edits in self._proposals(key):
            plan = self._plan(key, entry, edits)
            if plan is None:
                continue
            proposed, used = plan
            rewrite_inputs = set()
            for page, item in proposed.items():
                old = self.resident.get(page)
                if old is None or item is old:
                    continue
                rewrite_inputs.add(page)
                if old.base is not None:
                    rewrite_inputs.add(old.base)
                self.stats["rewrites"] += 1
                self.stats["rewritten_bytes"] += item.resident_bytes
                if self.verify:
                    self._verify(self.resident, page)
                    self._verify(proposed, page)
                    self.stats["verified_rewrites"] += 1
            self.stats["rewrite_read_bytes"] += sum(self.resident[x].input_bytes for x in rewrite_inputs)
            self.resident = proposed
            self.peak_bytes = max(self.peak_bytes, used)
            self.admission_formats[proposed[key].kind] += 1
            return
        self.stats["rejected_admissions"] += 1

    def load(self, key):
        self.requests[key] += 1
        frame = self.resident.get(key)
        if frame is None:
            self.stats["misses"] += 1
            return False
        self.stats["hits"] += 1
        self.stats["target_reads_bytes"] += frame.input_bytes
        if frame.base is not None:
            self.stats["delta_hits"] += 1
            self.stats["cold_base_reads_bytes"] += self.resident[frame.base].input_bytes
            self.resident.move_to_end(frame.base)
        self.resident.move_to_end(key)
        if self.verify:
            self._verify(self.resident, key)
            self.stats["verified_hits"] += 1
        return True


def _simulate(pages, events, reverse, decode_page, decode_patch, policy, capacity, cutoff, refill,
              verify=False):
    cache = Replay(pages, reverse, decode_page, decode_patch, policy, capacity, cutoff, verify)
    for _, op, key, _ in events:
        if op == "store" or (not cache.load(key) and refill):
            cache.admit(key)
    stats = cache.stats
    return {
        **dict(stats),
        "hits": stats["hits"],
        "misses": stats["misses"],
        "delta_hits": stats["delta_hits"],
        "rewrites": stats["rewrites"],
        "cold_encoded_read_bytes": stats["target_reads_bytes"] + stats["cold_base_reads_bytes"],
        "peak_resident_bytes": cache.peak_bytes,
        "resident_bytes_at_end": sum(item.resident_bytes for item in cache.resident.values()),
        "admission_formats": dict(cache.admission_formats),
    }


def _run_format(cohort, capacities, cutoff, refill, encode_page, decode_page, encode_patch, decode_patch):
    pages, events, scope = _prepare(
        cohort, encode_page=encode_page, decode_page=decode_page,
        encode_patch=encode_patch, decode_patch=decode_patch, allow_no_loads=True,
    )
    reverse = _reverse_candidates(pages, encode_patch, decode_patch)
    results = [
        {"capacity_bytes": capacity, **{
            policy: _simulate(pages, events, reverse, decode_page, decode_patch,
                              policy, capacity, cutoff, refill)
            for policy in POLICIES
        }}
        for capacity in capacities
    ]
    first_full = {
        policy: next((row["capacity_bytes"] for row in results
                      if scope["native_loads"] and row[policy]["misses"] == 0), None)
        for policy in POLICIES
    }
    audit = next(
        (row for row in results if all(row[policy]["misses"] == 0 for policy in POLICIES)),
        results[-1],
    )
    for policy in POLICIES:
        verified = _simulate(pages, events, reverse, decode_page, decode_patch, policy,
                             audit["capacity_bytes"], cutoff, refill, verify=True)
        if verified.get("verified_hits", 0) != audit[policy]["hits"]:
            raise ValueError("Replay changed its hit count during byte verification")
        if verified.get("verified_rewrites", 0) != audit[policy]["rewrites"]:
            raise ValueError("Replay changed its rewrite count during byte verification")
        audit[policy] = verified
    return {
        "scope": scope,
        "standalone_encoded_bytes": sum(page.independent.resident_bytes for page in pages.values()),
        "candidate_formats": dict(Counter(entry.kind for page in pages.values() for entry in page.deltas)),
        "audit_capacity_bytes": audit["capacity_bytes"],
        "first_full_capacity_bytes": first_full,
        "capacities": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", nargs="+", type=Path, help="One or more raw-page cohort directories")
    parser.add_argument("--capacity-mib", nargs="+", type=float, default=range(64, 165),
                        help="Committed cache capacities in MiB, including fractions")
    parser.add_argument("--min-delta-saving-pct", type=int, default=5)
    parser.add_argument("--no-refill", action="store_true", help="Replay only original recorded stores")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value * MIB < 1 for value in args.capacity_mib):
        parser.error("Cache capacities must be positive finite values")
    if not 0 <= args.min_delta_saving_pct <= 100:
        parser.error("Minimum delta saving must be between 0 and 100 percent")
    capacities = sorted({round(value * MIB) for value in args.capacity_mib})
    results = []
    for cohort in args.cohort:
        try:
            report = json.loads((cohort / "legacy-kv-page-report.json").read_text(encoding="utf-8"))
            options = {
                "current": (encode_independent, decode_independent, encode_delta, decode_delta),
                "packed_rows": (encode_independent, decode_independent, encode_packed_delta, decode_packed_delta),
                "packed_pages_and_rows": (encode_packed_independent, decode_packed_independent,
                                           encode_packed_delta, decode_packed_delta),
            }
            formats = {
                name: _run_format(cohort, capacities, args.min_delta_saving_pct,
                                  not args.no_refill, *codecs)
                for name, codecs in options.items()
            }
        except (KeyError, OSError, ValueError) as error:
            parser.exit(1, f"Packed KV replay stopped for {cohort}. {error}\n")
        results.append({"cohort": str(cohort), "turns": report["turns"], "formats": formats})
    print(json.dumps({
        "schema_version": 1,
        "model": {
            "minimum_delta_saving_pct": args.min_delta_saving_pct,
            "refill_after_miss": not args.no_refill,
            "cohort_scope": "Each archived conversation capture runs with a fresh logical cache",
            "capacity_scope": "Committed encoded frames and metadata; captured input tensors are available offline",
            "restore_scope": "One-hop dependencies; lossless SHA-256 checks at a common budget for each format",
        },
        "cohorts": results,
    }, indent=2))


if __name__ == "__main__":
    main()
