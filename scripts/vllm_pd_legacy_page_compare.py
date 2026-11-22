#!/usr/bin/env python3
"""Compare modeled row-replacement layouts for captured Full Blend KV pages."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from typing import NamedTuple

from vllm_pd_legacy_page_payloads import AVAILABLE, capture_records, load_payload


REPORT = "legacy-kv-page-report.json"
PAGE_HEADER_BYTES = 48
BASE_DESCRIPTOR_BYTES = 64
POLICIES = ("whole_page_one_base", "per_layer_up_to_one_base", "per_layer_up_to_two_bases")
RAW = -1


def _bitmap_bytes(bits):
    return (bits + 7) // 8


@dataclass(frozen=True)
class Geometry:
    block: int
    layers: int
    row_bytes: int
    dtype: str
    native_format: str

    @property
    def layer_bytes(self):
        return 2 * self.block * self.row_bytes

    @property
    def raw_bytes(self):
        return self.layers * self.layer_bytes


class BaseMatch(NamedTuple):
    store: dict
    changes: tuple
    layer_costs: tuple


class Plan(NamedTuple):
    encoded_bytes: int
    metadata_bytes: int
    kind: str
    layer_sources: tuple


def _geometry(row):
    shape = [row[key] for key in ("block_size", "layer_count", "row_bytes")]
    if any(type(value) is not int or value <= 0 for value in shape) or shape[0] != 16:
        raise ValueError("Expected a complete 16-token page with positive layer and row sizes")
    layout = Geometry(*shape, row["tensor_dtype"], row["native_kv_format"])
    if row["raw_bytes"] != layout.raw_bytes:
        raise ValueError(f"Incorrect tensor geometry for {row['tensor_sha256']}")
    return layout


def _fingerprint(raw, layout):
    view = memoryview(raw)
    return tuple(sha256(view[i:i + layout.row_bytes]).digest()
                 for i in range(0, len(raw), layout.row_bytes))


def _match(target, base, layout):
    block = layout.block
    counts = []
    for offset in range(0, len(target), 2 * block):
        k = sum(target[i] != base[i] for i in range(offset, offset + block))
        v = sum(target[i] != base[i] for i in range(offset + block, offset + 2 * block))
        counts.append((k, v))
    bitmap = _bitmap_bytes(2 * block)
    return tuple(counts), tuple(bitmap + (k + v) * layout.row_bytes for k, v in counts)


def _consider(candidate, current):
    # Keep an earlier simpler representation on a tie.
    return candidate if candidate.encoded_bytes < current.encoded_bytes else current


def _layered(indexes, matches, layout):
    costs, sources = [], []
    for layer in range(layout.layers):
        cost, source = min((layout.layer_bytes, RAW),
                           *((matches[index].layer_costs[layer], index) for index in indexes))
        costs.append(cost)
        sources.append(source)
    used = {source for source in sources if source != RAW}
    patched_layers = sum(source != RAW for source in sources)
    header = PAGE_HEADER_BYTES + len(used) * BASE_DESCRIPTOR_BYTES + _bitmap_bytes(2 * layout.layers)
    return Plan(sum(costs) + header, header + patched_layers * _bitmap_bytes(2 * layout.block),
                "layer_map", tuple(sources))


def _plans(matches, layout):
    raw = Plan(layout.raw_bytes, 0, "raw", (RAW,) * layout.layers)
    whole = one = raw
    whole_metadata = PAGE_HEADER_BYTES + BASE_DESCRIPTOR_BYTES + _bitmap_bytes(2 * layout.block * layout.layers)
    for index, match in enumerate(matches):
        replacements = sum(k + v for k, v in match.changes) * layout.row_bytes
        whole = _consider(Plan(whole_metadata + replacements, whole_metadata, "whole_page", (index,) * layout.layers),
                          whole)
        one = _consider(_layered((index,), matches, layout), one)
    one = _consider(one, whole)
    two = one
    for pair in combinations(range(len(matches)), 2):
        two = _consider(_layered(pair, matches, layout), two)
    return dict(zip(POLICIES, (whole, one, two), strict=True))


def _detail(plan, matches, layout):
    usage = Counter(plan.layer_sources)
    raw_layers = usage.pop(RAW, 0)
    references = []
    for index, layers in sorted(usage.items()):
        source = matches[index].store
        references.append({
            **{key: source[key] for key in ("store_index", "tensor_sha256", "version_sha256", "request_id",
                                            "worker_id", "pid", "raw_bytes")},
            "layers_used": layers,
        })
    raw = raw_layers * layout.layer_bytes
    return {
        "representation": plan.kind,
        "encoded_bytes": plan.encoded_bytes,
        "metadata_bytes": plan.metadata_bytes,
        "replacement_row_bytes": plan.encoded_bytes - raw - plan.metadata_bytes,
        "raw_layer_bytes": raw,
        "dependencies": references,
        "base_tensor_sha256_by_layer": [None if index == RAW else matches[index].store["tensor_sha256"]
                                         for index in plan.layer_sources],
        "changed_k_v_rows_by_layer": [None if index == RAW else matches[index].changes[layer]
                                      for layer, index in enumerate(plan.layer_sources)],
    }


def _summarize(targets, plans):
    dependencies = [base for plan in plans for base in plan["dependencies"]]
    unique = {(base["worker_id"], base["pid"], base["tensor_sha256"]): base["raw_bytes"] for base in dependencies}
    raw = sum(row["raw_bytes"] for row in targets)
    encoded = sum(plan["encoded_bytes"] for plan in plans)
    return {
        "raw_bytes": raw,
        "encoded_bytes": encoded,
        "gross_saved_bytes": raw - encoded,
        "gross_reduction": (raw - encoded) / raw,
        "metadata_bytes": sum(plan["metadata_bytes"] for plan in plans),
        "representations": dict(Counter(plan["representation"] for plan in plans)),
        "pages_by_dependency_count": dict(Counter(len(plan["dependencies"]) for plan in plans)),
        "dependency_references": len(dependencies),
        "unique_base_tensors_per_worker_process": len(unique),
        "unique_cold_base_raw_bytes": sum(unique.values()),
        "base_bytes_if_each_dependency_reads_full_page": sum(base["raw_bytes"] for base in dependencies),
    }


def _summarize_policy(targets, policy):
    plans = [row["plans"][policy] for row in targets]
    required = set()
    closed, promotions = [], []
    for row, plan in reversed(list(zip(targets, plans, strict=True))):
        if row["store_index"] in required and plan["representation"] != "raw":
            promotions.append({"store_index": row["store_index"],
                               "additional_raw_bytes": row["raw_bytes"] - plan["encoded_bytes"]})
            plan = {"representation": "raw", "encoded_bytes": row["raw_bytes"], "metadata_bytes": 0,
                    "dependencies": []}
        closed.append(plan)
        required.update(base["store_index"] for base in plan["dependencies"])
    return {
        **_summarize(targets, plans),
        "closed_depth_one": _summarize(targets, closed),
        "raw_base_promotions": sorted(promotions, key=lambda row: row["store_index"]),
    }


def compare(cohort):
    report = json.loads((cohort / REPORT).read_text(encoding="utf-8"))
    if report["schema_version"] != 1:
        raise ValueError("Unsupported native KV page report version")
    inventory = capture_records(report)
    if not inventory:
        raise ValueError("The natural replay report contains no verified captured KV page payloads")
    natural = {row["request_id"]: row["request_key"] for row in report["turns"]}
    measured = [row for row in report["stores"] if row["status"] == "measured"]
    captured = [row for row in measured if row.get("payload_capture_status") in AVAILABLE
                and row.get("native_transfer_matched") is True
                and natural.get(row.get("request_id")) == row.get("request_key")]
    fingerprints = {}
    pools = defaultdict(dict)
    targets = []
    for store_index, row in enumerate(sorted(captured, key=lambda item: item["ts_ns"])):
        layout = _geometry(row)
        digest = row["tensor_sha256"]
        key = digest, layout
        if key not in fingerprints:
            fingerprints[key] = _fingerprint(load_payload(cohort, digest, inventory[digest]), layout)
        hashes = fingerprints[key]
        pool = pools[row["worker_id"], row["pid"], layout]
        matches = []
        for earlier_digest, earlier in pool.items():
            if earlier["ts_ns"] < row["ts_ns"]:
                matches.append(BaseMatch(earlier, *_match(hashes, fingerprints[earlier_digest, layout], layout)))
        targets.append({
            "store_index": store_index,
            **{field: row[field] for field in ("target_page_id", "request_id", "ts_ns", "worker_id", "pid",
                                                "tensor_sha256", "raw_bytes")},
            "observed_encoding": row["encoding"],
            "observed_encoded_bytes": row["encoded_bytes"],
            "earlier_compatible_base_tensors": len(matches),
            "plans": {name: _detail(plan, matches, layout) for name, plan in _plans(matches, layout).items()},
        })
        pool.setdefault(digest, {**row, "store_index": store_index})
    return {
        "schema_version": 1,
        "status": "modeled",
        "cohort": cohort.name,
        "native_page_report": {
            "status": report["status"],
            "natural_requests": report["natural_requests"],
            "complete_measured_stores": len(measured),
            "compared_stores": len(targets),
            "capture_statuses": dict(Counter(row.get("payload_capture_status", "absent") for row in measured)),
            "observed_storage": report["storage"],
            "observed_encoded_bytes_for_compared_stores": sum(row["observed_encoded_bytes"] for row in targets),
        },
        "assumptions": {
            "rows": "SHA-256 fingerprints of native K/V rows at the same layer, plane, and within-page token position.",
            "candidate_pool": "Earlier captured buffers with the same worker, process, dtype, and geometry. Each distinct raw tensor can be a base once.",
            "availability": "The gross model assumes all earlier raw pages are available. The depth-one pass reserves required bases as raw, working backward through selected plans. It does not reoptimize these choices or model retention. Cold base bytes can overlap stored bytes.",
            "cost": "Raw bytes, or complete replacement rows plus modeled metadata. Layer maps use up to two bases in total and select one base or raw for each layer.",
            "page_header_bytes": PAGE_HEADER_BYTES,
            "base_descriptor_bytes": BASE_DESCRIPTOR_BYTES,
            "layer_selector_bits_per_layer": 2,
            "patch_bitmap": "One bit per K/V row across a whole page or each patched layer, rounded up to a byte.",
            "scope": "Offline model of storage bytes from captured raw tensors. Transport and latency are outside its scope.",
        },
        "comparison": {policy: _summarize_policy(targets, policy) for policy in POLICIES},
        "pages": targets,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path, help="Full Blend cohort with a page report and captured payloads")
    args = parser.parse_args()
    try:
        result = compare(args.cohort)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"KV page comparison stopped. {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
