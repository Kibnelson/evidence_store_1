#!/usr/bin/env python3
"""Freeze held-out MT-RAG conversations using the completed native replay."""

import argparse
from collections import defaultdict
from fractions import Fraction
import gzip
from itertools import zip_longest
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from vllm_pd_replay import sha256_file


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RUN = "native-vllm-30453-20261119T031129Z"
REFERENCE_COHORT = "1p3d-native"
DOMAINS = ("clapnq", "cloud", "fiqa", "govt")
COUNTS = {"clapnq": (29, 224), "cloud": (26, 205), "fiqa": (27, 199), "govt": (28, 214)}
CORPUS_SHA256 = {
    "clapnq": "faf5458bc8dbeb372a024e5ee7f2fee107413e2800975160a75d9089bdc84485",
    "cloud": "2ba213573e51d94e6f25a9bc71f2da99119c839e4fb804252b04ef027a6ecb47",
    "fiqa": "db8eef577eaf789b5238e00dfc51452078a33edda2e07a6cc4780f7da27c2079",
    "govt": "6500fb64683020effb1e23daf3692bab8365450e24e2ef91eea0a8c67a87d64b",
}
REQUEST_KEY = re.compile(r"mtrag:([0-9a-f]{32})<::>([1-9][0-9]*)")
EVIDENCE_FIELDS = ("doc_id", "chunk_id", "chunk_version", "rank")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_requests(root, domain):
    label = f"mtrag-{domain}"
    packet = root / "details" / label
    manifest = _read_json(packet / "archive-manifest.json")
    summary = _read_json(root / label / f"{REFERENCE_COHORT}.json")
    checkpoint = manifest["checkpoints"][REFERENCE_COHORT]
    count, turns = COUNTS[domain]
    if (manifest["campaign"] != REFERENCE_RUN or manifest["dataset"] != f"mtrag/{domain}"
            or summary["dataset"] != manifest["dataset"] or summary["campaign"] != REFERENCE_RUN
            or summary["cohort"] != REFERENCE_COHORT or summary["status"] != "passed"
            or summary["counter_reconciliation"]["status"] != "matched"
            or summary["multi_turn_conversations"] != count or summary["requests"] != turns
            or checkpoint["requests"] != turns):
        raise ValueError(f"Missing passing native replay for {domain}")

    path = packet / REFERENCE_COHORT / "requests.jsonl.gz"
    receipt = checkpoint["files"][path.name]
    digest = sha256_file(path)
    if digest != receipt["sha256"] or path.stat().st_size != receipt["bytes"] or receipt["rows"] != turns:
        raise ValueError(f"Archived requests failed size or checksum verification for {domain}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != turns:
        raise ValueError(f"Unexpected number of archived requests for {domain}")
    return rows, {
        "archive_requests_sha256": digest,
        "queries_sha256": summary["input_sha256"],
        "corpus_sha256": CORPUS_SHA256[domain],
        "conversations": count,
        "turns": turns,
    }


def _conversations(rows, domain):
    grouped = defaultdict(list)
    for row in rows:
        key = REQUEST_KEY.fullmatch(row["request_key"])
        if (key is None or row["dataset"] != "mtrag" or row["conversation_id"] != key[1]
                or row["turn"] != int(key[2])):
            raise ValueError(f"Malformed {domain} conversation request {row['request_key']}")
        chunks = sorted(row["evidence"], key=lambda chunk: chunk["rank"])
        if ([item["rank"] for item in chunks] != list(range(1, 7))
                or len({chunk["chunk_id"] for chunk in chunks}) != len(chunks)
                or any(not chunk["chunk_id"].startswith(f"mtrag:{domain}:chunk:") for chunk in chunks)):
            raise ValueError(f"Expected six distinct ranked chunks for {row['request_key']}")
        grouped[key[1]].append((row["turn"], {
            "request_key": row["request_key"],
            "evidence": [{field: chunk[field] for field in EVIDENCE_FIELDS} for chunk in chunks],
        }))
    if len(grouped) != COUNTS[domain][0]:
        raise ValueError(f"Unexpected number of {domain} conversations")
    conversations = []
    for conversation, items in grouped.items():
        items.sort(key=lambda item: item[0])
        turns = len(items)
        if turns < 2 or [turn for turn, _ in items] != list(range(1, turns + 1)):
            raise ValueError(f"Missing or duplicate turns in {domain} conversation {conversation}")
        seen = set()
        repeats = 0
        for _, request in items:
            identities = {tuple(chunk[field] for field in EVIDENCE_FIELDS[:-1]) for chunk in request["evidence"]}
            repeats += len(seen & identities)
            seen.update(identities)
        conversations.append({
            "domain": domain,
            "conversation_id": conversation,
            "turn_window": f"1:{turns}",
            "turns": turns,
            "archived_repeated_chunk_occurrences": repeats,
            "archived_chunk_occurrences": 6 * turns,
            "native_retrieval_reference": [request for _, request in items],
        })
    return conversations


def _diverse_order(conversations):
    remaining = sorted(conversations, key=lambda row: (row["turns"], row["conversation_id"]))
    ordered = []
    # Span conversation lengths in each wave. Alternate archived repeat-rate
    # percentiles across the length quintiles to cover both dimensions.
    repeat_percentiles = (0, 75, 25, 100, 50)
    for wave in range(3):
        width = len(remaining)
        chosen = []
        for bucket in range(5):
            candidates = sorted(
                remaining[bucket * width // 5:(bucket + 1) * width // 5],
                key=lambda row: (Fraction(row["archived_repeated_chunk_occurrences"], row["archived_chunk_occurrences"]),
                                 row["turns"], row["conversation_id"]),
            )
            percentile = repeat_percentiles[(bucket + wave) % 5]
            index = (percentile * (len(candidates) - 1) + 50) // 100
            chosen.append(candidates[index])
        ordered.extend(chosen)
        ids = {row["conversation_id"] for row in chosen}
        remaining = [row for row in remaining if row["conversation_id"] not in ids]
    return ordered + remaining


def build_plan(reference_root):
    legacy = _read_json(ROOT / "workloads/mtrag-multi-turn-repeat.json")
    excluded = {REQUEST_KEY.fullmatch(key)[1] for key in legacy["request_keys"]}
    if len(excluded) != 6:
        raise ValueError("Expected six original CLAPNQ regression conversations")
    domains, orders = {}, []
    for domain in DOMAINS:
        reference, domains[domain] = _reference_requests(reference_root, domain)
        rows = _conversations(reference, domain)
        if domain == "clapnq" and sum(row["turns"] for row in rows if row["conversation_id"] in excluded) != 40:
            raise ValueError("Original regression conversations do not match archived CLAPNQ turns")
        orders.append(_diverse_order([row for row in rows if row["conversation_id"] not in excluded]))

    queue = [row for group in zip_longest(*orders) for row in group if row is not None]
    if (len(queue) != 104 or sum(row["turns"] for row in queue) != 802
            or len({row["conversation_id"] for row in queue}) != len(queue)):
        raise ValueError("Expected 104 unique held-out conversations covering 802 turns")
    checkpoints = (20, 40, 60, 104)
    for position, row in enumerate(queue, start=1):
        row["stage_end"] = next(stage_end for stage_end in checkpoints if position <= stage_end)
    return {
        "schema_version": 1,
        "dataset": "mtrag",
        "native_reference_run": REFERENCE_RUN,
        "native_reference_cohort": REFERENCE_COHORT,
        "selection_method": "domain round-robin, length quintiles, alternating archived within-conversation repeat percentiles",
        "regression_conversations": sorted(excluded),
        "checkpoints": checkpoints,
        "domains": domains,
        "conversations": queue,
    }


def write_plan(path, plan):
    content = json.dumps(plan, separators=(",", ":"), allow_nan=False) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(content)
        try:
            path.hardlink_to(temporary)
        except FileExistsError:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"A different frozen plan already exists at {path}") from None
    finally:
        temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        write_plan(args.output, build_plan(ROOT / "reports/datasets" / REFERENCE_RUN))
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"MT-RAG plan stopped. {error}\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
