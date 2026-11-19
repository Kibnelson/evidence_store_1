#!/usr/bin/env python3
"""Prepare pinned MuSiQue inputs and verify the controlled smoke."""

import argparse
import json
from pathlib import Path
import time

from vllm_pd_replay import fixture_rows, load_tokenizer
from vllm_pd_stage_inputs import SOURCES

from evidencestore.retrieval import json_line, read_jsonl, sha256_file


# These questions retrieved the same versioned chunk at five prompt offsets
# in the archived full-corpus run.
MUSIQUE_KEYS = (
    "musique:3hop1__46571_87694_64412",
    "musique:3hop1__64957_87694_64412",
    "musique:3hop1__135793_87694_64412",
    "musique:3hop1__136016_87694_64412",
    "musique:3hop1__603558_87694_64412",
    "musique:3hop1__19768_19788_19761",
)
LMCACHE_CHUNK_SIZE = 256
MUSIQUE_DEV_REQUESTS = 2417


def musique_queries(prepared):
    manifest = json.loads((prepared / "manifest.json").read_text())
    if (
        manifest["dataset"] != "musique"
        or manifest["source_sha256"] != SOURCES["musique"][2]
        or manifest["query_rows"] != MUSIQUE_DEV_REQUESTS
    ):
        raise ValueError("Expected data prepared from the pinned full MuSiQue-Ans development split")
    return list(read_jsonl(prepared / "queries.jsonl")), manifest


def musique_sample(queries):
    rows = {row["request_key"]: row for row in queries if row["request_key"] in MUSIQUE_KEYS}
    missing = set(MUSIQUE_KEYS) - rows.keys()
    if missing:
        raise ValueError(f"Prepared MuSiQue queries are missing smoke rows: {sorted(missing)}")
    questions = [rows[key] for key in MUSIQUE_KEYS]
    first = {**questions[0], "pd_route": {"prefill": "p0", "decode": "d0"}}
    exact = {**first, "request_key": first["request_key"] + ":smoke-repeat"}
    return [first, exact, *questions[1:]]


def chunk_probes(tokenizer):
    route = {"prefill": "p0", "decode": "d0"}
    fixture = {
        row["case"]: row for row in fixture_rows(tokenizer, "2p2d", 8192, evidence_repeats=12)
        if row["pd_route"] == route
    }
    cold = fixture["cold"]
    moved = fixture["reordered"]
    if (
        len(cold["prompt_token_ids"]) <= 2 * LMCACHE_CHUNK_SIZE
        or cold["prompt_token_ids"] == moved["prompt_token_ids"]
        or cold["prompt_token_ids"][:LMCACHE_CHUNK_SIZE] == moved["prompt_token_ids"][:LMCACHE_CHUNK_SIZE]
    ):
        raise ValueError("Expected two cacheable prefix chunks and a reordered prompt that diverges in the first")
    return [
        {
            **row,
            "request_key": f"chunk-probes:{case}",
            "conversation_id": "chunk-probes",
            "case": case,
            "pd_route": {"prefill": prefill, "decode": "d0"},
        }
        for case, prefill, row in (
            ("cold", "p0", cold),
            ("warm_exact", "p0", cold),
            ("cross_prefill_exact", "p1", cold),
            ("reordered", "p1", moved),
        )
    ]


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as output:
        output.writelines(json_line(row) for row in rows)


def prepare_inputs(args):
    queries, data = musique_queries(args.prepared)
    if args.full:
        if (len(queries) != MUSIQUE_DEV_REQUESTS
                or len({row["request_key"] for row in queries}) != MUSIQUE_DEV_REQUESTS
                or any(row["dataset"] != "musique" or row.get("pd_route") for row in queries)):
            raise ValueError("Full MuSiQue expects 2,417 distinct questions with normal gateway routing")
        questions, probes = queries, []
    else:
        questions = musique_sample(queries)
        probes = chunk_probes(load_tokenizer(args.model, args.revision))
    corpus_hash = sha256_file(args.prepared / "corpus.jsonl")
    if args.retained_index:
        index = json.loads((args.retained_index / "manifest.json").read_text())
        if (
            index["rows"] != data["corpus_rows"]
            or index["corpus_sha256"] != corpus_hash
            or (index["embedding_model"], index["embedding_revision"])
            != (args.embedding_model, args.embedding_revision)
        ):
            raise ValueError("Retained HNSW index differs from the prepared corpus or pinned embedder")
    args.output.mkdir()
    write_jsonl(args.output / "queries.jsonl", questions)
    if probes:
        write_jsonl(args.output / "probes.jsonl", probes)
    (args.output / "manifest.json").write_text(json_line({
        "dataset": "musique",
        "workload": "full" if args.full else "smoke",
        "source_sha256": data["source_sha256"],
        "corpus_sha256": corpus_hash,
        "natural_queries": [row["request_key"] for row in questions],
        "probe_queries": [row["request_key"] for row in probes],
        "queries_sha256": sha256_file(args.output / "queries.jsonl"),
        **({"probes_sha256": sha256_file(args.output / "probes.jsonl")} if probes else {}),
    }), encoding="utf-8")


def lmcache_transfer_state(run_dir):
    submitted, completed = set(), set()
    for path in (run_dir / "trace").glob("lmcache-*.jsonl"):
        with path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                row = json.loads(line)
                event = row["event"]
                if event in {"lmcache_transfer_submit", "lmcache_transfer_complete"}:
                    identity = (row["pid"], row["transfer_key"])
                    (submitted if event == "lmcache_transfer_submit" else completed).add(identity)
    return submitted, completed


def drain_lmcache(args):
    deadline = time.monotonic() + 20
    previous = None
    while True:
        state = lmcache_transfer_state(args.run_dir)
        now = time.monotonic()
        if state != previous:
            previous, last_change = state, now
        submitted, completed = state
        # Wait for matching source completions and for the final enqueue to settle.
        if submitted and submitted == completed and now - last_change >= 0.5:
            print(f"LMCache drained {len(completed)} traced transfers before the counter snapshot")
            return
        if now >= deadline:
            break
        time.sleep(0.1)
    raise ValueError(f"LMCache transfer trace did not drain ({len(submitted)} submitted, {len(completed)} completed)")


def verify_pair(args):
    identities = []
    expected = json.loads((args.smoke_root / "inputs" / "manifest.json").read_text())["natural_queries"]
    if len(expected) != len(MUSIQUE_KEYS) + 1:
        raise ValueError("The MuSiQue smoke manifest must list six questions plus the seed repeat")
    fields = ("doc_id", "chunk_id", "chunk_version", "start_token", "end_token", "rank")
    for mode in ("native", "shared"):
        frozen = args.smoke_root / f"2p2d-{mode}" / "client" / "musique" / "frozen.jsonl"
        rows = list(read_jsonl(frozen))
        if [row["request_key"] for row in rows] != expected:
            raise ValueError(f"{mode} MuSiQue receipts differ from the selected request keys")
        prompts = [
            (row["prompt_sha256"], tuple(tuple(chunk[field] for field in fields) for chunk in row["evidence"]))
            for row in rows
        ]
        if not prompts[0][1] or prompts[0] != prompts[1]:
            raise ValueError(f"{mode} MuSiQue seed did not repeat the same prompt and evidence")
        identities.append(prompts)
    for request_key, native, shared in zip(expected, *identities, strict=True):
        if native != shared:
            raise ValueError(f"Native and shared prompt or chunk positions differ for {request_key}")
    print(f"Verified identical prompts and evidence spans for {len(expected)} live MuSiQue request pairs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--prepared", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--revision", required=True)
    prepare.add_argument("--retained-index", type=Path)
    prepare.add_argument("--embedding-model", required=True)
    prepare.add_argument("--embedding-revision", required=True)
    prepare.add_argument("--full", action="store_true", help="stage all questions once using normal routing")
    drain = subparsers.add_parser("drain")
    drain.add_argument("--run-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify-pair")
    verify.add_argument("--smoke-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        {"prepare": prepare_inputs, "drain": drain_lmcache, "verify-pair": verify_pair}[args.command](args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"MuSiQue smoke {args.command} failed: {error}\n")


if __name__ == "__main__":
    main()
