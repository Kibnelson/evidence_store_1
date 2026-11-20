"""Verify and load the repeat-heavy workload selections for Full Blend."""

from collections import defaultdict
from itertools import groupby
import json

from vllm_pd_replay import read_jsonl, sha256_file


SMOKE = "musique-smoke"
WORKLOADS = (SMOKE, "musique-single-turn-40", "mtrag-multi-turn-repeat")
DATASETS = {"musique-single-turn-40": "musique", "mtrag-multi-turn-repeat": "mtrag"}


def complete_conversations(sample, available):
    full = defaultdict(list)
    for row in available:
        full[row["conversation_id"]].append(row)
    counts = {}
    for conversation_id, group in groupby(sample, key=lambda row: row["conversation_id"]):
        turns = list(group)
        if (not conversation_id or conversation_id in counts or len(turns) < 2
                or turns != sorted(full[conversation_id], key=lambda row: row["turn"])):
            raise ValueError(f"Expected one complete turn-ordered block for MT-RAG conversation {conversation_id}")
        counts[conversation_id] = len(turns)
    return counts


def load_selected(path, workload, prepared, index_dir):
    selection = json.loads(path.read_text(encoding="utf-8"))
    dataset, domain = selection["dataset"], selection["domain"]
    keys = selection["request_keys"]
    if (selection["schema_version"] != 1 or selection["workload"] != workload
            or DATASETS[workload] != dataset or len(keys) != 40 or len(set(keys)) != len(keys)
            or (dataset == "mtrag" and not domain) or (dataset == "musique" and domain is not None)):
        raise ValueError(f"Invalid 40-request selection in {path}")

    queries = prepared / "queries.jsonl"
    available = {row["request_key"]: row for row in read_jsonl(queries)}
    sample = [available[key] for key in keys]
    if any(row["dataset"] != dataset or row.get("domain") != domain or row.get("pd_route") for row in sample):
        raise ValueError(f"Selected queries do not match the dataset or default gateway route for {workload}")
    if dataset == "musique" and (any(row.get("conversation_id") for row in sample)
                                 or len({row["question"] for row in sample}) != len(sample)):
        raise ValueError("Expected 40 distinct single-turn MuSiQue questions")
    conversations = complete_conversations(sample, available.values()) if dataset == "mtrag" else {}

    index = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    provenance = selection["provenance"]
    corpus_sha256 = sha256_file(prepared / "corpus.jsonl")
    if (index["corpus_sha256"] != corpus_sha256 or corpus_sha256 != provenance["corpus_sha256"]
            or index["embedding_model"] != "intfloat/e5-base-v2"
            or index["embedding_revision"] != "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
            or not all((index_dir / name).is_file() for name in ("index.faiss", "chunks.jsonl"))):
        raise ValueError(f"Expected the full matching corpus and HNSW index for {workload}")
    queries_sha256 = provenance.get("source_queries_sha256") or provenance.get("normalized_queries_sha256")
    if sha256_file(queries) != queries_sha256:
        raise ValueError(f"Full prepared query SHA-256 differs from the selection source for {workload}")
    return sample, {
        "dataset": dataset, "domain": domain, "workload": workload, "query_keys": keys,
        "prompt_format": "legacy-blend", "conversation_turn_counts": conversations,
        "selection_sha256": sha256_file(path), "selection_provenance": provenance,
        "selection_metrics": selection["selection"], "corpus_sha256": corpus_sha256,
        "index_sha256": index["index_sha256"],
    }
