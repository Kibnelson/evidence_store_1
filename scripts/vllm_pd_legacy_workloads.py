"""Verify and load the repeat-heavy workload selections for Full Blend."""

from collections import defaultdict
from itertools import groupby
import json
import re

from vllm_pd_mtrag_scale_select import DOMAINS, REFERENCE_COHORT, REFERENCE_RUN
from vllm_pd_replay import read_jsonl, sha256_file


SMOKE = "musique-smoke"
WORKLOADS = (SMOKE, "musique-single-turn-40", "mtrag-multi-turn-repeat")
DATASETS = {"musique-single-turn-40": "musique", "mtrag-multi-turn-repeat": "mtrag"}
PAGE_SMOKE_CONVERSATION = "1534a095279f2cb888fb0bea17bd70da"
PAGE_SMOKE_TURNS = (1, 2, 3, 4)


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


def load_scale_conversation(path, prepared, index_dir, *, domain, conversation_id, turn_window):
    """Check the frozen full-turn queue and the current domain's prepared inputs."""
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (plan["schema_version"] != 1 or plan["dataset"] != "mtrag" or domain not in DOMAINS
            or plan["native_reference_run"] != REFERENCE_RUN
            or plan["native_reference_cohort"] != REFERENCE_COHORT):
        raise ValueError(f"Invalid MT-RAG scale plan at {path}")
    matches = [row for row in plan["conversations"] if row["conversation_id"] == conversation_id]
    if len(matches) != 1 or conversation_id in plan["regression_conversations"]:
        raise ValueError(f"Expected one held-out conversation {conversation_id} in {path}")
    entry = matches[0]
    if entry["domain"] != domain or entry["turn_window"] != turn_window:
        raise ValueError(f"Use the full turn window and domain in the frozen plan for {conversation_id}")

    queries = prepared / "queries.jsonl"
    reference = entry["native_retrieval_reference"]
    keys = [row["request_key"] for row in reference]
    turns = list(range(1, entry["turns"] + 1))
    sample = [row for row in read_jsonl(queries) if row["conversation_id"] == conversation_id]
    if (len(turns) < 2 or turn_window != f"1:{len(turns)}" or [row["turn"] for row in sample] != turns
            or [row["request_key"] for row in sample] != keys
            or any(row["dataset"] != "mtrag" or row["domain"] != domain or row.get("pd_route") for row in sample)):
        raise ValueError(f"Full query order does not match the frozen plan for {conversation_id}")

    index = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    source = plan["domains"][domain]
    corpus_sha256 = sha256_file(prepared / "corpus.jsonl")
    if (sha256_file(queries) != source["queries_sha256"] or corpus_sha256 != source["corpus_sha256"]
            or index["corpus_sha256"] != corpus_sha256 or index["chunks_sha256"] != corpus_sha256
            or index["embedding_model"] != "intfloat/e5-base-v2"
            or index["embedding_revision"] != "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
            or (index["hnsw_m"], index["ef_construction"], index["ef_search"]) != (32, 200, 200)
            or not all((index_dir / name).is_file() for name in ("index.faiss", "chunks.jsonl"))):
        raise ValueError(f"Expected the full pinned queries, corpus, and HNSW settings for {domain}")
    return sample, {
        "dataset": "mtrag", "domain": domain, "workload": "mtrag-multi-turn-repeat", "query_keys": keys,
        "prompt_format": "legacy-blend", "conversation_turn_counts": {conversation_id: len(turns)},
        "selection_sha256": sha256_file(path), "selection_provenance": source,
        "selection_metrics": {"per_request": reference}, "corpus_sha256": corpus_sha256,
        "index_sha256": index["index_sha256"], "kv_page_trace": {
            "schema_version": 1, "conversation_id": conversation_id, "turns": turns,
        },
        "scale_selection": {
            "native_reference_run": REFERENCE_RUN, "native_reference_cohort": REFERENCE_COHORT,
            "archived_repeated_chunk_occurrences": entry["archived_repeated_chunk_occurrences"],
            "archived_chunk_occurrences": entry["archived_chunk_occurrences"], "stage_end": entry["stage_end"],
            "literal_separator_encoding": r"\u00b6",
        },
    }


def page_smoke(sample, manifest, *, conversation_id=None, turn_window=None):
    """Select one conversation window from the validated MT-RAG workload."""
    if manifest["workload"] != "mtrag-multi-turn-repeat" or manifest["domain"] != "clapnq":
        raise ValueError("KV page tracing requires the selected MT-RAG CLAPNQ workload")
    conversation = PAGE_SMOKE_CONVERSATION if conversation_id is None else conversation_id
    counts = manifest["conversation_turn_counts"]
    if conversation not in counts:
        raise ValueError(f"Unknown selected CLAPNQ conversation {conversation!r}. Choose from {', '.join(counts)}")
    first, last = PAGE_SMOKE_TURNS[0], PAGE_SMOKE_TURNS[-1]
    if turn_window is not None:
        match = re.fullmatch(r"([1-9][0-9]*):([1-9][0-9]*)", turn_window)
        if match is None:
            raise ValueError("--page-turns requires an inclusive FIRST:LAST range of positive turn numbers")
        first, last = map(int, match.groups())
    if not 1 <= first <= last <= counts[conversation]:
        raise ValueError(f"Expected an inclusive page turn window within 1:{counts[conversation]} for {conversation}")
    expected = list(range(first, last + 1))
    turns = [row for row in sample if row["conversation_id"] == conversation and first <= row["turn"] <= last]
    if [row["turn"] for row in turns] != expected:
        raise ValueError(f"Missing ordered turns {first}:{last} for {conversation}")
    keys = [row["request_key"] for row in turns]
    selection = manifest["selection_metrics"]
    references = [row for row in selection["per_request"] if row["request_key"] in keys]
    if [row["request_key"] for row in references] != keys:
        raise ValueError("The KV page turns do not match their selected retrieval references")
    return turns, {
        **manifest,
        "query_keys": keys,
        "conversation_turn_counts": {conversation: len(turns)},
        "selection_metrics": {"per_request": references},
        "kv_page_trace": {"schema_version": 1, "conversation_id": conversation, "turns": expected},
    }
