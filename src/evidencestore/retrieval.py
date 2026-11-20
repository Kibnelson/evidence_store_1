"""Repeatable corpus preparation, Faiss HNSW retrieval, and prompt identity."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import zipfile
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from time import monotonic_ns


DOMAINS = ("clapnq", "cloud", "fiqa", "govt")
PROMPT_FORMATS = ("standard", "legacy-blend", "legacy-blend-openclaw")
LEGACY_BLEND_SEPARATOR = "¶"


def json_line(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object")
                yield row


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_sha256(token_ids):
    encoded = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def chunk_record(namespace, source_doc_id, source_chunk_id, text):
    """Corpus IDs survive HNSW rebuilds and changes to vector insertion order."""
    doc_hash = sha256_text(source_doc_id)
    chunk_hash = sha256_text(json_line([source_doc_id, source_chunk_id]))
    return {
        "doc_id": f"{namespace}:doc:{doc_hash}",
        "chunk_id": f"{namespace}:chunk:{chunk_hash}",
        "chunk_version": sha256_text(text),
        "source_doc_id": source_doc_id,
        "source_chunk_id": source_chunk_id,
        "text": text,
    }


def _write_jsonl(path, rows):
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_line(row))
            count += 1
    return count


def _prepare_labeled_dev(dataset, source, output):
    corpus, queries = {}, []
    unmapped_supporting_facts = []
    examples = read_jsonl(source) if dataset == "musique" else json.loads(source.read_text(encoding="utf-8"))
    for example in examples:
        support = []
        if dataset == "musique":
            for passage in example["paragraphs"]:
                title, text = passage["title"], passage["paragraph_text"]
                chunk = chunk_record(dataset, title, sha256_text(text), text)
                corpus.setdefault(chunk["chunk_id"], chunk)
                if passage["is_supporting"]:
                    support.append(chunk["chunk_id"])
            request_key = example["id"]
            aliases = example.get("answer_aliases", [])
        else:
            request_key = example["_id"]
            supporting_titles = {title for title, _ in example["supporting_facts"]}
            sentence_counts = {}
            for title, sentences in example["context"]:
                text = "".join(sentences)
                chunk = chunk_record(dataset, title, sha256_text(json_line(sentences)), text)
                offsets = [0]
                for sentence in sentences:
                    offsets.append(offsets[-1] + len(sentence))
                chunk["source_sentence_offsets"] = offsets
                corpus.setdefault(chunk["chunk_id"], chunk)
                if title in supporting_titles:
                    # Supporting paragraph identity survives an invalid sentence index.
                    support.append(chunk["chunk_id"])
                    sentence_counts.setdefault(title, set()).add(len(sentences))
            for title, sentence_index in example["supporting_facts"]:
                counts = sorted(sentence_counts.get(title, ()))
                if not any(0 <= sentence_index < count for count in counts):
                    unmapped_supporting_facts.append({
                        "request_key": f"{dataset}:{request_key}",
                        "title": title,
                        "sentence_index": sentence_index,
                        "context_sentence_counts": counts,
                        "reason": "sentence_index_out_of_range" if counts else "title_missing",
                    })
            aliases = []
        queries.append({
            "dataset": dataset,
            "request_key": f"{dataset}:{request_key}",
            "question": example["question"],
            "answers": [example["answer"], *aliases],
            "supporting_chunk_ids": sorted(set(support)),
        })
    if not corpus or not queries:
        raise ValueError(f"The labeled {dataset} input is empty")
    if len({query["request_key"] for query in queries}) != len(queries):
        raise ValueError(f"Duplicate {dataset} question IDs")
    manifest = {
        "corpus_rows": _write_jsonl(output / "corpus.jsonl", corpus.values()),
        "query_rows": _write_jsonl(output / "queries.jsonl", queries),
        "multi_turn_conversations": 0,
        "source_sha256": sha256_file(source),
        "corpus_scope": "union of all contexts in supplied labeled split, including distractors",
    }
    if dataset == "2wiki":
        manifest.update({
            "unmapped_supporting_facts": unmapped_supporting_facts,
            "unmapped_supporting_fact_count": len(unmapped_supporting_facts),
            "queries_with_unmapped_supporting_facts": len({
                fact["request_key"] for fact in unmapped_supporting_facts
            }),
        })
    return manifest


def _mtrag_corpus(archive_path, domain):
    suffix = r"_\d+-\d+-\d+-\d+$" if domain == "clapnq" else r"-\d+-\d+$"
    with zipfile.ZipFile(archive_path) as archive, archive.open(f"{domain}.jsonl") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            passage_id = row["_id"]
            document_id = re.sub(suffix, "", passage_id)
            yield chunk_record(f"mtrag:{domain}", document_id, passage_id, row["text"])


def _prepare_mtrag(source, output):
    tasks_path = source / "mtrag-human/generation_tasks/reference.jsonl"
    retrieval = source / "mtrag-human/retrieval_tasks"
    tasks = {domain: [] for domain in DOMAINS}
    for task in read_jsonl(tasks_path):
        collection = task.get("Collection") or task.get("collection") or ""
        match = re.match(r"^mt-rag-([a-z]+)-", collection.lower())
        domain = match[1].removeprefix("ibm") if match else ""
        if domain not in tasks:
            raise ValueError(f"Unknown MT-RAG collection: {collection!r}")
        tasks[domain].append(task)
    manifest = {"reference_tasks_sha256": sha256_file(tasks_path), "domains": {}}
    for domain in DOMAINS:
        domain_dir = output / domain
        domain_dir.mkdir()
        archive = source / "corpora/passage_level" / f"{domain}.jsonl.zip"
        rewrites_path = retrieval / domain / f"{domain}_rewrite.jsonl"
        rewrites = {row["_id"]: row["text"].removeprefix("|user|:").strip() for row in read_jsonl(rewrites_path)}
        queries = []
        for task in tasks[domain]:
            prior, current = task["input"][:-1], task["input"][-1]
            question = current["text"]
            queries.append({
                "dataset": "mtrag",
                "domain": domain,
                "request_key": f"mtrag:{task['task_id']}",
                "conversation_id": task["conversation_id"],
                "turn": int(task["turn"]),
                "question": question,
                "retrieval_query": rewrites.get(task["task_id"]) or question,
                "reference_history": [{"speaker": entry["speaker"], "text": entry["text"]} for entry in prior],
                "answers": [target["text"] for target in task["targets"]],
                "answerability": task.get("Answerability", []),
                "reference_passage_ids": [context["document_id"] for context in task["contexts"]],
            })
        queries.sort(key=lambda row: (row["conversation_id"], row["turn"]))
        if not queries or len({row["request_key"] for row in queries}) != len(queries):
            raise ValueError(f"Empty or duplicate MT-RAG tasks for {domain}")
        manifest["domains"][domain] = {
            "corpus_rows": _write_jsonl(domain_dir / "corpus.jsonl", _mtrag_corpus(archive, domain)),
            "query_rows": _write_jsonl(domain_dir / "queries.jsonl", queries),
            "multi_turn_conversations": len({row["conversation_id"] for row in queries}),
            "archive_sha256": sha256_file(archive),
            "rewrites_sha256": sha256_file(rewrites_path),
        }
    manifest["query_rows"] = sum(domain["query_rows"] for domain in manifest["domains"].values())
    manifest["multi_turn_conversations"] = len({
        task["conversation_id"] for group in tasks.values() for task in group
    })
    return manifest


def prepare_data(dataset, source, output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Dataset output directory already exists: {output}")
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "prepared"
        staged.mkdir()
        if dataset == "mtrag":
            manifest = _prepare_mtrag(Path(source), staged)
        else:
            manifest = _prepare_labeled_dev(dataset, Path(source), staged)
        manifest["dataset"] = dataset
        (staged / "manifest.json").write_text(json_line(manifest), encoding="utf-8")
        staged.rename(output)
    return manifest


def prepare_smoke_sample(queries_path, corpus_path, output):
    """Select two complete MT-RAG conversations and two isolated opening turns."""
    groups = {}
    for row in read_jsonl(queries_path):
        if row.get("dataset") != "mtrag" or not row.get("conversation_id"):
            raise ValueError("Smoke selection requires normalized MT-RAG queries")
        groups.setdefault(row["conversation_id"], []).append(row)

    ranked = sorted(groups, key=lambda group: (len(groups[group]), group))
    if len(ranked) < 4:
        raise ValueError("Smoke selection requires at least four conversations")
    complete = [sorted(groups[group], key=lambda row: row["turn"]) for group in ranked[:2]]
    isolated = [min(groups[group], key=lambda row: row["turn"]) for group in ranked[2:4]]
    if any(len(group) < 2 or [row["turn"] for row in group] != list(range(1, len(group) + 1))
           for group in complete) or any(row["turn"] != 1 for row in isolated):
        raise ValueError("Smoke selection requires complete, consecutively numbered conversations")
    selected = isolated + [row for group in complete for row in group]
    if len({row["request_key"] for row in selected}) != len(selected):
        raise ValueError("Smoke queries must have unique request keys")

    references = {passage for row in selected for passage in row["reference_passage_ids"]}
    found = set()
    passages = {}
    for index, row in enumerate(read_jsonl(corpus_path)):
        source_id = row["source_chunk_id"]
        if index < 64 or source_id in references:
            passages.setdefault(row["chunk_id"], row)
            if source_id in references:
                found.add(source_id)
        if index >= 63 and found == references:
            break
    if found != references:
        raise ValueError(f"Smoke corpus is missing reference passages: {sorted(references - found)}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Smoke sample directory already exists: {output}")
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "sample"
        staged.mkdir()
        _write_jsonl(staged / "queries.jsonl", selected)
        _write_jsonl(staged / "corpus.jsonl", passages.values())
        manifest = {
            "requests": len(selected),
            "isolated_requests": [row["request_key"] for row in isolated],
            "conversations": [{"id": group[0]["conversation_id"],
                               "request_keys": [row["request_key"] for row in group]}
                              for group in complete],
            "corpus_rows": len(passages),
            "reference_passages": len(references),
            "queries_sha256": sha256_file(staged / "queries.jsonl"),
            "corpus_sha256": sha256_file(staged / "corpus.jsonl"),
        }
        (staged / "manifest.json").write_text(json_line(manifest), encoding="utf-8")
        staged.rename(output)
    return manifest


class E5Embedder:
    """Use the pinned E5 retrieval prefixes, masked mean pooling and unit norm."""

    def __init__(self, model_name, revision, device):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
        self.device = device

    def encode(self, texts, prefix):
        batch = self.tokenizer(
            [f"{prefix}: {text}" for text in texts],
            max_length=512, padding=True, truncation=True, return_tensors="pt",
        )
        batch = {name: value.to(self.device) for name, value in batch.items()}
        with self.torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            unit_vectors = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
        return unit_vectors.cpu().numpy()


def build_hnsw(corpus, index_dir, *, embedding_model, embedding_revision, embedding_device,
               embedding_batch_size, hnsw_m, ef_construction, ef_search):
    import faiss

    corpus, index_dir = Path(corpus), Path(index_dir).resolve()
    if index_dir.exists():
        raise FileExistsError(f"Index directory already exists: {index_dir}")
    if min(embedding_batch_size, hnsw_m, ef_construction, ef_search) <= 0:
        raise ValueError("Embedding batch size and HNSW settings must be positive")
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    embedder = E5Embedder(embedding_model, embedding_revision, embedding_device)
    with tempfile.TemporaryDirectory(dir=index_dir.parent) as directory:
        staged = Path(directory) / "index"
        staged.mkdir()
        seen = set()
        index = None
        with (staged / "chunks.jsonl").open("x", encoding="utf-8") as output:
            source = iter(read_jsonl(corpus))
            while batch := list(islice(source, embedding_batch_size)):
                for chunk in batch:
                    chunk_id = chunk["chunk_id"]
                    if chunk_id in seen or sha256_text(chunk["text"]) != chunk["chunk_version"]:
                        raise ValueError(f"Duplicate or mismatched chunk: {chunk_id}")
                    seen.add(chunk_id)
                vectors = embedder.encode([chunk["text"] for chunk in batch], "passage")
                if index is None:
                    index = faiss.IndexHNSWFlat(vectors.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
                    index.hnsw.efConstruction = ef_construction
                    index.hnsw.efSearch = ef_search
                index.add(vectors)
                output.writelines(json_line(chunk) for chunk in batch)
        if index is None:
            raise ValueError(f"Empty corpus: {corpus}")
        faiss.write_index(index, str(staged / "index.faiss"))
        manifest = {
            "rows": len(seen),
            "dimension": index.d,
            "metric": "inner_product_unit_norm",
            "embedding_model": embedding_model,
            "embedding_revision": embedding_revision,
            "hnsw_m": hnsw_m,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "faiss_version": faiss.__version__,
            "corpus_sha256": sha256_file(corpus),
            "chunks_sha256": sha256_file(staged / "chunks.jsonl"),
            "index_sha256": sha256_file(staged / "index.faiss"),
        }
        (staged / "manifest.json").write_text(json_line(manifest), encoding="utf-8")
        staged.rename(index_dir)
    return manifest


class HnswRetriever:
    """Translate stock Faiss row labels to stable passage metadata."""

    def __init__(self, index_dir, embedding_device):
        import faiss

        index_dir = Path(index_dir)
        self.manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        self.chunks_path = index_dir / "chunks.jsonl"
        index_path = index_dir / "index.faiss"
        if (
            sha256_file(self.chunks_path) != self.manifest["chunks_sha256"]
            or sha256_file(index_path) != self.manifest["index_sha256"]
        ):
            raise ValueError("Faiss file or row metadata does not match the index manifest")
        self.index = faiss.read_index(str(index_path))
        self.offsets = []
        with self.chunks_path.open("rb") as handle:
            offset = 0
            for line in handle:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        if self.index.ntotal != len(self.offsets) or self.index.ntotal != self.manifest["rows"]:
            raise ValueError("Faiss rows and corpus chunks differ")
        self.embedder = E5Embedder(
            self.manifest["embedding_model"], self.manifest["embedding_revision"], embedding_device,
        )

    def embed(self, query):
        return self.embedder.encode([query], "query")

    def search(self, vector, top_k):
        return self.index.search(vector, top_k)

    def materialize(self, scores, labels, *, chunk_stages=None):
        hits = []
        reads = [] if chunk_stages is not None else None
        metadata_bytes = 0
        with self.chunks_path.open("rb") as handle:
            for label, score in zip(labels[0], scores[0]):
                if label >= 0:
                    offset = self.offsets[label]
                    read_start = monotonic_ns() if reads is not None else None
                    handle.seek(offset)
                    encoded = handle.readline()
                    read_end = monotonic_ns() if reads is not None else None
                    metadata_bytes += len(encoded)
                    chunk = json.loads(encoded)
                    chunk["score"] = float(score)
                    hits.append(chunk)
                    if reads is not None:
                        reads.append((chunk, read_start, read_end, int(label), offset, len(encoded)))
        hits.sort(key=lambda chunk: (-chunk["score"], chunk["chunk_id"]))
        for rank, chunk in enumerate(hits, 1):
            chunk["rank"] = rank
        if reads is not None:
            for chunk, start_ns, end_ns, label, offset, row_bytes in reads:
                _append_stage(chunk_stages, "corpus_row_read", start_ns, end_ns,
                              **_chunk_stage_identity(chunk), row_bytes=row_bytes,
                              row_label=label, byte_offset=offset)
        return hits, metadata_bytes


@dataclass(frozen=True)
class RenderedPrompt:
    token_ids: list[int]
    evidence: list[dict]
    dropped_chunk_ids: list[str]
    blend_separator_token_spans: list[tuple[int, int]] | None = None


def _append_stage(stages, stage, start_ns, end_ns, **fields):
    stages.append({"stage": stage, "start_ns": start_ns, "end_ns": end_ns, **fields})


def _chunk_stage_identity(chunk):
    return {"doc_id": chunk["doc_id"], "chunk_id": chunk["chunk_id"],
            "chunk_version": chunk["chunk_version"], "rank": chunk.get("rank")}


def _render_parts(question, history, hits, *, prompt_format, evidence_order, chunk_stages=None):
    start_ns = monotonic_ns() if chunk_stages is not None else None
    ordered_hits = sorted(hits, key=evidence_order) if evidence_order is not None else hits
    blend = prompt_format != "standard"
    openclaw = prompt_format == "legacy-blend-openclaw"
    instruction = "Answer the question using the evidence below."
    if openclaw:
        parts = [instruction, LEGACY_BLEND_SEPARATOR, "\n\nConversation:\n"]
    else:
        parts = [f"{instruction}\n"]
    if prompt_format == "legacy-blend":
        parts.append("Evidence:")
    else:
        parts.extend(f"{turn['speaker']}: {turn['text']}\n" for turn in history)
        parts.append(f"Question: {question}\nEvidence:" if openclaw else f"Question: {question}\nEvidence:\n")
    cursor = sum(map(len, parts))
    if chunk_stages is not None:
        _append_stage(chunk_stages, "prompt_template_prefix", start_ns, monotonic_ns(),
                      char_start=0, char_end=cursor)
    spans = []
    for hit in ordered_hits:
        start_ns = monotonic_ns() if chunk_stages is not None else None
        text = hit["text"]
        if blend:
            parts.append(LEGACY_BLEND_SEPARATOR)
            cursor += len(LEGACY_BLEND_SEPARATOR)
            # Source whitespace can merge into the next separator token.
            text = f"[Doc_{hit['chunk_id']}] {text}\n[/Doc]"
        spans.append((cursor, cursor + len(text)))
        parts.append(text)
        cursor += len(text)
        if not blend:
            parts.append("\n\n")
            cursor += 2
        if chunk_stages is not None:
            _append_stage(chunk_stages, "prompt_chunk_append", start_ns, monotonic_ns(),
                          **_chunk_stage_identity(hit), char_start=spans[-1][0],
                          char_end=spans[-1][1], retained=False)
    start_ns = monotonic_ns() if chunk_stages is not None else None
    suffix = "Answer:"
    if openclaw:
        suffix = f"{LEGACY_BLEND_SEPARATOR}{suffix}"
    elif blend:
        history_text = "".join(f"{turn['speaker']}: {turn['text']}\n" for turn in history)
        suffix = f"{LEGACY_BLEND_SEPARATOR}\n\nConversation:\n{history_text}Question: {question}\nAnswer:"
    parts.append(suffix)
    if chunk_stages is not None:
        _append_stage(chunk_stages, "prompt_template_suffix", start_ns, monotonic_ns(),
                      char_start=cursor, char_end=cursor + len(parts[-1]))
        start_ns = monotonic_ns()
    text = "".join(parts)
    if chunk_stages is not None:
        _append_stage(chunk_stages, "prompt_join", start_ns, monotonic_ns(), prompt_char_count=len(text))
    return text, spans, ordered_hits


def _legacy_blend_separator_spans(text, encoded, delimiter_ids, expected_count):
    if not delimiter_ids:
        raise ValueError("The legacy Blend separator encodes to an empty token sequence")
    positions = [offset for offset, character in enumerate(text) if character == LEGACY_BLEND_SEPARATOR]
    if len(positions) != expected_count:
        raise ValueError("Legacy Blend input contains the reserved separator ¶")
    tokens = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(tokens):
        raise ValueError("Legacy Blend tokenizer returned a different number of tokens and offsets")
    width = len(delimiter_ids)
    spans = [(start, start + width) for start in range(len(tokens) - width + 1)
             if tokens[start:start + width] == delimiter_ids]
    if len(spans) != len(positions) or any(
        any(tuple(offset) != (position, position + 1) for offset in offsets[start:end])
        for position, (start, end) in zip(positions, spans)
    ):
        raise ValueError("Legacy Blend separator tokens do not match their whole-prompt character spans")
    return spans


def render_prompt(
    tokenizer, question, history, hits, max_prompt_tokens, *, prompt_format="standard", chunk_stages=None,
    evidence_order=None,
):
    """Map evidence against the final tokenization while trimming in retrieval order."""
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for exact prompt-token offsets")
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(f"Unsupported prompt format {prompt_format!r}")
    blend = prompt_format != "standard"
    leading_segments = 2 if prompt_format == "legacy-blend-openclaw" else 1
    kept = list(hits)
    dropped = []
    attempt = 0
    while True:
        attempt_stages = [] if chunk_stages is not None else None
        text, spans, ordered_hits = _render_parts(
            question, history, kept, prompt_format=prompt_format, evidence_order=evidence_order,
            chunk_stages=attempt_stages,
        )
        tokenize_start = monotonic_ns() if attempt_stages is not None else None
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=not blend, truncation=False)
        tokenize_end = monotonic_ns() if attempt_stages is not None else None
        token_count = len(encoded["input_ids"])
        fits = token_count <= max_prompt_tokens
        if attempt_stages is not None:
            attempt += 1
            _append_stage(attempt_stages, "prompt_tokenize", tokenize_start, tokenize_end,
                          token_count=token_count, max_prompt_tokens=max_prompt_tokens,
                          dropped_chunk_id=kept[-1]["chunk_id"] if kept and not fits else None)
            for stage in attempt_stages:
                stage.update(attempt=attempt, accepted_attempt=fits)
            chunk_stages.extend(attempt_stages)
        if fits:
            break
        if not kept:
            raise ValueError(f"Question and reference history exceed {max_prompt_tokens} tokens")
        dropped.append(kept.pop()["chunk_id"])
    separator_spans = None
    if blend:
        delimiter_ids = list(tokenizer.encode(LEGACY_BLEND_SEPARATOR, add_special_tokens=False))
        separator_spans = _legacy_blend_separator_spans(text, encoded, delimiter_ids, len(kept) + leading_segments)
    evidence = []
    append_stages = ([stage for stage in attempt_stages if stage["stage"] == "prompt_chunk_append"]
                     if attempt_stages is not None else None)
    for index, (hit, (char_start, char_end)) in enumerate(zip(ordered_hits, spans, strict=True)):
        map_start = monotonic_ns() if append_stages is not None else None
        indices = [index for index, (start, end) in enumerate(encoded["offset_mapping"])
                   if start < char_end and end > char_start and end > start]
        if separator_spans is not None:
            preceding_separator = index + leading_segments - 1
            segment_start = separator_spans[preceding_separator][1]
            segment_end = separator_spans[preceding_separator + 1][0]
            if not indices or (indices[0], indices[-1] + 1) != (segment_start, segment_end):
                raise ValueError(f"Legacy Blend evidence does not cover its token segment for {hit['chunk_id']}")
        if append_stages is not None:
            map_end = monotonic_ns()
            append_stages[index]["retained"] = bool(indices)
            _append_stage(chunk_stages, "prompt_chunk_offset_map", map_start, map_end,
                          **_chunk_stage_identity(hit), attempt=attempt, accepted_attempt=True,
                          retained=bool(indices), char_start=char_start, char_end=char_end,
                          start_token=indices[0] if indices else None, end_token=indices[-1] + 1 if indices else None)
        if not indices:
            dropped.append(hit["chunk_id"])
            continue
        row = {field: hit[field] for field in ("doc_id", "chunk_id", "chunk_version")}
        row.update(start_token=indices[0], end_token=indices[-1] + 1)
        for field in ("rank", "score"):
            if field in hit:
                row[field] = hit[field]
        evidence.append(row)
    return RenderedPrompt(encoded["input_ids"], evidence, list(reversed(dropped)), separator_spans)
