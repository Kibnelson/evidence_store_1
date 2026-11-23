#!/usr/bin/env python3
"""Prepare retrieval data and replay frozen or Faiss HNSW prompts against native PD."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError
from urllib.request import Request, urlopen


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evidencestore.retrieval import (  # noqa: E402
    HnswRetriever,
    LEGACY_BLEND_SEPARATOR,
    PROMPT_FORMATS,
    build_hnsw,
    chunk_record,
    json_line,
    prepare_data,
    prepare_smoke_sample,
    prompt_sha256,
    read_jsonl,
    render_prompt,
    sha256_file,
    sha256_text,
)


GENERATION_KEYS = ("max_tokens", "min_tokens", "ignore_eos", "temperature", "top_p", "seed")


def load_tokenizer(name, revision):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)


def fixture_rows(tokenizer, topology, max_prompt_tokens, evidence_repeats=1):
    facts = (
        "A single observation links the blue notebook to the western station. "
        "The station opened before the bridge. Each surviving record contains a witness account.",
        "The second witness checked the timetable at the eastern station. "
        "That timetable lists both stops. The western station is listed before the eastern station.",
        "A final map puts the bridge midway between both stations. "
        "The publisher identified the notebook and the timetable as separate sources.",
    )
    if evidence_repeats < 1:
        raise ValueError("--evidence-repeats must be positive")
    chunks = [
        chunk_record("fixture", f"witness-{index}", "body", " ".join([text] * evidence_repeats))
        for index, text in enumerate(facts)
    ]
    pairs = [(0, 0), (0, 1), (0, 2)] if topology == "1p3d" else [(0, 0), (0, 1), (1, 0), (1, 1)]
    pair_labels = ("amber", "bravo", "citron", "dragon")
    for label, (prefill, decode) in zip(pair_labels, pairs):
        route = {"prefill": f"p{prefill}", "decode": f"d{decode}"}
        # Change the first content token per route, so one pair's warm-up cannot
        # create a full native prefix on a subsequent pair sharing the same P.
        question = f"{label}: Which of the two stations opened first?"
        for case, hits in (("cold", chunks), ("warm_exact", chunks),
                           ("reordered", [chunks[2], chunks[0], chunks[1]])):
            rendered = render_prompt(tokenizer, question, [], hits, max_prompt_tokens)
            if len(rendered.evidence) != len(chunks) or rendered.dropped_chunk_ids:
                raise ValueError("The fixture prompt budget excludes a controlled evidence chunk")
            yield {
                "request_key": f"fixture:{route['prefill']}:{route['decode']}:{case}",
                "dataset": "fixture",
                "case": case,
                "pd_route": route,
                "conversation_id": f"fixture:{route['prefill']}:{route['decode']}",
                "prompt_token_ids": rendered.token_ids,
                "prompt_sha256": prompt_sha256(rendered.token_ids),
                "evidence": rendered.evidence,
            }


def load_frozen(path):
    if path.suffix == ".json":
        source = json.loads(path.read_text(encoding="utf-8"))
        generation = {key: source["generation"][key] for key in GENERATION_KEYS if key in source["generation"]}
        for phase in ("warmup", "measured"):
            for conv in source["conversations"]:
                turns = [conv["warmup_turn"]] if phase == "warmup" else conv["turns"]
                for turn in turns:
                    tokens = turn["prompt_token_ids"]
                    token_hash = prompt_sha256(tokens)
                    if len(tokens) != turn["prompt_token_count"] or token_hash != turn["token_ids_sha256"]:
                        raise ValueError(f"Prior frozen tokens changed for {turn['request_key']}")
                    yield {
                        "request_key": f"docs8k:{phase}:{turn['request_key']}",
                        "dataset": "docs8k",
                        "phase": phase,
                        "start_offset_us": conv["start_offset_us"],
                        "conversation_id": conv["conversation_id"],
                        "turn": turn["turn_ordinal"],
                        "prompt_token_ids": tokens,
                        "prompt_sha256": token_hash,
                        "source_token_ids_sha256": turn["token_ids_sha256"],
                        "generation": generation,
                        "evidence": [],
                    }
    else:
        yield from read_jsonl(path)


def sse_messages(response):
    event = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if event:
                yield "\n".join(event)
                event.clear()
        elif line.startswith("data:"):
            event.append(line[5:].removeprefix(" "))
    if event:
        yield "\n".join(event)


def stream_completion(gateway, body, timeout, tracer, fields):
    request = Request(
        gateway.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                 "X-Request-Id": fields["request_id"]},
        method="POST",
    )
    received = []
    output_token_ids = []
    all_ids_observed = True
    chunk_count = 0
    usage_tokens = finish_reason = None
    last_token_clock = None
    done = False
    with urlopen(request, timeout=timeout) as response:
        if response.headers.get_content_type() != "text/event-stream":
            raise ValueError(f"Expected SSE from gateway, got {response.headers.get_content_type()}")
        tracer.emit("client_response_headers", **fields, status=response.status)
        for raw in sse_messages(response):
            received_ns = time.monotonic_ns()
            received_wall_ns = time.time_ns()
            if done:
                raise ValueError("Stream sent data after [DONE]")
            if raw == "[DONE]":
                done = True
                # Read through EOF so the gateway can finish the upstream relay.
                continue
            event = json.loads(raw)
            if "error" in event or event.get("object") == "error":
                raise ValueError(f"Streaming error: {event.get('error', event)}")
            received.append({"received_monotonic_ns": received_ns, "event": event})
            for choice in event.get("choices", []):
                ids = choice.get("token_ids") or []
                fragment = choice.get("text", "")
                finish_reason = choice.get("finish_reason") or finish_reason
                if ids or fragment:
                    if chunk_count == 0:
                        tracer.emit("client_first_token", **fields,
                                    ts_ns=received_ns, wall_ns=received_wall_ns)
                    last_token_clock = (received_ns, received_wall_ns)
                    chunk_count += 1
                    output_token_ids.extend(ids)
                    all_ids_observed = all_ids_observed and bool(ids)
                    tracer.emit("client_stream_chunk", **fields, chunk_number=chunk_count,
                                token_count=len(ids) if ids else None,
                                ts_ns=received_ns, wall_ns=received_wall_ns)
            usage = event.get("usage") or {}
            usage_tokens = usage.get("completion_tokens", usage_tokens)
    if not done or not finish_reason or not chunk_count:
        raise ValueError("Stream ended without generated output, terminal finish reason, or [DONE]")
    completion_tokens = usage_tokens if usage_tokens is not None else len(output_token_ids) or None
    if output_token_ids and usage_tokens is not None and len(output_token_ids) != usage_tokens:
        raise ValueError(f"Stream token IDs ({len(output_token_ids)}) disagree with usage ({usage_tokens})")
    complete_ids = output_token_ids and (usage_tokens is not None or all_ids_observed)
    output_hash = prompt_sha256(output_token_ids) if complete_ids else None
    tracer.emit("client_last_token", **fields, ts_ns=last_token_clock[0], wall_ns=last_token_clock[1])
    tracer.emit("client_response_end", **fields, completion_tokens=completion_tokens,
                stream_chunks=chunk_count, finish_reason=finish_reason, output_token_ids_sha256=output_hash)
    return {"stream_chunks": chunk_count, "completion_tokens": completion_tokens,
            "finish_reason": finish_reason, "output_token_ids_sha256": output_hash, "sse_events": received}


def _request_fields(row, label):
    return {
        "request_id": "pd-" + sha256_text(f"{label}\0{row['request_key']}")[:24],
        "request_key": row["request_key"],
        "dataset": row.get("dataset"),
        "domain": row.get("domain"),
        "case": row.get("case"),
        "phase": row.get("phase", "measured"),
        "pd_route": row.get("pd_route"),
        "conversation_id": row.get("conversation_id"),
        "turn": row.get("turn"),
        "label": label,
    }


def _chunk_identity(chunk):
    return (chunk["doc_id"], chunk["chunk_id"], chunk["chunk_version"])


def _escape_legacy_input(question, history, hits):
    """Protect literal input from being interpreted as a Blend boundary."""
    count = 0

    def escape(text):
        nonlocal count
        count += text.count(LEGACY_BLEND_SEPARATOR)
        return text.replace(LEGACY_BLEND_SEPARATOR, r"\u00b6")

    question = escape(question)
    history = [{**turn, "speaker": escape(turn["speaker"]), "text": escape(turn["text"])} for turn in history]
    hits = [{**hit, "text": escape(hit["text"])} for hit in hits]
    return question, history, hits, count


def serve_row(args, row, tracer, retriever, tokenizer, evidence_order=None):
    fields = _request_fields(row, args.label)
    frozen = None
    tracer.emit("client_request_start", **fields)
    try:
        if retriever is None:
            tokens = row["prompt_token_ids"]
            evidence = row.get("evidence", [])
            prompt_hash = prompt_sha256(tokens)
            if row.get("prompt_sha256", prompt_hash) != prompt_hash:
                raise ValueError("Frozen prompt token hash differs")
            frozen = row
        else:
            tracer.emit("embedding_start", **fields)
            vector = retriever.embed(row.get("retrieval_query") or row["question"])
            tracer.emit("embedding_end", **fields, embedding_dimension=vector.shape[1])
            tracer.emit("hnsw_search_start", **fields, top_k=args.top_k)
            scores, labels = retriever.search(vector, args.top_k)
            tracer.emit("hnsw_search_end", **fields, returned_rows=int((labels[0] >= 0).sum()))
            materialize_stage_fields = {"chunk_stages": []} if args.trace_chunk_stages else {}
            tracer.emit("chunk_materialize_start", **fields)
            hits, metadata_bytes = retriever.materialize(scores, labels, **materialize_stage_fields)
            trace_hits = [{key: hit[key] for key in ("doc_id", "chunk_id", "chunk_version", "rank", "score")}
                          for hit in hits]
            tracer.emit("chunk_materialize_end", **fields, hits=trace_hits, corpus_metadata_bytes=metadata_bytes,
                        **materialize_stage_fields)
            render_stage_fields = {"chunk_stages": []} if args.trace_chunk_stages else {}
            tracer.emit("prompt_render_start", **fields)
            question, history = row["question"], row.get("reference_history", [])
            format_fields = {}
            if args.escape_legacy_separator:
                question, history, hits, escaped = _escape_legacy_input(question, history, hits)
                format_fields["escaped_input_separator_count"] = escaped
            rendered = render_prompt(tokenizer, question, history, hits, args.max_prompt_tokens, prompt_format=args.prompt_format,
                                     evidence_order=evidence_order,
                                     **render_stage_fields)
            tokens, evidence = rendered.token_ids, rendered.evidence
            prompt_hash = prompt_sha256(tokens)
            if rendered.blend_separator_token_spans is not None:
                format_fields.update(prompt_format=args.prompt_format,
                                     blend_separator_token_spans=rendered.blend_separator_token_spans)
            if evidence_order is not None:
                format_fields["evidence_order"] = args.evidence_order
            tracer.emit("prompt_render_end", **fields, prompt_token_count=len(tokens),
                        prompt_sha256=prompt_hash, dropped_chunk_ids=rendered.dropped_chunk_ids,
                        **format_fields, **render_stage_fields)
            frozen = {**row, "prompt_token_ids": tokens, "prompt_sha256": prompt_hash,
                      "retrieval_hits": trace_hits, "evidence": evidence,
                      "dropped_chunk_ids": rendered.dropped_chunk_ids,
                      **format_fields,
                      "generation": {"max_tokens": args.max_tokens, "temperature": 0.0}}
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Prompt must contain nonnegative integer token IDs")
        for chunk in evidence:
            if not 0 <= chunk["start_token"] < chunk["end_token"] <= len(tokens):
                raise ValueError(f"Invalid prompt span for {chunk['chunk_id']}")
        payload = {
            "model": args.model,
            "prompt": tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            **{key: row["generation"][key] for key in GENERATION_KEYS if key in row.get("generation", {})},
            "evidence": evidence,
        }
        if row.get("pd_route"):
            payload["pd_route"] = row["pd_route"]
        body = json_line(payload).encode("utf-8")
        tracer.emit("client_dispatch", **fields, prompt_token_count=len(tokens),
                    prompt_sha256=prompt_hash, evidence=evidence, http_request_body_bytes=len(body))
        response = stream_completion(args.gateway, body, args.timeout, tracer, fields)
        return frozen, {**fields, "success": True, "prompt_sha256": prompt_hash, **response}
    except (HTTPException, OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
        message = str(exc)
        if isinstance(exc, HTTPError):
            with exc:
                message += ": " + exc.read(1024).decode("utf-8", errors="replace")
        tracer.emit("client_request_error", **fields, message=message)
        return frozen, {**fields, "success": False, "message": message}


def replay(args):
    from evidencestore.trace import TraceWriter

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.label):
        raise ValueError("--label must contain only letters, digits, underscores, and hyphens")
    if min(args.concurrency, args.top_k, args.max_prompt_tokens, args.max_tokens, args.timeout) <= 0:
        raise ValueError("Replay limits, concurrency, and timeout must be positive")
    retriever = tokenizer = None
    if args.frozen:
        if args.index_dir or args.tokenizer or args.tokenizer_revision:
            raise ValueError("Use index and tokenizer options with --queries")
        input_path = args.frozen
        rows = list(load_frozen(input_path))
    else:
        if not all((args.index_dir, args.tokenizer, args.tokenizer_revision)):
            raise ValueError("--queries requires --index-dir, --tokenizer, and --tokenizer-revision")
        input_path = args.queries
        rows = list(read_jsonl(input_path))
        if args.evidence_order == "seen-first" and any(
            row.get("conversation_id") or row.get("reference_history") for row in rows
        ):
            raise ValueError("Seen-first evidence ordering requires single-turn queries")
        if args.evidence_order == "previous-turn-first" and any(
            row.get("dataset") != "mtrag" or not row.get("conversation_id") for row in rows
        ):
            raise ValueError("Previous-turn-first evidence ordering requires MT-RAG conversation turns")
        retriever = HnswRetriever(args.index_dir, args.embedding_device)
        tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_revision)
    if not rows or len({row["request_key"] for row in rows}) != len(rows):
        raise ValueError("Replay input must contain nonempty unique request keys")
    output = args.run_dir / "client" / args.label
    output.mkdir(parents=True)
    tracer = TraceWriter(args.run_dir, "client")
    phases = defaultdict(dict)
    for index, row in enumerate(rows):
        phase = row.get("phase", "measured")
        group = row.get("conversation_id") or row["request_key"]
        phases[phase].setdefault(group, []).append((index, row))
    frozen_by_index = {}
    failed = 0
    lock = Lock()
    seen_chunks = {}
    previous_turns = {}
    previous_turn_first = args.evidence_order == "previous-turn-first"

    def first_seen(chunk):
        return seen_chunks.get(_chunk_identity(chunk), len(seen_chunks))

    def run_group(group, epoch):
        offset = group[0][1].get("start_offset_us", 0)
        delay = (epoch + offset * 1000 - time.monotonic_ns()) / 1e9
        if delay > 0:
            time.sleep(delay)
        if previous_turn_first:
            conversation_id = group[0][1]["conversation_id"]
            previous_order = previous_turns.setdefault(conversation_id, {})

            def previous_prompt_order(chunk):
                identity = _chunk_identity(chunk)
                if identity in previous_order:
                    return 0, previous_order[identity]
                return 1, seen_chunks.get(identity, len(seen_chunks))

            evidence_order = previous_prompt_order
        else:
            evidence_order = first_seen if args.evidence_order == "seen-first" else None

        for index, row in group:
            frozen, response = serve_row(args, row, tracer, retriever, tokenizer, evidence_order)
            if evidence_order is not None and response["success"]:
                identities = [_chunk_identity(chunk) for chunk in frozen["evidence"]]
                for identity in identities:
                    seen_chunks.setdefault(identity, len(seen_chunks))
                if previous_turn_first:
                    previous_order.clear()
                    previous_order.update((identity, position) for position, identity in enumerate(identities))
            yield index, frozen, response

    with (output / "responses.jsonl").open("x", encoding="utf-8") as receipt:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for groups in phases.values():
                epoch = time.monotonic_ns()
                # Executors run whole conversations concurrently. Each thread writes
                # a finished request immediately, then sends that conversation's next turn.
                def run_and_record(group):
                    nonlocal failed
                    for index, frozen, response in run_group(group, epoch):
                        with lock:
                            failed += int(not response["success"])
                            if frozen is not None:
                                frozen_by_index[index] = frozen
                            receipt.write(json_line(response))
                            receipt.flush()

                futures = [pool.submit(run_and_record, group) for group in groups.values()]
                for future in futures:
                    future.result()
    with (output / "frozen.jsonl").open("x", encoding="utf-8") as handle:
        handle.writelines(json_line(frozen_by_index[index]) for index in sorted(frozen_by_index))
    manifest = {
        "mode": "frozen" if args.frozen else "live_hnsw",
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "top_k": args.top_k,
        "max_prompt_tokens": args.max_prompt_tokens,
        "concurrency": args.concurrency,
        "requested": len(rows),
        "succeeded": len(rows) - failed,
        "failed": failed,
        "frozen_rows": len(frozen_by_index),
        "frozen_sha256": sha256_file(output / "frozen.jsonl"),
        "index": retriever.manifest if retriever else None,
    }
    if args.trace_chunk_stages:
        manifest["trace_chunk_stages"] = retriever is not None
    if args.escape_legacy_separator:
        manifest["escape_legacy_separator"] = True
    formats = {row.get("prompt_format", "standard") for row in rows} if args.frozen else {args.prompt_format}
    for prompt_format in PROMPT_FORMATS[1:]:
        if formats == {prompt_format}:
            manifest["prompt_format"] = prompt_format
    if args.evidence_order != "retrieval":
        manifest["evidence_order"] = args.evidence_order
    (output / "manifest.json").write_text(json_line(manifest), encoding="utf-8")
    print(f"{manifest['succeeded']}/{manifest['requested']} complete. Receipts: {output}")
    return int(failed > 0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare-data", help="Normalize complete labeled datasets and record source checksums")
    data.add_argument("--dataset", choices=("musique", "2wiki", "mtrag"), required=True)
    data.add_argument("--source", type=Path, required=True)
    data.add_argument("--output", type=Path, required=True)

    smoke = sub.add_parser("smoke-sample", help="Select a small MT-RAG query and passage sample")
    smoke.add_argument("--queries", type=Path, required=True)
    smoke.add_argument("--corpus", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)

    build = sub.add_parser("build-index", help="Build stock Faiss HNSW and stable row metadata")
    build.add_argument("--corpus", type=Path, required=True)
    build.add_argument("--index-dir", type=Path, required=True)
    build.add_argument("--embedding-model", default="intfloat/e5-base-v2")
    build.add_argument("--embedding-revision", required=True)
    build.add_argument("--embedding-device", default="cpu")
    build.add_argument("--embedding-batch-size", type=int, default=32)
    build.add_argument("--hnsw-m", type=int, default=32)
    build.add_argument("--ef-construction", type=int, default=200)
    build.add_argument("--ef-search", type=int, default=200)

    fixture = sub.add_parser("fixture", help="Freeze controlled cold, repeat, and reordered request tokens")
    fixture.add_argument("--topology", choices=("1p3d", "2p2d"), required=True)
    fixture.add_argument("--tokenizer", required=True)
    fixture.add_argument("--tokenizer-revision", required=True)
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--max-prompt-tokens", type=int, default=8192)
    fixture.add_argument("--evidence-repeats", type=int, default=1)

    send = sub.add_parser("replay", help="Replay the complete input through the gateway")
    send.add_argument("--gateway", required=True)
    send.add_argument("--run-dir", type=Path, required=True)
    send.add_argument("--label", required=True)
    send.add_argument("--model", required=True)
    inputs = send.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--frozen", type=Path)
    inputs.add_argument("--queries", type=Path)
    send.add_argument("--index-dir", type=Path)
    send.add_argument("--tokenizer")
    send.add_argument("--tokenizer-revision")
    send.add_argument("--embedding-device", default="cpu")
    send.add_argument("--top-k", type=int, default=6)
    send.add_argument("--max-prompt-tokens", type=int, default=8192)
    send.add_argument("--max-tokens", type=int, default=32)
    send.add_argument("--timeout", type=float, default=300)
    send.add_argument("--concurrency", type=int, default=1)
    send.add_argument("--trace-chunk-stages", action="store_true",
                      help="trace live corpus-row reads, prompt chunks, and whole-prompt assembly")
    send.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="standard",
                      help="render live queries with the selected prompt format")
    send.add_argument("--escape-legacy-separator", action="store_true",
                      help="escape literal ¶ in legacy Blend prompt inputs")
    send.add_argument("--evidence-order", choices=("retrieval", "seen-first", "previous-turn-first"),
                      default="retrieval", help="order retained evidence using retrieval rank or replay history")
    args = parser.parse_args(argv)
    if args.command == "replay" and args.frozen and args.prompt_format != "standard":
        parser.error("--prompt-format applies to live --queries only")
    if args.command == "replay" and args.escape_legacy_separator and (args.frozen or args.prompt_format == "standard"):
        parser.error("--escape-legacy-separator requires live queries and a legacy Blend prompt format")
    if args.command == "replay" and args.evidence_order != "retrieval" and (
        args.frozen or args.concurrency != 1
    ):
        parser.error("Replay-based evidence ordering requires live --queries with --concurrency 1")
    return args


def main():
    args = parse_args()
    if args.command == "prepare-data":
        print(json.dumps(prepare_data(args.dataset, args.source, args.output), indent=2))
    elif args.command == "smoke-sample":
        print(json.dumps(prepare_smoke_sample(args.queries, args.corpus, args.output), indent=2))
    elif args.command == "build-index":
        print(json.dumps(build_hnsw(
            args.corpus, args.index_dir, embedding_model=args.embedding_model,
            embedding_revision=args.embedding_revision, embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size, hnsw_m=args.hnsw_m,
            ef_construction=args.ef_construction, ef_search=args.ef_search,
        ), indent=2))
    elif args.command == "fixture":
        tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_revision)
        rows = list(fixture_rows(tokenizer, args.topology, args.max_prompt_tokens, args.evidence_repeats))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.writelines(json_line(row) for row in rows)
        print(f"Wrote {len(rows)} fixture requests to {args.output}")
    else:
        return replay(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
