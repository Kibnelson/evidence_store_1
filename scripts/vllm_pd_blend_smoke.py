#!/usr/bin/env python3
"""Smoke-test CacheBlend with controlled probes or live MuSiQue retrieval."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from vllm_pd_cluster import (
    HOST, LMCACHE_TRACE_SOURCE, MODEL, REVISION, ROOT, TOPOLOGIES, assigned_gpus, await_health,
    check_children, start, stop_children, write_json,
)
from vllm_pd_replay import load_tokenizer

from evidencestore.retrieval import (
    LEGACY_BLEND_SEPARATOR, chunk_record, json_line, prompt_sha256, read_jsonl, render_prompt, sha256_file,
)


CHUNK_SIZE = 256
CACHE_PORT = 6555
HTTP_PORT = 8090
CACHE_HTTP = f"http://{HOST}:{HTTP_PORT}"
SCRIPT = Path(__file__).resolve()
CASES = {"prefix": ("seed-exact", "seed-shifted"), "blend": ("exact", "shifted")}
# Each group of five retrieved a common versioned chunk at different offsets
# in the previous MuSiQue archive. The first group's chunk was 610 tokens.
MUSIQUE_BLEND_KEYS = (
    "musique:3hop1__604797_443779_52195",
    "musique:3hop1__144439_443779_52195",
    "musique:3hop1__743797_443779_52195",
    "musique:3hop1__275261_443779_52195",
    "musique:4hop1__93963_170667_443779_52195",
    "musique:3hop1__248929_160713_77246",
    "musique:3hop1__475351_160713_77246",
    "musique:3hop1__632623_160713_77246",
    "musique:3hop1__106847_160713_77246",
    "musique:3hop1__23444_160713_77246",
)


def musique_sample(prepared, index_dir):
    from vllm_pd_chunk_smoke_inputs import musique_queries

    queries, manifest = musique_queries(prepared)
    rows = {query["request_key"]: query for query in queries}
    missing = set(MUSIQUE_BLEND_KEYS) - rows.keys()
    if missing:
        raise ValueError(f"Pinned MuSiQue queries are missing {', '.join(sorted(missing))}")
    sample = [rows[key] for key in MUSIQUE_BLEND_KEYS]
    if len({row["question"] for row in sample}) != len(sample) or any(row.get("pd_route") for row in sample):
        raise ValueError("Natural Blend smoke expects distinct questions and normal gateway routing")
    index = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    corpus_digest = sha256_file(prepared / "corpus.jsonl")
    if (index["rows"] != manifest["corpus_rows"] or index["corpus_sha256"] != corpus_digest
            or index["embedding_model"] != "intfloat/e5-base-v2"
            or index["embedding_revision"] != "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
            or not all((index_dir / name).is_file() for name in ("index.faiss", "chunks.jsonl"))):
        raise ValueError("The Blend smoke needs the full matching MuSiQue HNSW index")
    return sample, {
        "dataset": "musique", "query_keys": list(MUSIQUE_BLEND_KEYS), "prompt_format": "legacy-blend",
        "source_sha256": manifest["source_sha256"], "index_sha256": index["index_sha256"],
        "corpus_sha256": corpus_digest, "selection_reference": "native-vllm-30453-20261119T031129Z",
    }


def write_replay_sample(directory, sample, manifest):
    directory.mkdir(parents=True)
    path = directory / "queries.jsonl"
    path.write_text("".join(json_line(row) for row in sample), encoding="utf-8")
    write_json(directory / "manifest.json", {**manifest, "queries_sha256": sha256_file(path)})


def verify_replay_pair(root, label="musique", expected_requests=len(MUSIQUE_BLEND_KEYS)):
    def receipts(topology):
        run = root / topology / "blend/client" / label
        fields = ("doc_id", "chunk_id", "chunk_version", "rank", "start_token", "end_token")
        return [
            (row["request_key"], row["prompt_sha256"], [tuple(chunk[field] for field in fields) for chunk in row["evidence"]])
            for row in read_jsonl(run / "frozen.jsonl")
        ]

    first = receipts("1p3d")
    if len(first) != expected_requests or first != receipts("2p2d"):
        raise ValueError(f"{label} request count, prompts, or evidence spans differ between the Blend layouts")
    write_json(root / f"{label}-pair.json", {
        "status": "passed", "layouts": ["1b3d", "2b2d"], "requests_per_layout": expected_requests,
        "matched_fields": ["request_key", "prompt_sha256", "evidence_identity_rank_and_token_span"],
    })


def fixture_doc(name, topic):
    text = " ".join(f"{name} entry {index} records an observation about {topic}." for index in range(68))
    return chunk_record("blend-smoke", name, "body", text)


def prepare_probes(tokenizer, directory, topology):
    question = "Summarize the evidence in one sentence."
    exact = [fixture_doc("harbor", "boats leaving the harbor"), fixture_doc("furnace", "fuel consumed overnight")]
    shifted = [fixture_doc("orchard", "apples gathered in autumn"), fixture_doc("archive", "letters kept in the archive")]
    seed_slot, blend_slot = ("p1", "p0") if topology == "2p2d" else ("p0", "p0")
    shifted_slot = "p1" if topology == "2p2d" else "p0"
    cases = (
        ("seed-exact", exact, seed_slot, "d0"),
        ("seed-shifted", shifted, "p0", "d1"),
        ("exact", exact, blend_slot, "d0"),
        ("shifted", list(reversed(shifted)), shifted_slot, "d2" if topology == "1p3d" else "d1"),
    )
    rows = {}
    for case, docs, prefill, decode in cases:
        rendered = render_prompt(tokenizer, question, [], docs, 8192, prompt_format="legacy-blend")
        if len(rendered.evidence) != len(docs) or rendered.dropped_chunk_ids:
            raise ValueError(f"Evidence was dropped from the {case} probe")
        rows[case] = {
            "request_key": f"{topology}:{case}", "dataset": "blend-smoke", "case": case,
            "pd_route": {"prefill": prefill, "decode": decode}, "prompt_format": "legacy-blend",
            "prompt_token_ids": rendered.token_ids, "prompt_sha256": prompt_sha256(rendered.token_ids),
            "evidence": rendered.evidence, "blend_separator_token_spans": rendered.blend_separator_token_spans,
        }
    validate_probes(rows)
    directory.mkdir(parents=True)
    for case, row in rows.items():
        (directory / f"{case}.jsonl").write_text(json_line(row), encoding="utf-8")
    write_json(directory / "manifest.json", {
        "model": MODEL, "revision": REVISION, "topology": topology, "chunk_size": CHUNK_SIZE,
        "separator": LEGACY_BLEND_SEPARATOR,
        "separator_token_ids": tokenizer.encode(LEGACY_BLEND_SEPARATOR, add_special_tokens=False),
        "prompt_hashes": {case: row["prompt_sha256"] for case, row in rows.items()},
    })


def validate_probes(rows):
    if rows["seed-exact"]["prompt_sha256"] != rows["exact"]["prompt_sha256"]:
        raise ValueError("Exact seed and repeat prompts differ")
    seed, repeat = (rows[case] for case in ("seed-shifted", "shifted"))
    old_ids, new_ids = seed["prompt_token_ids"], repeat["prompt_token_ids"]
    if old_ids[:CHUNK_SIZE] == new_ids[:CHUNK_SIZE]:
        raise ValueError("Shifted prompt must break the prefix before the first LMCache chunk")
    exact_ids = rows["seed-exact"]["prompt_token_ids"]
    exact_windows = {tuple(exact_ids[start:start + CHUNK_SIZE])
                     for start in range(0, len(exact_ids) - CHUNK_SIZE + 1, CHUNK_SIZE)}
    if any(tuple(new_ids[start:start + CHUNK_SIZE]) in exact_windows
           for start in range(len(new_ids) - CHUNK_SIZE + 1)):
        raise ValueError("The exact trial could prewarm a cache window in the shifted trial")
    old_spans = {chunk["chunk_id"]: chunk for chunk in seed["evidence"]}
    for chunk in repeat["evidence"]:
        old = old_spans[chunk["chunk_id"]]
        before, after = old["start_token"], chunk["start_token"]
        aligned = -(-before // CHUNK_SIZE) * CHUNK_SIZE
        if (before == after or aligned + CHUNK_SIZE > old["end_token"]
                or old_ids[before:old["end_token"]] != new_ids[after:chunk["end_token"]]):
            raise ValueError(f"No stable shifted cache window for {chunk['chunk_id']}")


def replay_cases(directory, mode):
    from vllm_pd_blend_receipts import capture_case

    run_dir = Path(os.environ["EVIDENCESTORE_RUN_DIR"])
    gateway = os.environ["EVIDENCESTORE_GATEWAY"]
    for case in CASES[mode]:
        command = [sys.executable, str(ROOT / "scripts/vllm_pd_replay.py"), "replay", "--gateway", gateway,
                   "--run-dir", str(run_dir), "--label", case, "--model", MODEL, "--max-tokens", "16",
                   "--frozen", str(directory / f"{case}.jsonl")]
        capture_case(CACHE_HTTP, run_dir, case, command)


def replay_musique(directory, index_dir, trace_enabled):
    run_dir = Path(os.environ["EVIDENCESTORE_RUN_DIR"])
    command = [sys.executable, str(ROOT / "scripts/vllm_pd_replay.py"), "replay",
               "--gateway", os.environ["EVIDENCESTORE_GATEWAY"], "--run-dir", str(run_dir),
               "--label", "musique", "--model", MODEL, "--queries", str(directory / "queries.jsonl"),
               "--index-dir", str(index_dir), "--tokenizer", MODEL, "--tokenizer-revision", REVISION,
               "--prompt-format", "legacy-blend", "--concurrency", "1"]
    if trace_enabled:
        command.append("--trace-chunk-stages")
    subprocess.run(command, check=True)
    if trace_enabled:
        from vllm_pd_chunk_smoke_inputs import wait_for_lmcache_transfers

        wait_for_lmcache_transfers(run_dir.parent / "lmcache-trace")


def run_topology(directory, topology, gpus, cache_source, *, trace_enabled, index_dir=None):
    from vllm_pd_blend_receipts import verify_topology

    (directory / "logs").mkdir()
    trace_mode = "on" if trace_enabled else "off"
    trace_gate = "1" if trace_enabled else "0"
    cache_trace = directory / "lmcache-trace"
    cache_cmd = [sys.executable, "-m", "lmcache.cli.main", "server", "--engine-type", "blend",
                 "--supported-transfer-mode", "lmcache_driven", "--host", HOST, "--port", str(CACHE_PORT),
                 "--http-host", HOST, "--http-port", str(HTTP_PORT), "--chunk-size", str(CHUNK_SIZE),
                 "--l1-align-bytes", "16777216", "--l1-size-gb", os.environ.get("PD_LMCACHE_SIZE_GB", "8"),
                 "--eviction-policy", "LRU", "--instance-id", f"blend-{topology}-{os.environ['SLURM_JOB_ID']}"]
    cache_env = {
        "CUDA_VISIBLE_DEVICES": ",".join(gpus[:TOPOLOGIES[topology][0]]),
        "EVIDENCESTORE_TRACE_LMCACHE_TRANSFERS": trace_gate,
        "EVIDENCESTORE_TRACE_LMCACHE_BLEND": trace_gate,
        "EVIDENCESTORE_TRACE_DIR": str(cache_trace),
        "EVIDENCESTORE_RUN_ID": f"{directory.parent.name}/{topology}",
    }
    server = start("lmcache", cache_cmd, cache_env, directory)
    try:
        write_json(directory / "lmcache-server.json", {
            "pid": server.process.pid, "argv": cache_cmd, "environment": cache_env, "trace_enabled": trace_enabled,
        })
        await_health({"lmcache": CACHE_HTTP}, [server], 60, lambda: False, path="/healthcheck")
        workloads = [(mode, mode, len(cases)) for mode, cases in CASES.items()]
        if index_dir is not None:
            requests = sum(1 for _ in read_jsonl(directory / "inputs/queries.jsonl"))
            workloads = [("blend", "musique", requests)]
        for mode, workload_kind, count in workloads:
            print(f"Starting {topology} {mode}; LMCache PID {server.process.pid}", flush=True)
            command = [sys.executable, str(ROOT / "scripts/vllm_pd_cluster.py"), "--run-dir", str(directory / mode),
                       "--topology", topology, "--prefill-mode", mode, "--lmcache", "--external-lmcache",
                       "--lmcache-port", str(CACHE_PORT), "--lmcache-http-port", str(HTTP_PORT),
                       "--lmcache-source", str(cache_source), "--prefix-cache", "off", "--trace", trace_mode,
                       "--minimum-nixl-reads", str(count),
                       "--", sys.executable, str(SCRIPT),
                       "--run-root", str(directory / "inputs"), "--workload", workload_kind, "--trace", trace_mode]
            if index_dir is not None:
                command.extend(("--index-dir", str(index_dir)))
            launcher = start(f"{mode}-launch", command, {"EVIDENCESTORE_TRACE_GPU": trace_gate}, directory)
            try:
                while launcher.process.poll() is None:
                    check_children([server])
                    time.sleep(0.5)
                if launcher.process.returncode:
                    raise RuntimeError(f"{mode} launch failed; inspect {launcher.log}")
            finally:
                if launcher.process.poll() is None:
                    launcher.process.terminate()
                    try:
                        launcher.process.wait(timeout=25)
                    except subprocess.TimeoutExpired:
                        stop_children([launcher])
    finally:
        cleanup = stop_children([server])
        write_json(directory / "lmcache-cleanup.json", cleanup)
        if cleanup["remaining_process_groups"] or cleanup["unverified_process_groups"]:
            raise RuntimeError(f"LMCache cleanup could not be verified in {directory}")
    if index_dir is not None:
        from vllm_pd_blend_natural_report import verify_natural_replay

        verify_natural_replay(directory, trace_enabled=trace_enabled)
        print(f"Natural MuSiQue Blend report: {directory / 'natural-blend-summary.json'}", flush=True)
        return
    summary = verify_topology(directory, topology, CASES, trace_enabled=trace_enabled)
    try:
        if trace_enabled:
            from vllm_pd_blend_trace_join import join_lmcache_trace

            joined = join_lmcache_trace(directory)
            summary["lmcache_request_trace"] = joined.name
            summary["measurement_scope"]["lmcache_scatter_h2d_payload"] = "Completed host-to-GPU payload in the joined trace."
            print(f"Joined LMCache trace: {joined}", flush=True)
        elif any(cache_trace.glob("*.jsonl")):
            raise ValueError(f"Trace-off Blend run wrote LMCache events in {cache_trace}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        summary.update(status="failed", lmcache_trace_error=str(error))
        raise ValueError(f"LMCache trace verification failed: {error}") from error
    finally:
        write_json(directory / "blend-summary.json", summary)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="A fresh node-local output directory")
    parser.add_argument("--topology", choices=("both", *TOPOLOGIES), default="both")
    parser.add_argument("--trace", choices=("on", "off"), default="on",
                        help="toggle client, gateway, engine, NIXL, and LMCache request traces")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare the smoke input files without launching GPUs")
    parser.add_argument("--musique-prepared", type=Path, help="Replay ten distinct questions using the full prepared MuSiQue index")
    parser.add_argument("--index-dir", type=Path, help="Override the index in the prepared MuSiQue directory")
    parser.add_argument("--workload", choices=(*CASES, "musique"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.run_root.resolve()
    try:
        trace_enabled = args.trace == "on"
        if args.workload == "musique":
            if args.index_dir is None:
                parser.error("A live MuSiQue workload requires --index-dir")
            replay_musique(root, args.index_dir.resolve(), trace_enabled)
            return
        if args.workload:
            replay_cases(root, args.workload)
            return
        if args.index_dir and not args.musique_prepared:
            parser.error("--index-dir requires --musique-prepared")
        index_dir = None
        if args.musique_prepared:
            prepared = args.musique_prepared.resolve()
            index_dir = (args.index_dir or prepared / "index").resolve()
            sample, manifest = musique_sample(prepared, index_dir)
        gpus = cache_source = plugin = None
        if not args.prepare_only:
            if not os.environ.get("SLURM_JOB_ID"):
                raise ValueError("Run this GPU smoke inside a four-GPU Slurm allocation")
            if importlib.util.find_spec("lmcache_cacheblend"):
                plugin = importlib.util.find_spec("lmcache_cacheblend.connector")
            if not plugin or not plugin.origin:
                raise ValueError("The approved lmcache-cacheblend plugin providing CBKVConnector is missing")
            cache_spec = importlib.util.find_spec("lmcache")
            if not cache_spec or not cache_spec.origin:
                raise ValueError("Activate the verified vLLM and LMCache runtime first")
            cache_source = Path(cache_spec.origin).resolve().parents[1]
            if trace_enabled and cache_source != LMCACHE_TRACE_SOURCE.resolve():
                raise ValueError("Trace-on Blend requires a runtime built with EVIDENCESTORE_LMCACHE_TRACE=1")
            gpus = assigned_gpus()
            # Check the fixed ports before starting a cache process. An active
            # PD replay in this allocation will already own its worker ports.
            for port in (8000, *range(8100, 8104), *range(5600, 5604), CACHE_PORT, HTTP_PORT):
                with socket.socket() as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((HOST, port))
        root.mkdir(parents=True)
        print(f"Blend smoke artifacts: {root}", flush=True)
        if plugin:
            write_json(root / "plugin-module.json", {"path": plugin.origin, "sha256": sha256_file(plugin.origin)})
        tokenizer = load_tokenizer(MODEL, REVISION) if index_dir is None else None
        for topology in TOPOLOGIES if args.topology == "both" else (args.topology,):
            directory = root / topology
            if index_dir is None:
                prepare_probes(tokenizer, directory / "inputs", topology)
            else:
                write_replay_sample(directory / "inputs", sample, manifest)
            if gpus is not None:
                run_topology(directory, topology, gpus, cache_source, trace_enabled=trace_enabled, index_dir=index_dir)
        if gpus is not None and index_dir is not None and args.topology == "both":
            verify_replay_pair(root)
        print("Blend prompts prepared." if args.prepare_only else "Blend GPU smoke passed.")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Blend smoke failed: {error}\n")


if __name__ == "__main__":
    def stop(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
