#!/usr/bin/env python3
"""Seed one LMCache server with prefix prefill, then test CacheBlend on it."""

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
    HOST, MODEL, REVISION, ROOT, TOPOLOGIES, assigned_gpus, await_health,
    check_children, start, stop_children, write_json,
)
from vllm_pd_replay import load_tokenizer

from evidencestore.retrieval import (
    LEGACY_BLEND_SEPARATOR, chunk_record, json_line, prompt_sha256, render_prompt, sha256_file,
)


CHUNK_SIZE = 256
CACHE_PORT = 6555
HTTP_PORT = 8090
CACHE_HTTP = f"http://{HOST}:{HTTP_PORT}"
SCRIPT = Path(__file__).resolve()
CASES = {"prefix": ("seed-exact", "seed-shifted"), "blend": ("exact", "shifted")}


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


def run_topology(directory, topology, gpus, cache_source):
    from vllm_pd_blend_receipts import verify_topology

    (directory / "logs").mkdir()
    cache_cmd = [sys.executable, "-m", "lmcache.cli.main", "server", "--engine-type", "blend",
                 "--supported-transfer-mode", "lmcache_driven", "--host", HOST, "--port", str(CACHE_PORT),
                 "--http-host", HOST, "--http-port", str(HTTP_PORT), "--chunk-size", str(CHUNK_SIZE),
                 "--l1-align-bytes", "16777216", "--l1-size-gb", os.environ.get("PD_LMCACHE_SIZE_GB", "8"),
                 "--eviction-policy", "LRU", "--instance-id", f"blend-{topology}-{os.environ['SLURM_JOB_ID']}"]
    cache_env = {
        "CUDA_VISIBLE_DEVICES": ",".join(gpus[:TOPOLOGIES[topology][0]]),
        "EVIDENCESTORE_TRACE_LMCACHE_TRANSFERS": "0",
    }
    server = start("lmcache", cache_cmd, cache_env, directory)
    try:
        write_json(directory / "lmcache-server.json", {
            "pid": server.process.pid, "argv": cache_cmd, "environment": cache_env,
        })
        await_health({"lmcache": CACHE_HTTP}, [server], 60, lambda: False, path="/healthcheck")
        for mode in CASES:
            print(f"Starting {topology} {mode}; LMCache PID {server.process.pid}", flush=True)
            command = [sys.executable, str(ROOT / "scripts/vllm_pd_cluster.py"), "--run-dir", str(directory / mode),
                       "--topology", topology, "--prefill-mode", mode, "--lmcache", "--external-lmcache",
                       "--lmcache-port", str(CACHE_PORT), "--lmcache-http-port", str(HTTP_PORT),
                       "--lmcache-source", str(cache_source), "--prefix-cache", "off", "--trace", "on",
                       "--", sys.executable, str(SCRIPT),
                       "--run-root", str(directory / "inputs"), "--workload", mode]
            launcher = start(f"{mode}-launch", command, {"EVIDENCESTORE_TRACE_GPU": "1"}, directory)
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
        verify_topology(directory, topology, CASES)
    finally:
        cleanup = stop_children([server])
        write_json(directory / "lmcache-cleanup.json", cleanup)
        if cleanup["remaining_process_groups"] or cleanup["unverified_process_groups"]:
            raise RuntimeError(f"LMCache cleanup could not be verified in {directory}")


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="A fresh node-local output directory")
    parser.add_argument("--topology", choices=("both", *TOPOLOGIES), default="both")
    parser.add_argument("--prepare-only", action="store_true", help="Validate prompt fixtures on CPU")
    parser.add_argument("--workload", choices=tuple(CASES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.run_root.resolve()
    try:
        if args.workload:
            replay_cases(root, args.workload)
            return
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
        tokenizer = load_tokenizer(MODEL, REVISION)
        for topology in TOPOLOGIES if args.topology == "both" else (args.topology,):
            directory = root / topology
            prepare_probes(tokenizer, directory / "inputs", topology)
            if gpus is not None:
                run_topology(directory, topology, gpus, cache_source)
        print("Blend prompts prepared." if args.prepare_only else "Blend GPU smoke passed.")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Blend smoke failed: {error}\n")


if __name__ == "__main__":
    def stop(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
