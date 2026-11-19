#!/usr/bin/env python3
"""Run the public in-process Full Blend smoke with native NIXL decode."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from vllm_pd_blend_smoke import (
    fixture_doc, musique_sample, verify_musique_pair, write_musique_sample,
)
from vllm_pd_cluster import MODEL, REVISION, ROOT, TOPOLOGIES
from vllm_pd_replay import load_tokenizer

from evidencestore.retrieval import json_line, prompt_sha256, render_prompt


SCRIPT = Path(__file__).resolve()


def prepare_controlled(tokenizer, topology, output):
    rows = []
    for slot in range(TOPOLOGIES[topology][0]):
        documents = [fixture_doc(f"legacy-{slot}-{topic}", topic) for topic in ("harbor", "observatory")]
        old_spans = {}
        for case, evidence in (("cold", documents), ("reordered", documents[::-1])):
            prompt = render_prompt(tokenizer, f"{case.capitalize()} probe: summarize the evidence.", [], evidence,
                                   8192, prompt_format="legacy-blend")
            if len(prompt.evidence) != len(documents) or prompt.dropped_chunk_ids:
                raise ValueError("A controlled Blend evidence chunk was dropped")
            for span in prompt.evidence:
                key = span["chunk_id"]
                start, stop = span["start_token"], span["end_token"]
                tokens = prompt.token_ids[start:stop]
                if case == "cold":
                    old_spans[key] = (start, tokens)
                elif old_spans[key][0] == start or old_spans[key][1] != tokens:
                    raise ValueError(f"Controlled evidence {key} did not move with identical tokens")
            rows.append({
                "dataset": "legacy-probes", "request_key": f"legacy:p{slot}:{case}", "case": case,
                "pd_route": {"prefill": f"p{slot}", "decode": f"d{slot}"},
                "prompt_format": "legacy-blend", "prompt_token_ids": prompt.token_ids,
                "prompt_sha256": prompt_sha256(prompt.token_ids), "evidence": prompt.evidence,
                "blend_separator_token_spans": prompt.blend_separator_token_spans,
            })
    output.write_text("".join(json_line(row) for row in rows), encoding="utf-8")
    return len(rows)


def replay(args):
    run_dir = Path(os.environ["EVIDENCESTORE_RUN_DIR"])
    directory = run_dir / "inputs"
    shutil.copytree(args.inputs, directory)
    command = [sys.executable, str(ROOT / "scripts/vllm_pd_replay.py"), "replay",
               "--gateway", os.environ["EVIDENCESTORE_GATEWAY"], "--run-dir", str(run_dir),
               "--model", MODEL, "--concurrency", "1"]
    natural = ["--label", "musique", "--queries", str(directory / "queries.jsonl"),
               "--index-dir", str(args.index_dir), "--tokenizer", MODEL, "--tokenizer-revision", REVISION,
               "--prompt-format", "legacy-blend"]
    if args.trace == "on":
        natural.append("--trace-chunk-stages")
    subprocess.run([*command, *natural], check=True)
    subprocess.run([*command, "--label", "legacy-probes", "--max-tokens", "8",
                    "--frozen", str(directory / "controlled.jsonl")], check=True)


def run(args):
    prepared = args.musique_prepared.resolve()
    index = prepared / "index"
    sample, manifest = musique_sample(prepared, index)
    tokenizer = load_tokenizer(MODEL, REVISION)
    layouts = TOPOLOGIES if args.topology == "both" else (args.topology,)
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    print(f"Legacy Full Blend artifacts: {root}", flush=True)
    for topology in layouts:
        layout = root / topology
        inputs = layout / "inputs"
        write_musique_sample(inputs, sample, manifest)
        probes = prepare_controlled(tokenizer, topology, inputs / "controlled.jsonl")
        if args.prepare_only:
            continue
        cohort = layout / "blend"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/vllm_pd_cluster.py"), "--legacy-full-blend",
            "--run-dir", str(cohort), "--topology", topology, "--trace", args.trace,
            "--source", os.environ["VLLM_SOURCE"], "--lmcache-source", os.environ["LMCACHE_SOURCE"],
            "--lmcache-size-gb", "25", "--minimum-nixl-reads", str(len(sample) + probes),
            "--", sys.executable, str(SCRIPT), "replay", "--inputs", str(inputs),
            "--index-dir", str(index), "--trace", args.trace,
        ], check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/vllm_pd_legacy_blend_report.py"),
                        "--run-dir", str(cohort), "--trace", args.trace], check=True)
    if not args.prepare_only and args.topology == "both":
        verify_musique_pair(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    smoke = modes.add_parser("run", help="prepare and launch the layout smoke")
    smoke.add_argument("--topology", choices=(*TOPOLOGIES, "both"), default="1p3d")
    smoke.add_argument("--run-root", type=Path, required=True)
    smoke.add_argument("--musique-prepared", type=Path, required=True)
    smoke.add_argument("--prepare-only", action="store_true")
    workload = modes.add_parser("replay", help="replay both request groups after the workers are ready")
    workload.add_argument("--inputs", type=Path, required=True)
    workload.add_argument("--index-dir", type=Path, required=True)
    for mode in (smoke, workload):
        mode.add_argument("--trace", choices=("on", "off"), default="on")
    args = parser.parse_args()
    try:
        (run if args.mode == "run" else replay)(args)
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Legacy Full Blend stopped: {error}\n")


if __name__ == "__main__":
    os.umask(0o077)
    main()
