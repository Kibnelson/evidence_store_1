#!/usr/bin/env python3
"""Archive a complete MuSiQue cohort or pair without copying raw run inputs."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


COHORTS = ("2p2d-native", "2p2d-shared")
REQUESTS = 2417
MANIFEST = "archive-manifest.json"


def read_summary(path):
    contents = path.read_bytes()
    summary = json.loads(contents)
    if not isinstance(summary, dict) or summary.get("status") != "passed":
        raise ValueError(f"Expected a passed summary at {path}")
    return contents, summary


def read_cohort_summary(run_root, cohort, occurrences=None):
    name = Path(cohort) / "chunk-review" / "summary.json"
    contents, summary = read_summary(run_root / name)
    chunk_count = summary.get("chunk_occurrences")
    if (
        summary.get("labels") != ["musique"] or summary.get("requests") != REQUESTS
        or type(chunk_count) is not int or chunk_count <= 0
        or (occurrences is not None and chunk_count != occurrences)
        or summary.get("nixl_counter_reconciliation", {}).get("status") != "matched"
        or (cohort.endswith("-shared")
            and (summary.get("lmcache_counter_reconciliation") or {}).get("status") != "matched")
    ):
        raise ValueError(f"Expected verified MuSiQue request and transfer counts in {run_root / name}")
    return name, contents, summary


def verify_summaries(run_root):
    contents, repeat = read_summary(run_root / "repeat-summary.json")
    occurrences = repeat.get("chunk_occurrences_per_cohort")
    repeats = repeat.get("repeat_occurrences")
    if (
        repeat.get("dataset") != "musique"
        or repeat.get("cohorts") != list(COHORTS)
        or repeat.get("requests_per_cohort") != REQUESTS
        or type(occurrences) is not int
        or occurrences <= 0
        or type(repeats) is not int
        or not 0 <= repeats <= occurrences
    ):
        raise ValueError("Repeat summary does not describe the complete two-cohort MuSiQue replay")
    summaries = {Path("repeat-summary.json"): contents}
    for cohort in COHORTS:
        name, contents, _ = read_cohort_summary(run_root, cohort, occurrences)
        summaries[name] = contents
    return summaries, occurrences, repeats


def verify_single_cohort(run_root, cohort):
    name, contents, summary = read_cohort_summary(run_root, cohort)
    occurrences, repeats = summary["chunk_occurrences"], summary.get("repeat_occurrences")
    if type(repeats) is not int or not 0 <= repeats <= occurrences:
        raise ValueError(f"Invalid repeat count in {run_root / name}")
    root = run_root / cohort
    run = json.loads((root / "run_manifest.json").read_bytes())
    replay = json.loads((root / "client" / "musique" / "manifest.json").read_bytes())
    inputs = json.loads((run_root / "inputs" / "manifest.json").read_bytes())
    topology, mode = cohort.split("-")
    cache = run.get("lmcache")
    prefix_policy = "on" if mode == "shared" else "off"
    if (
        (run.get("run_id"), run.get("topology")) != (cohort, topology)
        or run.get("engine_trace_enabled") is not True
        or run.get("prefix_cache_by_role") != {"P": prefix_policy, "D": prefix_policy}
        or bool(cache) != (mode == "shared")
        or replay.get("mode") != "live_hnsw" or replay.get("trace_chunk_stages") is not True
        or replay.get("concurrency") != 1
        or tuple(replay.get(key) for key in ("requested", "succeeded", "frozen_rows", "failed"))
        != (REQUESTS, REQUESTS, REQUESTS, 0)
        or (inputs.get("dataset"), inputs.get("workload")) != ("musique", "full")
        or len(inputs.get("natural_queries", ())) != REQUESTS
        or len(set(inputs["natural_queries"])) != REQUESTS or inputs.get("probe_queries")
        or replay["input_sha256"] != inputs["queries_sha256"]
        or replay["index"]["corpus_sha256"] != inputs["corpus_sha256"]
        or replay["model"] != run["model"] or replay["tokenizer_revision"] != run["revision"]
    ):
        raise ValueError(f"Expected a completed live MuSiQue replay with matching provenance in {root}")
    provenance = {
        "topology": topology,
        "model": run["model"],
        "model_revision": run["revision"],
        "prefix_cache_by_role": run["prefix_cache_by_role"],
        "vllm_commit": run["vllm_source"]["commit"],
        "lmcache": {"mode": cache["mode"], "commit": cache["source"]["commit"]} if cache else None,
        "source_sha256": inputs["source_sha256"],
        "corpus_sha256": inputs["corpus_sha256"],
        "input_sha256": replay["input_sha256"],
        "frozen_sha256": replay["frozen_sha256"],
        "index_sha256": replay["index"]["index_sha256"],
    }
    summaries = {name: contents}
    dataset_path = Path(cohort) / "datasets" / "musique" / "summary.json"
    if (run_root / dataset_path).exists():
        contents, dataset = read_summary(run_root / dataset_path)
        if (
            (dataset["campaign"], dataset["cohort"], dataset["dataset"], dataset["requests"])
            != (run_root.name, cohort, "musique", REQUESTS)
            or dataset["evidence"]["occurrences"] != occurrences
            or dataset["counter_reconciliation"]["status"] != "matched"
            or any(dataset[field] != provenance[field] for field in ("input_sha256", "frozen_sha256"))
            or any(dataset["provenance"][field] != provenance[field] for field in (
                "topology", "model", "model_revision", "prefix_cache_by_role", "vllm_commit", "lmcache",
            ))
        ):
            raise ValueError(f"Dataset summary differs from the completed chunk review at {run_root / dataset_path}")
        summaries[dataset_path] = contents
    traced = [worker["trace"] for worker in summary["nixl_counter_reconciliation"]["workers"].values()]
    nixl_totals = {
        "count": sum(worker["count"] for worker in traced),
        "bytes": sum(worker["sum"] for worker in traced),
    }
    if nixl_totals["count"] <= 0 or nixl_totals["bytes"] <= 0:
        raise ValueError(f"Expected completed NIXL transfers in {root}")
    return summaries, occurrences, repeats, provenance, nixl_totals


def file_receipt(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size}


def write_jsonl_gzip(source, destination, expected, *, expected_nixl=None):
    raw_digest = hashlib.sha256()
    raw_bytes = count = 0
    nixl = {"count": 0, "bytes": 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream, destination.open("xb") as compressed:
        with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as writer:
            for count, line in enumerate(stream, start=1):
                row = json.loads(line)
                if not line.endswith(b"\n") or not isinstance(row, dict):
                    raise ValueError(f"Invalid JSONL record {count} in {source}")
                if expected_nixl is not None and row.get("path") == "nixl":
                    nixl["count"] += 1
                    nixl["bytes"] += row["completed_bytes"]
                raw_digest.update(line)
                raw_bytes += len(line)
                writer.write(line)
    if expected is not None and count != expected:
        raise ValueError(f"{source} contains {count} rows, expected {expected}")
    if expected_nixl is not None and nixl != expected_nixl:
        raise ValueError(f"NIXL transfer count or bytes in {source} differ from the reconciled summary")
    return {
        **file_receipt(destination),
        "source_sha256": raw_digest.hexdigest(),
        "source_bytes": raw_bytes,
        "rows": count,
    }


def archive_musique(run_root: Path, target: Path, *, cohort=None) -> dict:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Archive already exists: {target}")
    provenance = nixl_totals = None
    if cohort is None:
        selected = COHORTS
        summaries, occurrences, repeats = verify_summaries(run_root)
        reports = {Path("repeated-chunks.jsonl"): repeats}
    else:
        if cohort not in COHORTS:
            raise ValueError(f"Unknown MuSiQue cohort: {cohort}")
        selected = (cohort,)
        summaries, occurrences, repeats, provenance, nixl_totals = verify_single_cohort(run_root, cohort)
        reports = {}
    for name in selected:
        folder = Path(name) / "chunk-review"
        reports[folder / "requests.jsonl"] = REQUESTS
        reports[folder / "chunks.jsonl"] = occurrences
        reports[folder / "kv-transfers.jsonl"] = None

    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".musique-archive-", dir=target.parent) as temporary:
        staged = Path(temporary) / "archive"
        staged.mkdir()
        files = {}
        for name, contents in summaries.items():
            destination = staged / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
            files[str(name)] = file_receipt(destination)
        for name, count in reports.items():
            compressed = name.with_suffix(".jsonl.gz")
            files[str(compressed)] = write_jsonl_gzip(
                run_root / name, staged / compressed, count,
                expected_nixl=nixl_totals if name.name == "kv-transfers.jsonl" else None,
            )
        manifest = {
            "schema_version": 1,
            "run": run_root.name,
            "dataset": "musique",
            "cohorts": list(selected),
            "requests_per_cohort": REQUESTS,
            "chunk_occurrences_per_cohort": occurrences,
            "repeat_occurrences": repeats,
            **({"provenance": provenance} if provenance is not None else {}),
            "files": files,
        }
        (staged / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Archive already exists: {target}")
        staged.rename(target)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Path to a new review archive directory")
    parser.add_argument("--cohort", choices=COHORTS, help="Archive a single complete cohort")
    args = parser.parse_args()
    target = args.output.expanduser().absolute()
    try:
        manifest = archive_musique(args.run_root.resolve(), target, cohort=args.cohort)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"MuSiQue archive failed: {error}\n")
    print(f"Archived {manifest['repeat_occurrences']} repeated occurrences from "
          f"{', '.join(manifest['cohorts'])} in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
