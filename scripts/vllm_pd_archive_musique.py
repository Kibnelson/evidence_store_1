#!/usr/bin/env python3
"""Archive the two-cohort MuSiQue review without copying raw run inputs."""

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
        name = Path(cohort) / "chunk-review" / "summary.json"
        contents, summary = read_summary(run_root / name)
        if (
            summary.get("labels") != ["musique"]
            or summary.get("requests") != REQUESTS
            or summary.get("chunk_occurrences") != occurrences
            or summary.get("nixl_counter_reconciliation", {}).get("status") != "matched"
            or (cohort.endswith("-shared")
                and (summary.get("lmcache_counter_reconciliation") or {}).get("status") != "matched")
        ):
            raise ValueError(f"Expected verified MuSiQue request and transfer counts in {run_root / name}")
        summaries[name] = contents
    return summaries, occurrences, repeats


def file_receipt(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size}


def write_jsonl_gzip(source, destination, expected):
    raw_digest = hashlib.sha256()
    raw_bytes = count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream, destination.open("xb") as compressed:
        with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as writer:
            for count, line in enumerate(stream, start=1):
                if not line.endswith(b"\n") or not isinstance(json.loads(line), dict):
                    raise ValueError(f"Invalid JSONL record {count} in {source}")
                raw_digest.update(line)
                raw_bytes += len(line)
                writer.write(line)
    if expected is not None and count != expected:
        raise ValueError(f"{source} contains {count} rows, expected {expected}")
    return {
        **file_receipt(destination),
        "source_sha256": raw_digest.hexdigest(),
        "source_bytes": raw_bytes,
        "rows": count,
    }


def archive_musique(run_root: Path, target: Path) -> dict:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Archive already exists: {target}")
    summaries, occurrences, repeats = verify_summaries(run_root)
    reports = {Path("repeated-chunks.jsonl"): repeats}
    for cohort in COHORTS:
        folder = Path(cohort) / "chunk-review"
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
            files[str(compressed)] = write_jsonl_gzip(run_root / name, staged / compressed, count)
        manifest = {
            "schema_version": 1,
            "run": run_root.name,
            "dataset": "musique",
            "cohorts": list(COHORTS),
            "requests_per_cohort": REQUESTS,
            "chunk_occurrences_per_cohort": occurrences,
            "repeat_occurrences": repeats,
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
    args = parser.parse_args()
    target = args.output.expanduser().absolute()
    try:
        manifest = archive_musique(args.run_root.resolve(), target)
    except (OSError, TypeError, ValueError) as error:
        parser.exit(1, f"MuSiQue archive failed: {error}\n")
    print(f"Archived {manifest['repeat_occurrences']} repeated occurrences and both cohort reports in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
