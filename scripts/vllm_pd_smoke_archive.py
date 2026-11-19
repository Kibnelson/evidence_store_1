#!/usr/bin/env python3
"""Archive a verified native/shared PD smoke pair or verify a pulled archive."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidencestore.retrieval import sha256_file  # noqa: E402


MANIFEST = "archive-manifest.json"
TOPOLOGIES = ("1p3d", "2p2d")
MODES = ("native", "shared")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_cohorts(root, topology):
    sample = read_json(root / "smoke/fiqa-sample/manifest.json")
    results = {}
    for mode in MODES:
        cohort = root / "smoke" / f"{topology}-{mode}"
        launch = read_json(cohort / "run_manifest.json")
        summary = read_json(cohort / "summary.json")
        receipt = read_json(cohort / "smoke-check.json")
        cleanup = read_json(cohort / "cleanup.json")
        if (launch["topology"] != topology or launch["run_id"] != cohort.name
                or (launch.get("lmcache") or {}).get("mode") != ("shared_pd" if mode == "shared" else None)):
            raise ValueError(f"Unexpected topology or cache mode at {cohort}")
        if (summary["issues"] != [] or summary["nixl_reconciliation"]["status"] != "matched"
                or receipt["status"] != "passed" or receipt["sample"] != sample
                or len(receipt["request_ids"]) != sample["requests"]
                or cleanup["remaining_process_groups"] != [] or cleanup["unverified_process_groups"] != []):
            raise ValueError(f"Incomplete or failed smoke at {cohort}")
        if mode == "shared" and (summary["lmcache"]["status"] != "captured"
                                 or not read_json(cohort / "lmcache-status.json")["verified"]):
            raise ValueError(f"Unverified shared LMCache counters or worker registration at {cohort}")
        replay_counts = []
        for label in ("controlled", "mtrag-smoke"):
            replay = read_json(cohort / "client" / label / "manifest.json")
            if replay["requested"] != replay["succeeded"] or replay["failed"] != 0:
                raise ValueError(f"Incomplete {label} replay at {cohort}")
            replay_counts.append(replay["succeeded"])
        if replay_counts[1] != sample["requests"] or sum(replay_counts) != summary["requests"]:
            raise ValueError(f"Smoke request counts disagree at {cohort}")
        results[mode] = {"controlled_requests": replay_counts[0], "fiqa_requests": replay_counts[1],
                         "nixl_counters": "matched", "vllm_commit": launch["vllm_source"]["commit"]}
    if results["native"]["vllm_commit"] != results["shared"]["vllm_commit"]:
        raise ValueError("Native and shared smoke used different vLLM source commits")
    return results


def collect(root, topology, check_root):
    paths = ["e5-model.txt", "python-packages.txt", f"smoke/controlled-{topology}.jsonl",
             "smoke/fiqa-sample/manifest.json", "smoke/fiqa-sample/index/manifest.json"]
    for mode in MODES:
        cohort = root / "smoke" / f"{topology}-{mode}"
        files = ["run_manifest.json", "summary.json", "smoke-check.json", "cleanup.json",
                 "endpoints.json", "requests.jsonl", "timeline.json", "metrics-before.json", "metrics-after.json",
                 "client/controlled/manifest.json", "client/mtrag-smoke/manifest.json"]
        endpoints = read_json(cohort / "endpoints.json")
        for worker in endpoints["prefill"] + endpoints["decode"]:
            files.extend(f"native-metrics-{stage}-{worker['id']}.prom" for stage in ("before", "after"))
        if mode == "shared":
            files.extend(("lmcache-status.json", "lmcache-metrics-before.prom", "lmcache-metrics-after.prom"))
        traces = sorted((cohort / "trace").glob("*.jsonl"))
        if not traces:
            raise ValueError(f"No raw traces at {cohort}")
        paths.extend((cohort / name).relative_to(root) for name in files)
        paths.extend(path.relative_to(root) for path in traces)
    sources = {Path(name): root / name for name in paths}
    if check_root:
        for name in ("runtime.json", "python.txt", "cuda-toolkit.txt", "vllm-build.txt", "experiment-source.txt"):
            if (check_root / name).is_file():
                sources[Path("setup") / name] = check_root / name
        for path in sorted(check_root.glob("experiment-source-refresh-*.txt")):
            sources[Path("setup") / path.name] = path
    return sources


def fingerprint(root):
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink in archive: {path}")
        if path.is_file() and path != root / MANIFEST:
            files[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return files


def verify(root):
    manifest = read_json(root / MANIFEST)
    if manifest["schema_version"] != 1 or manifest["run_id"] != root.name or manifest["topology"] not in TOPOLOGIES:
        raise ValueError(f"Invalid smoke archive manifest at {root}")
    if fingerprint(root) != manifest["files"]:
        raise ValueError(f"Files differ from their archived sizes or SHA-256 hashes at {root}")
    if check_cohorts(root, manifest["topology"]) != manifest["cohorts"]:
        raise ValueError(f"Cohort receipts differ from the archive manifest at {root}")
    return manifest


def archive(root, topology, check_root):
    results = check_cohorts(root, topology)
    sources = collect(root, topology, check_root)
    target = ROOT / "reports/smoke" / root.name
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as directory:
        staged = Path(directory) / root.name
        for name, source in sources.items():
            parent = check_root if name.parts[0] == "setup" else root
            if not source.resolve().is_relative_to(parent) or source.is_symlink() or not source.is_file():
                raise ValueError(f"Expected a regular smoke artifact at {source}")
            destination = staged / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        manifest = {"schema_version": 1, "run_id": root.name, "topology": topology,
                    "cohorts": results, "files": fingerprint(staged)}
        (staged / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        verify(staged)
        if target.exists():
            if verify(target) != manifest:
                raise FileExistsError(f"A different archive already exists at {target}")
        else:
            staged.rename(target)
    return target.relative_to(ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run-root", type=Path, help="create a tracked archive from a completed scratch run")
    action.add_argument("--verify", type=Path, help="check a published archive and print its cohort receipts")
    parser.add_argument("--topology", choices=TOPOLOGIES, default="1p3d")
    parser.add_argument("--check-root", type=Path, help="include setup provenance from the activated runtime")
    args = parser.parse_args()
    try:
        if args.verify:
            verified = verify(args.verify.resolve())
            verified["files"] = len(verified["files"])
            print(json.dumps(verified, indent=2))
        else:
            setup = args.check_root.resolve() if args.check_root else None
            print(archive(args.run_root.resolve(), args.topology, setup))
    except (OSError, ValueError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
