#!/usr/bin/env python3
"""Publish verified CLAPNQ raw page packets in bounded Git pushes."""

import argparse
import json
from pathlib import Path
import re

from vllm_pd_legacy_blend_archive import MANIFEST, ROOT, RUN_NAME, archive, read_json, write_json
from vllm_pd_legacy_page_payloads import PAYLOAD_DIRECTORY
from vllm_pd_publish_dataset import git


PAYLOAD_PART_BYTES = 1024 ** 3
MAX_BLOB_BYTES = 100 * 1024 ** 2
PUSH_TIMEOUT = 1800


def campaign_receipts(receipts):
    rows = [json.loads(line) for line in receipts.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    jobs = set()
    if not rows:
        raise ValueError("The campaign has no verified archive receipts")
    for row in rows:
        conversation = row["conversation_id"]
        source = Path(row["archive"]).resolve()
        run = source.name.removesuffix(".tar.gz")
        match = RUN_NAME.fullmatch(run)
        if (not re.fullmatch(r"[0-9a-f]{32}", conversation) or conversation in seen
                or source.parent != receipts.parent / conversation or not source.is_file()
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                or not match or match[1] != "mtrag-multi-turn-repeat"
                or not source.name.endswith(".tar.gz")):
            raise ValueError(f"Invalid or duplicate raw archive receipt for {conversation}")
        seen.add(conversation)
        jobs.add(match[2])
        row["run_id"] = run
    if len(jobs) != 1:
        raise ValueError("Use verified archives from one Slurm campaign")
    return jobs.pop(), rows


def push(branch):
    result = git("-c", "pack.threads=2", "-c", "pack.window=0", "push", "origin",
                 f"HEAD:refs/heads/{branch}", timeout=PUSH_TIMEOUT)
    if result.stderr.strip():
        print(result.stderr.strip(), flush=True)


def parts(entries):
    batch = []
    size = 0
    for path in entries:
        file_size = (ROOT / path).stat().st_size
        if file_size > MAX_BLOB_BYTES:
            raise ValueError(f"GitHub file limit exceeded for {path}")
        if batch and size + file_size > PAYLOAD_PART_BYTES:
            yield batch, size
            batch, size = [], 0
        batch.append(path)
        size += file_size
    if batch:
        yield batch, size


def commit_items(branch, files, message):
    expected = set(files)
    if staged_paths() - expected:
        raise ValueError("Publishing checkout contains files staged outside the next packet part")
    git("add", "--sparse", "--", *files, timeout=PUSH_TIMEOUT)
    if staged_paths() != expected:
        raise ValueError("Staged Git files do not match the packet part")
    git("commit", "-m", message, timeout=PUSH_TIMEOUT)
    push(branch)


def staged_paths():
    return set(filter(None, git("diff", "--cached", "--name-only", "-z").stdout.split("\x00")))


def _published_manifest(relative, digest):
    result = git("show", f"HEAD:{relative}/{MANIFEST}", check=False)
    if result.returncode:
        return False
    manifest = json.loads(result.stdout)
    if manifest["source_archive_sha256"] != digest or manifest.get("include_raw_kv_pages") is not True:
        raise ValueError(f"Published packet {relative} differs from this verified receipt")
    return True


def publish_packet(row, branch):
    relative = f"reports/smoke/{row['run_id']}"
    target = ROOT / relative
    if _published_manifest(relative, row["sha256"]):
        print(f"Already published {relative}", flush=True)
        return relative

    print(f"Verifying and staging {row['conversation_id']} turns {row['turn_window']}", flush=True)
    if archive(Path(row["archive"]), row["sha256"], include_raw_kv_pages=True) != target:
        raise ValueError(f"Unexpected archive destination {target}")
    sample = read_json(target / "1p3d/blend/provenance.json")["sample"]
    first, last = map(int, row["turn_window"].split(":"))
    trace = sample["kv_page_trace"]
    if (trace["conversation_id"] != row["conversation_id"] or trace["turns"] != list(range(first, last + 1))
            or len(trace["turns"]) != row["turns"]):
        raise ValueError(f"Staged provenance differs from the receipt for {row['conversation_id']}")

    tracked = set(git("ls-tree", "-r", "--name-only", "-z", "HEAD", "--", relative).stdout.split("\x00"))
    pending = [p for p in sorted(target.rglob("*")) if p.is_file() and p.relative_to(ROOT).as_posix() not in tracked]
    payloads = [p.relative_to(ROOT).as_posix() for p in pending if p.parent.name == PAYLOAD_DIRECTORY]
    metadata = [p.relative_to(ROOT).as_posix() for p in pending if p.parent.name != PAYLOAD_DIRECTORY]
    for index, (part, size) in enumerate(parts(payloads), 1):
        print(f"Pushing {row['run_id']} page payload part {index}, {size / 1024**2:.1f} MiB", flush=True)
        commit_items(branch, part, f"Archive {row['run_id']} raw KV pages part {index}")
    # The archive manifest marks a packet that can be verified and replayed.
    manifest = f"{relative}/{MANIFEST}"
    if manifest not in metadata:
        raise ValueError(f"Missing pending archive manifest for {relative}")
    for part, _ in parts([p for p in metadata if p != manifest] + [manifest]):
        commit_items(branch, part, f"Complete verified KV packet {row['run_id']}")
    print(f"Published {relative}", flush=True)
    evict_sparse_paths()
    return relative


def evict_sparse_paths():
    sparse = git("config", "--bool", "core.sparseCheckout", check=False)
    if sparse.stdout.strip() == "true":
        git("sparse-checkout", "reapply", timeout=PUSH_TIMEOUT)


def publish_campaign(receipts):
    job, rows = campaign_receipts(receipts.resolve(strict=True))
    branch = f"results/clapnq-{job}"
    if git("branch", "--show-current").stdout.strip() != branch:
        raise ValueError(f"Use the campaign publication checkout on branch {branch}")
    if git("diff", "--quiet", check=False).returncode:
        raise ValueError("The publication checkout has unstaged changes")
    print(f"Publishing {len(rows)} verified captures on {branch}", flush=True)
    # Retry a commit whose preceding push was interrupted before adding another part.
    push(branch)
    captures = []
    for row in rows:
        relative = publish_packet(row, branch)
        fields = ("conversation_id", "turn_window", "turns", "stores", "observed_loads",
                  "unique_raw_pages", "raw_capture_bytes", "sha256")
        captures.append({**{key: row[key] for key in fields}, "cohort": f"{relative}/1p3d/blend"})
    index = f"reports/campaigns/clapnq-{job}.json"
    summary = {"schema_version": 1, "slurm_job_id": job, "results_branch": branch, "captures": captures}
    committed = git("show", f"HEAD:{index}", check=False)
    prior = None if committed.returncode else json.loads(committed.stdout)
    if prior is not None and any(prior[key] != summary[key] for key in ("schema_version", "slurm_job_id", "results_branch")):
        raise ValueError(f"Published campaign index differs at {index}")
    if prior is not None and prior["captures"] != captures[:len(prior["captures"])]:
        raise ValueError(f"Published campaign captures are not a prefix of {receipts}")
    if prior != summary:
        path = ROOT / index
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, summary)
        commit_items(branch, [index], f"Index CLAPNQ campaign {job}")
    evict_sparse_paths()
    print(f"Published campaign {job} with {len(captures)} verified captures on {branch}\n{index}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, help="The campaign archives.jsonl on node scratch")
    args = parser.parse_args()
    try:
        publish_campaign(args.receipts)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Campaign publication stopped. {error}\n"
                       "Keep this publication checkout and rerun the same command to retry.\n")


if __name__ == "__main__":
    main()
