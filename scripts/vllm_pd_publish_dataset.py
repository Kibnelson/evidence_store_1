#!/usr/bin/env python3
"""Commit and push a completed dataset checkpoint from the shared checkout."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
COHORTS = {"1p3d-native", "1p3d-shared", "2p2d-native", "2p2d-shared"}
GIT_TIMEOUT = 35


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=8")
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"git {args[0]} timed out after {GIT_TIMEOUT}s") from error
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {args[0]} failed")
    return result


def identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid {field} identifier")
    return value


def archive_report(source: Path) -> str:
    report = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("Expected a dataset report with schema_version 1")
    if report.get("status") != "passed":
        raise ValueError("Expected a passed dataset report")
    reconciliation = report.get("counter_reconciliation")
    if not isinstance(reconciliation, dict) or reconciliation.get("status") != "matched":
        raise ValueError("Expected matched dataset-window counter reconciliation")

    campaign = identifier(report.get("campaign"), "campaign")
    cohort = identifier(report.get("cohort"), "cohort")
    if cohort not in COHORTS:
        raise ValueError(f"Unknown PD cohort {cohort}")
    dataset = report.get("dataset")
    if not isinstance(dataset, str):
        raise ValueError("Expected a dataset identifier")
    for part in dataset.split("/"):
        identifier(part, "dataset")
    label = dataset.replace("/", "-")
    if report.get("label") != label:
        raise ValueError(f"Expected dataset label {label}")

    relative_path = f"reports/datasets/{campaign}/{label}/{cohort}.json"
    target = ROOT / relative_path
    if target.resolve() != target:
        raise ValueError(f"Dataset archive path contains a symlink: {target}")
    contents = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != contents:
            raise ValueError(f"A different dataset checkpoint already exists at {target}")
    else:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            staging_path = Path(temporary.name)
            try:
                temporary.write(contents)
                temporary.close()
                staging_path.replace(target)
            finally:
                staging_path.unlink(missing_ok=True)

    git("add", "--", relative_path)
    staged = git("diff", "--cached", "--quiet", "--", relative_path, check=False)
    if staged.returncode == 1:
        git("commit", "--only", "-m", f"Archive PD dataset {campaign} {label} {cohort}", "--", relative_path)
    elif staged.returncode != 0:
        raise RuntimeError(staged.stderr.strip() or f"Unable to inspect staged changes for {relative_path}")
    print(f"Dataset checkpoint archived at {target}", flush=True)
    return campaign


def push_campaign(campaign: str) -> None:
    head = git("rev-parse", "HEAD").stdout.strip()
    git("fetch", "--no-tags", "origin", "refs/heads/main")
    remote = git("rev-parse", "FETCH_HEAD").stdout.strip()
    if git("merge-base", "--is-ancestor", remote, head, check=False).returncode:
        raise RuntimeError("origin/main and local main have diverged. Synchronize them and retry publication.")

    campaign_directory = f"reports/datasets/{campaign}"
    unrelated = git("log", "--full-history", "--format=%H", f"{remote}..{head}", "--", ".",
                    f":(exclude){campaign_directory}").stdout
    if unrelated.strip():
        raise RuntimeError(f"Local main has unpublished changes outside {campaign_directory}")
    if head != remote:
        git("push", "--no-follow-tags", "origin", f"{head}:refs/heads/main")
    print(f"Published campaign {campaign} dataset checkpoints through {head[:12]} to origin/main")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--report", type=Path, help="completed dataset summary.json to archive and push")
    actions.add_argument("--retry", metavar="CAMPAIGN", help="push queued commits for this campaign")
    args = parser.parse_args()

    try:
        git_dir = Path(git("rev-parse", "--absolute-git-dir").stdout.strip())
        if Path(git("rev-parse", "--show-toplevel").stdout.strip()).resolve() != ROOT:
            raise RuntimeError("Run the publisher from the shared EvidenceStore checkout")
        if git("symbolic-ref", "--quiet", "HEAD").stdout.strip() != "refs/heads/main":
            raise RuntimeError("The shared EvidenceStore checkout must be on main")
        with (git_dir / "vllm-pd-publish-dataset.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another dataset publication is in progress") from error
            campaign = archive_report(args.report) if args.report else identifier(args.retry, "campaign")
            push_campaign(campaign)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Dataset publication pending: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
