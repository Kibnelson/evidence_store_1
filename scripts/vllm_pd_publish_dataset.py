#!/usr/bin/env python3
"""Commit and push dataset checkpoints or the detailed campaign review archive."""

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
PUSH_TIMEOUT = 300


def git(*args: str, check: bool = True, timeout: int = GIT_TIMEOUT) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=8")
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"git {args[0]} timed out after {timeout}s") from error
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {args[0]} failed")
    return result


def identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid {field} identifier")
    return value


def commit_path(path: str, message: str) -> None:
    git("add", "--", path)
    staged = git("diff", "--cached", "--quiet", "--", path, check=False)
    if staged.returncode == 1:
        git("commit", "--only", "-m", message, "--", path)
    elif staged.returncode != 0:
        raise RuntimeError(staged.stderr.strip() or f"Unable to inspect staged changes for {path}")


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

    commit_path(relative_path, f"Archive PD dataset {campaign} {label} {cohort}")
    print(f"Dataset checkpoint archived at {target}", flush=True)
    return campaign


def stage_details(target, build):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".details-", dir=target.parent) as tmp:
        staged = Path(tmp) / target.name
        build(staged)
        staged.rename(target)


def archive_details(source: Path, dataset: str | None = None) -> str:
    from vllm_pd_detail_archive import (
        DATASETS,
        MANIFEST,
        build_archive,
        build_dataset_archive,
        dataset_label,
        verify_archive,
        verify_dataset_archive,
    )

    source = source.resolve(strict=True)
    campaign = identifier(source.parent.name if source.name == "run" else source.name, "campaign")
    reports = ROOT / "reports" / "datasets" / campaign
    details = reports / "details"
    if details.resolve() != details:
        raise ValueError(f"Dataset archive path contains a symlink: {details}")
    if not reports.is_dir():
        raise ValueError(f"Pull the published campaign summaries before archiving details: {reports}")

    datasets = (dataset,) if dataset is not None else DATASETS
    labels = [dataset_label(name) for name in datasets]
    manifest = details / MANIFEST
    if manifest.exists() or manifest.is_symlink() or (dataset is None and not details.exists()):
        if details.exists():
            verify_archive(details, reports)
        else:
            stage_details(details, lambda staged: build_archive(source, reports, staged))
        commit_path(details.relative_to(ROOT).as_posix(), f"Archive PD campaign review details {campaign}")
        print(f"Campaign review details archived at {details}", flush=True)
        return campaign

    for name, label in zip(datasets, labels, strict=True):
        target = details / label
        if target.resolve() != target:
            raise ValueError(f"Dataset archive path contains a symlink: {target}")
        if target.exists():
            verify_dataset_archive(target, reports, name)
        else:
            stage_details(target, lambda staged: build_dataset_archive(source, reports, staged, name))
        commit_path(target.relative_to(ROOT).as_posix(), f"Archive PD dataset review details {campaign} {label}")
        print(f"Dataset review details archived at {target}", flush=True)
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
        git("push", "--no-follow-tags", "origin", f"{head}:refs/heads/main", timeout=PUSH_TIMEOUT)
    print(f"Published campaign {campaign} results through {head[:12]} to origin/main")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--report", type=Path, help="completed dataset summary.json to archive and push")
    actions.add_argument("--details", type=Path, metavar="RUN_ROOT", help="archive the completed run's review details")
    actions.add_argument("--retry", metavar="CAMPAIGN", help="push queued commits for this campaign")
    parser.add_argument("--dataset", metavar="NAME", help="limit --details to one dataset across all four cohorts")
    args = parser.parse_args()
    if args.dataset is not None and args.details is None:
        parser.error("--dataset requires --details")

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
            if args.report:
                campaign = archive_report(args.report)
            elif args.details:
                campaign = archive_details(args.details, args.dataset)
            else:
                campaign = identifier(args.retry, "campaign")
            push_campaign(campaign)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Dataset publication pending: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
