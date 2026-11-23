#!/usr/bin/env python3
"""Run cumulative MT-RAG milestones using the already published campaign plan."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from vllm_pd_mtrag_scale_publish import GitCheckpoints, canonical, command


ROOT = Path(__file__).resolve().parents[1]
MILESTONES = (1, 20, 40, 60, 80, 104)
DEFAULT_REPO = "Kibnelson/evidence_store_1"


def owned_directory(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir() or path.resolve() != path or path.stat().st_uid != os.getuid():
        raise ValueError(f"Expected an owned scratch directory at {path}")
    return path


def frozen_plan(checkpoints, args):
    checkpoints.refresh()
    plan = checkpoints.read(checkpoints.plan_path)
    if (not plan or plan["schema_version"] != 1 or plan["campaign"] != args.campaign
            or not re.fullmatch(r"[0-9a-f]{40}", plan["source_commit"])
            or hashlib.sha256(canonical(plan["plan"])).hexdigest() != plan["plan_sha256"]
            or len(plan["plan"]["conversations"]) < args.through):
        raise ValueError("The published plan does not cover the requested campaign milestones")
    backend = plan.get("archive_backend", "release")
    if (backend == "git-ssh" and plan["archive_repo"] == args.repo
            or backend == "release" and plan["release_repo"] == args.repo):
        return plan, backend
    raise ValueError("The published plan names an incompatible archive destination")


def pinned_checkout(source, fetch_url, commit, plan_file):
    if not (source / ".git").is_dir():
        command(["git", "init", "--quiet", str(source)])
    git = ["git", "-C", str(source)]
    if command([*git, "cat-file", "-e", f"{commit}^{{commit}}"], check=False).returncode:
        command([*git, "fetch", "--depth=1", "--no-tags", fetch_url, commit])
    subprocess.run([*git, "checkout", "--detach", commit], check=True)
    subprocess.run([sys.executable, "-B", str(source / "scripts/vllm_pd_mtrag_scale_select.py"),
                    "--output", str(plan_file)], cwd=source, check=True)


def verify_checkpoint(receipt, plan, through):
    if (receipt["through"] != through or receipt["campaign"] != plan["campaign"]
            or receipt["plan_sha256"] != plan["plan_sha256"]
            or receipt["source_commit"] != plan["source_commit"]
            or receipt["totals"]["conversations"] != through):
        raise ValueError(f"Published milestone {through} disagrees with the campaign plan")


def run(args, scratch):
    fetch = f"https://github.com/{args.repo}.git"
    push = f"git@github.com:{args.repo}.git"
    milestones = MILESTONES[:MILESTONES.index(args.through) + 1]
    with TemporaryDirectory(prefix="milestone-refs-", dir=scratch) as workspace:
        checkpoints = GitCheckpoints(Path(workspace), ROOT, args.campaign, fetch, push)
        plan, backend = frozen_plan(checkpoints, args)
        print(f"Campaign {args.campaign} · {backend} · source {plan['source_commit']}", flush=True)
        source = scratch / f"mtrag-scale-source-{args.campaign}"
        plan_file = scratch / f"{args.campaign}.plan.json"
        initialized = False
        for through in milestones:
            relative = f"{checkpoints.prefix}/milestones/{through:03d}.json"
            completed = checkpoints.read(relative)
            if completed is None:
                if not initialized:
                    pinned_checkout(owned_directory(source), fetch, plan["source_commit"], plan_file)
                    initialized = True
                print(f"Running through {through} conversations", flush=True)
                runner = [sys.executable, "-B", str(source / "scripts/vllm_pd_mtrag_scale_run.py"),
                          "--campaign", args.campaign, "--release-repo", args.repo,
                          "--archive-backend", backend, "--plan", str(plan_file),
                          "--through", str(through), "--capture-mib", str(args.capture_mib)]
                if args.read_back:
                    runner.append("--read-back")
                subprocess.run(runner, cwd=source, check=True)
                subprocess.run([sys.executable, "-B", str(ROOT / "scripts/vllm_pd_mtrag_scale_milestone.py"),
                                "--campaign", args.campaign, "--through", str(through), "--repo", args.repo,
                                "--scratch-dir", str(scratch), "--git-fetch-url", fetch, "--git-push-url", push],
                               cwd=ROOT, stdout=subprocess.DEVNULL, check=True)
                checkpoints.refresh()
                completed = checkpoints.read(relative)
                if completed is None:
                    raise ValueError(f"Milestone {through} is missing from the results branch after publication")
            verify_checkpoint(completed, plan, through)
            print(f"Milestone {through} verified · {completed['totals']['turns']} turns", flush=True)
            print(f"https://github.com/{args.repo}/blob/results/{args.campaign}/{relative}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, help="Campaign ID from a verified pilot")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub owner/repository")
    parser.add_argument("--through", type=int, choices=MILESTONES, default=104,
                        help="Last cumulative milestone (default: 104)")
    parser.add_argument("--capture-mib", type=int, default=6144, help="Raw page capture MiB per conversation")
    parser.add_argument("--read-back", action="store_true", help="Read and hash every newly published archive")
    args = parser.parse_args()
    if (not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", args.campaign)
            or not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", args.repo) or args.capture_mib < 1):
        parser.error("Use a valid campaign ID, owner/repository, and positive capture size")
    if (not re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")) or
            os.environ.get("SLURM_JOB_NUM_NODES", os.environ.get("SLURM_NNODES", "1")) != "1"):
        parser.error("Run inside a single-node four-GPU Slurm allocation")
    os.umask(0o077)
    try:
        user = command(["id", "-un"]).stdout.decode().strip()
        scratch = owned_directory(Path("/tmp") / user)
        lock = scratch / f"{args.campaign}.milestones.lock"
        with lock.open("a", encoding="utf-8") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"Another milestone launcher owns {lock}") from error
            run(args, scratch)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Milestone loop stopped. {error}\nRerun this command to resume.\n")


if __name__ == "__main__":
    main()
