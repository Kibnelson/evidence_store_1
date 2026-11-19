#!/usr/bin/env python3
"""Materialize or verify the pinned upstream seed checkouts."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "third_party"
CHECKOUTS = SOURCE_ROOT / "checkouts"
GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}


def git(path, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=check,
        env=GIT_ENV,
        stdout=subprocess.PIPE,
        text=True,
    )


def verify(path, source):
    if path.is_symlink() or not (path / ".git").is_dir():
        raise ValueError(f"{path} is not a standalone source checkout")
    actual = git(path, "rev-parse", "HEAD").stdout.strip()
    if actual != source["commit"]:
        raise ValueError(f"{path.name} is at {actual}, expected {source['commit']}")
    if git(path, "symbolic-ref", "--quiet", "HEAD", check=False).returncode != 1:
        raise ValueError(f"{path.name} must have a detached HEAD")
    if git(path, "remote").stdout.strip():
        raise ValueError(f"{path.name} has a configured Git remote")
    if git(path, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise ValueError(f"{path.name} has modified tracked files")


def checkout(source, verify_only):
    name, commit = source["name"], source["commit"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError(f"Invalid source name or Git commit for {name}")
    destination = CHECKOUTS / name
    if destination.exists() or destination.is_symlink() or verify_only:
        verify(destination, source)
    else:
        CHECKOUTS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=CHECKOUTS) as temp:
            staged = Path(temp) / name
            staged.mkdir()
            git(staged, "init", "--quiet")
            git(staged, "fetch", "--quiet", "--no-tags", "--depth=1", source["url"], commit)
            git(staged, "switch", "--quiet", "--detach", "FETCH_HEAD")
            verify(staged, source)
            staged.rename(destination)
    print(f"{name}: {commit} (detached, no remotes, tracked files clean)", flush=True)


def main():
    sources = json.loads((SOURCE_ROOT / "sources.json").read_text(encoding="utf-8"))["sources"]
    selections = {source["name"]: source for source in sources}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify locally without fetching")
    parser.add_argument("sources", nargs="*", metavar="SOURCE", help="source name, or all if omitted")
    args = parser.parse_args()
    names = args.sources or list(selections)
    unknown = [name for name in names if name not in selections]
    if unknown:
        parser.error(f"unknown sources: {', '.join(unknown)}")

    failed = False
    for name in names:
        try:
            checkout(selections[name], args.verify)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print(f"{name}: {error}", flush=True)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
