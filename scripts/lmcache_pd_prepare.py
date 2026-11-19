#!/usr/bin/env python3
"""Restore a repo-local LMCache fork, or export the instrumented PD fork."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from vllm_pd_prepare import THIRD_PARTY, git, require_ancestor, require_local_repo, sha256


SEED = THIRD_PARTY / "checkouts/lmcache"
FORK = THIRD_PARTY / "forks/lmcache"
BUNDLE = THIRD_PARTY / "lmcache-pd.bundle"
RECEIPT = THIRD_PARTY / "lmcache-pd-fork.json"
BRANCH = "evidencestore/native-pd"
REF = f"refs/heads/{BRANCH}"
LEGACY_SEED = SEED.with_name("lmcache-legacy-base")
LEGACY_FORK = FORK.with_name("lmcache-full-blend")
LEGACY_BUNDLE = THIRD_PARTY / "lmcache-full-blend.bundle"
LEGACY_RECEIPT = THIRD_PARTY / "lmcache-full-blend-fork.json"
LEGACY_REF = "refs/heads/evidencestore/legacy-full-blend"


def seed_pin(seed=SEED):
    entries = json.loads((THIRD_PARTY / "sources.json").read_text())["sources"]
    pin = next(item["commit"] for item in entries if item["name"] == seed.name)
    if not isinstance(pin, str) or not re.fullmatch(r"[0-9a-f]{40}", pin):
        raise ValueError("LMCache needs a full Git source pin")
    require_local_repo(seed)
    if (
        git(seed, "rev-parse", "HEAD").stdout.strip() != pin
        or git(seed, "symbolic-ref", "--quiet", "HEAD", check=False).returncode != 1
        or git(seed, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    ):
        raise ValueError("The LMCache seed must be clean and detached at the upstream pin")
    return pin


def require_bundle_head(bundle, source, *, seed=SEED, ref=REF):
    heads = git(seed, "bundle", "list-heads", bundle).stdout.splitlines()
    if len(heads) != 1 or heads[0].split() != [source, ref]:
        raise ValueError(f"The LMCache bundle must contain exactly {source} {ref}")


def require_fork(path, pin):
    require_local_repo(path)
    require_ancestor(path, pin, "LMCache source does not descend from the upstream pin")
    if git(path, "symbolic-ref", "--quiet", "HEAD", check=False).stdout.strip() != REF:
        raise ValueError(f"Switch the LMCache fork to {BRANCH}")
    if git(path, "status", "--porcelain").stdout.strip():
        raise ValueError("Commit or remove local LMCache fork changes before preparing it")
    return git(path, "rev-parse", "HEAD").stdout.strip()


def restore(pin):
    receipt = json.loads(RECEIPT.read_text())
    source = receipt["source_commit"]
    if (
        receipt["base_commit"] != pin or receipt["ref"] != REF
        or not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{40}", source)
        or source == pin or sha256(BUNDLE) != receipt["bundle_sha256"]
    ):
        raise ValueError("The LMCache fork receipt, source pin, and bundle digest disagree")
    require_bundle_head(BUNDLE, source)
    git(SEED, "bundle", "verify", BUNDLE)

    if FORK.exists() or FORK.is_symlink():
        local = require_fork(FORK, pin)
        if local != source:
            git(FORK, "fetch", "--quiet", "--no-tags", BUNDLE, REF)
            git(FORK, "merge", "--quiet", "--ff-only", source)
        if require_fork(FORK, pin) != source:
            raise ValueError("The local LMCache HEAD differs from the bundle receipt")
        return

    FORK.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".lmcache-pd-", dir=FORK.parent) as tmp:
        checkout = Path(tmp) / "repo"
        checkout.mkdir()
        git(checkout, "init", "--quiet")
        git(checkout, "fetch", "--quiet", "--no-tags", "--depth=1", SEED.as_uri(), pin)
        git(checkout, "switch", "--quiet", "--detach", pin)
        git(checkout, "bundle", "verify", BUNDLE)
        git(checkout, "fetch", "--quiet", "--no-tags", BUNDLE, f"{REF}:{REF}")
        git(checkout, "switch", "--quiet", BRANCH)
        if require_fork(checkout, pin) != source:
            raise ValueError("The restored LMCache HEAD differs from the bundle receipt")
        checkout.rename(FORK)


def export(pin):
    source = require_fork(FORK, pin)
    if source == pin:
        raise ValueError("The local LMCache fork has no committed changes")
    with TemporaryDirectory(prefix=".lmcache-pd-export-", dir=THIRD_PARTY) as tmp:
        bundle = Path(tmp) / BUNDLE.name
        receipt = Path(tmp) / RECEIPT.name
        git(FORK, "bundle", "create", bundle, REF, f"^{pin}")
        require_bundle_head(bundle, source)
        git(SEED, "bundle", "verify", bundle)
        receipt.write_text(json.dumps({
            "base_commit": pin, "source_commit": source, "ref": REF, "bundle_sha256": sha256(bundle),
        }, indent=2) + "\n")
        bundle.replace(BUNDLE)
        receipt.replace(RECEIPT)


def require_legacy_fork(path, pin, source, tree):
    require_local_repo(path)
    revisions = git(path, "rev-parse", "HEAD", "HEAD^{tree}", "HEAD^").stdout.splitlines()
    if (
        revisions != [source, tree, pin]
        or git(path, "symbolic-ref", "--quiet", "HEAD", check=False).returncode != 1
        or git(path, "status", "--porcelain").stdout.strip()
    ):
        raise ValueError(f"Expected the clean detached Full Blend snapshot at {path}")


def restore_legacy(pin):
    receipt = json.loads(LEGACY_RECEIPT.read_text())
    source, tree = receipt["source_commit"], receipt["historical_tree"]
    if (
        receipt["base_commit"] != pin or receipt["ref"] != LEGACY_REF
        or sha256(LEGACY_BUNDLE) != receipt["bundle_sha256"]
    ):
        raise ValueError("The legacy Full Blend receipt, source pin, and bundle digest disagree")
    require_bundle_head(LEGACY_BUNDLE, source, seed=LEGACY_SEED, ref=LEGACY_REF)
    git(LEGACY_SEED, "bundle", "verify", LEGACY_BUNDLE)

    if LEGACY_FORK.exists() or LEGACY_FORK.is_symlink():
        require_legacy_fork(LEGACY_FORK, pin, source, tree)
        return

    LEGACY_FORK.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".lmcache-full-blend-", dir=LEGACY_FORK.parent) as tmp:
        checkout = Path(tmp) / "repo"
        checkout.mkdir()
        git(checkout, "init", "--quiet")
        git(checkout, "fetch", "--quiet", "--no-tags", "--depth=1", LEGACY_SEED.as_uri(), pin)
        git(checkout, "fetch", "--quiet", "--no-tags", LEGACY_BUNDLE, LEGACY_REF)
        git(checkout, "switch", "--quiet", "--detach", "FETCH_HEAD")
        require_legacy_fork(checkout, pin, source, tree)
        checkout.rename(LEGACY_FORK)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export", action="store_true", help="Package committed local PD fork changes")
    mode.add_argument("--legacy-full-blend", action="store_true", help="Restore the detached job 29538 source")
    args = parser.parse_args()
    try:
        if args.legacy_full_blend:
            restore_legacy(seed_pin(LEGACY_SEED))
        else:
            (export if args.export else restore)(seed_pin())
    except (KeyError, OSError, StopIteration, ValueError, subprocess.CalledProcessError) as exc:
        print(getattr(exc, "stderr", "") or str(exc), file=sys.stderr)
        return 1
    print(LEGACY_FORK if args.legacy_full_blend else FORK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
