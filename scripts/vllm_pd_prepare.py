#!/usr/bin/env python3
"""Restore or export the locally committed native PD vLLM fork."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
SEED = THIRD_PARTY / "checkouts/vllm"
FORK = THIRD_PARTY / "forks/vllm"
BUNDLE = THIRD_PARTY / "vllm-native-pd.bundle"
RECEIPT = THIRD_PARTY / "vllm-fork.json"
BRANCH = "evidencestore/native-pd"
REF = f"refs/heads/{BRANCH}"
LEGACY_SEED = SEED.with_name("vllm-legacy-base")
LEGACY_FORK = FORK.with_name("vllm-full-blend")
LEGACY_BUNDLE = THIRD_PARTY / "vllm-full-blend.bundle"
LEGACY_RECEIPT = THIRD_PARTY / "vllm-full-blend-fork.json"


def git(root, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(root), *map(str, args)],
        check=check,
        text=True,
        capture_output=True,
    )


def source_pin(seed=SEED):
    entries = json.loads((THIRD_PARTY / "sources.json").read_text())["sources"]
    pin = next((item["commit"] for item in entries if item["name"] == seed.name), None)
    if not isinstance(pin, str) or len(pin) != 40 or any(char not in "0123456789abcdef" for char in pin):
        raise ValueError(f"The source manifest needs a full Git commit pin for {seed.name}")
    return pin


def require_local_repo(path):
    if path.is_symlink() or not (path / ".git").is_dir():
        raise ValueError(f"Expected an independent local Git checkout at {path}")
    if git(path, "remote").stdout.strip():
        raise ValueError(f"The source checkout at {path} must have zero remotes")


def require_ancestor(path, ancestor, message):
    if git(path, "merge-base", "--is-ancestor", ancestor, "HEAD", check=False).returncode:
        raise ValueError(f"{message}: {path}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_bundle_head(repo, bundle, source, *, ref=REF):
    heads = git(repo, "bundle", "list-heads", bundle).stdout.splitlines()
    if len(heads) != 1 or heads[0].split() != [source, ref]:
        raise ValueError(f"The source bundle must contain exactly {source} {ref}")


def require_clean_detached(path, *revisions):
    require_local_repo(path)
    objects = ("HEAD", "HEAD^{tree}", "HEAD^")[:len(revisions)]
    if (
        git(path, "rev-parse", *objects).stdout.splitlines() != list(revisions)
        or git(path, "symbolic-ref", "--quiet", "HEAD", check=False).returncode != 1
        or git(path, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    ):
        raise ValueError(f"Expected a clean detached checkout at the recorded revision in {path}")


def restore_detached_fork(seed, fork, bundle, receipt_path):
    pin = source_pin(seed)
    require_clean_detached(seed, pin)
    receipt = json.loads(receipt_path.read_text())
    source, tree, ref = receipt["source_commit"], receipt["historical_tree"], receipt["ref"]
    if receipt["base_commit"] != pin or sha256(bundle) != receipt["bundle_sha256"]:
        raise ValueError(f"The source receipt, base pin, and bundle digest disagree for {fork.name}")
    require_bundle_head(seed, bundle, source, ref=ref)
    git(seed, "bundle", "verify", bundle)
    if fork.exists() or fork.is_symlink():
        require_clean_detached(fork, source, tree, pin)
        return

    fork.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{fork.name}-", dir=fork.parent) as tmp:
        checkout = Path(tmp) / "repo"
        checkout.mkdir()
        git(checkout, "init", "--quiet")
        git(checkout, "fetch", "--quiet", "--no-tags", "--depth=1", seed.as_uri(), pin)
        git(checkout, "fetch", "--quiet", "--no-tags", bundle, ref)
        git(checkout, "switch", "--quiet", "--detach", "FETCH_HEAD")
        require_clean_detached(checkout, source, tree, pin)
        checkout.rename(fork)


def prepare(pin):
    require_clean_detached(SEED, pin)

    receipt = json.loads(RECEIPT.read_text())
    if receipt["base_commit"] != pin or receipt["ref"] != REF:
        raise ValueError("The vLLM fork receipt does not match the upstream pin and local branch")
    if sha256(BUNDLE) != receipt["bundle_sha256"]:
        raise ValueError("The vLLM bundle does not match its recorded SHA-256")
    source = receipt["source_commit"]
    require_bundle_head(SEED, BUNDLE, source)

    if FORK.exists() or FORK.is_symlink():
        require_local_repo(FORK)
        require_ancestor(FORK, pin, "Existing source does not descend from the upstream pin")
        if git(FORK, "merge-base", "--is-ancestor", source, "HEAD", check=False).returncode == 0:
            return
        if git(FORK, "symbolic-ref", "--quiet", "HEAD", check=False).stdout.strip() != REF:
            raise ValueError(f"Switch the editable fork to {BRANCH} before updating from the bundle")
        if git(FORK, "status", "--porcelain").stdout.strip():
            raise ValueError("Commit local vLLM changes before updating from the bundle")
        git(FORK, "bundle", "verify", BUNDLE)
        git(FORK, "fetch", "--quiet", "--no-tags", BUNDLE, REF)
        if git(FORK, "merge-base", "--is-ancestor", "HEAD", source, check=False).returncode:
            raise ValueError("The local vLLM branch has diverged from the bundled source")
        git(FORK, "merge", "--quiet", "--ff-only", source)
        return

    FORK.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".vllm-", dir=FORK.parent) as tmp:
        checkout = Path(tmp) / "repo"
        checkout.mkdir()
        git(checkout, "init", "--quiet")
        git(checkout, "fetch", "--quiet", "--no-tags", "--depth=1", SEED.as_uri(), pin)
        git(checkout, "switch", "--quiet", "--detach", pin)
        git(checkout, "bundle", "verify", BUNDLE)
        git(checkout, "fetch", "--quiet", "--no-tags", BUNDLE, f"{REF}:{REF}")
        git(checkout, "switch", "--quiet", BRANCH)
        require_local_repo(checkout)
        require_ancestor(checkout, pin, "Bundled source does not descend from the upstream pin")
        if (
            git(checkout, "rev-parse", "HEAD").stdout.strip() != source
            or git(checkout, "symbolic-ref", "HEAD").stdout.strip() != REF
        ):
            raise ValueError("The restored vLLM source does not match the recorded commit")
        checkout.rename(FORK)


def export(pin):
    require_local_repo(FORK)
    require_ancestor(FORK, pin, "The editable fork must descend from the upstream pin")
    if git(FORK, "symbolic-ref", "--quiet", "HEAD", check=False).stdout.strip() != REF:
        raise ValueError(f"Switch the editable fork to {BRANCH} before exporting")
    if git(FORK, "status", "--porcelain").stdout.strip():
        raise ValueError("Commit all vLLM source changes before exporting the fork")
    source = git(FORK, "rev-parse", "HEAD").stdout.strip()
    if source == pin:
        raise ValueError("The editable fork has no source changes beyond the upstream pin")

    with TemporaryDirectory(prefix=".vllm-export-", dir=THIRD_PARTY) as tmp:
        temp_bundle = Path(tmp) / BUNDLE.name
        temp_receipt = Path(tmp) / RECEIPT.name
        git(FORK, "bundle", "create", temp_bundle, REF, f"^{pin}")
        require_bundle_head(FORK, temp_bundle, source)
        receipt = {
            "base_commit": pin,
            "source_commit": source,
            "ref": REF,
            "bundle_sha256": sha256(temp_bundle),
        }
        temp_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        temp_bundle.replace(BUNDLE)
        temp_receipt.replace(RECEIPT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export", action="store_true", help="Package committed changes from the local PD fork")
    mode.add_argument("--legacy-full-blend", action="store_true", help="Restore the detached Full Blend vLLM source")
    args = parser.parse_args()
    try:
        if args.legacy_full_blend:
            restore_detached_fork(LEGACY_SEED, LEGACY_FORK, LEGACY_BUNDLE, LEGACY_RECEIPT)
        else:
            (export if args.export else prepare)(source_pin())
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(getattr(exc, "stderr", "") or str(exc), file=sys.stderr)
        return 1
    print(LEGACY_FORK if args.legacy_full_blend else FORK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
