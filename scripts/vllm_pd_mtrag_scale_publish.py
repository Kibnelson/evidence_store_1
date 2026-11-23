#!/usr/bin/env python3
"""Publish exact MT-RAG archives and checkpoint them in Git."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid

from vllm_pd_mtrag_scale_git import GitArchives


RELEASE_REPO = "Kibnelson/evidence_store_1"
ARCHIVE_BACKENDS = ("release", "git-ssh")
PART_BYTES = 1900 * 1024**2
BUFFER_BYTES = 4 * 1024**2
CAPTURE_FIELDS = (
    "conversation_id", "domain", "turn_window", "turns", "stores", "observed_loads",
    "unique_raw_pages", "raw_capture_bytes", "source_commit",
)
SCHEMA = 1


def command(args, *, data=None, check=True):
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0", GH_PROMPT_DISABLED="1")
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    try:
        result = subprocess.run(args, input=data, capture_output=True, timeout=3600, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{args[0]} exceeded its one-hour timeout") from None
    if check and result.returncode:
        error = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"{args[0]} failed ({result.returncode}). {error}")
    return result


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(path.read_bytes())


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_id(conversation):
    if not re.fullmatch(r"[0-9a-f]{32}", conversation):
        raise ValueError(f"Invalid conversation ID {conversation!r}")


class GitCheckpoints:
    def __init__(self, scratch, repo_root, campaign, fetch_url, push_url):
        self.root = scratch / "git"
        self.branch = f"refs/heads/results/{campaign}"
        self.prefix = f"reports/campaigns/{campaign}"
        self.plan_path = f"{self.prefix}/plan.json"
        self.git("init", "--quiet", str(self.root), local=False)
        self.git("remote", "add", "origin", fetch_url)
        self.git("remote", "set-url", "--push", "origin", push_url)
        for key, last_commit_field in (("user.name", "%an"), ("user.email", "%ae")):
            config = command(["git", "-C", str(repo_root), "config", "--get", key], check=False)
            identity = config.stdout.decode().strip()
            if not identity:
                identity = command(["git", "-C", str(repo_root), "log", "-1",
                                    f"--format={last_commit_field}"]).stdout.decode().strip()
            self.git("config", key, identity)

    def git(self, *args, local=True, data=None, check=True):
        prefix = ["git", "-C", str(self.root)] if local else ["git"]
        return command([*prefix, *args], data=data, check=check)

    def refresh(self, *, allow_init=False):
        if not self.git("ls-remote", "--refs", "origin", self.branch).stdout.strip():
            if not allow_init:
                raise ValueError(f"Campaign branch {self.branch} is missing. Initialize the campaign first.")
            source = "refs/heads/main"
        else:
            source = self.branch
        # Only commit/tree metadata and explicitly requested small files enter this checkout.
        self.git("fetch", "--quiet", "--depth=1", "--filter=blob:none", "--no-tags", "origin", source)
        return source == self.branch

    def read(self, name):
        if not self.git("ls-tree", "FETCH_HEAD", "--", name).stdout.strip():
            return None
        return json.loads(self.git("show", f"FETCH_HEAD:{name}").stdout)

    def write(self, files, message):
        parent = self.git("rev-parse", "FETCH_HEAD").stdout.decode().strip()
        self.git("read-tree", parent)
        for path, value in files.items():
            blob = self.git("hash-object", "-w", "--stdin", data=canonical(value)).stdout.decode().strip()
            self.git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        # The parent is already on the remote. Its unchanged blobs stay in the remote's object store.
        tree = self.git("write-tree", "--missing-ok").stdout.decode().strip()
        commit = self.git("commit-tree", tree, "-p", parent, "-m", message).stdout.decode().strip()
        self.git("push", "origin", f"{commit}:{self.branch}")
        self.refresh()
        if any(self.read(path) != value for path, value in files.items()):
            raise ValueError("Remote Git checkpoint differs after push")

    def capture_path(self, conversation):
        return f"{self.prefix}/captures/{conversation}.json"

    def study_path(self, conversation):
        return f"{self.prefix}/studies/{conversation}.json"


class Releases:
    def __init__(self, repo, tag):
        self.repo = repo
        self.tag = tag
        self.api_path = f"repos/{repo}/releases"

    def lookup(self):
        result = command(["gh", "api", f"{self.api_path}/tags/{self.tag}"], check=False)
        if result.returncode and b"(HTTP 404)" in result.stderr:
            return None
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace").strip())
        return json.loads(result.stdout)

    def ensure_release(self, source_commit):
        release = self.lookup()
        if release is None:
            command(["gh", "release", "create", self.tag, "--repo", self.repo, "--target", source_commit,
                     "--title", f"MT-RAG KV capture {self.tag}", "--prerelease",
                     "--notes", f"Exact MT-RAG KV capture archives for campaign {self.tag}."])
            release = self.lookup()
        if (not release or release["tag_name"] != self.tag or release["draft"]
                or self.tag_commit() != source_commit):
            raise ValueError(f"Release {self.repo}/{self.tag} does not match the pinned source commit")
        return release["id"]

    def tag_commit(self):
        root = f"repos/{self.repo}/git"
        ref = json.loads(command(["gh", "api", f"{root}/ref/tags/{self.tag}"]).stdout)["object"]
        for _ in range(8):
            if ref["type"] == "commit":
                return ref["sha"]
            if ref["type"] != "tag":
                break
            ref = json.loads(command(["gh", "api", f"{root}/tags/{ref['sha']}"]).stdout)["object"]
        raise ValueError(f"Tag {self.tag} does not resolve to a commit")

    def assets(self, release_id):
        result = command(["gh", "api", "--paginate", f"{self.api_path}/{release_id}/assets?per_page=100",
                          "--jq", ".[]"])
        assets = map(json.loads, result.stdout.splitlines())
        return {asset["name"]: asset for asset in assets}

    def asset(self, asset_id):
        return json.loads(command(["gh", "api", f"{self.api_path}/assets/{asset_id}"]).stdout)

    def upload(self, path, release_id, size, digest, existing):
        names = re.compile(re.escape(path.name) + r"(?:\.retry-[0-9a-f]{12})?")
        for name, asset in existing.items():
            if names.fullmatch(name) and asset["state"] != "starter":
                return artifact(asset, name, size, digest)
        upload_path = path
        if path.name in existing:
            # A failed upload can leave an empty starter asset. Keep it and retry under a fresh name.
            upload_path = path.with_name(f"{path.name}.retry-{uuid.uuid4().hex[:12]}")
            upload_path.symlink_to(path)
        upload = command(["gh", "release", "upload", self.tag, str(upload_path), "--repo", self.repo],
                         check=False)
        # Check the remote even after a failed response, which can arrive after upload completion.
        existing.update(self.assets(release_id))
        asset = existing.get(upload_path.name)
        if upload.returncode and (not asset or asset["state"] == "starter"):
            raise RuntimeError(upload.stderr.decode(errors="replace").strip())
        return artifact(asset, upload_path.name, size, digest)

    def verify(self, part):
        remote = self.asset(part["id"])
        if artifact(remote, part["name"], part["bytes"], part["sha256"]) != part:
            raise ValueError(f"Remote asset identity changed for {part['name']}")

    def download(self, part, destination, aggregate):
        args = ["gh", "release", "download", self.tag, "--repo", self.repo,
                "--pattern", part["name"], "--output", "-"]
        digest = hashlib.sha256()
        size = 0
        with subprocess.Popen(args, stdout=subprocess.PIPE, env={**os.environ, "GH_PROMPT_DISABLED": "1"}) as process:
            try:
                for block in iter(lambda: process.stdout.read(BUFFER_BYTES), b""):
                    size += len(block)
                    if size > part["bytes"]:
                        raise ValueError(f"Too many bytes received for {part['name']}")
                    digest.update(block)
                    aggregate.update(block)
                    if destination is not None:
                        destination.write(block)
                if process.wait() or size != part["bytes"] or digest.hexdigest() != part["sha256"]:
                    raise ValueError(f"Downloaded bytes differ for {part['name']}")
            finally:
                if process.poll() is None:
                    process.kill()


def artifact(remote, name, size, digest):
    if (not remote or remote["name"] != name or remote["state"] != "uploaded"
            or remote["size"] != size or remote.get("digest") != f"sha256:{digest}"):
        raise ValueError(f"GitHub asset {name} is missing, incomplete, or has a different SHA-256")
    link = remote["browser_download_url"]
    return {"id": remote["id"], "name": name, "bytes": size, "sha256": digest, "url": link}


def validate_plan(plan):
    seen = set()
    if not plan["conversations"]:
        raise ValueError("Capture plan is empty")
    for row in plan["conversations"]:
        conversation = row["conversation_id"]
        ensure_id(conversation)
        if (conversation in seen or row["domain"] not in {"clapnq", "cloud", "fiqa", "govt"}
                or not re.fullmatch(r"1:[1-9][0-9]*", row["turn_window"])):
            raise ValueError(f"Invalid or duplicate plan entry for {conversation}")
        seen.add(conversation)


def plan_entry(plan, conversation):
    ensure_id(conversation)
    for row in plan["plan"]["conversations"]:
        if row["conversation_id"] == conversation:
            return row
    raise ValueError(f"Conversation {conversation} is outside the frozen plan")


def validate_capture(plan, capture):
    row = plan_entry(plan, capture["conversation_id"])
    if (capture["domain"] != row["domain"] or capture["turn_window"] != row["turn_window"]
            or capture["turns"] != int(row["turn_window"].split(":")[1])
            or capture["source_commit"] != plan["source_commit"]):
        raise ValueError(f"Capture for {capture['conversation_id']} differs from its plan entry")


def study_hash(study, capture, digest):
    if study is None:
        return None
    replayed = study["input"]["capture"]
    fields = ("conversation_id", "turn_window", "turns", "stores", "observed_loads",
              "unique_raw_pages", "raw_capture_bytes")
    if replayed["sha256"] != digest or any(replayed[field] != capture[field] for field in fields):
        raise ValueError(f"CPU study describes a different archive for {capture['conversation_id']}")
    return hashlib.sha256(canonical(study)).hexdigest()


def init_campaign(args, checkpoints):
    plan = read_json(args.plan)
    validate_plan(plan)
    oid = command(["git", "-C", str(args.repo_root), "rev-parse", "--verify",
                   f"{args.source_commit or 'HEAD'}^{{commit}}"]).stdout.decode().strip()
    result = {"schema_version": SCHEMA, "campaign": args.campaign, "source_commit": oid,
              "plan_sha256": hashlib.sha256(canonical(plan)).hexdigest(), "plan": plan}
    initialized = checkpoints.refresh(allow_init=True)
    previous = checkpoints.read(checkpoints.plan_path) if initialized else None
    if args.archive_backend == "git-ssh":
        result.update(archive_backend="git-ssh", archive_repo=args.release_repo)
        if initialized and previous != result:
            raise ValueError("Campaign already exists with a different plan, source commit, or archive target")
    else:
        result.update(release_repo=args.release_repo, release_tag=args.campaign)
        if initialized and (not previous or previous != {**result, "release_id": previous.get("release_id")}):
            raise ValueError("Campaign already exists with a different plan, source commit, or archive target")
        result["release_id"] = Releases(args.release_repo, args.campaign).ensure_release(oid)
        if initialized and previous != result:
            raise ValueError("The initialized campaign's Release was replaced")
    if not initialized:
        checkpoints.write({checkpoints.plan_path: result}, f"Initialize KV capture campaign {args.campaign}")
    return result


def get_checkpoint(checkpoints, archives, plan, conversation, *, metadata_only=False):
    plan_entry(plan, conversation)
    value = checkpoints.read(checkpoints.capture_path(conversation))
    if value is None:
        return None
    validate_capture(plan, value)
    if (value["conversation_id"] != conversation or value["schema_version"] != SCHEMA
            or value["plan_sha256"] != plan["plan_sha256"]):
        raise ValueError(f"Remote checkpoint does not match plan entry {conversation}")
    study = checkpoints.read(checkpoints.study_path(conversation))
    if study_hash(study, value, value["archive"]["sha256"]) != value["study_sha256"]:
        raise ValueError(f"Remote CPU study differs for {conversation}")
    archive = value["archive"]
    parts = archive["parts"]
    if (not parts or len({part["id"] for part in parts}) != len(parts)
            or not re.fullmatch(r"[0-9a-f]{64}", archive["sha256"])
            or sum(part["bytes"] for part in parts) != archive["bytes"]):
        raise ValueError(f"Remote archive receipt is inconsistent for {conversation}")
    if plan.get("archive_backend") == "git-ssh":
        prefix = f"{checkpoints.prefix}/payloads/{archive['sha256']}"
        if any(part["git_path"] != f"{prefix}/part-{position:04d}" for position, part in enumerate(parts, 1)):
            raise ValueError(f"Remote Git archive parts are out of order for {conversation}")
    if not metadata_only:
        for part in parts:
            archives.verify(part)
    return value


def stage_part(stream, length, path, aggregate):
    digest = hashlib.sha256()
    with path.open("xb") as output:
        remaining = length
        while remaining:
            block = stream.read(min(remaining, BUFFER_BYTES))
            if not block:
                raise ValueError("Capture archive shrank while uploading")
            digest.update(block)
            aggregate.update(block)
            output.write(block)
            remaining -= len(block)
    return digest.hexdigest()


def upload_archive(path, digest, size, scratch, releases, release_id):
    existing = releases.assets(release_id)
    prefix = f"{path.name.removesuffix('.tar.gz')}-{digest}"
    if size <= PART_BYTES:
        alias = scratch / f"{prefix}.tar.gz"
        alias.symlink_to(path)
        return [releases.upload(alias, release_id, size, digest, existing)]
    parts = []
    aggregate = hashlib.sha256()
    total_parts = (size + PART_BYTES - 1) // PART_BYTES
    with path.open("rb") as archive:
        for index in range(total_parts):
            size_part = min(PART_BYTES, size - index * PART_BYTES)
            part = scratch / f"{prefix}.tar.gz.part{index + 1:04d}"
            part_digest = stage_part(archive, size_part, part, aggregate)
            parts.append(releases.upload(part, release_id, size_part, part_digest, existing))
            part.unlink()
        if archive.read(1) or aggregate.hexdigest() != digest:
            raise ValueError("Capture archive changed while uploading")
    return parts


def read_back(archives, archive, destination=None):
    aggregate = hashlib.sha256()
    for part in archive["parts"]:
        archives.download(part, destination, aggregate)
    if aggregate.hexdigest() != archive["sha256"]:
        raise ValueError("The ordered remote parts do not match the complete archive SHA-256")


def publish(args, checkpoints, archives, plan, scratch):
    receipt = read_json(args.receipt)
    validate_capture(plan, receipt)
    conversation = receipt["conversation_id"]
    study = read_json(args.study_report) if args.study_report else None
    study_digest = study_hash(study, receipt, receipt["sha256"])
    prior = get_checkpoint(checkpoints, archives, plan, conversation)
    if prior is not None:
        if prior["archive"]["sha256"] != receipt["sha256"] or prior["study_sha256"] != study_digest:
            raise ValueError(f"A different capture is already verified for {conversation}")
        if args.read_back:
            read_back(archives, prior["archive"])
        return prior

    path = Path(receipt["archive"]).resolve(strict=True)
    if not re.fullmatch(r"mtrag-multi-turn-repeat-legacy-full-blend-\d+-\d{8}T\d{6}Z\.tar\.gz", path.name):
        raise ValueError(f"Unexpected MT-RAG archive name {path.name}")
    size = path.stat().st_size
    digest = receipt["sha256"]
    if not path.is_file() or not size or sha256_file(path) != digest:
        raise ValueError(f"Archive {path} does not match the verified capture receipt")

    print(f"Uploading {conversation} capture ({size / 1024**2:.1f} MiB)", file=sys.stderr, flush=True)
    if plan.get("archive_backend") == "git-ssh":
        parts = archives.upload_archive(path, digest, size)
    else:
        parts = upload_archive(path, digest, size, scratch, archives, plan["release_id"])
    archive = {"filename": path.name, "bytes": size, "sha256": digest, "parts": parts}
    if args.read_back:
        read_back(archives, archive)

    value = {**{field: receipt[field] for field in CAPTURE_FIELDS}, "schema_version": SCHEMA,
             "plan_sha256": plan["plan_sha256"], "study_sha256": study_digest, "archive": archive}
    # Refresh after the upload so a sibling checkpoint that finished first is kept.
    checkpoints.refresh()
    prior = get_checkpoint(checkpoints, archives, plan, conversation)
    if prior is not None:
        if prior != value:
            raise ValueError(f"A different capture was published concurrently for {conversation}")
        return prior
    files = {checkpoints.capture_path(conversation): value}
    if study is not None:
        files[checkpoints.study_path(conversation)] = study
    checkpoints.write(files, f"Checkpoint MT-RAG KV capture {conversation}")
    return get_checkpoint(checkpoints, archives, plan, conversation)


def download(archives, checkpoint, output):
    archive = checkpoint["archive"]
    if output.exists():
        if output.stat().st_size != archive["bytes"] or sha256_file(output) != archive["sha256"]:
            raise ValueError(f"Existing output {output} differs from the verified archive")
        return
    with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", dir=output.parent) as staging:
        read_back(archives, archive, staging)
        staging.flush()
        os.link(staging.name, output)


def run(args, scratch):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", args.campaign):
        raise ValueError("Campaign IDs must contain lowercase ASCII letters, digits, and hyphens")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", args.release_repo):
        raise ValueError("Expected a GitHub owner/repository")
    checkpoints = GitCheckpoints(scratch, args.repo_root, args.campaign,
                                 args.git_fetch_url or f"https://github.com/{args.release_repo}.git",
                                 args.git_push_url or f"git@github.com:{args.release_repo}.git")
    if args.command == "init":
        return init_campaign(args, checkpoints)
    checkpoints.refresh()
    plan = checkpoints.read(checkpoints.plan_path)
    if (not plan or plan["schema_version"] != SCHEMA or plan["campaign"] != args.campaign
            or plan["plan_sha256"] != hashlib.sha256(canonical(plan["plan"])).hexdigest()):
        raise ValueError("Remote plan does not match this campaign or GitHub repository")
    backend = plan.get("archive_backend", "release")
    if args.archive_backend is not None and args.archive_backend != backend:
        raise ValueError(f"Campaign {args.campaign} uses archive backend {backend}")
    if backend == "git-ssh" and plan["archive_repo"] == args.release_repo:
        archives = GitArchives(checkpoints, scratch)
    elif backend == "release" and plan["release_repo"] == args.release_repo and plan["release_tag"] == args.campaign:
        archives = Releases(args.release_repo, args.campaign)
    else:
        raise ValueError("Remote plan does not match this campaign or GitHub repository")
    if args.command == "plan":
        return plan
    if args.command == "publish":
        return publish(args, checkpoints, archives, plan, scratch)
    checkpoint = get_checkpoint(checkpoints, archives, plan, args.conversation,
                                metadata_only=getattr(args, "metadata_only", False))
    if checkpoint is not None and args.command == "download":
        download(archives, checkpoint, args.output)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--release-repo", default=RELEASE_REPO, help="GitHub owner/repository for publication")
    parser.add_argument("--archive-backend", choices=ARCHIVE_BACKENDS,
                        help="Use Release assets (init default) or Git over SSH")
    parser.add_argument("--scratch-dir", type=Path, help="Temporary Git metadata and archive parts")
    parser.add_argument("--git-fetch-url", help="Override the HTTPS fetch URL for a private repo or local test")
    parser.add_argument("--git-push-url", help="Override the SSH push URL")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Initialize or verify the campaign plan and archive destination")
    init.add_argument("--plan", required=True, type=Path)
    init.add_argument("--source-commit", help="Pin source commit, defaulting to --repo-root HEAD")
    commands.add_parser("plan", help="Read the frozen campaign plan from the results branch")
    status = commands.add_parser("status", help="Verify one published checkpoint, exiting 3 if absent")
    status.add_argument("--conversation", required=True)
    status.add_argument("--metadata-only", action="store_true", help="Read the Git receipt without verifying archive parts")
    upload = commands.add_parser("publish", help="Publish and verify a conversation archive and CPU study")
    upload.add_argument("--receipt", required=True, type=Path)
    upload.add_argument("--study-report", type=Path)
    upload.add_argument("--read-back", action="store_true", help="Download and hash the bytes before checkpointing")
    restore = commands.add_parser("download", help="Download the exact archive and verify every digest")
    restore.add_argument("--conversation", required=True)
    restore.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix="mtrag-publish-", dir=args.scratch_dir) as work:
            value = run(args, Path(work).resolve())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"KV capture publication stopped. {error}\n")
    if value is None:
        parser.exit(3)
    print(canonical(value).decode(), end="")


if __name__ == "__main__":
    main()
