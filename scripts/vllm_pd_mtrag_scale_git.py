"""Publish exact capture archives as bounded Git payloads on the results branch."""

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory


PART_BYTES = 64 * 1024**2
PARTS_PER_PUSH = 4
BUFFER_BYTES = 4 * 1024**2


class GitArchives:
    def __init__(self, checkpoints, scratch):
        self.checkpoints = checkpoints
        self.scratch = scratch
        self.campaign = checkpoints.branch.removeprefix("refs/heads/results/")
        self.fetch_url = checkpoints.git("remote", "get-url", "origin").stdout.decode().strip()
        self.push_url = checkpoints.git("remote", "get-url", "--push", "origin").stdout.decode().strip()

    def _workspace(self, directory):
        # A fresh shallow, filtered repo keeps previous payload objects out of this workspace.
        repo = type(self.checkpoints)(Path(directory), self.checkpoints.root, self.campaign,
                                      self.fetch_url, self.push_url)
        repo.refresh()
        return repo

    @staticmethod
    def _remote_blob(repo, git_path):
        entry = repo.git("ls-tree", "-z", "FETCH_HEAD", "--", git_path).stdout
        if not entry:
            return None
        match = re.fullmatch(rb"100644 blob ([0-9a-f]{40}|[0-9a-f]{64})\t([^\x00]+)\x00", entry)
        if match is None or match[2].decode() != git_path:
            raise ValueError(f"Unexpected Git tree entry at {git_path}")
        return match[1].decode()

    def _part(self, path, digest, index, data, algorithm):
        git_path = f"{self.checkpoints.prefix}/payloads/{digest}/part-{index:04d}"
        git_hash = hashlib.new(algorithm, usedforsecurity=False)
        git_hash.update(f"blob {len(data)}\0".encode())
        git_hash.update(data)
        return {"id": git_path, "git_path": git_path, "git_blob": git_hash.hexdigest(),
                "name": f"{path.name}.part{index:04d}", "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()}

    def _push_group(self, repo, archive, path, digest, size, count, first, aggregate):
        parent = repo.git("rev-parse", "FETCH_HEAD").stdout.decode().strip()
        algorithm = repo.git("rev-parse", "--show-object-format").stdout.decode().strip()
        repo.git("read-tree", parent)
        parts = []
        added = 0
        for index in range(first, min(first + PARTS_PER_PUSH, count + 1)):
            length = min(PART_BYTES, size - (index - 1) * PART_BYTES)
            data = archive.read(length)
            if len(data) != length:
                raise ValueError("Capture archive shrank while uploading")
            aggregate.update(data)
            part = self._part(path, digest, index, data, algorithm)
            git_path, expected = part["git_path"], part["git_blob"]
            prior = self._remote_blob(repo, git_path)
            if prior is None:
                blob = repo.git("hash-object", "-w", "--stdin", data=data).stdout.decode().strip()
                if blob != expected:
                    raise ValueError(f"Git wrote an unexpected object ID for {git_path}")
                repo.git("update-index", "--add", "--cacheinfo", f"100644,{blob},{git_path}")
                added += 1
            elif prior != expected:
                raise ValueError(f"Published Git payload differs at {git_path}")
            parts.append(part)
            del data
        if added:
            tree = repo.git("write-tree", "--missing-ok").stdout.decode().strip()
            message = f"Archive MT-RAG {path.name} payload {first}–{first + len(parts) - 1}"
            commit = repo.git("commit-tree", tree, "-p", parent, "-m", message).stdout.decode().strip()
            print(f"Pushing Git payload parts {first}–{first + len(parts) - 1} of {count}",
                  file=sys.stderr, flush=True)
            repo.git("-c", "pack.threads=2", "-c", "pack.window=0", "push", "origin",
                     f"{commit}:{repo.branch}")
        return parts

    def upload_archive(self, path, digest, size):
        parts = []
        aggregate = hashlib.sha256()
        count = (size + PART_BYTES - 1) // PART_BYTES
        with path.open("rb") as archive:
            for first in range(1, count + 1, PARTS_PER_PUSH):
                with TemporaryDirectory(prefix="git-payload-push-", dir=self.scratch) as directory:
                    repo = self._workspace(directory)
                    group = self._push_group(repo, archive, path, digest, size, count, first, aggregate)
                self.checkpoints.refresh()
                for part in group:
                    self.verify(part)
                parts.extend(group)
            if archive.read(1) or aggregate.hexdigest() != digest:
                raise ValueError("Capture archive changed while uploading")
        return parts

    def _verify(self, repo, part):
        path = part["git_path"]
        pattern = re.escape(f"{self.checkpoints.prefix}/payloads/") + r"[0-9a-f]{64}/part-[0-9]{4,}"
        if (part["id"] != path or not re.fullmatch(pattern, path)
                or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", part["git_blob"])
                or not re.fullmatch(r"[0-9a-f]{64}", part["sha256"])
                or not 0 < part["bytes"] <= PART_BYTES):
            raise ValueError(f"Invalid Git payload checkpoint for {path}")
        if self._remote_blob(repo, path) != part["git_blob"]:
            raise ValueError(f"Git payload is absent or differs on the results branch at {path}")

    def verify(self, part):
        self._verify(self.checkpoints, part)

    def download(self, part, destination, aggregate):
        # Read each blob from a fresh remote-backed repository. Previous uploads and other parts
        # cannot supply the bytes or accumulate in this download's object store.
        with TemporaryDirectory(prefix="git-payload-read-", dir=self.scratch) as directory:
            repo = self._workspace(directory)
            self._verify(repo, part)
            env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
            env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
            args = ["git", "-C", str(repo.root), "cat-file", "blob", part["git_blob"]]
            digest = hashlib.sha256()
            size = 0
            with subprocess.Popen(args, stdout=subprocess.PIPE, env=env) as process:
                try:
                    for block in iter(lambda: process.stdout.read(BUFFER_BYTES), b""):
                        size += len(block)
                        if size > part["bytes"]:
                            raise ValueError(f"Too many bytes received for {part['git_path']}")
                        digest.update(block)
                        aggregate.update(block)
                        if destination is not None:
                            destination.write(block)
                    if process.wait() or size != part["bytes"] or digest.hexdigest() != part["sha256"]:
                        raise ValueError(f"Downloaded bytes differ for {part['git_path']}")
                finally:
                    if process.poll() is None:
                        process.kill()
