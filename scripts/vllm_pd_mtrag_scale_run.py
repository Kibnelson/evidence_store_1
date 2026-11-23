#!/usr/bin/env python3
"""Capture, replay, and publish a frozen MT-RAG queue one conversation at a time."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import mkdtemp, NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DOMAINS = {"clapnq", "cloud", "fiqa", "govt"}
WINDOW = re.compile(r"1:[1-9][0-9]*")
CONVERSATION = re.compile(r"[0-9a-f]{32}")
RUNTIME = re.compile(r"es-legacy-blend-[0-9]+\.[A-Za-z0-9]+")
RUN = re.compile(r"mtrag-multi-turn-repeat-legacy-full-blend-[0-9]+-[0-9]{8}T[0-9]{6}Z")
CAPACITIES = (96, 128, 160, 192, 224, 256, 384, 512, 768, 1024, 1536, 2048, 3072)


def write_receipt(path, contents):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(contents, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_receipt(path):
    return json.loads(path.read_text(encoding="utf-8"))


def command(*argv, output=False, allow_absent=False, env=None):
    result = subprocess.run(argv, cwd=ROOT, text=True, stdout=subprocess.PIPE if output else None,
                            check=False, env=env)
    if allow_absent and result.returncode == 3:
        return None
    result.check_returncode()
    return result.stdout.strip() if output else None


def script(name, *argv, **options):
    return command(sys.executable, "-B", SCRIPTS / name, *argv, **options)


def plan_entries(path, through):
    entries = read_receipt(path)["conversations"]
    seen = set()
    for entry in entries:
        conversation = entry["conversation_id"]
        if (not CONVERSATION.fullmatch(conversation) or conversation in seen
                or entry["domain"] not in DOMAINS or not WINDOW.fullmatch(entry["turn_window"])):
            raise ValueError(f"Invalid or duplicate conversation in {path}: {entry}")
        seen.add(conversation)
    if not 1 <= through <= len(entries):
        raise ValueError(f"--through must be between 1 and {len(entries)}")
    return entries[:through]


def owned_directory(path):
    path.mkdir(mode=0o700, exist_ok=True)
    if path.resolve() != path or not path.is_dir() or path.stat().st_uid != os.getuid():
        raise ValueError(f"Expected an owned directory with no symlink components at {path}")
    return path


def check_runtime(runtime, node_root, source_commit):
    if (runtime.parent != node_root or not RUNTIME.fullmatch(runtime.name)
            or runtime.resolve() != runtime or runtime.stat().st_uid != os.getuid()
            or not (runtime / "activate.sh").is_file() or not (runtime / "check/runtime.json").is_file()
            or (runtime / "check/experiment-source.txt").read_text().strip() != source_commit):
        raise ValueError(f"Expected a verified runtime for source {source_commit} at {runtime}")
    return runtime


def get_runtime(state, node_root, source_commit):
    pointer = state / "runtime.json"
    if pointer.is_file():
        runtime = Path(read_receipt(pointer)["runtime"])
        if runtime.exists():
            return check_runtime(runtime, node_root, source_commit)
    activation = state / "runtime-activation.txt"
    command("bash", SCRIPTS / "vllm_pd_legacy_blend_runtime.sh", activation)
    runtime = Path(activation.read_text().strip()).parent
    check_runtime(runtime, node_root, source_commit)
    write_receipt(pointer, {"runtime": str(runtime)})
    return runtime


def capture(entry, case, runtime, plan, source_commit, budget):
    attempt = Path(mkdtemp(prefix="capture-", dir=case))
    conversation, window, domain = (entry[key] for key in ("conversation_id", "turn_window", "domain"))
    env = {**os.environ, "EVIDENCESTORE_LEGACY_RUNTIME": str(runtime),
           "EVIDENCESTORE_LEGACY_REPORT_DIR": str(attempt),
           "EVIDENCESTORE_KV_PAGE_CAPTURE_MIB": str(budget)}
    env.pop("MTRAG_PREPARED", None)
    command("bash", "-o", "pipefail", "-c", 'log=$1; shift; bash "$@" 2>&1 | tee "$log"',
            "mtrag-capture", attempt / "launch.log", SCRIPTS / "vllm_pd_legacy_blend_run.sh",
            "1b3d", "on", "mtrag-multi-turn-repeat", "previous-turn-first", "--trace-kv-pages",
            "--capture-kv-pages", "--page-conversation", conversation, "--page-turns", window,
            "--mtrag-scale-plan", plan, "--mtrag-domain", domain, env=env)
    receipt = script("vllm_pd_legacy_page_campaign.py", "record", runtime / "repo", attempt,
                     conversation, window, str(budget), output=True)
    receipt = {**json.loads(receipt), "domain": domain, "source_commit": source_commit,
               "runtime": str(runtime), "capture_budget_mib": budget}
    write_receipt(case / "capture.json", receipt)
    return receipt


def local_paths(receipt, case, node_root, source_commit):
    archive = Path(receipt["archive"])
    run_name = archive.name.removesuffix(".tar.gz")
    runtime = Path(receipt["runtime"])
    run = runtime / "repo/reports/runs" / run_name
    if (receipt["source_commit"] != source_commit or archive.parent.parent != case
            or not archive.parent.name.startswith("capture-") or archive.suffixes[-2:] != [".tar", ".gz"]
            or not RUN.fullmatch(run_name) or runtime.parent != node_root or not RUNTIME.fullmatch(runtime.name)
            or archive.resolve() != archive or run.resolve() != run):
        raise ValueError(f"Local capture paths are outside the conversation's own scratch: {archive}")
    return archive, run


def check_entry(receipt, entry, source_commit):
    keys = ("conversation_id", "domain", "turn_window")
    if any(receipt[key] != entry[key] for key in keys) or receipt["source_commit"] != source_commit:
        raise ValueError(f"Capture does not match the frozen plan entry for {entry['conversation_id']}")


def clean_capture(archive, run):
    if run.exists():
        cleanup = read_receipt(run / "1p3d/blend/cleanup.json")
        if cleanup["remaining_process_groups"] or cleanup["unverified_process_groups"]:
            raise ValueError(f"Cannot reclaim a run with unverified worker cleanup: {run}")
        shutil.rmtree(run)
    archive.unlink(missing_ok=True)
    print(f"Scratch reclaimed for {archive.name}", flush=True)


def replay(receipt, receipt_path, archive, run_dir):
    if run_dir.is_dir():
        source = ("--capture-receipt", receipt_path, "--cohort", run_dir / "1p3d/blend")
    else:
        source = ("--archive", archive, "--sha256", receipt["sha256"],
                  "--conversation", receipt["conversation_id"], "--turn-window", receipt["turn_window"],
                  "--scratch-dir", archive.parent)
    study_dir = archive.parent / "cpu"
    script("vllm_pd_legacy_exponent_replay.py", *source, "--output-dir", study_dir,
           "--capacity-mib", *map(str, CAPACITIES))
    return study_dir / f"{receipt['conversation_id']}.json"


def run(args, entries):
    if not re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")) or (
        os.environ.get("SLURM_JOB_NUM_NODES", os.environ.get("SLURM_NNODES", "1")) != "1"
    ):
        raise ValueError("Run inside a single-node four-GPU Slurm allocation")
    source_commit = command("git", "rev-parse", "HEAD", output=True)
    command("git", "diff", "--quiet", "HEAD", "--")
    command("git", "ls-files", "--error-unmatch", "--", *(f"scripts/{name}" for name in (
        "vllm_pd_mtrag_scale_run.py", "vllm_pd_mtrag_scale_publish.py", "vllm_pd_mtrag_scale_select.py",
        "vllm_pd_mtrag_scale_git.py",
        "vllm_pd_legacy_exponent_replay.py", "vllm_pd_legacy_exponent_codec.py", "vllm_pd_legacy_exponent_codec.c",
    )), output=True)
    user = command("id", "-un", output=True)
    node_root = owned_directory(Path("/tmp") / user)
    state = owned_directory(node_root / f"mtrag-scale-{args.campaign}")
    print(f"Campaign scratch {state}\nSource {source_commit}", flush=True)
    command("df", "-h", state)

    def publish(*argv, **options):
        return script("vllm_pd_mtrag_scale_publish.py", "--repo-root", ROOT, "--campaign", args.campaign,
                      "--release-repo", args.release_repo, "--archive-backend", args.archive_backend, *argv,
                      env={**os.environ, "TMPDIR": str(state)}, **options)

    with (state / "run.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another process already owns {state / 'run.lock'}") from error
        command(sys.executable, "-B", "-c",
                "import sys\nsys.path.insert(0, sys.argv[1])\n"
                "from vllm_pd_legacy_exponent_codec import _field_library, _zstd_library\n"
                "_field_library()\n_zstd_library(None)", SCRIPTS)
        publish("init", "--plan", args.plan, "--source-commit", source_commit, output=True)
        for position, entry in enumerate(entries, 1):
            conversation = entry["conversation_id"]
            print(f"[{position}/{len(entries)}] {entry['domain']} {conversation} turns {entry['turn_window']}",
                  flush=True)
            case = owned_directory(state / conversation)
            receipt_path = case / "capture.json"
            receipt = read_receipt(receipt_path) if receipt_path.is_file() else None
            checkpoint = publish("status", "--conversation", conversation, output=True, allow_absent=True)
            if checkpoint is None:
                if receipt is not None:
                    check_entry(receipt, entry, source_commit)
                    archive, _ = local_paths(receipt, case, node_root, source_commit)
                    if not archive.is_file():
                        print(f"Local archive disappeared for {conversation}. Starting a new capture.", flush=True)
                        receipt = None
                if receipt is None:
                    runtime = get_runtime(state, node_root, source_commit)
                    receipt = capture(entry, case, runtime, args.plan, source_commit, args.capture_mib)
                check_entry(receipt, entry, source_commit)
                archive, run_dir = local_paths(receipt, case, node_root, source_commit)
                study_report = replay(receipt, receipt_path, archive, run_dir)
                options = ["--read-back"] if args.read_back else []
                checkpoint = publish("publish", "--receipt", receipt_path, "--study-report",
                                     study_report, *options, output=True)

            checkpoint = json.loads(checkpoint)
            check_entry(checkpoint, entry, source_commit)
            if not re.fullmatch(r"[0-9a-f]{64}", checkpoint.get("study_sha256") or ""):
                raise ValueError(f"Published checkpoint is missing a verified CPU study for {conversation}")
            if receipt is not None:
                check_entry(receipt, entry, source_commit)
                if checkpoint["archive"]["sha256"] != receipt["sha256"]:
                    raise ValueError(f"Local and published archives disagree for {conversation}")
            write_receipt(case / "published.json", checkpoint)
            if receipt is not None:
                archive, run_dir = local_paths(receipt, case, node_root, source_commit)
                if not args.keep_local and (archive.exists() or run_dir.exists()):
                    clean_capture(archive, run_dir)
            print(f"Verified published checkpoint for {conversation}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, help="Stable campaign ID for results and archives")
    parser.add_argument("--release-repo", default="Kibnelson/evidence_store_1")
    parser.add_argument("--archive-backend", choices=("release", "git-ssh"), default="release",
                        help="Publish raw archives as Release assets or split Git blobs over SSH")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--through", type=int, default=20, help="Stop after this entry in the frozen queue")
    parser.add_argument("--capture-mib", type=int, default=6144, help="Raw page capture MiB per conversation")
    parser.add_argument("--read-back", action="store_true", help="Download new archives to verify the full SHA-256")
    parser.add_argument("--keep-local", action="store_true", help="Keep archives and run trees after publication")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected queue without Slurm or GitHub")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", args.campaign) or args.capture_mib < 1:
        parser.error("Use a lowercase campaign ID and a positive --capture-mib")
    args.plan = args.plan.resolve()
    os.umask(0o077)
    try:
        entries = plan_entries(args.plan, args.through)
        if args.dry_run:
            for entry in entries:
                print(entry["domain"], entry["conversation_id"], entry["turn_window"], sep="\t")
        else:
            run(args, entries)
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Campaign stopped: {error}\nPending scratch is retained.\n")


if __name__ == "__main__":
    main()
