#!/usr/bin/env python3
"""Run a native vLLM PD cohort on four GPUs in an existing Slurm allocation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
SOURCE = ROOT / "third_party/forks/vllm"
LMCACHE_SOURCE = ROOT / "third_party/checkouts/lmcache"
LMCACHE_TRACE_SOURCE = ROOT / "third_party/forks/lmcache"
TOPOLOGIES = {"1p3d": (1, 3), "2p2d": (2, 2)}
# All cohort traffic stays on the allocated node, even when HTTP_PROXY is set.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
LMCACHE_WORKER_ID = re.compile(r"LMCache MP worker adapter created with instance_id=(\d+)")
BLEND_MODULE = "lmcache_cacheblend.connector"
BLEND_CONNECTOR = "CBKVConnector"
BLEND_SETTINGS = {"cb.check_layer": 1, "cb.recomp_ratio": 0.15}


@dataclass(frozen=True)
class Worker:
    id: str
    role: str
    gpu: str
    port: int
    nixl_port: int

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}"


@dataclass
class Child:
    name: str
    process: subprocess.Popen
    log: Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--topology", choices=TOPOLOGIES, default="1p3d")
    parser.add_argument("--prefix-cache", choices=("off", "on"), default="off")
    parser.add_argument("--trace", choices=("off", "on"), default="on", help="gate native source instrumentation")
    parser.add_argument("--minimum-nixl-reads", type=int, default=0,
                        help="wait for this many new decode reads after a successful workload")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--max-model-len", type=int, default=10240)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--port", type=int, default=8000, help="router port")
    parser.add_argument("--worker-port-base", type=int, default=8100)
    parser.add_argument("--nixl-port-base", type=int, default=5600)
    parser.add_argument("--lmcache", action="store_true", help="use one shared prefill-side LMCache MP server")
    parser.add_argument("--legacy-full-blend", action="store_true",
                        help="use the historical in-process LMCache on each producer")
    parser.add_argument("--prefill-mode", choices=("prefix", "blend"),
                        help="use the selected prefiller connector with an LMCache blend server")
    parser.add_argument("--external-lmcache", action="store_true",
                        help="reuse the loopback blend server on the configured LMCache ports")
    parser.add_argument(
        "--lmcache-on-decode", action="store_true",
        help="also connect the decode workers to the same LMCache MP server",
    )
    parser.add_argument("--lmcache-source", type=Path)
    parser.add_argument("--trace-lmcache-transfers", action="store_true",
                        help="record completed LMCache staging transfers and verified token ranges")
    parser.add_argument("--trace-kv-storage", action="store_true",
                        help="sample differential KV storage on legacy Full Blend producers")
    parser.add_argument("--trace-kv-pages", action="store_true",
                        help="measure differential storage for native P pages in the Full Blend smoke")
    parser.add_argument("--capture-kv-pages", action="store_true",
                        help="capture raw page payloads from the four-turn KV page smoke")
    parser.add_argument("--lmcache-size-gb", type=float, default=8)
    parser.add_argument("--lmcache-chunk-size", type=int, default=256)
    parser.add_argument("--lmcache-port", type=int, default=6555)
    parser.add_argument("--lmcache-http-port", type=int, default=8090)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--dry-run", action="store_true", help="print commands without starting workers")
    parser.add_argument("client", nargs=argparse.REMAINDER, help="optional workload after --")
    args = parser.parse_args()
    if args.client and args.client[0] == "--":
        args.client = args.client[1:]
    if args.legacy_full_blend and (
        args.lmcache or args.prefill_mode or args.prefix_cache != "off"
        or not math.isfinite(args.lmcache_size_gb) or args.lmcache_size_gb <= 0
    ):
        parser.error("--legacy-full-blend requires prefix cache off and a positive per-producer cache size")
    if args.prefill_mode:
        args.lmcache = True
        if args.lmcache_on_decode or args.lmcache_chunk_size != 256:
            parser.error("--prefill-mode requires bare NIXL decode and --lmcache-chunk-size 256")
    if args.external_lmcache and not args.prefill_mode:
        parser.error("--external-lmcache requires --prefill-mode")
    if args.lmcache_on_decode and not args.lmcache:
        parser.error("--lmcache-on-decode requires --lmcache")
    if args.trace_lmcache_transfers and (not args.lmcache or args.trace != "on"):
        parser.error("--trace-lmcache-transfers requires --lmcache and --trace on")
    if args.trace_lmcache_transfers and args.prefill_mode:
        parser.error("--prefill-mode pilots use blend-server counters and do not support --trace-lmcache-transfers")
    if args.trace_kv_storage and (not args.legacy_full_blend or args.trace != "on"):
        parser.error("--trace-kv-storage requires --legacy-full-blend and --trace on")
    if args.trace_kv_pages and (not args.legacy_full_blend or args.trace != "on" or args.topology != "1p3d"
                                or args.trace_kv_storage):
        parser.error("--trace-kv-pages requires legacy Full Blend, 1p3d, trace on, and no object storage trace")
    if args.capture_kv_pages and not args.trace_kv_pages:
        parser.error("--capture-kv-pages requires --trace-kv-pages")
    if args.minimum_nixl_reads < 0 or (args.minimum_nixl_reads and not args.client):
        parser.error("--minimum-nixl-reads requires a workload and a nonnegative count")
    args.trace_blend_server = bool(args.prefill_mode and not args.external_lmcache and args.trace == "on")
    if args.lmcache_source is None:
        args.lmcache_source = LMCACHE_TRACE_SOURCE if args.trace_lmcache_transfers or args.trace_blend_server else LMCACHE_SOURCE
    if args.trace_blend_server and args.lmcache_source.expanduser().resolve() != LMCACHE_TRACE_SOURCE.resolve():
        parser.error("For a traced Blend server, install and use the verified LMCache trace fork")
    if args.max_model_len < 16 or args.max_num_seqs < 1:
        parser.error("max-model-len must be at least 16 and max-num-seqs must be positive")
    if (not 0 < args.gpu_memory_utilization <= 1 or not math.isfinite(args.startup_timeout)
            or args.startup_timeout <= 0):
        parser.error("invalid GPU memory utilization or startup timeout")
    ports = [args.port, *range(args.worker_port_base, args.worker_port_base + 4),
             *range(args.nixl_port_base, args.nixl_port_base + 4)]
    if args.lmcache:
        if not math.isfinite(args.lmcache_size_gb) or args.lmcache_size_gb <= 0:
            parser.error("lmcache-size-gb must be finite and positive")
        if args.lmcache_chunk_size <= 0 or args.lmcache_chunk_size % 16:
            parser.error("lmcache-chunk-size must be a positive multiple of the vLLM block size (16)")
        ports.extend((args.lmcache_port, args.lmcache_http_port))
    if len(set(ports)) != len(ports) or any(not 1024 <= port < 65536 for port in ports):
        parser.error("router, worker, NIXL, and selected LMCache ports must be distinct unprivileged ports")
    args.run_dir = args.run_dir.expanduser().resolve()
    args.source = args.source.expanduser().resolve()
    args.lmcache_source = args.lmcache_source.expanduser().resolve()
    return args


def assigned_gpus() -> list[str]:
    # PyTorch respects the parent Slurm CUDA mask. The inventory check excludes
    # MIG. Probe in a subprocess so measurement starts with no parent CUDA context.
    probe = (
        "import json, torch\n"
        "print(json.dumps([str(torch.cuda.get_device_properties(i).uuid) "
        "for i in range(torch.cuda.device_count())]))"
    )
    visible = json.loads(subprocess.check_output([sys.executable, "-c", probe], text=True))
    if len(visible) != 4:
        raise ValueError(f"Expected four CUDA devices visible to the Slurm step, got {len(visible)}")
    visible = [value if value.startswith("GPU-") else f"GPU-{value}" for value in visible]
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(set(visible)) != 4 or not set(visible).issubset({row.strip() for row in inventory}):
        raise ValueError(f"Expected four distinct physical CUDA GPUs, got {visible}")
    return visible


def workers_for(args: argparse.Namespace, gpus: list[str]) -> list[Worker]:
    prefill, decode = TOPOLOGIES[args.topology]
    roles = [("P", index) for index in range(prefill)] + [("D", index) for index in range(decode)]
    return [
        Worker(f"{role.lower()}{index}", role, gpus[slot],
               args.worker_port_base + slot, args.nixl_port_base + slot)
        for slot, (role, index) in enumerate(roles)
    ]


def uses_lmcache(args: argparse.Namespace, role: str) -> bool:
    return args.lmcache and (role == "P" or args.lmcache_on_decode)


def prefix_cache_for(args: argparse.Namespace, role: str) -> str:
    if role == "D" and args.lmcache and not args.lmcache_on_decode:
        return "off"
    return args.prefix_cache


def worker_command(args: argparse.Namespace, worker: Worker) -> list[str]:
    connector = {
        "kv_connector": "NixlConnector" if args.legacy_full_blend else "NixlPullConnector",
        "kv_role": "kv_producer" if worker.role == "P" else "kv_consumer",
        "kv_load_failure_policy": "fail",
    }
    if args.legacy_full_blend:
        connector.update(kv_buffer_device="cuda", kv_connector_extra_config={
            "backends": ["UCX"], "enforce_handshake_compat": True, "kv_recompute_threshold": 0,
        })
        if worker.role == "P":
            cache = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both",
                     "kv_connector_extra_config": {"use_native": False, "use_layerwise": True}}
            connector = {"kv_connector": "MultiConnector", "kv_role": "kv_producer",
                         "kv_load_failure_policy": "fail", "kv_connector_extra_config": {
                             "connectors": [connector, cache],
                         }}
    elif uses_lmcache(args, worker.role):
        blend = args.prefill_mode == "blend"
        cache_extra = {
            "lmcache.mp.host": f"tcp://{HOST}",
            "lmcache.mp.port": str(args.lmcache_port) if args.prefill_mode else args.lmcache_port,
        }
        if blend:
            cache_extra.update(BLEND_SETTINGS)
        elif args.prefill_mode == "prefix":
            cache_extra["lmcache.mp.mp_transfer_mode"] = "lmcache_driven"
        cache = {
            "kv_connector": BLEND_CONNECTOR if blend else "LMCacheMPConnector",
            "kv_connector_module_path": BLEND_MODULE if blend else "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": cache_extra,
        }
        connector = {
            "kv_connector": "MultiConnector",
            "kv_role": connector["kv_role"],
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {"connectors": [connector, cache]},
        }
    prefix_flag = "--enable-prefix-caching" if prefix_cache_for(args, worker.role) == "on" else "--no-enable-prefix-caching"
    pilot_flags = ["--pipeline-parallel-size", "1", "--no-enable-chunked-prefill"] if (
        (args.prefill_mode or args.legacy_full_blend) and worker.role == "P"
    ) else []
    runner = ["env", "VLLM_USE_V2_MODEL_RUNNER=0"] if args.legacy_full_blend else []
    return [
        *runner, sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
        "--host", HOST, "--port", str(worker.port),
        "--revision", args.revision, "--tokenizer-revision", args.revision,
        "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--block-size", "16",
        "--seed", "1024", "--generation-config", "vllm", "--disable-hybrid-kv-cache-manager",
        "--max-model-len", str(args.max_model_len),
        "--max-num-batched-tokens", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        prefix_flag, "--enforce-eager", "--no-async-scheduling",
        *pilot_flags,
        "--kv-transfer-config", json.dumps(connector),
    ]


def worker_env(args: argparse.Namespace, worker: Worker) -> dict[str, str]:
    env = {
        "CUDA_VISIBLE_DEVICES": worker.gpu,
        "VLLM_NIXL_SIDE_CHANNEL_HOST": HOST,
        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(worker.nixl_port),
        "VLLM_BATCH_INVARIANT": "1",
        "UCX_TLS": os.environ.get("UCX_TLS", "all"),
        "UCX_NET_DEVICES": os.environ.get("UCX_NET_DEVICES", "all"),
        "EVIDENCESTORE_PD_TRACE": "1" if args.trace == "on" else "0",
        "EVIDENCESTORE_TRACE_KV_STORAGE": "1" if args.trace_kv_storage and worker.role == "P" else "0",
        "EVIDENCESTORE_TRACE_KV_PAGES": "1" if args.trace_kv_pages and worker.role == "P" else "0",
        "EVIDENCESTORE_CAPTURE_KV_PAGES": "1" if args.capture_kv_pages and worker.role == "P" else "0",
        "EVIDENCESTORE_TRACE_DIR": str(args.run_dir / "trace"),
        "EVIDENCESTORE_RUN_ID": args.run_dir.name,
        "EVIDENCESTORE_WORKER_ID": worker.id,
        "EVIDENCESTORE_ROLE": worker.role,
    }
    if uses_lmcache(args, worker.role):
        env["LMCACHE_LOG_LEVEL"] = "INFO"
    if args.legacy_full_blend:
        env.update(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="1", VLLM_ATTENTION_BACKEND="FLASH_ATTN",
                   VLLM_KV_CACHE_LAYOUT="HND", PYTHONHASHSEED="0")
        if worker.role == "P":
            env.update(LMCACHE_CONFIG_FILE=str(args.run_dir / f"lmcache-{worker.id}.json"),
                       LMCACHE_BLEND_SYNC_MODE="events", LMCACHE_TRACK_USAGE="false")
    return env


def lmcache_command(args: argparse.Namespace) -> list[str]:
    instance_id = "evidencestore-shared-pd" if args.lmcache_on_decode else "evidencestore-prefill"
    mode = "blend" if args.prefill_mode else "default"
    blend_flags = ["--supported-transfer-mode", "lmcache_driven", "--l1-align-bytes", "16777216"] if (
        args.prefill_mode
    ) else []
    return [
        sys.executable, "-m", "lmcache.cli.main", "server", "--engine-type", mode, *blend_flags,
        "--host", HOST, "--port", str(args.lmcache_port),
        "--http-host", HOST, "--http-port", str(args.lmcache_http_port),
        "--l1-size-gb", str(args.lmcache_size_gb), "--eviction-policy", "LRU",
        "--chunk-size", str(args.lmcache_chunk_size), "--instance-id", instance_id,
    ]


def lmcache_env(args: argparse.Namespace, workers: list[Worker]) -> dict[str, str]:
    trace_transfers = args.trace_lmcache_transfers or args.trace_blend_server
    env = {"CUDA_VISIBLE_DEVICES": ",".join(worker.gpu for worker in workers if uses_lmcache(args, worker.role)),
           "EVIDENCESTORE_TRACE_LMCACHE_TRANSFERS": "1" if trace_transfers else "0"}
    if args.prefill_mode:
        env["EVIDENCESTORE_TRACE_LMCACHE_BLEND"] = "1" if args.trace_blend_server else "0"
    if trace_transfers:
        env.update(EVIDENCESTORE_TRACE_DIR=str(args.run_dir / "trace"), EVIDENCESTORE_RUN_ID=args.run_dir.name)
    return env


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def start(name: str, command: list[str], env: dict[str, str], run_dir: Path) -> Child:
    log_path = run_dir / "logs" / f"{name}.log"
    process_env = {**os.environ, **env}
    # The older run card selected a runner removed by this vLLM pin.
    process_env.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=process_env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return Child(name, process, log_path)


def worker_exit_excerpt(log_path: Path) -> str:
    limit = 64 * 1024
    failures = (b"EngineCore encountered a fatal error.", b"EngineCore failed to start.")
    with log_path.open("rb") as stream:
        for line in stream:
            if any(message in line for message in failures):
                payload = line[:limit] + stream.read(max(0, limit - len(line)))
                lines = payload.decode("utf-8", errors="replace").splitlines()[:100]
                return "First EngineCore failure:\n" + "\n".join(lines)
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - limit))
        lines = stream.read().decode("utf-8", errors="replace").splitlines()[-80:]
    return "Last lines from the worker log:\n" + "\n".join(lines) if lines else ""


def check_children(children: list[Child]) -> None:
    for child in children:
        status = child.process.poll()
        if status is not None:
            message = f"{child.name} exited with {status}; see {child.log}"
            try:
                excerpt = worker_exit_excerpt(child.log)
                if excerpt:
                    message += "\n" + excerpt
            except OSError as error:
                message += f"\nCould not read worker log: {error}"
            raise RuntimeError(message)


def await_health(endpoints: dict[str, str], children: list[Child], timeout: float, stopped,
                 *, path: str = "/health") -> None:
    deadline = time.monotonic() + timeout
    pending = dict(endpoints)
    while pending:
        if stopped():
            raise InterruptedError("worker startup interrupted")
        check_children(children)
        for name, url in list(pending.items()):
            try:
                with OPENER.open(f"{url}{path}", timeout=1) as response:
                    if response.status == 200:
                        del pending[name]
            except OSError:
                continue
        if pending and time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out starting {', '.join(pending)}; see worker logs")
        if pending:
            time.sleep(0.3)


def lmcache_status(url):
    with OPENER.open(f"{url}/status", timeout=2) as response:
        status = json.load(response)
    if not isinstance(status, dict):
        raise RuntimeError("LMCache status must be a JSON object")
    return status


def require_blend_server(status):
    if (
        status.get("is_healthy") is not True
        or status.get("chunk_size") != 256
        or not isinstance(status.get("registered_cb_rope_instances"), list)
        or not isinstance(status.get("cb_rope_meta"), dict)
        or "active_cb_lookups" not in status
    ):
        raise ValueError("Expected a healthy LMCache blend server reporting 256-token chunks and CB RoPE status")


def await_idle_blend_server(url, children, stopped, timeout, receipt):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stopped():
            raise InterruptedError("LMCache server check interrupted")
        check_children(children)
        status = lmcache_status(url)
        require_blend_server(status)
        ready = (
            status.get("registered_gpu_ids") == []
            and status.get("cache_context_meta") == {}
            and status["registered_cb_rope_instances"] == []
            and status["cb_rope_meta"] == {}
        )
        write_json(receipt, {"verified": ready, "server_engine": "blend", "server_status": status})
        if ready:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Previous LMCache GPU or RoPE registrations did not clear; see {receipt}")


def await_lmcache_registrations(url, workers, children, stopped, receipt, *, prefill_mode=None, model=None):
    deadline = time.monotonic() + 30
    child_logs = {child.name: child.log for child in children}
    while time.monotonic() < deadline:
        if stopped():
            raise InterruptedError("LMCache registration interrupted")
        check_children(children)
        try:
            last = lmcache_status(url)
        except (OSError, ValueError):
            time.sleep(0.3)
            continue
        if prefill_mode:
            require_blend_server(last)
        ids = last.get("registered_gpu_ids", [])
        contexts = last.get("cache_context_meta", {})
        valid_ids = isinstance(ids, list) and all(type(item) is int for item in ids)
        registered = set(ids) if valid_ids else set()
        registrations = {}
        for worker in workers:
            log = child_logs[worker.id].read_text(errors="replace")
            matches = registered.intersection(map(int, LMCACHE_WORKER_ID.findall(log)))
            if len(matches) == 1:
                registrations[worker.id] = {"gpu_uuid": worker.gpu, "instance_id": matches.pop()}
        matched = {entry["instance_id"] for entry in registrations.values()}
        complete = (
            valid_ids and len(ids) == len(registered) == len(workers)
            and isinstance(contexts, dict) and set(contexts) == {str(item) for item in ids}
        )
        verified = complete and registered == matched
        details = {}
        if prefill_mode:
            complete = complete and all(
                isinstance(context, dict) and context.get("model_name") == model
                and context.get("world_size") == 1 for context in contexts.values()
            )
            cb_ids = last["registered_cb_rope_instances"]
            cb_contexts = last["cb_rope_meta"]
            if prefill_mode == "blend":
                verified = complete and len(cb_ids) == len(registered) and all(
                    type(item) is int for item in cb_ids
                ) and set(cb_ids) == registered and cb_contexts == {
                    str(item): [model, 1] for item in registered
                }
                method = "cb_rope_contexts"
            else:
                verified = complete and registered == matched and cb_ids == [] and cb_contexts == {}
                method = "mp_worker_log"
            details = {
                "prefill_mode": prefill_mode,
                "server_engine": "blend",
                "registration_method": method,
                "registered_instance_ids": sorted(registered),
                "cb_rope_instance_ids": cb_ids,
            }
        write_json(receipt, {"verified": verified, "worker_registrations": registrations,
                             "server_status": last, **details})
        if verified:
            return
        time.sleep(0.3)
    logs = "the prefiller logs" if prefill_mode else "logs/lmcache.log"
    raise RuntimeError(f"Expected {len(workers)} LMCache worker registrations; inspect {receipt} and {logs}")


def group_exists(child: Child) -> bool:
    # poll() also reaps the direct child, so its zombie cannot extend the grace period.
    child.process.poll()
    try:
        os.killpg(child.process.pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_children(children: list[Child]) -> dict:
    forced = []
    denied: dict[str, str] = {}
    for child in reversed(children):
        try:
            os.killpg(child.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as error:
            denied[child.name] = str(error)
            try:
                child.process.terminate()
            except PermissionError:
                pass
    accessible = [child for child in children if child.name not in denied]
    deadline = time.monotonic() + 15
    while any(group_exists(child) for child in accessible) and time.monotonic() < deadline:
        time.sleep(0.2)
    for child in reversed(accessible):
        if group_exists(child):
            try:
                os.killpg(child.process.pid, signal.SIGKILL)
                forced.append(child.name)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    while any(group_exists(child) for child in accessible) and time.monotonic() < deadline:
        time.sleep(0.2)
    remaining = [child.name for child in accessible if group_exists(child)]
    return {"forced_kill": forced, "remaining_process_groups": remaining,
            "unverified_process_groups": denied,
            "exit_status": {child.name: child.process.poll() for child in children}}


def capture_metrics(run_dir: Path, period: str, *, minimum_nixl_reads: int = 0) -> None:
    command = [sys.executable, str(ROOT / "scripts/vllm_pd_report.py"), "--run-dir", str(run_dir), "--capture", period]
    if minimum_nixl_reads:
        command.extend(("--minimum-nixl-reads", str(minimum_nixl_reads)))
    with (run_dir / "logs" / f"metrics-{period}.log").open("x", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
            env={**os.environ, "NO_PROXY": "localhost,127.0.0.1", "no_proxy": "localhost,127.0.0.1"},
        )


def lmcache_preflight(args: argparse.Namespace, source_pins: list[dict]) -> dict:
    root = args.lmcache_source
    spec = importlib.util.find_spec("lmcache")
    if spec is None or not spec.origin or Path(spec.origin).resolve() != root / "lmcache/__init__.py":
        raise ValueError(f"Install {root} editable in the active Python environment for --lmcache")
    upstream = next(item["commit"] for item in source_pins if item["name"] == "lmcache")
    expected = upstream
    if args.trace_lmcache_transfers or (args.prefill_mode and root == LMCACHE_TRACE_SOURCE.resolve()):
        receipt = json.loads((ROOT / "third_party/lmcache-pd-fork.json").read_text())
        if receipt["base_commit"] != upstream:
            raise ValueError("The LMCache tracing fork receipt does not match the upstream pin")
        expected = receipt["source_commit"]
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError(f"LMCache source {actual} differs from the selected commit {expected}")
    for command in (("remote",), ("status", "--porcelain", "--untracked-files=no")):
        if subprocess.check_output(["git", "-C", str(root), *command], text=True).strip():
            raise ValueError("The pinned LMCache checkout must have zero remotes and clean tracked files")
    return {"commit": actual, "base_commit": upstream, "editable_source": str(root), "module": spec.origin}


def blend_plugin_preflight() -> dict:
    probe = """
import inspect
import json
import sys
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

config = KVTransferConfig(kv_connector=sys.argv[1], kv_connector_module_path=sys.argv[2], kv_role="kv_both")
connector = KVConnectorFactory.get_connector_class(config)
if not issubclass(connector, KVConnectorBase_V1):
    raise TypeError("Expected a vLLM v1 KV connector")
print(json.dumps(inspect.getfile(connector)))
"""
    try:
        lines = subprocess.check_output(
            [sys.executable, "-c", probe, BLEND_CONNECTOR, BLEND_MODULE], text=True, timeout=60
        ).splitlines()
        source = Path(json.loads(lines[-1])).resolve(strict=True)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        raise ValueError(f"The active runtime cannot load {BLEND_MODULE}.{BLEND_CONNECTOR}: {error}") from error
    return {"connector": BLEND_CONNECTOR, "module_path": BLEND_MODULE, "source_file": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "settings": BLEND_SETTINGS}


def legacy_runtime_identity(args: argparse.Namespace) -> dict:
    receipt = Path(os.environ["CHECK_ROOT"]) / "runtime.json"
    identity = json.loads(receipt.read_text(encoding="utf-8"))
    for module, expected, source in (("vllm", "0.23.0", args.source),
                                     ("lmcache", "0.5.0", args.lmcache_source)):
        spec = importlib.util.find_spec(module)
        if (version(module) != expected or not spec or not spec.origin or
                Path(spec.origin).resolve() != source / f"{module}/__init__.py"):
            raise ValueError(f"Activate the verified editable legacy {module} {expected} runtime")
    if args.trace_kv_storage and not (args.source / "vllm/_evidencestore_legacy_storage_trace.py").is_file():
        raise ValueError("Build a fresh Full Blend runtime with the KV storage trace overlay")
    if args.trace_kv_pages and not (args.source / "vllm/_evidencestore_legacy_page_trace.py").is_file():
        raise ValueError("Build a fresh Full Blend runtime with the KV page trace overlay")
    return {"vllm_source": identity, "lmcache_source": str(args.lmcache_source), "blend_plugin": None}


def preflight(args: argparse.Namespace, workers: list[Worker]) -> dict:
    ports = [args.port, *(worker.port for worker in workers), *(worker.nixl_port for worker in workers)]
    if args.lmcache and not args.external_lmcache:
        ports.extend((args.lmcache_port, args.lmcache_http_port))
    for port in ports:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((HOST, port))
    if args.legacy_full_blend:
        return legacy_runtime_identity(args)
    root = args.source
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin or Path(spec.origin).resolve() != root / "vllm/__init__.py":
        raise ValueError(f"Install {root} editable in the active Python environment before launch")
    source_pin = json.loads((ROOT / "third_party/sources.json").read_text())["sources"]
    expected = next(item["commit"] for item in source_pin if item["name"] == "vllm")
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    ancestry = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", expected, actual])
    if ancestry.returncode:
        raise ValueError(f"vLLM source {actual} does not descend from the approved pin {expected}")
    if subprocess.check_output(["git", "-C", str(root), "remote"], text=True).strip():
        raise ValueError("The local vLLM fork must have no Git remote")
    if not (root / "vllm/_evidencestore_trace.py").is_file():
        raise ValueError("The editable vLLM fork is missing its native tracing source")
    tree = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True).strip()
    changes = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).splitlines()
    cache_source = lmcache_preflight(args, source_pin) if args.lmcache else None
    blend_plugin = blend_plugin_preflight() if args.prefill_mode == "blend" else None
    return {"vllm_source": {"base_commit": expected, "commit": actual, "git_tree": tree,
                           "editable_source": str(root), "module": spec.origin, "working_tree_changes": changes},
            "lmcache_source": cache_source, "blend_plugin": blend_plugin}


def run(args: argparse.Namespace) -> int:
    if args.dry_run:
        workers = workers_for(args, ["GPU-slot-0", "GPU-slot-1", "GPU-slot-2", "GPU-slot-3"])
        if args.external_lmcache:
            print(json.dumps({"id": "lmcache", "server_ownership": "external", "server_engine": "blend",
                              "url": f"http://{HOST}:{args.lmcache_http_port}", "port": args.lmcache_port}))
        elif args.lmcache:
            print(json.dumps({"id": "lmcache", "argv": lmcache_command(args), "environment": lmcache_env(args, workers)}))
        for worker in workers:
            print(json.dumps({**asdict(worker), "argv": worker_command(args, worker),
                              "environment": worker_env(args, worker)}))
        return 0

    job_id = os.environ.get("SLURM_JOB_ID")
    node_count = os.environ.get("SLURM_JOB_NUM_NODES", os.environ.get("SLURM_NNODES", "1"))
    if not job_id or node_count != "1":
        raise ValueError("Run this launcher inside an existing single-node Slurm allocation")
    workers = workers_for(args, assigned_gpus())
    identity = preflight(args, workers)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "logs").mkdir()
    (args.run_dir / "trace").mkdir()
    router = f"http://{HOST}:{args.port}"
    endpoints = {
        "prefill": [{"id": worker.id, "url": worker.url} for worker in workers if worker.role == "P"],
        "decode": [{"id": worker.id, "url": worker.url} for worker in workers if worker.role == "D"],
    }
    if args.lmcache:
        endpoints["lmcache"] = {"id": "lmcache", "url": f"http://{HOST}:{args.lmcache_http_port}"}
    write_json(args.run_dir / "endpoints.json", endpoints)
    plans = [(worker, worker_command(args, worker), worker_env(args, worker)) for worker in workers]
    manifest = {
        "run_id": args.run_dir.name, "slurm_job_id": job_id, "host": socket.gethostname(),
        "started_wall_ns": time.time_ns(), "topology": args.topology, "router": router,
        "model": args.model, "revision": args.revision, "prefix_cache": args.prefix_cache,
        "prefix_cache_by_role": {role: prefix_cache_for(args, role) for role in ("P", "D")},
        "full_nixl_handoff": prefix_cache_for(args, "D") == "off",
        "engine_trace_enabled": args.trace == "on",
        "gpu_event_timing": args.trace == "on" and os.environ.get("EVIDENCESTORE_TRACE_GPU", "1") != "0",
        "vllm_source": identity["vllm_source"],
        "workers": [dict(asdict(worker), url=worker.url, argv=command, environment=env)
                    for worker, command, env in plans],
    }
    if args.lmcache:
        manifest["lmcache"] = {
            **endpoints["lmcache"], "source": identity["lmcache_source"],
            "mode": "shared_pd" if args.lmcache_on_decode else "prefill_only",
            "transfer_trace_enabled": args.trace_lmcache_transfers,
            "port": args.lmcache_port, "l1_size_gb": None if args.external_lmcache else args.lmcache_size_gb,
            "chunk_size": args.lmcache_chunk_size,
            "prefill_workers": [worker.id for worker in workers if worker.role == "P"],
            "decode_workers": [worker.id for worker in workers if worker.role == "D" and uses_lmcache(args, worker.role)],
            "argv": None if args.external_lmcache else lmcache_command(args),
            "environment": None if args.external_lmcache else lmcache_env(args, workers),
        }
    if args.prefill_mode:
        manifest["prefill_mode"] = args.prefill_mode
        manifest["request_trace_enabled"] = args.trace == "on"
        manifest["lmcache"].update(
            server_engine="blend", server_ownership="external" if args.external_lmcache else "launcher",
            blend_plugin=identity["blend_plugin"],
            blend_trace_enabled=None if args.external_lmcache else args.trace_blend_server,
        )
    if args.legacy_full_blend:
        manifest["legacy_full_blend"] = {"mode": "in_process", "cache_per_producer_gib": args.lmcache_size_gb,
                                         "lmcache_source": identity["lmcache_source"],
                                         "kv_storage_trace_enabled": args.trace_kv_storage,
                                         "kv_page_trace_enabled": args.trace_kv_pages,
                                         "raw_page_capture_enabled": args.capture_kv_pages}
        for worker, _, env in plans:
            if worker.role == "P":
                write_json(Path(env["LMCACHE_CONFIG_FILE"]), {
                    "local_cpu": True, "max_local_cpu_size": args.lmcache_size_gb,
                    "chunk_size": 256, "use_layerwise": True, "enable_blending": True,
                    "blend_special_str": "¶", "blend_check_layers": [1], "blend_recompute_ratios": [0.15],
                    "enable_async_loading": False, "save_unfull_chunk": True,
                    "pre_caching_hash_algorithm": "builtin",
                })
    write_json(args.run_dir / "run_manifest.json", manifest)

    children: list[Child] = []
    stopped = False
    http_env = {"NO_PROXY": "localhost,127.0.0.1", "no_proxy": "localhost,127.0.0.1"}
    if args.prefill_mode or args.legacy_full_blend:
        http_env["EVIDENCESTORE_TRACE_REQUESTS"] = "1" if args.trace == "on" else "0"

    def stop(_signal, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    status = 0
    try:
        if args.lmcache:
            if not args.external_lmcache:
                children.append(start("lmcache", lmcache_command(args), lmcache_env(args, workers), args.run_dir))
            await_health({"lmcache": endpoints["lmcache"]["url"]}, children,
                         args.startup_timeout, lambda: stopped, path="/healthcheck")
            if args.prefill_mode:
                await_idle_blend_server(endpoints["lmcache"]["url"], children, lambda: stopped,
                                        args.startup_timeout, args.run_dir / "lmcache-initial-status.json")
        for worker, command, env in plans:
            if stopped:
                raise InterruptedError("worker launch interrupted")
            children.append(start(worker.id, command, env, args.run_dir))
        await_health({worker.id: worker.url for worker in workers}, children,
                     args.startup_timeout, lambda: stopped)
        if args.lmcache:
            await_lmcache_registrations(
                endpoints["lmcache"]["url"], [worker for worker in workers if uses_lmcache(args, worker.role)],
                children, lambda: stopped, args.run_dir / "lmcache-status.json",
                prefill_mode=args.prefill_mode, model=args.model,
            )
        children.append(start(
            "router", [sys.executable, str(ROOT / "scripts/vllm_pd_gateway.py"), "--run-dir",
                       str(args.run_dir), "--port", str(args.port)],
            http_env, args.run_dir,
        ))
        await_health({"router": router}, children, 30, lambda: stopped)
        print(f"Native PD ready at {router} in {args.run_dir}", flush=True)
        if args.client:
            capture_metrics(args.run_dir, "before")
            if stopped:
                raise InterruptedError("workload launch interrupted")
            client = start(
                "workload", args.client,
                {"EVIDENCESTORE_RUN_DIR": str(args.run_dir), "EVIDENCESTORE_GATEWAY": router,
                 **http_env},
                args.run_dir,
            )
            children.append(client)
            while not stopped and client.process.poll() is None:
                check_children(children[:-1])
                time.sleep(0.2)
            if not stopped:
                status = client.process.returncode or 0
                capture_metrics(args.run_dir, "after", minimum_nixl_reads=args.minimum_nixl_reads if not status else 0)
                if status:
                    print(f"Workload exited with {status}; see {client.log}", file=sys.stderr)
        else:
            while not stopped:
                check_children(children)
                time.sleep(0.3)
    except InterruptedError as error:
        print(error, file=sys.stderr)
        status = 130
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        status = 1
    finally:
        cleanup = stop_children(children)
        write_json(args.run_dir / "cleanup.json", cleanup)
    if cleanup["remaining_process_groups"] or cleanup["unverified_process_groups"]:
        print(f"Cleanup could not verify all process groups; see {args.run_dir / 'cleanup.json'}", file=sys.stderr)
        return 1
    if stopped and args.client:
        return 130
    return status if status >= 0 else 128 - status


def main() -> int:
    try:
        return run(arguments())
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
