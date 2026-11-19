"""Copied into the historical vLLM build to measure its existing PD paths."""

import itertools
import json
import logging
import os
from pathlib import Path
import socket
import time


ENABLED = os.environ.get("EVIDENCESTORE_PD_TRACE") == "1"
WORKER = os.environ.get("EVIDENCESTORE_WORKER_ID", "worker")
ROLE = os.environ.get("EVIDENCESTORE_ROLE", "")
RUN_ID = os.environ.get("EVIDENCESTORE_RUN_ID", "")
TRACE_DIR = os.environ.get("EVIDENCESTORE_TRACE_DIR")
HOST = socket.gethostname()
if ENABLED and not TRACE_DIR:
    raise ValueError("EVIDENCESTORE_PD_TRACE=1 requires EVIDENCESTORE_TRACE_DIR")

_LOG = logging.getLogger(__name__)
_SEQUENCE = itertools.count()
_READS = {}
_WRITE_FAILED = False


def emit(event, *, component="vllm", **fields):
    global _WRITE_FAILED
    if not ENABLED or _WRITE_FAILED:
        return
    pid = os.getpid()
    row = {
        "schema_version": 1, "event": event, "component": component,
        "run_id": RUN_ID, "worker_id": WORKER, "role": ROLE, "pid": pid,
        "host_id": HOST, "ts_ns": time.monotonic_ns(), "wall_ns": time.time_ns(),
        **fields,
    }
    path = Path(TRACE_DIR) / f"legacy-{WORKER}-{pid}.jsonl"
    try:
        payload = (json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode()
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            if os.write(fd, payload) != len(payload):
                raise OSError("short trace write")
        finally:
            os.close(fd)
    except (OSError, TypeError, ValueError):
        _WRITE_FAILED = True
        _LOG.exception("Legacy Full Blend trace write failed")


def read_submitted(worker, handle, request_id, remote_request_id, local, remote,
                   remote_block_size, remote_physical_per_logical):
    details = {
        "transfer_id": f"{WORKER}:{os.getpid()}:{next(_SEQUENCE)}",
        "native_request_id": request_id, "remote_request_id": remote_request_id,
        "remote_block_ids": [list(group) for group in remote],
        "local_block_counts": [len(group) for group in local],
        "block_size": worker.block_size, "remote_block_size": remote_block_size,
        "remote_physical_per_logical": remote_physical_per_logical,
    }
    _READS[id(worker), handle] = details
    emit("nixl_read_submit", **details)


def read_done(worker, handle, telemetry):
    details = _READS.pop((id(worker), handle), None)
    if details is not None:
        emit("nixl_read_complete", **details, status="done",
             completed_bytes=int(telemetry.totalBytes),
             transfer_duration_us=float(telemetry.xferDuration),
             post_duration_us=float(telemetry.postDuration))


def read_failed(worker, handle, request_id):
    details = _READS.pop((id(worker), handle), {})
    emit("nixl_read_failed", **{**details, "native_request_id": request_id}, status="error")


def gpu_forward_start(torch, scheduler_output):
    details = dict(request_tokens=dict(scheduler_output.num_scheduled_tokens),
                   batch_id=f"{WORKER}:{os.getpid()}:{next(_SEQUENCE)}")
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    emit("gpu_forward_start", **details)
    return begin, end, details


def gpu_forward_end(ticket):
    begin, end, details = ticket
    end.record()
    emit("gpu_forward_end", **details)
    # Diagnostic only. This timer covers the ordinary forward after the
    # connector has run its separate layerwise Blend work.
    end.synchronize()
    emit("gpu_forward_duration", **details, clock="cuda_event",
         duration_ns=round(begin.elapsed_time(end) * 1e6))


def _position_ranges(values):
    bounds = []
    for position in values:
        if bounds and bounds[-1][1] == position:
            bounds[-1][1] = position + 1
        else:
            bounds.append([position, position + 1])
    return bounds


def blend_complete(request_id, tokens, layer_positions, started_ns, torch):
    # The legacy generator has completed the connector's native GPU fence.
    # A trace-only sync also confirms the model's work before reading positions.
    torch.cuda.synchronize(tokens.device)
    emit("legacy_blend_complete", component="lmcache", native_request_id=request_id,
         started_ns=started_ns, input_token_count=len(tokens),
         computed_ranges_by_layer=[_position_ranges(positions.tolist())
                                   for positions in layer_positions])
