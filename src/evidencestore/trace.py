"""Write the client and gateway side of the native PD trace."""

import json
import os
from pathlib import Path
import socket
import time


class TraceWriter:
    def __init__(self, run_dir, component):
        self.enabled = os.environ.get("EVIDENCESTORE_TRACE_REQUESTS", "1") == "1"
        if not self.enabled:
            return
        run_dir = Path(run_dir).resolve()
        trace_dir = run_dir / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = trace_dir / f"{component}-{os.getpid()}.jsonl"
        self.common = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "component": component,
            "host_id": socket.gethostname(),
            "pid": os.getpid(),
        }

    def emit(self, event, **fields):
        if not self.enabled:
            return
        row = {
            **self.common,
            "event": event,
            "ts_ns": time.monotonic_ns(),
            "wall_ns": time.time_ns(),
            **fields,
        }
        with self.path.open("ab", buffering=0) as stream:
            stream.write((json.dumps(row, separators=(",", ":")) + "\n").encode())
