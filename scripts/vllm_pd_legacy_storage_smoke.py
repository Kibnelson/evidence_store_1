"""Verify observed LMCache storage targets with exact MT-RAG prompt replays."""

from pathlib import Path
import subprocess

from vllm_pd_replay import json_line, prompt_sha256, read_jsonl

from evidencestore.pd_report import load_events, request_joins


VERIFICATION_LABEL = "legacy-storage-verification"
VERIFICATION_INPUT = Path("inputs/storage-verification.jsonl")


def _storage_key(row):
    return row["worker_id"], row["pid"], row["layer_id"], row["key_sha256"]


def replay_storage_targets(run_dir, command, label):
    """Replay at most one recently stored natural delta target per prefill worker."""
    if label != "mtrag":
        raise ValueError("Storage verification requires the live MT-RAG workload")
    events = list(load_events(run_dir))
    puts = [row for row in events if row["event"] == "legacy_kv_storage_put" and row.get("status") == "measured"]
    latest = {_storage_key(row): row for row in sorted(puts, key=lambda row: row["ts_ns"])}
    active_deltas = {key: row for key, row in latest.items() if row["encoding"] == "delta"}
    if not active_deltas:
        print(f"No measured LMCache delta targets to replay in {run_dir}", flush=True)
        return 0

    client = run_dir / "client" / label
    frozen = {row["request_key"]: row for row in read_jsonl(client / "frozen.jsonl")}
    responses = {row["request_id"]: row for row in read_jsonl(client / "responses.jsonl")
                 if row["success"] is True and row["label"] == label}
    groups, _ = request_joins(events)
    selected = {}
    for (request_id, _attempt), rows in groups.items():
        if request_id not in responses:
            continue
        candidates = [row for row in rows if row["event"] == "legacy_kv_storage_put"
                      and row.get("status") == "measured" and active_deltas.get(_storage_key(row)) == row]
        if not candidates:
            continue
        put = max(candidates, key=lambda row: row["ts_ns"])
        worker = put["worker_id"]
        if worker not in selected or put["ts_ns"] > selected[worker][0]:
            selected[worker] = (put["ts_ns"], request_id, rows)

    if not selected:
        raise ValueError("Measured LMCache delta targets are missing natural MT-RAG request joins")

    targets = []
    for worker, (_, source_id, rows) in sorted(selected.items()):
        route = {}
        for role in ("prefill", "decode"):
            matches = [row["worker_id"] for row in rows if row["event"] == f"gateway_route_{role}"]
            if len(matches) != 1:
                raise ValueError(f"Expected one {role} route for {source_id}")
            route[role] = matches[0]
        source = responses[source_id]
        prompt = dict(frozen[source["request_key"]])
        if route["prefill"] != worker or prompt_sha256(prompt["prompt_token_ids"]) != source["prompt_sha256"]:
            raise ValueError(f"MT-RAG storage target differs from its source request {source_id}")
        prompt.pop("evidence_order", None)
        prompt.update(request_key=f"{VERIFICATION_LABEL}:{source_id}",
                      case="storage-target-replay", pd_route=route)
        targets.append(prompt)

    path = run_dir / VERIFICATION_INPUT
    with path.open("x", encoding="utf-8") as handle:
        handle.writelines(json_line(row) for row in targets)
    print(f"Replaying {len(targets)} LMCache storage targets from {path}", flush=True)
    subprocess.run([*command, "--label", VERIFICATION_LABEL, "--frozen", str(path)], check=True)
    return len(targets)
