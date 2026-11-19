"""Native NIXL pull gateway with out-of-band evidence telemetry."""

from contextlib import asynccontextmanager
from itertools import cycle
import json
from pathlib import Path
import re
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
import httpx

from .trace import TraceWriter


def _validate_evidence(evidence, prompt):
    if not isinstance(evidence, list):
        raise HTTPException(422, "evidence must be a list")
    if evidence and not isinstance(prompt, list):
        raise HTTPException(422, "evidence spans require an exact token-ID prompt")
    for entry in evidence:
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(field), str) or not entry[field]
            for field in ("doc_id", "chunk_id", "chunk_version")
        ):
            raise HTTPException(422, "evidence requires doc_id, chunk_id, and chunk_version")
        start, end = entry.get("start_token"), entry.get("end_token")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(prompt):
            raise HTTPException(422, "evidence token span must be inside the prompt")


def create_app(run_dir):
    endpoints = json.loads((Path(run_dir) / "endpoints.json").read_text())
    workers = {
        role: {worker["id"]: worker["url"].rstrip("/") for worker in endpoints[role]}
        for role in ("prefill", "decode")
    }
    if not all(workers.values()):
        raise ValueError("The endpoint manifest needs P and D workers")
    choices = {role: cycle(routes) for role, routes in workers.items()}
    tracer = TraceWriter(run_dir, "gateway")

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=10), trust_env=False) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(request: Request):
        received_ns = time.monotonic_ns()
        received_wall_ns = time.time_ns()
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(400, "Expected a JSON request") from exc
        if not isinstance(body, dict):
            raise HTTPException(422, "Expected a completion object")
        body_request_id = body.pop("request_id", None)
        request_id = request.headers.get("x-request-id") or body_request_id or uuid.uuid4().hex
        if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request_id):
            raise HTTPException(422, "Invalid x-request-id")
        attempt_id = uuid.uuid4().hex

        def emit(event, **fields):
            tracer.emit(event, request_id=request_id, attempt_id=attempt_id, **fields)

        emit("gateway_received", ts_ns=received_ns, wall_ns=received_wall_ns)
        prompt = body.get("prompt")
        if not isinstance(prompt, str) and not (
            isinstance(prompt, list) and prompt and all(type(token) is int for token in prompt)
        ):
            raise HTTPException(422, "Supply one text prompt or one sequence of token IDs")
        if body.get("stream") is not True or body.get("n", 1) != 1 or "kv_transfer_params" in body:
            raise HTTPException(422, "Native PD replay expects stream=true and one completion")
        evidence = body.pop("evidence", [])
        route = body.pop("pd_route", {})
        _validate_evidence(evidence, prompt)
        if not isinstance(route, dict) or any(
            role not in workers or not isinstance(worker, str) or worker not in workers[role]
            for role, worker in route.items()
        ):
            raise HTTPException(422, "Unknown P or D worker in pd_route")
        emit("gateway_prompt_ready", evidence=evidence, prompt_tokens=len(prompt) if isinstance(prompt, list) else None)
        client = request.app.state.client

        p_worker = route.get("prefill") or next(choices["prefill"])
        p_header_id = f"es-{attempt_id}-p"
        emit("gateway_route_prefill", worker_id=p_worker, forced="prefill" in route)
        stub = {**body, "stream": False, "max_tokens": 1, "min_tokens": 0}
        stub.pop("stream_options", None)
        stub["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        p_request = client.build_request(
            "POST", f"{workers['prefill'][p_worker]}/v1/completions",
            json=stub, headers={"X-Request-Id": p_header_id},
        )
        emit(
            "gateway_dispatch_prefill", worker_id=p_worker, engine_header_id=p_header_id,
            http_request_body_bytes=len(p_request.content),
        )
        try:
            prefill = await client.send(p_request)
            prefill.raise_for_status()
            transfer = prefill.json().get("kv_transfer_params")
            if not isinstance(transfer, dict) or not transfer.get("do_remote_prefill"):
                raise ValueError("Prefill did not return native KV coordinates")
        except (httpx.HTTPError, AttributeError, ValueError) as exc:
            emit("gateway_request_error", stage="prefill", error=str(exc))
            raise HTTPException(502, "Prefill worker did not complete the native handoff") from exc
        emit(
            "gateway_prefill_response", worker_id=p_worker,
            remote_request_id=transfer.get("remote_request_id"), http_response_body_bytes=len(prefill.content),
        )

        d_worker = route.get("decode") or next(choices["decode"])
        d_header_id = f"es-{attempt_id}-d"
        emit("gateway_route_decode", worker_id=d_worker, forced="decode" in route)
        d_request = client.build_request(
            "POST", f"{workers['decode'][d_worker]}/v1/completions",
            json={**body, "kv_transfer_params": transfer}, headers={"X-Request-Id": d_header_id},
        )
        emit(
            "gateway_dispatch_decode", worker_id=d_worker, engine_header_id=d_header_id,
            http_request_body_bytes=len(d_request.content),
        )
        try:
            upstream = await client.send(d_request, stream=True)
        except httpx.HTTPError as exc:
            emit("gateway_request_error", stage="decode_dispatch", error=str(exc))
            raise HTTPException(502, "Decode worker did not accept the request") from exc

        emit("gateway_decode_headers", worker_id=d_worker, status_code=upstream.status_code)
        if upstream.is_error:
            try:
                content = await upstream.aread()
            finally:
                await upstream.aclose()
            emit("gateway_request_error", stage="decode_response", status_code=upstream.status_code)
            return Response(content, status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))

        async def relay():
            first = True
            completed = False
            bytes_streamed = 0
            try:
                async for chunk in upstream.aiter_bytes():
                    if first:
                        emit("gateway_first_decode_bytes", worker_id=d_worker)
                        first = False
                    bytes_streamed += len(chunk)
                    yield chunk
                completed = True
            finally:
                await upstream.aclose()
                emit("gateway_stream_end", worker_id=d_worker, completed=completed, sse_payload_bytes=bytes_streamed)

        return StreamingResponse(
            relay(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            headers={"X-Request-Id": request_id},
        )

    return app
