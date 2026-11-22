"""Sample storage-only KV deltas without modifying LMCache objects or reads."""

from array import array
from collections import OrderedDict
from dataclasses import dataclass, replace
from hashlib import sha256
import logging
import os
import struct
from threading import Lock
from time import monotonic_ns

from vllm._evidencestore_legacy_trace import ENABLED as TRACE_ENABLED, emit


ENABLED = TRACE_ENABLED and os.environ.get("EVIDENCESTORE_TRACE_KV_STORAGE") == "1"
BACKEND = "LocalCPUBackend"
TILE_BYTES = 4096
MAX_OBJECT_BYTES = 8 * 1024**2
MAX_PATCH_BYTES = 32 * 1024**2
MAX_TRACKED = 512
MAX_CANDIDATES = 4
MAX_SAMPLE_TOKENS = 4096
MIN_PREFIX_TOKENS = 8
# Magic, target size, base size, single-plane row width, tile width and count,
# then target key, target tensor, base key and base tensor digests.
_HEADER = struct.Struct(">4sIIIII32s32s32s32s")
_MAGIC = b"ESK1"
_LOG = logging.getLogger(__name__)
_LOCK = Lock()
_VERSIONS = OrderedDict()
_PATCH_BYTES = 0


@dataclass(frozen=True)
class _Geometry:
    shape: tuple
    dtype: str
    row_bytes: int


@dataclass(frozen=True)
class _Delta:
    base: object
    record: bytes
    manifest_bytes: int
    changed_tiles: int
    common_prefix_tokens: int


@dataclass(frozen=True)
class _Version:
    key: object
    key_sha256: str
    target_sha256: str
    geometry: _Geometry
    raw_bytes: int
    start_token: int
    end_token: int
    token_bytes: bytes
    delta: _Delta | None = None


def _digest(value):
    return sha256(value).hexdigest()


def _event(event, request_id, layer_id, started, **fields):
    emit(event, component="lmcache", native_request_id=request_id, backend=BACKEND,
         layer_id=layer_id, duration_ns=monotonic_ns() - started, **fields)


def _snapshot(obj, count):
    size = obj.get_size()
    if not 0 < size <= MAX_OBJECT_BYTES or not 0 < count <= MAX_SAMPLE_TOKENS:
        return "oversized", size, None, None
    tensor = obj.tensor
    if (tensor is None or tensor.device.type != "cpu" or not tensor.is_contiguous()
            or getattr(obj.get_memory_format(), "name", None) != "KV_2TD"
            or len(tensor.shape) != 3 or tuple(tensor.shape[:2]) != (2, count)):
        return "unsupported", size, None, None
    shape = tuple(tensor.shape)
    row_bytes = shape[2] * tensor.element_size()
    if size != 2 * count * row_bytes:
        return "unsupported", size, None, None
    payload = memoryview(obj.byte_array)
    if payload.nbytes != size:
        return "unsupported", size, None, None
    return "measured", size, _Geometry(shape, str(tensor.dtype), row_bytes), bytes(payload)


def _resident(backend, key, count, expected=None):
    with backend.cpu_lock:
        obj = backend.hot_cache.get(key)
        if obj is None or (expected is not None and obj is not expected):
            return "not_resident", None, None, None
        obj.ref_count_up()
    try:
        return _snapshot(obj, count)
    finally:
        obj.ref_count_down()


def _pack_tokens(tokens, start, end):
    segment = tokens[start:end]
    if hasattr(segment, "device"):
        if segment.device.type != "cpu":
            return b""
        segment = segment.tolist()
    return array("I", segment).tobytes()


def _prefix_length(left, right):
    left_ids, right_ids = memoryview(left).cast("I"), memoryview(right).cast("I")
    for index, (earlier, current) in enumerate(zip(left_ids, right_ids)):
        if earlier != current:
            return index
    return min(len(left_ids), len(right_ids))


def _candidates(backend, target, layer_id):
    candidates = []
    with _LOCK:
        recent = tuple(reversed(_VERSIONS.items()))
    for (backend_id, _, base_layer_id), base in recent:
        if (backend_id != id(backend) or base_layer_id != layer_id or base.delta is not None
                or base.key == target.key
                or getattr(base.key, "tags", None) != getattr(target.key, "tags", None)
                or base.geometry.dtype != target.geometry.dtype
                or base.geometry.row_bytes != target.geometry.row_bytes):
            continue
        prefix = _prefix_length(base.token_bytes, target.token_bytes)
        if prefix >= MIN_PREFIX_TOKENS:
            candidates.append((prefix, base))
    # Prefer longer shared prefixes and then matching positions. Stable sort
    # retains recency order among equal candidates.
    candidates.sort(key=lambda row: (row[0], row[1].start_token == target.start_token), reverse=True)
    return candidates[:MAX_CANDIDATES]


def _remember(backend, layer_id, version):
    global _PATCH_BYTES
    identity = (id(backend), version.key, layer_id)
    with _LOCK:
        previous = _VERSIONS.pop(identity, None)
        if previous is not None and previous.delta is not None:
            _PATCH_BYTES -= len(previous.delta.record)
        if version.delta is not None:
            _PATCH_BYTES += len(version.delta.record)
        _VERSIONS[identity] = version
        while len(_VERSIONS) > MAX_TRACKED or _PATCH_BYTES > MAX_PATCH_BYTES:
            _, expired = _VERSIONS.popitem(last=False)
            if expired.delta is not None:
                _PATCH_BYTES -= len(expired.delta.record)


def _resize_base(base, target_bytes):
    if len(base) == target_bytes:
        return base
    old_plane, new_plane = len(base) // 2, target_bytes // 2
    if old_plane < new_plane:
        padding = bytes(new_plane - old_plane)
        return base[:old_plane] + padding + base[old_plane:] + padding
    return base[:new_plane] + base[old_plane:old_plane + new_plane]


def _new_tail_intersects(start, end, base_bytes, target_bytes):
    if base_bytes >= target_bytes:
        return False
    old_plane, new_plane = base_bytes // 2, target_bytes // 2
    return (start < new_plane and end > old_plane) or end > new_plane + old_plane


def _encode(base, old_bytes, target, raw, prefix):
    tile_count = (len(raw) + TILE_BYTES - 1) // TILE_BYTES
    bitmap = bytearray((tile_count + 7) // 8)
    manifest_bytes = _HEADER.size + len(bitmap)
    if manifest_bytes >= len(raw):
        return None
    normalized = _resize_base(old_bytes, len(raw))
    patch = bytearray()
    changed_tiles = 0
    for tile, start in enumerate(range(0, len(raw), TILE_BYTES)):
        end = min(start + TILE_BYTES, len(raw))
        current = raw[start:end]
        if (current != normalized[start:end]
                or _new_tail_intersects(start, end, len(old_bytes), len(raw))):
            bitmap[tile // 8] |= 1 << (tile % 8)
            patch.extend(current)
            changed_tiles += 1
            if manifest_bytes + len(patch) >= len(raw):
                return None
    header = _HEADER.pack(
        _MAGIC, len(raw), len(old_bytes), target.geometry.row_bytes, TILE_BYTES, tile_count,
        bytes.fromhex(target.key_sha256), bytes.fromhex(target.target_sha256),
        bytes.fromhex(base.key_sha256), bytes.fromhex(base.target_sha256),
    )
    return _Delta(base, header + bitmap + patch, manifest_bytes, changed_tiles, prefix)


def _decode(target, base):
    delta = target.delta
    (magic, target_bytes, base_bytes, row_bytes, tile_bytes, tile_count,
     target_key, target_digest, base_key, base_digest) = _HEADER.unpack_from(delta.record)
    if (magic != _MAGIC or target_bytes != target.raw_bytes or base_bytes != len(base)
            or row_bytes != target.geometry.row_bytes or tile_bytes != TILE_BYTES
            or tile_count != (target_bytes + TILE_BYTES - 1) // TILE_BYTES
            or target_key.hex() != target.key_sha256 or target_digest.hex() != target.target_sha256
            or base_key.hex() != delta.base.key_sha256 or base_digest.hex() != delta.base.target_sha256
            or delta.manifest_bytes != _HEADER.size + (tile_count + 7) // 8):
        raise ValueError("Storage delta header differs")
    reconstructed = bytearray(_resize_base(base, target_bytes))
    cursor = delta.manifest_bytes
    for tile, start in enumerate(range(0, target_bytes, TILE_BYTES)):
        if delta.record[_HEADER.size + tile // 8] & (1 << (tile % 8)):
            end = min(start + TILE_BYTES, target_bytes)
            next_cursor = cursor + end - start
            if next_cursor > len(delta.record):
                raise ValueError("Incomplete storage delta tile")
            reconstructed[start:end] = delta.record[cursor:next_cursor]
            cursor = next_cursor
    if cursor != len(delta.record):
        raise ValueError("Unexpected storage delta payload")
    return reconstructed


def _fields(version):
    delta = version.delta
    base = delta.base if delta is not None else None
    return {
        "key_sha256": version.key_sha256, "target_sha256": version.target_sha256,
        "raw_bytes": version.raw_bytes, "tensor_shape": version.geometry.shape,
        "tensor_dtype": version.geometry.dtype, "row_bytes": version.geometry.row_bytes,
        "encoding": "delta" if delta is not None else "raw",
        "encoded_bytes": len(delta.record) if delta is not None else version.raw_bytes,
        "base_key_sha256": base.key_sha256 if base is not None else None,
        "base_sha256": base.target_sha256 if base is not None else None,
        "base_raw_bytes": base.raw_bytes if base is not None else None,
        "base_start_token": base.start_token if base is not None else None,
        "base_end_token": base.end_token if base is not None else None,
        "manifest_bytes": delta.manifest_bytes if delta is not None else 0,
        "patch_bytes": len(delta.record) - delta.manifest_bytes if delta is not None else 0,
        "tile_bytes": TILE_BYTES,
        "tile_count": (version.raw_bytes + TILE_BYTES - 1) // TILE_BYTES,
        "changed_tiles": delta.changed_tiles if delta is not None else 0,
        "common_prefix_tokens": delta.common_prefix_tokens if delta is not None else 0,
    }


def sample_put(manager, location, request_id, layer_id, keys, objects, starts, ends, tokens):
    if not ENABLED or location not in (None, BACKEND):
        return
    backend = manager.storage_backends.get(BACKEND)
    if backend is None:
        return
    for key, obj, start, end in zip(keys, objects, starts, ends):
        started = monotonic_ns()
        fields = {"start_token": start, "end_token": end, "candidate_count": 0}
        try:
            key_digest = _digest(key.to_string().encode())
            fields["key_sha256"] = key_digest
            status, size, geometry, raw = _resident(backend, key, end - start, expected=obj)
            fields["raw_bytes"] = size
            if status == "measured":
                version = _Version(key, key_digest, _digest(raw), geometry, size, start, end,
                                   _pack_tokens(tokens, start, end))
                chosen = None
                for prefix, base in _candidates(backend, version, layer_id):
                    base_status, _, base_geometry, old = _resident(
                        backend, base.key, base.end_token - base.start_token
                    )
                    if (base_status != "measured" or base_geometry != base.geometry
                            or _digest(old) != base.target_sha256):
                        continue
                    fields["candidate_count"] += 1
                    delta = _encode(base, old, version, raw, prefix)
                    if delta is not None and (chosen is None or len(delta.record) < len(chosen.record)):
                        chosen = delta
                version = replace(version, delta=chosen)
                _remember(backend, layer_id, version)
                fields.update(_fields(version))
            _event("legacy_kv_storage_put", request_id, layer_id, started, status=status, **fields)
        except Exception:
            _LOG.exception("Legacy KV storage sample failed")
            _event("legacy_kv_storage_put", request_id, layer_id, started, status="error", **fields)


def sample_restore(manager, location, request_id, layer_id, keys, objects, starts, ends):
    if not ENABLED or location != BACKEND:
        return
    backend = manager.storage_backends.get(BACKEND)
    if backend is None:
        return
    for key, obj, start, end in zip(keys, objects, starts, ends):
        with _LOCK:
            version = _VERSIONS.get((id(backend), key, layer_id))
        if version is None or version.delta is None:
            continue
        started = monotonic_ns()
        fields = _fields(version)
        fields.update(start_token=start, end_token=end, observed_sha256=None,
                      reconstructed_sha256=None, decode_duration_ns=None)
        try:
            target_status, _, geometry, actual = _snapshot(obj, end - start)
            if target_status == "measured":
                fields["observed_sha256"] = _digest(actual)
            if geometry != version.geometry or fields["observed_sha256"] != version.target_sha256:
                status = "target_mismatch"
            else:
                base = version.delta.base
                with _LOCK:
                    tracked_base = _VERSIONS.get((id(backend), base.key, layer_id))
                if tracked_base is not None and tracked_base.delta is not None:
                    base_status, base_geometry, old = "not_resident", None, None
                else:
                    base_status, _, base_geometry, old = _resident(
                        backend, base.key, base.end_token - base.start_token
                    )
                if base_status == "not_resident":
                    status = "missing_base"
                elif base_geometry != base.geometry or _digest(old) != base.target_sha256:
                    status = "base_mismatch"
                else:
                    decode_started = monotonic_ns()
                    try:
                        reconstructed = _decode(version, old)
                    finally:
                        fields["decode_duration_ns"] = monotonic_ns() - decode_started
                    fields["reconstructed_sha256"] = _digest(reconstructed)
                    status = "verified" if reconstructed == actual else "reconstruction_mismatch"
            _event("legacy_kv_storage_restore", request_id, layer_id, started, status=status, **fields)
        except Exception:
            _LOG.exception("Legacy KV storage reconstruction failed")
            _event("legacy_kv_storage_restore", request_id, layer_id, started, status="error", **fields)
