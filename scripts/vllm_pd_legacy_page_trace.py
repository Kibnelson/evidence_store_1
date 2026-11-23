"""Shadow measurement of complete native pages in the legacy LMCache CPU tier."""

from __future__ import annotations

from array import array
from collections import OrderedDict
from dataclasses import dataclass
import gzip
from hashlib import sha256
import logging
import os
from pathlib import Path
import stat
import struct
from tempfile import NamedTemporaryFile
from threading import Lock
from time import monotonic_ns

from vllm._evidencestore_legacy_trace import ENABLED as TRACE_ENABLED, emit


def _capture_budget_mib():
    value = os.environ.get("EVIDENCESTORE_KV_PAGE_CAPTURE_MIB", "512")
    if value.isascii() and value.isdecimal():
        budget = int(value)
        if budget:
            return budget
    raise ValueError("EVIDENCESTORE_KV_PAGE_CAPTURE_MIB must be a positive integer")


ENABLED = TRACE_ENABLED and os.environ.get("EVIDENCESTORE_TRACE_KV_PAGES") == "1"
CAPTURE_ENABLED = ENABLED and os.environ.get("EVIDENCESTORE_CAPTURE_KV_PAGES") == "1"
_CAPTURE_BUDGET_MIB = _capture_budget_mib() if CAPTURE_ENABLED else 512
_EXPANDED_CAPTURE = _CAPTURE_BUDGET_MIB > 512
BACKEND = "LocalCPUBackend"
MAX_OBJECTS = 1024 if _EXPANDED_CAPTURE else 128
MAX_VERSIONS = 8192 if _EXPANDED_CAPTURE else 512
MAX_ENCODED_BYTES = (4096 if _EXPANDED_CAPTURE else 64) * 1024**2
MAX_PAGES_PER_REQUEST = 512 if _EXPANDED_CAPTURE else 128
MAX_CANDIDATES = 2
MAX_CAPTURE_RAW_BYTES = _CAPTURE_BUDGET_MIB * 1024**2
MIN_PREFIX_TOKENS = 8
# Magic, block size, layer count, K/V row width, and digests for the
# target/base versions and base tensor. The bitmap follows the header.
_HEADER = struct.Struct(">4sIII32s32s32s")
_MAGIC = b"EPG1"
_LOG = logging.getLogger(__name__)
_LOCK = Lock()
_OBJECTS = OrderedDict()
_VERSIONS = OrderedDict()
_ENCODED_BYTES = 0
_CAPTURE_LOCK = Lock()
_CAPTURED_RAW_BYTES = 0


@dataclass(frozen=True)
class Layout:
    block: int
    blocks: int
    layers: int
    row_bytes: int
    dtype: str
    native_format: str

    @property
    def raw_bytes(self):
        return self.block * self.layers * self.row_bytes * 2


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    blocks: tuple[int, ...]
    fragment: str | None


@dataclass(frozen=True)
class StoredObject:
    keys: tuple
    key_id: str
    scope_id: str
    start: int
    end: int
    tokens: bytes
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class Page:
    obj: StoredObject
    span: Span
    logical_id: str
    version_id: str
    tensor_digest: str
    layout: Layout
    codec: Codec | None


@dataclass(frozen=True)
class Codec:
    base: Page
    record: bytes
    bitmap_bytes: int
    layers: tuple[dict, ...]
    prefix: int

    @property
    def manifest_bytes(self):
        return _HEADER.size + self.bitmap_bytes


def _digest(data):
    return sha256(data).hexdigest()


def _existing_capture(path, current):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("Existing KV page payload is not a private regular file")
        with gzip.GzipFile(fileobj=stream, mode="rb") as compressed:
            if compressed.read(len(current) + 1) != current:
                raise ValueError("Existing KV page payload differs from the observed page")
        return info.st_size


def _capture_payload(tensor_digest, current):
    global _CAPTURED_RAW_BYTES
    if not CAPTURE_ENABLED:
        return {}
    try:
        directory = Path(os.environ["EVIDENCESTORE_TRACE_DIR"]).parent / "page-payloads"
        target = directory / f"{tensor_digest}.bin.gz"
        # Serialize writes and budget updates in this GPU worker. Exclusive
        # publication also deduplicates matching pages across workers.
        with _CAPTURE_LOCK:
            directory.mkdir(mode=0o700, exist_ok=True)
            info = directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError("KV page payload directory is not private")
            size = _existing_capture(target, current)
            if size is not None:
                return {"payload_capture_status": "reused", "payload_capture_compressed_bytes": size}
            if _CAPTURED_RAW_BYTES + len(current) > MAX_CAPTURE_RAW_BYTES:
                return {"payload_capture_status": "budget_exhausted"}
            temporary = None
            try:
                with NamedTemporaryFile(mode="w+b", prefix=".page-", dir=directory, delete=False) as stream:
                    temporary = Path(stream.name)
                    with gzip.GzipFile(filename="", fileobj=stream, mode="wb", compresslevel=1, mtime=0) as compressed:
                        compressed.write(current)
                    size = stream.tell()
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, target, follow_symlinks=False)
                except FileExistsError:
                    size = _existing_capture(target, current)
                    if size is None:
                        raise ValueError("Concurrent KV page payload publication disappeared")
                    return {"payload_capture_status": "reused", "payload_capture_compressed_bytes": size}
                _CAPTURED_RAW_BYTES += len(current)
                return {"payload_capture_status": "captured", "payload_capture_compressed_bytes": size}
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    except Exception:
        _LOG.exception("Could not capture diagnostic KV page payload")
        return {"payload_capture_status": "write_error"}


def _emit(kind, request_id, **fields):
    emit(kind, component="lmcache", native_request_id=request_id, backend=BACKEND, **fields)


def _backend(engine, location):
    if location not in (None, BACKEND):
        return None
    return engine.storage_manager.storage_backends.get(BACKEND)


def _layout(engine):
    from lmcache.v1.gpu_connector.utils import get_block_size, get_num_blocks

    connector = engine.gpu_connector
    caches, fmt = connector.kvcaches, connector.engine_kv_format
    block = get_block_size(caches, fmt)
    shape = connector.get_shape(block)
    if (block != 16 or len(caches) != engine.num_layers or tuple(shape[:2]) != (2, block)
            or any(get_block_size(caches, fmt, layer) != block for layer in range(engine.num_layers))):
        raise ValueError("Storage page trace expects uniform dense 16-token KV blocks")
    return Layout(block, get_num_blocks(caches, fmt), engine.num_layers,
                  shape[2] * connector.element_size, str(connector.dtype), str(fmt))


def _object(keys, start, end, tokens, spans):
    key = keys[0]
    # Use layer zero's key for the object identity. Native key tags scope
    # candidates even when a matching token prefix occurs in other requests.
    key_id = _digest(key.to_string().encode())
    scope = (key.model_name, key.world_size, key.worker_id, str(key.dtype), key.tags)
    return StoredObject(tuple(keys), key_id, _digest(repr(scope).encode()), start, end, tokens, tuple(spans))


def _packed_tokens(tokens, start, end):
    return array("I", tokens[start:end]).tobytes()


def _partition(slots, start, end, layout):
    if not 0 <= start < end <= len(slots):
        yield Span(start, end, (), "invalid_slot_mapping")
        return
    pos = start
    limit = layout.blocks * layout.block
    while pos < end:
        slot = slots[pos]
        valid = 0 <= slot < limit
        block = slot // layout.block if valid else None
        boundary = pos + 1
        while boundary < end:
            other = slots[boundary]
            if valid:
                if other != slot + boundary - pos or other // layout.block != block:
                    break
            elif 0 <= other < limit:
                break
            boundary += 1
        reason = None
        if not valid:
            reason = "invalid_slot_mapping"
        elif slot % layout.block or boundary - pos != layout.block:
            reason = "object_boundary" if pos == start or boundary == end else "invalid_slot_mapping"
        yield Span(pos, boundary, (block,) if valid else (), reason)
        pos = boundary


def _landing(slots, start, end, layout):
    if (not 0 <= start < end <= len(slots)
            or any(not 0 <= slot < layout.blocks * layout.block for slot in slots[start:end])):
        return Span(start, end, (), "invalid_slot_mapping")
    landing = slots[start:end]
    for previous, current in zip(landing, landing[1:]):
        if (previous + 1) % layout.block:
            valid = previous + 1 == current
        else:
            valid = current % layout.block == 0 and previous // layout.block != current // layout.block
        if not valid:
            return Span(start, end, (), "invalid_slot_mapping")
    blocks = tuple(dict.fromkeys(slot // layout.block for slot in landing))
    return Span(start, end, blocks, None)


def _geometry(layout):
    fields = {
        "block_size": layout.block, "layer_count": layout.layers,
        "row_bytes": layout.row_bytes, "tensor_dtype": layout.dtype, "native_kv_format": layout.native_format,
        "max_pages_per_request": MAX_PAGES_PER_REQUEST, "max_shadow_encoded_bytes": MAX_ENCODED_BYTES,
        "max_tracked_pages": MAX_VERSIONS, "max_tracked_objects": MAX_OBJECTS, "max_candidates": MAX_CANDIDATES,
    }
    if CAPTURE_ENABLED:
        fields["max_capture_raw_bytes"] = MAX_CAPTURE_RAW_BYTES
    return fields


def _positions(obj, span):
    return {
        "object_key_sha256": obj.key_id, "object_start_token": obj.start, "object_end_token": obj.end,
        "start_token": span.start, "end_token": span.end, "relative_start_token": span.start - obj.start,
        "worker_block_id": span.blocks[0] if len(span.blocks) == 1 else None,
        "destination_block_ids": span.blocks,
    }


def _fragment(request_id, operation, obj, span, layout):
    _emit("legacy_kv_page_fragment", request_id, operation=operation,
          fragment_reason=span.fragment, status="fragment",
          raw_bytes=(span.end - span.start) * layout.layers * 2 * layout.row_bytes,
          **_positions(obj, span), **_geometry(layout))


def _layer_bytes(memory, obj_tokens, offset, layout):
    tensor = memory.tensor
    plane_size = obj_tokens * layout.row_bytes
    if (tensor is None or tensor.device.type != "cpu" or not tensor.is_contiguous()
            or getattr(memory.get_memory_format(), "name", None) != "KV_2TD"
            or len(tensor.shape) != 3 or tuple(tensor.shape[:2]) != (2, obj_tokens)
            or str(tensor.dtype) != layout.dtype or tensor.shape[2] * tensor.element_size() != layout.row_bytes
            or memory.get_size() != plane_size * 2 or not 0 <= offset <= obj_tokens - layout.block):
        return None
    data = memoryview(memory.byte_array).cast("B")
    if data.nbytes != plane_size * 2:
        return None
    lo = offset * layout.row_bytes
    hi = lo + layout.block * layout.row_bytes
    return bytes(data[lo:hi]) + bytes(data[plane_size + lo:plane_size + hi])


def _resident_layer(backend, key, tokens, offset, layout, expected=None):
    with backend.cpu_lock:
        memory = backend.hot_cache.get(key)
        if memory is None or (expected is not None and memory is not expected):
            return "not_resident", None
        memory.ref_count_up()
    try:
        payload = _layer_bytes(memory, tokens, offset, layout)
        return ("measured", payload) if payload is not None else ("unsupported", None)
    finally:
        memory.ref_count_down()


def _read_page(backend, obj, span, layout, expected=None):
    layers = []
    for layer, key in enumerate(obj.keys):
        status, payload = _resident_layer(
            backend, key, obj.end - obj.start, span.start - obj.start, layout,
            None if expected is None else expected[layer],
        )
        if status != "measured":
            return status, None
        layers.append(payload)
    return "measured", b"".join(layers)


def _remember(identity, page):
    global _ENCODED_BYTES
    with _LOCK:
        previous = _VERSIONS.pop(identity, None)
        if previous is not None and previous.codec is not None:
            _ENCODED_BYTES -= len(previous.codec.record)
        if page.codec is not None:
            _ENCODED_BYTES += len(page.codec.record)
        _VERSIONS[identity] = page
        while len(_VERSIONS) > MAX_VERSIONS or _ENCODED_BYTES > MAX_ENCODED_BYTES:
            _, previous = _VERSIONS.popitem(last=False)
            if previous.codec is not None:
                _ENCODED_BYTES -= len(previous.codec.record)


def _prefix(left, right):
    for index, (first, second) in enumerate(zip(memoryview(left).cast("I"), memoryview(right).cast("I"))):
        if first != second:
            return index
    return min(len(left), len(right)) // array("I").itemsize


def _candidate_pages(backend, page):
    relative = page.span.start - page.obj.start
    candidates = []
    with _LOCK:
        recent = tuple(reversed(_VERSIONS.items()))
    for identity, earlier in recent:
        if (identity[0] == id(backend) and earlier.codec is None and earlier.obj.key_id != page.obj.key_id
                and identity[2] == relative and earlier.obj.scope_id == page.obj.scope_id
                and earlier.layout == page.layout):
            prefix = _prefix(earlier.obj.tokens, page.obj.tokens)
            if prefix >= MIN_PREFIX_TOKENS:
                in_page = min(page.layout.block, max(0, prefix - relative))
                candidates.append((in_page, prefix, earlier))
    candidates.sort(key=lambda row: (row[0], row[1], row[2].span.start == page.span.start), reverse=True)
    return candidates[:MAX_CANDIDATES]


def _pack(base, target, earlier, current, prefix):
    layout = target.layout
    row_width, block, layers = layout.row_bytes, layout.block, layout.layers
    bitmap = bytearray((block * layers * 2 + 7) // 8)
    patch = bytearray()
    counts = []
    for layer in range(layers):
        k_changed_rows = v_changed_rows = 0
        for index in range(2 * block):
            row = layer * 2 * block + index
            offset = row * row_width
            value = current[offset:offset + row_width]
            if value != earlier[offset:offset + row_width]:
                bitmap[row // 8] |= 1 << (row % 8)
                patch.extend(value)
                if index < block:
                    k_changed_rows += 1
                else:
                    v_changed_rows += 1
        counts.append({"layer_id": layer, "k_changed_rows": k_changed_rows, "v_changed_rows": v_changed_rows})
    header = _HEADER.pack(_MAGIC, block, layers, row_width, bytes.fromhex(target.version_id),
                          bytes.fromhex(base.version_id), bytes.fromhex(base.tensor_digest))
    return Codec(base, header + bitmap + patch, len(bitmap), tuple(counts), prefix)


def _page_fields(page):
    codec = page.codec
    base = codec.base if codec is not None else None
    patch_bytes = len(codec.record) - codec.manifest_bytes if codec is not None else 0
    return {
        "logical_page_sha256": page.logical_id, "version_sha256": page.version_id,
        "tensor_sha256": page.tensor_digest,
        "encoding": ("patch" if patch_bytes else "reference") if codec is not None else "raw",
        "raw_bytes": page.layout.raw_bytes, "encoded_bytes": len(codec.record) if codec else page.layout.raw_bytes,
        "manifest_bytes": codec.manifest_bytes if codec else 0, "patch_bytes": patch_bytes,
        "base_logical_page_sha256": base.logical_id if base else None,
        "base_object_key_sha256": base.obj.key_id if base else None,
        "base_version_sha256": base.version_id if base else None,
        "base_tensor_sha256": base.tensor_digest if base else None,
        "base_raw_bytes": base.layout.raw_bytes if base else None,
        "base_start_token": base.span.start if base else None, "base_end_token": base.span.end if base else None,
        "common_prefix_tokens": codec.prefix if codec else 0, "layers": codec.layers if codec else (),
        "common_in_page_prefix_tokens": min(page.layout.block, max(0, codec.prefix - page.span.start + page.obj.start))
        if codec else 0,
    }


def _sample_store(backend, request_id, obj, span, layout, expected):
    started = monotonic_ns()
    common = {**_geometry(layout), **_positions(obj, span)}
    status, current = _read_page(backend, obj, span, layout, expected)
    if status != "measured":
        _emit("legacy_kv_page_put", request_id, status=status, raw_bytes=layout.raw_bytes,
              duration_ns=monotonic_ns() - started, **common)
        return
    relative = span.start - obj.start
    token_width = array("I").itemsize
    token_bytes = obj.tokens[relative * token_width:(relative + layout.block) * token_width]
    logical = _digest(obj.scope_id.encode() + struct.pack(">II", layout.block, relative) + token_bytes)
    tensor = _digest(current)
    version = _digest((logical + obj.key_id + tensor).encode())
    page = Page(obj, span, logical, version, tensor, layout, None)
    codec = None
    best = None
    candidate_count = 0
    reason = "no_compatible_base"
    for _, prefix, base in _candidate_pages(backend, page):
        base_status, earlier = _read_page(backend, base.obj, base.span, layout)
        if base_status != "measured":
            reason = "base_unavailable"
            continue
        if _digest(earlier) != base.tensor_digest:
            reason = "base_mismatch"
            continue
        candidate_count += 1
        candidate = _pack(base, page, earlier, current, prefix)
        if best is None or len(candidate.record) < len(best.record):
            best = candidate
    if best is not None and len(best.record) < len(current):
        codec = best
    if codec is not None:
        page = Page(obj, span, logical, version, tensor, layout, codec)
        reason = "selected"
    elif candidate_count:
        reason = "not_smaller"
    _remember((id(backend), obj.key_id, relative), page)
    page_fields = _page_fields(page)
    duration = monotonic_ns() - started
    capture_fields = _capture_payload(tensor, current)
    _emit("legacy_kv_page_put", request_id, status="measured", **common, **page_fields,
          candidate_status=reason, candidate_count=candidate_count, base_available=None,
          compared_layers=best.layers if best is not None and codec is None else (),
          compared_base_version_sha256=best.base.version_id if best is not None and codec is None else None,
          duration_ns=duration, **capture_fields)


def sample_put(engine, request_id, keys, objects, starts, ends, tokens, slots):
    if not ENABLED:
        return
    try:
        backend = _backend(engine, engine.store_location)
        if backend is None:
            return
        layout = _layout(engine)
        slots = slots.tolist()
        if not isinstance(tokens, list):
            tokens = tokens.tolist()
        count = 0
        for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
            obj = _object([row[i] for row in keys], start, end,
                          _packed_tokens(tokens, start, end), _partition(slots, start, end, layout))
            identity = (id(backend), obj.key_id)
            with _LOCK:
                _OBJECTS[identity] = obj
                _OBJECTS.move_to_end(identity)
                if len(_OBJECTS) > MAX_OBJECTS:
                    _OBJECTS.popitem(last=False)
            expected = [row[i] for row in objects]
            for span in obj.spans:
                if span.fragment:
                    _fragment(request_id, "put", obj, span, layout)
                elif count >= MAX_PAGES_PER_REQUEST:
                    _emit("legacy_kv_page_put", request_id, status="sample_limit",
                          raw_bytes=layout.raw_bytes, **_positions(obj, span), **_geometry(layout))
                else:
                    count += 1
                    _sample_store(backend, request_id, obj, span, layout, expected)
    except Exception:
        _LOG.exception("Legacy LMCache page put sampling failed")
        _emit("legacy_kv_page_put", request_id, status="error")


class RestorePage:
    def __init__(self, obj, span, page, status):
        self.obj, self.span, self.page, self.status = obj, span, page, status
        self.observed = sha256()
        self.reconstructed = sha256()
        self.base_hash = sha256()
        self.duration_ns = self.decode_ns = self.layers = 0
        self.base_read_bytes = 0
        self.cursor = page.codec.manifest_bytes if page is not None and page.codec else None
        self.base_available = None
        self.exact = True

    def sample(self, backend, layer, memory, layout):
        if self.page is None or self.status not in ("pending", "missing_base", "base_mismatch"):
            return
        started = monotonic_ns()
        try:
            current = _layer_bytes(memory, self.obj.end - self.obj.start, self.span.start - self.obj.start, layout)
            if current is None:
                self.status = "target_mismatch"
                return
            self.observed.update(current)
            self.layers += 1
            if self.cursor is None or self.status != "pending":
                return
            base = self.page.codec.base
            with _LOCK:
                latest = _VERSIONS.get((id(backend), base.obj.key_id, base.span.start - base.obj.start))
            if latest is not None and (latest.codec is not None or latest.version_id != base.version_id):
                self.status, self.base_available = "missing_base", False
                return
            status, source = _resident_layer(
                backend, base.obj.keys[layer], base.obj.end - base.obj.start,
                base.span.start - base.obj.start, layout,
            )
            if status == "not_resident":
                self.status, self.base_available = "missing_base", False
                return
            if status != "measured":
                self.status, self.base_available = "base_mismatch", False
                return
            self.base_read_bytes += len(source)
            self.base_hash.update(source)
            decode_started = monotonic_ns()
            try:
                restored, self.cursor = _decode_layer(self.page, layer, source, self.cursor)
            finally:
                self.decode_ns += monotonic_ns() - decode_started
            self.reconstructed.update(restored)
            self.exact = self.exact and restored == current
        except Exception:
            self.status = "error"
            _LOG.exception("Legacy LMCache page reconstruction failed")
        finally:
            self.duration_ns += monotonic_ns() - started

    def finish(self, request_id, layout):
        fields = {**_positions(self.obj, self.span), **_geometry(layout), "raw_bytes": layout.raw_bytes}
        status, observed, reconstructed = self.status, None, None
        page = self.page
        if page is not None:
            fields.update(_page_fields(page), old_start_token=page.span.start, old_end_token=page.span.end,
                          source_worker_block_id=page.span.blocks[0])
            if self.layers == layout.layers:
                observed = self.observed.hexdigest()
            if status == "pending":
                if observed != page.tensor_digest:
                    status = "target_mismatch"
                elif page.codec is None:
                    status = "raw"
                elif self.base_hash.hexdigest() != page.codec.base.tensor_digest:
                    status, self.base_available = "base_mismatch", False
                else:
                    self.base_available = True
                    reconstructed = self.reconstructed.hexdigest()
                    status = ("verified" if self.cursor == len(page.codec.record) and self.exact
                              and reconstructed == observed
                              else "reconstruction_mismatch")
        _emit("legacy_kv_page_restore", request_id, status=status, **fields,
              observed_sha256=observed, reconstructed_sha256=reconstructed,
              base_available=self.base_available, duration_ns=self.duration_ns,
              base_read_bytes=self.base_read_bytes,
              decode_duration_ns=self.decode_ns if self.cursor is not None else None)


def _decode_layer(page, layer, source, cursor):
    codec = page.codec
    layout = page.layout
    if layer == 0:
        header = _HEADER.unpack_from(codec.record)
        if header != (_MAGIC, layout.block, layout.layers, layout.row_bytes, bytes.fromhex(page.version_id),
                      bytes.fromhex(codec.base.version_id), bytes.fromhex(codec.base.tensor_digest)):
            raise ValueError("Storage page manifest differs from the selected base")
    restored = bytearray(source)
    for index in range(layout.block * 2):
        row = layer * layout.block * 2 + index
        if codec.record[_HEADER.size + row // 8] & (1 << (row % 8)):
            first = index * layout.row_bytes
            end = cursor + layout.row_bytes
            if end > len(codec.record):
                raise ValueError("Incomplete storage page replacement row")
            restored[first:first + layout.row_bytes] = codec.record[cursor:end]
            cursor = end
    return restored, cursor


class RestoreSample:
    def __init__(self, backend, request_id, layout, pages, fragments):
        self.backend, self.request_id, self.layout = backend, request_id, layout
        self.pages, self.fragments = pages, fragments

    def sample_layer(self, layer, objects):
        for index, page in self.pages:
            if index < len(objects):
                page.sample(self.backend, layer, objects[index], self.layout)
            else:
                page.status = "target_mismatch"

    def finish(self):
        for obj, span in self.fragments:
            _fragment(self.request_id, "restore", obj, span, self.layout)
        for _, page in self.pages:
            page.finish(self.request_id, self.layout)


def start_restore(engine, request_id, location, keys, starts, ends, slots):
    if not ENABLED or location != BACKEND:
        return None
    try:
        backend = _backend(engine, location)
        if backend is None:
            return None
        layout = _layout(engine)
        slots = slots.tolist()
        pages, fragments = [], []
        for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
            obj = _object([row[i] for row in keys], start, end, b"", _partition(slots, start, end, layout))
            with _LOCK:
                prior = _OBJECTS.get((id(backend), obj.key_id))
            if prior is None or prior.end - prior.start != end - start:
                source_spans = obj.spans
                source_start = start
            else:
                source_spans = prior.spans
                source_start = prior.start
            for stored in source_spans:
                relative = stored.start - source_start
                landing = _landing(slots, start + relative, start + relative + stored.end - stored.start, layout)
                if stored.fragment or landing.fragment:
                    fragments.append((obj, Span(landing.start, landing.end, landing.blocks,
                                                landing.fragment or stored.fragment)))
                    continue
                with _LOCK:
                    page = _VERSIONS.get((id(backend), obj.key_id, relative))
                if len(pages) >= MAX_PAGES_PER_REQUEST:
                    pages.append((i, RestorePage(obj, landing, None, "sample_limit")))
                elif page is None or page.layout != layout:
                    pages.append((i, RestorePage(obj, landing, None, "untracked")))
                else:
                    pages.append((i, RestorePage(obj, landing, page, "pending")))
        return RestoreSample(backend, request_id, layout, pages, fragments)
    except Exception:
        _LOG.exception("Legacy LMCache page restore setup failed")
        _emit("legacy_kv_page_restore", request_id, status="error")
        return None
