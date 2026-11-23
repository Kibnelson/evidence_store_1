"""Exact compact row frames for small native KV pages."""

from vllm_pd_legacy_replay_codec import (
    _page_rows,
    apply_row_patch,
    build_row_patch,
    decode_delta,
    decode_independent,
    encode_delta,
    encode_independent,
)


_LITERAL = 255
_MAX_BLOCK = 16
_BITMAP = "row_bitmap"
_SPARSE = "row_sparse"
_TAGS = {_BITMAP: b"\x03", _SPARSE: b"\x04"}


def _write_varint(out: bytearray, value: int) -> None:
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)


def _read_varint(encoded: bytes, start: int, maximum: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, max(1, maximum.bit_length()), 7):
        if start == len(encoded):
            break
        digit = encoded[start]
        start += 1
        value |= (digit & 127) << shift
        if digit < 128 and value <= maximum and (shift == 0 or digit):
            return value, start
        if digit < 128:
            break
    raise ValueError("Invalid packed row varint")


def _bitmap_prefix(refs: list[tuple[int, int]], row_count: int) -> bytes:
    bitmap = bytearray((row_count + 7) // 8)
    slots = bytearray((len(refs) + 1) // 2)
    for index, (row, slot) in enumerate(refs):
        bitmap[row // 8] |= 1 << (row % 8)
        slots[index // 2] |= slot << (4 * (index % 2))
    return _TAGS[_BITMAP] + bitmap + slots


def _sparse_prefix(refs: list[tuple[int, int]]) -> bytes:
    out = bytearray(_TAGS[_SPARSE])
    _write_varint(out, len(refs))
    previous = -1
    for row, slot in refs:
        _write_varint(out, (row - previous - 1) << 4 | slot)
        previous = row
    return bytes(out)


def encode_packed_delta(target: bytes, base: bytes, block_size: int, row_bytes: int) -> tuple[str, bytes]:
    """Choose the shortest complete frame, keeping the original codec on ties."""
    original_kind, original = encode_delta(target, base, block_size, row_bytes)
    if original_kind == "reference" or block_size > _MAX_BLOCK:
        return original_kind, original

    row_count = len(base) // row_bytes
    if original_kind == "row":
        selectors = original[1:1 + row_count]
        literal_frame = original[1 + row_count:]
    else:
        selectors, literals = build_row_patch(target, base, block_size, row_bytes)
        if len(literals) == len(target):
            return original_kind, original
        literal_frame = encode_independent(literals)

    refs = [(row, slot) for row, slot in enumerate(selectors) if slot != _LITERAL]
    kind, prefix = min(
        ((_BITMAP, _bitmap_prefix(refs, row_count)), (_SPARSE, _sparse_prefix(refs))),
        key=lambda frame: len(frame[1]),
    )
    if len(prefix) + len(literal_frame) < len(original):
        return kind, prefix + literal_frame
    return original_kind, original


def _read_bitmap(encoded: bytes, row_count: int) -> tuple[bytes, int]:
    bitmap_end = 1 + (row_count + 7) // 8
    if bitmap_end >= len(encoded):
        raise ValueError("Truncated row bitmap")
    bitmap = encoded[1:bitmap_end]
    if row_count % 8 and bitmap[-1] >> (row_count % 8):
        raise ValueError("Packed row bitmap refers outside the page")
    count = sum(byte.bit_count() for byte in bitmap)
    packed_end = bitmap_end + (count + 1) // 2
    if packed_end >= len(encoded) or (count % 2 and encoded[packed_end - 1] >> 4):
        raise ValueError("Invalid packed row slot stream")

    selectors = bytearray([_LITERAL]) * row_count
    index = 0
    for row in range(row_count):
        if bitmap[row // 8] & (1 << (row % 8)):
            packed = encoded[bitmap_end + index // 2]
            selectors[row] = (packed >> (4 * (index % 2))) & 15
            index += 1
    return bytes(selectors), packed_end


def _read_sparse(encoded: bytes, row_count: int) -> tuple[bytes, int]:
    count, cursor = _read_varint(encoded, 1, row_count)
    selectors = bytearray([_LITERAL]) * row_count
    previous = -1
    for _ in range(count):
        entry, cursor = _read_varint(encoded, cursor, ((row_count - 1) << 4) | 15)
        row = previous + 1 + (entry >> 4)
        if row >= row_count:
            raise ValueError("Packed row reference is outside the page")
        selectors[row] = entry & 15
        previous = row
    return bytes(selectors), cursor


def decode_packed_delta(kind: str, encoded: bytes, base: bytes, block_size: int, row_bytes: int) -> bytes:
    """Decode packed row frames and the three original delta kinds."""
    if kind not in _TAGS:
        return decode_delta(kind, encoded, base, block_size, row_bytes)
    row_count = _page_rows(base, base, block_size, row_bytes)
    if block_size > _MAX_BLOCK or encoded[:1] != _TAGS[kind]:
        raise ValueError("Invalid packed KV row frame")
    if kind == _BITMAP:
        selectors, literal_start = _read_bitmap(encoded, row_count)
    else:
        selectors, literal_start = _read_sparse(encoded, row_count)
    literals = decode_independent(encoded[literal_start:])
    return apply_row_patch(selectors, literals, base, block_size, row_bytes)
