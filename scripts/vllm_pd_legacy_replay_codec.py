"""Lossless CPU codecs for canonical little-endian BF16 KV page bytes."""

from functools import lru_cache
from struct import Struct
import zlib


_FRAMED_LENGTH = Struct(">I")
_RAW = b"\x00"
_BF16 = b"\x01"
_DELTA_TAGS = {"reference": b"\x00", "xor": b"\x01", "row": b"\x02"}
_LITERAL = 255


@lru_cache(maxsize=4)
def _masks(size: int) -> tuple[int, int, int]:
    return tuple(int.from_bytes(bytes([value]) * size, "little") for value in (1, 127, 128))


def _split_bf16(raw: bytes) -> tuple[bytes, bytes]:
    size = len(raw) // 2
    _, magnitude_bits, top_bit = _masks(size)
    low = int.from_bytes(raw[::2], "little")
    high = int.from_bytes(raw[1::2], "little")
    exponents = ((high & magnitude_bits) << 1) | ((low & top_bit) >> 7)
    sign_fractions = (high & top_bit) | (low & magnitude_bits)
    return exponents.to_bytes(size, "little"), sign_fractions.to_bytes(size, "little")


def _join_bf16(exponents: bytes, sign_fractions: bytes) -> bytes:
    if len(exponents) != len(sign_fractions):
        raise ValueError("BF16 streams have different lengths")
    size = len(exponents)
    first_bit, magnitude_bits, top_bit = _masks(size)
    exponent = int.from_bytes(exponents, "little")
    sign_fraction = int.from_bytes(sign_fractions, "little")
    low = (sign_fraction & magnitude_bits) | ((exponent & first_bit) << 7)
    high = (sign_fraction & top_bit) | ((exponent >> 1) & magnitude_bits)
    raw = bytearray(2 * size)
    raw[::2] = low.to_bytes(size, "little")
    raw[1::2] = high.to_bytes(size, "little")
    return bytes(raw)


def _inflate(stream: bytes) -> bytes:
    decoder = zlib.decompressobj()
    data = decoder.decompress(stream)
    if not decoder.eof or decoder.unused_data:
        raise ValueError("Invalid zlib stream in BF16 frame")
    return data


def encode_independent(raw: bytes) -> bytes:
    """Encode a page without a base, keeping raw bytes if compression expands it."""
    if raw and len(raw) % 2 == 0:
        exponents, sign_fractions = _split_bf16(raw)
        exponent_stream = zlib.compress(exponents, level=1)
        sign_fraction_stream = zlib.compress(sign_fractions, level=1)
        if _FRAMED_LENGTH.size + len(exponent_stream) + len(sign_fraction_stream) < len(raw):
            return _BF16 + _FRAMED_LENGTH.pack(len(exponent_stream)) + exponent_stream + sign_fraction_stream
    return _RAW + raw


def decode_independent(encoded: bytes) -> bytes:
    tag = encoded[:1]
    if tag == _RAW:
        return encoded[1:]
    if tag == _BF16 and len(encoded) > 1 + _FRAMED_LENGTH.size:
        start = 1 + _FRAMED_LENGTH.size
        end = start + _FRAMED_LENGTH.unpack_from(encoded, 1)[0]
        if start < end < len(encoded):
            return _join_bf16(_inflate(encoded[start:end]), _inflate(encoded[end:]))
    raise ValueError("Invalid independent KV frame")


def _page_rows(raw: bytes, base: bytes, block_size: int, row_bytes: int) -> int:
    if (type(block_size) is not int or type(row_bytes) is not int
            or block_size <= 0 or row_bytes <= 0 or len(raw) != len(base) or not raw
            or len(base) % (2 * block_size * row_bytes)):
        raise ValueError("Delta buffers must contain complete, matching K and V planes")
    return len(base) // row_bytes


def xor_bytes(raw: bytes, base: bytes) -> bytes:
    """XOR two equally sized buffers while preserving leading zero bytes."""
    if len(raw) != len(base):
        raise ValueError("XOR buffers must have equal lengths")
    return (int.from_bytes(raw, "little") ^ int.from_bytes(base, "little")).to_bytes(len(base), "little")


def build_row_patch(raw: bytes, base: bytes, block_size: int, row_bytes: int) -> tuple[bytes, bytes]:
    """Return base slot selectors and raw literal rows in target row order."""
    _page_rows(raw, base, block_size, row_bytes)
    if block_size > _LITERAL:
        raise ValueError("KV row patch requires at most 255 slots per plane")
    selectors = bytearray()
    literals = bytearray()
    plane_bytes = block_size * row_bytes
    for plane in range(0, len(base), plane_bytes):
        base_rows = {}
        for slot, position in enumerate(range(plane, plane + plane_bytes, row_bytes)):
            base_rows.setdefault(base[position:position + row_bytes], slot)
        for position in range(plane, plane + plane_bytes, row_bytes):
            row = raw[position:position + row_bytes]
            source_slot = base_rows.get(row, _LITERAL)
            selectors.append(source_slot)
            if source_slot == _LITERAL:
                literals.extend(row)
    return bytes(selectors), bytes(literals)


def apply_row_patch(selectors: bytes, literals: bytes, base: bytes, block_size: int, row_bytes: int) -> bytes:
    """Reconstruct a target from raw literal rows and exact base slot references."""
    row_count = _page_rows(base, base, block_size, row_bytes)
    if block_size > _LITERAL:
        raise ValueError("KV row patch requires at most 255 slots per plane")
    if len(selectors) != row_count:
        raise ValueError("KV row patch has the wrong number of selectors")
    if len(literals) != selectors.count(_LITERAL) * row_bytes:
        raise ValueError("KV row frame has the wrong number of literal rows")
    restored = bytearray()
    literal_start = 0
    for slot, source_slot in enumerate(selectors):
        if source_slot == _LITERAL:
            restored.extend(literals[literal_start:literal_start + row_bytes])
            literal_start += row_bytes
        elif source_slot < block_size:
            start = (slot - slot % block_size + source_slot) * row_bytes
            restored.extend(base[start:start + row_bytes])
        else:
            raise ValueError("KV row reference is outside its K or V plane")
    return bytes(restored)


def encode_delta(raw: bytes, base: bytes, block_size: int, row_bytes: int) -> tuple[str, bytes]:
    """Encode a delta against one base. The caller accounts for its base identifier."""
    _page_rows(raw, base, block_size, row_bytes)
    if raw == base:
        return "reference", _DELTA_TAGS["reference"]
    xor = _DELTA_TAGS["xor"] + encode_independent(xor_bytes(raw, base))
    if block_size <= _LITERAL:
        selectors, literals = build_row_patch(raw, base, block_size, row_bytes)
        if len(literals) < len(raw):
            rows = _DELTA_TAGS["row"] + selectors + encode_independent(literals)
            if len(rows) < len(xor):
                return "row", rows
    return "xor", xor


def decode_delta(kind: str, encoded: bytes, base: bytes, block_size: int, row_bytes: int) -> bytes:
    row_count = _page_rows(base, base, block_size, row_bytes)
    if not encoded or _DELTA_TAGS.get(kind) != encoded[:1]:
        raise ValueError("KV delta frame does not match its codec kind")
    if kind == "reference" and len(encoded) == 1:
        return base
    if kind == "xor":
        residual = decode_independent(encoded[1:])
        if len(residual) == len(base):
            return xor_bytes(residual, base)
    if kind == "row" and block_size <= _LITERAL and len(encoded) > 1 + row_count:
        selectors = encoded[1:1 + row_count]
        literals = decode_independent(encoded[1 + row_count:])
        return apply_row_patch(selectors, literals, base, block_size, row_bytes)
    raise ValueError("Invalid KV delta frame")
