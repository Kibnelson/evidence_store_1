"""Lossless independent KV frames using a per-page BF16 exponent palette."""

from collections import Counter
from struct import Struct
import zlib

from vllm_pd_legacy_replay_codec import (
    _BF16,
    _FRAMED_LENGTH,
    _inflate,
    _join_bf16,
    _split_bf16,
    decode_independent,
    encode_independent,
)


_TAG = b"\x02"
# Tag, compressed-escape flag, palette, code count, and lengths of the two
# exponent streams. The remaining bytes hold the zlib sign/fraction stream.
_HEADER = Struct(">cB15sIII")
_ESCAPE = 15
_LOW = bytes(value & 15 for value in range(256))
_HIGH = bytes(value >> 4 for value in range(256))


def encode_packed_independent(raw: bytes) -> bytes:
    """Choose the smallest complete independent frame, preserving legacy ties."""
    original = encode_independent(raw)
    if not raw or len(raw) % 2 or len(raw) // 2 > 0xFFFFFFFF:
        return original

    exponents, sign_fractions = _split_bf16(raw)
    counts = Counter(exponents)
    palette = bytes(sorted(range(256), key=lambda value: (-counts[value], value))[:_ESCAPE])
    table = bytearray([_ESCAPE]) * 256
    for code, exponent in enumerate(palette):
        table[exponent] = code
    codes = exponents.translate(table)
    packed = int.from_bytes(codes[::2], "little") | (int.from_bytes(codes[1::2], "little") << 4)
    code_stream = zlib.compress(packed.to_bytes((len(codes) + 1) // 2, "little"), level=1)
    escapes = exponents.translate(None, palette)
    compressed_escapes = zlib.compress(escapes, level=1)
    compress_escapes = len(compressed_escapes) < len(escapes)
    escape_stream = compressed_escapes if compress_escapes else escapes
    if original[:1] == _BF16:
        sf_offset = 1 + _FRAMED_LENGTH.size + _FRAMED_LENGTH.unpack_from(original, 1)[0]
        sf_stream = original[sf_offset:]
    else:
        sf_stream = zlib.compress(sign_fractions, level=1)

    frame_size = _HEADER.size + len(code_stream) + len(escape_stream) + len(sf_stream)
    if frame_size >= len(original):
        return original
    header = _HEADER.pack(_TAG, compress_escapes, palette, len(codes), len(code_stream), len(escape_stream))
    return header + code_stream + escape_stream + sf_stream


def decode_packed_independent(frame: bytes) -> bytes:
    """Decode a packed BF16 frame or an existing independent frame exactly."""
    if frame[:1] != _TAG:
        return decode_independent(frame)
    if len(frame) < _HEADER.size:
        raise ValueError("Truncated BF16 palette header")
    _, flags, palette, count, code_len, escape_len = _HEADER.unpack_from(frame)
    code_end = _HEADER.size + code_len
    escape_end = code_end + escape_len
    if flags not in (0, 1) or len(set(palette)) != _ESCAPE or not count or not code_len or escape_end >= len(frame):
        raise ValueError("Invalid BF16 palette header")

    packed = _inflate(frame[_HEADER.size:code_end])
    sign_fractions = _inflate(frame[escape_end:])
    if len(sign_fractions) != count or len(packed) != (count + 1) // 2 or (count % 2 and packed[-1] >> 4):
        raise ValueError("Invalid BF16 palette code stream length or padding")
    codes = bytearray(count)
    codes[::2] = packed.translate(_LOW)
    codes[1::2] = packed.translate(_HIGH)[:count // 2]

    stream = frame[code_end:escape_end]
    escapes = _inflate(stream) if flags else stream
    if len(escapes) != codes.count(_ESCAPE) or len(escapes.translate(None, palette)) != len(escapes):
        raise ValueError("Invalid BF16 palette escape stream")
    exponents = codes.translate(palette + bytes(256 - _ESCAPE))
    position = -1
    for exponent in escapes:
        position = codes.find(_ESCAPE, position + 1)
        exponents[position] = exponent
    return _join_bf16(bytes(exponents), sign_fractions)
