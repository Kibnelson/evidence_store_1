"""Lossless BF16 field codec for the offline exponent-only replay."""

import ctypes as ct
from ctypes.util import find_library
from functools import cache
import os
from pathlib import Path
from shlex import split
from struct import Struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from vllm_pd_legacy_replay_codec import _page_rows, xor_bytes


_FRAME_LENGTH = Struct(">I")
_FIELD_TAG = b"Z"
_REFERENCE = b"R"
_ZSTD_LEVEL = 3


@cache
def _field_library() -> ct.CDLL:
    source = Path(__file__).with_suffix(".c")
    mac = sys.platform == "darwin"
    flags = ["-dynamiclib"] if mac else ["-shared", "-fPIC"]
    suffix = ".dylib" if mac else ".so"
    compiler = split(os.environ.get("CC", "cc"))
    if not compiler:
        raise ValueError("CC must name a C compiler")
    with TemporaryDirectory(prefix="es-exponent-codec-") as scratch:
        output = Path(scratch) / f"fields{suffix}"
        try:
            subprocess.run([*compiler, "-O3", "-std=c11", *flags, str(source), "-o", str(output)],
                           check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Compiling the BF16 field codec failed:\n{error.stderr}") from error
        except OSError as error:
            raise RuntimeError(f"The BF16 field codec requires a C compiler: {error}") from error
        library = ct.CDLL(str(output))

    for name, inputs in (("split_fields", 1), ("join_fields", 2), ("join_delta", 3)):
        routine = getattr(library, name)
        routine.argtypes = [ct.c_char_p] * inputs + [ct.c_void_p, ct.c_size_t]
        routine.restype = None
    return library


@cache
def _zstd_library(library_path: str | None) -> ct.CDLL:
    if library_path is None:
        names = [find_library("zstd")]
        if sys.platform == "darwin":
            prefixes = ("/opt/homebrew", "/usr/local")
            names.extend(f"{prefix}/opt/zstd/lib/libzstd.dylib" for prefix in prefixes)
            names.append("libzstd.dylib")
        else:
            names.extend(("libzstd.so.1", "libzstd.so"))
    else:
        names = [library_path]
    for name in names:
        if not name:
            continue
        try:
            library = ct.CDLL(name)
        except OSError as error:
            if library_path is not None:
                raise RuntimeError(f"Could not load libzstd at {library_path}: {error}") from error
            continue
        break
    else:
        raise RuntimeError("Install libzstd or pass the path to its shared library")

    library.ZSTD_compressBound.argtypes = [ct.c_size_t]
    library.ZSTD_compressBound.restype = ct.c_size_t
    library.ZSTD_compress.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_char_p, ct.c_size_t, ct.c_int]
    library.ZSTD_compress.restype = ct.c_size_t
    library.ZSTD_decompress.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_char_p, ct.c_size_t]
    library.ZSTD_decompress.restype = ct.c_size_t
    library.ZSTD_isError.argtypes = [ct.c_size_t]
    library.ZSTD_isError.restype = ct.c_uint
    library.ZSTD_getErrorName.argtypes = [ct.c_size_t]
    library.ZSTD_getErrorName.restype = ct.c_char_p
    library.ZSTD_versionString.argtypes = []
    library.ZSTD_versionString.restype = ct.c_char_p
    return library


def _exponent_end(blob: bytes) -> int:
    if len(blob) <= _FRAME_LENGTH.size:
        raise ValueError("Truncated BF16 field frame")
    end = _FRAME_LENGTH.size + _FRAME_LENGTH.unpack_from(blob)[0]
    if not _FRAME_LENGTH.size < end < len(blob):
        raise ValueError("Invalid BF16 field frame length")
    return end


class ExponentCodec:
    """Encode standalone fields or an exponent-only delta for one page size."""

    def __init__(self, raw_bytes: int, library_path: str | None = None):
        if type(raw_bytes) is not int or raw_bytes <= 0 or raw_bytes % 2:
            raise ValueError("BF16 page size must be a positive, even number of bytes")
        self.raw_bytes = raw_bytes
        self.field_bytes = raw_bytes // 2
        self._zstd = _zstd_library(library_path)
        self._native = _field_library()

    @property
    def zstd_version(self) -> str:
        return self._zstd.ZSTD_versionString().decode("ascii")

    def _native_call(self, name: str, *inputs: bytes) -> bytes:
        out = ct.create_string_buffer(self.raw_bytes)
        getattr(self._native, name)(*inputs, out, self.field_bytes)
        return out.raw

    def split_page(self, raw: bytes) -> tuple[bytes, bytes]:
        if len(raw) != self.raw_bytes:
            raise ValueError("Raw page size differs from the BF16 codec layout")
        fields = self._native_call("split_fields", raw)
        return fields[:self.field_bytes], fields[self.field_bytes:]

    def _zstd_count(self, count: int) -> int:
        if self._zstd.ZSTD_isError(count):
            reason = self._zstd.ZSTD_getErrorName(count).decode("ascii")
            raise ValueError(f"Invalid Zstd operation: {reason}")
        return count

    def _compress(self, field: bytes) -> bytes:
        out = ct.create_string_buffer(self._zstd.ZSTD_compressBound(len(field)))
        count = self._zstd_count(self._zstd.ZSTD_compress(out, len(out), field, len(field), _ZSTD_LEVEL))
        return _FIELD_TAG + out[:count]

    def _decompress(self, frame: bytes) -> bytes:
        if frame[:1] != _FIELD_TAG:
            raise ValueError("Invalid Zstd field tag")
        out = ct.create_string_buffer(self.field_bytes)
        data = frame[1:]
        count = self._zstd_count(self._zstd.ZSTD_decompress(out, len(out), data, len(data)))
        if count != self.field_bytes:
            raise ValueError("Zstd field decoded to the wrong byte length")
        return out.raw

    def _encode_fields(self, exponent: bytes, sign_fraction: bytes) -> bytes:
        encoded_exponent = self._compress(exponent)
        return _FRAME_LENGTH.pack(len(encoded_exponent)) + encoded_exponent + self._compress(sign_fraction)

    def _decode_fields(self, blob: bytes) -> tuple[bytes, bytes]:
        end = _exponent_end(blob)
        return self._decompress(blob[_FRAME_LENGTH.size:end]), self._decompress(blob[end:])

    def encode_independent(self, raw: bytes) -> bytes:
        return self._encode_fields(*self.split_page(raw))

    def decode_independent(self, blob: bytes) -> bytes:
        return self._native_call("join_fields", *self._decode_fields(blob))

    def decode_base_exponent(self, base_frame: bytes) -> bytes:
        """Decode just the exponent field of an independently encoded base."""
        return self._decompress(base_frame[_FRAME_LENGTH.size:_exponent_end(base_frame)])

    def encode_delta_from_exponent(self, raw: bytes, base_exponent: bytes) -> bytes:
        exponent, sign_fraction = self.split_page(raw)
        return self._encode_fields(xor_bytes(exponent, base_exponent), sign_fraction)

    def restore_from_exponent(self, blob: bytes, base_exponent: bytes) -> bytes:
        if len(base_exponent) != self.field_bytes:
            raise ValueError("Base exponent size differs from the BF16 codec layout")
        return self._native_call("join_delta", base_exponent, *self._decode_fields(blob))

    def encode_delta(self, raw: bytes, base: bytes, block_size: int, row_bytes: int) -> tuple[str, bytes]:
        _page_rows(raw, base, block_size, row_bytes)
        if len(base) != self.raw_bytes:
            raise ValueError("Delta page size differs from the BF16 codec layout")
        if raw == base:
            return "reference", _REFERENCE
        return "exponent_xor", self.encode_delta_from_exponent(raw, self.split_page(base)[0])

    def decode_delta(self, kind: str, blob: bytes, base: bytes, block_size: int, row_bytes: int) -> bytes:
        _page_rows(base, base, block_size, row_bytes)
        if len(base) != self.raw_bytes:
            raise ValueError("Delta page size differs from the BF16 codec layout")
        if kind == "reference" and blob == _REFERENCE:
            return base
        if kind == "exponent_xor":
            return self.restore_from_exponent(blob, self.split_page(base)[0])
        raise ValueError("Invalid BF16 delta kind or reference frame")

    def restore_delta(self, kind: str, blob: bytes, base_frame: bytes) -> bytes:
        if kind == "reference" and blob == _REFERENCE:
            return self.decode_independent(base_frame)
        if kind == "exponent_xor":
            return self.restore_from_exponent(blob, self.decode_base_exponent(base_frame))
        raise ValueError("Invalid BF16 delta kind or reference frame")

    def encoded_base_read_bytes(self, base_frame: bytes, kind: str) -> int:
        """Count the independently encoded base bytes read on an encoded-base hit."""
        if kind == "reference":
            return len(base_frame)
        if kind == "exponent_xor":
            return _exponent_end(base_frame)
        raise ValueError(f"Unknown BF16 delta kind {kind}")
