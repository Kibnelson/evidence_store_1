"""Read and verify content-addressed native KV page captures."""

import gzip
from hashlib import sha256
from pathlib import Path
import re
import stat
import zlib


PAYLOAD_DIRECTORY = "page-payloads"
CAPTURE_STATUSES = {"captured", "reused", "budget_exhausted", "write_error"}
AVAILABLE = {"captured", "reused"}
CAPTURE_LAYOUT = "layer, K then V, token, row bytes"


def payload_path(tensor_sha256: str) -> Path:
    if not isinstance(tensor_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", tensor_sha256) is None:
        raise ValueError("Invalid page tensor digest")
    return Path(PAYLOAD_DIRECTORY) / f"{tensor_sha256}.bin.gz"


def load_payload(cohort: Path, tensor_sha256: str, raw_bytes: int) -> bytes:
    """Read only a regular capture and verify length and SHA-256 of its tensor."""
    if type(raw_bytes) is not int or raw_bytes <= 0:
        raise ValueError("Invalid native page length")
    path = cohort / payload_path(tensor_sha256)
    if path.parent.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Expected a regular page capture at {path}")
    try:
        with gzip.open(path, "rb") as source:
            payload = source.read(raw_bytes + 1)
    except (EOFError, OverflowError, zlib.error) as error:
        raise ValueError(f"Incomplete page capture or invalid length at {path}") from error
    if len(payload) != raw_bytes or sha256(payload).hexdigest() != tensor_sha256:
        raise ValueError(f"Page capture length or digest differs at {path}")
    return payload


def capture_records(report: dict) -> dict[str, int]:
    """Select only verified store captures in the recorded natural workload."""
    if "raw_page_capture" not in report:
        return {}
    capture = report["raw_page_capture"]
    if capture["schema_version"] != 1 or capture["layout"] != CAPTURE_LAYOUT or report["replay_label"] != "mtrag":
        raise ValueError("Unexpected raw page capture contract")
    natural = {row["request_id"]: row["request_key"] for row in report["turns"]}
    records = {}
    for row in report["stores"]:
        if (row.get("payload_capture_status") not in AVAILABLE or row["status"] != "measured"
                or row.get("native_transfer_matched") is not True
                or row.get("request_id") not in natural or natural[row["request_id"]] != row["request_key"]):
            continue
        digest, size = row["tensor_sha256"], row["raw_bytes"]
        payload_path(digest)
        if (type(size) is not int or size <= 0
                or size != 2 * row["block_size"] * row["layer_count"] * row["row_bytes"]
                or records.setdefault(digest, size) != size):
            raise ValueError("Raw page capture lengths do not match store geometry")
    return records
