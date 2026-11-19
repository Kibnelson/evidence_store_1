#!/usr/bin/env python3
"""Stage one author-released, labeled development split for the PD campaign."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidencestore.retrieval import sha256_file  # noqa: E402


SOURCES = {
    "musique": (
        "https://drive.usercontent.google.com/download?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download&confirm=t",
        "data/musique_ans_v1.0_dev.jsonl",
        "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b",
    ),
    "2wiki": (
        "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1",
        "dev.json",
        "79f77ae104088ea8e25b1a65dbece768d45771194663bc5660ec9a98070dadf5",
    ),
}


def check_split(path, expected):
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{path} differs from the pinned development split")


def stage(dataset, output):
    url, member, fingerprint = SOURCES[dataset]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / Path(member).name
    if destination.exists():
        check_split(destination, fingerprint)
        print(f"Using verified {dataset} development split at {destination}", file=sys.stderr)
        return destination

    with TemporaryDirectory(prefix=f".{dataset}-", dir=output) as directory:
        archive = Path(directory) / "download.zip"
        extracted = Path(directory) / destination.name
        print(f"Downloading the official {dataset} archive", file=sys.stderr)
        subprocess.run(
            ["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "20",
             "--output", str(archive), url],
            check=True,
        )
        with ZipFile(archive) as package, package.open(member) as source, extracted.open("wb") as target:
            shutil.copyfileobj(source, target)
        check_split(extracted, fingerprint)
        extracted.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SOURCES)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        print(stage(args.dataset, args.output))
    except (BadZipFile, KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Could not stage {args.dataset}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
