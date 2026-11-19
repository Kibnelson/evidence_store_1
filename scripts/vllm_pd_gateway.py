#!/usr/bin/env python3
"""Run the native PD gateway against a prepared endpoint manifest."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    from evidencestore.gateway import create_app

    uvicorn.run(create_app(args.run_dir), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
