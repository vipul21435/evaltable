"""Read Inspect logs end to end and compare with the accuracy each log reports itself.

Usage:
    python scripts/fetch_public_data.py DATA
    python examples/inspect_logs.py DATA
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

from evaltable.readers import read
from evaltable.summary import summarize


def reported_accuracy(path: Path) -> float | None:
    """The first scorer's ``accuracy`` from the log's own ``results`` block."""
    if path.suffix == ".eval":
        with zipfile.ZipFile(path) as archive:
            header = json.loads(archive.read("header.json"))
    else:
        header = json.loads(path.read_text(encoding="utf-8"))
    scores = (header.get("results") or {}).get("scores") or []
    if not scores:
        return None
    accuracy = scores[0].get("metrics", {}).get("accuracy")
    return None if accuracy is None else float(accuracy["value"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="directory given to fetch_public_data.py")
    args = parser.parse_args(argv)
    paths = sorted((args.data / "inspect").glob("*"))
    for path in paths:
        table = read(path, run_id=path.stem[:32])
        (s,) = summarize(table)
        task = table.rows[0].metadata.get("task")
        mine = "n/a" if s.strict is None else f"{s.strict:.4f}"
        own = reported_accuracy(path)
        theirs = "none" if own is None else f"{own:.4f}"
        print(
            f"{path.name}: task {task}, {s.n_reference} items, ok {s.n_ok}, "
            f"missing {s.n_missing}, error {s.n_error}; "
            f"errors as failures {mine}, log's own accuracy {theirs}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
