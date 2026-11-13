"""How far SWE-bench scores move between the two conventions, over public submissions.

Usage:
    python scripts/fetch_public_data.py DATA
    python examples/swebench_conventions.py DATA [--split verified]

The reference item set is the union of instance ids listed by any submission of the split.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from evaltable.readers import ReadOptions, read_many
from evaltable.summary import summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="directory given to fetch_public_data.py")
    parser.add_argument("--split", default="verified")
    parser.add_argument("--top", type=int, default=5, help="largest gaps to list")
    args = parser.parse_args(argv)

    paths = sorted(
        (args.data / "swebench" / "evaluation" / args.split).glob("*/results/results.json")
    )
    if not paths:
        print("no results.json files found; run scripts/fetch_public_data.py first")
        return 1
    ids: set[str] = set()
    old_format = 0
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        old_format += "generated" in data
        for value in data.values():
            if isinstance(value, list):
                ids.update(str(v) for v in value)
    table = read_many(paths, fmt="swebench", options=ReadOptions(items=tuple(sorted(ids))))
    runs = summarize(table, items=ids)

    with_missing = [s for s in runs if s.n_missing]
    with_errors = [s for s in runs if s.n_error or s.n_timeout]
    affected = [s for s in runs if s.n_missing or s.n_error or s.n_timeout]
    gaps = [s.gap for s in runs if s.gap is not None]
    affected_gaps = [s.gap for s in affected if s.gap is not None]

    print(f"split: {args.split}")
    rest = len(runs) - old_format
    print(f"submissions: {len(runs)} ({old_format} list every outcome, {rest} do not)")
    print(f"reference instances: {len(ids)}")
    print(f"with missing instances (no patch): {len(with_missing)}")
    print(f"with errored or timed-out instances: {len(with_errors)}")
    print(f"with either: {len(affected)}")
    print(f"gap, all submissions: median {statistics.median(gaps):.4f}, max {max(gaps):.4f}")
    if affected_gaps:
        print(
            f"gap, affected submissions: median {statistics.median(affected_gaps):.4f}, "
            f"mean {statistics.fmean(affected_gaps):.4f}"
        )
    for threshold in (0.01, 0.05, 0.10):
        count = sum(1 for g in gaps if g >= threshold)
        print(f"gap >= {threshold:.2f}: {count}")
    print()
    print(f"largest {args.top} gaps (errors as failures -> errors left out):")
    for s in sorted(runs, key=lambda s: s.gap or 0.0, reverse=True)[: args.top]:
        print(
            f"  {s.run_id}: {s.strict:.4f} -> {s.lenient:.4f} (gap {s.gap:+.4f}; "
            f"missing {s.n_missing}, error {s.n_error}, timeout {s.n_timeout})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
