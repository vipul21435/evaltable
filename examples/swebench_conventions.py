"""How far SWE-bench scores move between the two conventions, over public submissions.

Usage:
    python scripts/fetch_public_data.py DATA
    python examples/swebench_conventions.py DATA [--split verified] [--write-items ids.txt]

The reference item set is the union of instance ids listed by any submission of the split.
The counts describe what each entry's results.json lists, which reflects what was uploaded
with the entry; they say nothing about why an instance has no result.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from evaltable.readers import ReadOptions, read_many
from evaltable.summary import summarize

# Outcome labels, as the reader records them in the swebench_outcome metadata column.
OUTCOMES = [
    ("no_generation", "ids under no_generation (no log folder or no patch.diff)"),
    ("no_logs", "ids under no_logs (patch.diff without test output)"),
    ("test_timeout", "ids under test_timeout"),
    ("install_fail", "ids under install_fail"),
    ("reset_failed", "ids under reset_failed"),
    ("test_errored", "ids under test_errored"),
    ("not_listed", "reference ids listed under no outcome"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="directory given to fetch_public_data.py")
    parser.add_argument("--split", default="verified")
    parser.add_argument("--write-items", type=Path, default=None, help="write the reference ids")
    args = parser.parse_args(argv)

    paths = sorted(
        (args.data / "swebench" / "evaluation" / args.split).glob("*/results/results.json")
    )
    if not paths:
        print("no results.json files found; run scripts/fetch_public_data.py first")
        return 1
    ids: set[str] = set()
    old_format = 0
    entries: dict[str, set[str]] = {key: set() for key, _ in OUTCOMES}
    totals: Counter[str] = Counter()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        old_format += "generated" in data
        for key, value in data.items():
            if isinstance(value, list):
                listed = {str(v) for v in value}
                ids.update(listed)
                if key in entries and listed:
                    entries[key].add(path.parent.parent.name)
                    totals[key] += len(listed)
    if args.write_items is not None:
        args.write_items.write_text("".join(f"{i}\n" for i in sorted(ids)), encoding="utf-8")
    table = read_many(paths, fmt="swebench", options=ReadOptions(items=tuple(sorted(ids))))
    runs = summarize(table, items=ids)

    # Ids an entry lists under no outcome are found by the reader, not in the file.
    for row in table:
        if row.metadata.get("swebench_outcome") == "not_listed":
            entries["not_listed"].add(row.run_id)
            totals["not_listed"] += 1
    affected = [s for s in runs if s.n_missing or s.n_error or s.n_timeout]
    gaps = [s.gap for s in runs if s.gap is not None]
    affected_gaps = [s.gap for s in affected if s.gap is not None]

    print(f"split: {args.split}")
    rest = len(runs) - old_format
    print(f"entries: {len(runs)} ({old_format} list every outcome, {rest} do not)")
    print(f"reference instances: {len(ids)}")
    for key, label in OUTCOMES:
        print(f"entries with {label}: {len(entries[key])} ({totals[key]} distinct ids)")
    print(f"entries with any of these: {len(affected)}")
    print(f"gap, all entries: median {statistics.median(gaps):.4f}, max {max(gaps):.4f}")
    if affected_gaps:
        print(
            f"gap, entries with any of these: median {statistics.median(affected_gaps):.4f}, "
            f"mean {statistics.fmean(affected_gaps):.4f}"
        )
    for threshold in (0.01, 0.05, 0.10):
        count = sum(1 for g in gaps if g >= threshold)
        print(f"gap >= {threshold:.2f}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
