"""Checks across runs for differences that make their scores incomparable."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from .model import EvalTable, Status
from .summary import collapse


@dataclass(frozen=True)
class CheckOptions:
    """What ``check`` tolerates. The defaults tolerate nothing."""

    allow_missing: bool = False
    allow_duplicates: bool = False
    allow_sample_mismatch: bool = False
    max_gap: float = 0.0


@dataclass
class Finding:
    check: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckReport:
    findings: list[Finding]

    @property
    def comparable(self) -> bool:
        return all(f.passed for f in self.findings)

    def as_dict(self) -> dict[str, Any]:
        return {"comparable": self.comparable, "findings": [asdict(f) for f in self.findings]}


def check_item_sets(
    table: EvalTable, items: Iterable[str] | None = None, allow: bool = False
) -> Finding:
    """Which reference items each run lacks (no row, or only ``missing`` rows)."""
    reference = sorted(set(items)) if items is not None else table.items()
    lacking: dict[str, list[str]] = {}
    for run in table.runs():
        present = set(table.items(run, present_only=True))
        absent = [item for item in reference if item not in present]
        if absent:
            lacking[run] = absent
    if not lacking:
        message = f"all runs cover the same {len(reference)} items"
    else:
        parts = [f"{run} lacks {len(absent)}" for run, absent in lacking.items()]
        message = f"item sets differ over {len(reference)} items: " + ", ".join(parts)
    return Finding(
        "item_sets",
        passed=allow or not lacking,
        message=message,
        details={"n_reference": len(reference), "lacking": lacking},
    )


def check_duplicates(table: EvalTable, allow: bool = False) -> Finding:
    """Rows that repeat the same (run, item, sample index)."""
    counts = Counter(row.key() for row in table)
    dups = [
        {"run_id": run, "item_id": item, "sample_index": sample, "count": count}
        for (run, item, sample), count in sorted(counts.items())
        if count > 1
    ]
    message = "no duplicated rows" if not dups else f"{len(dups)} (run, item, sample) keys repeat"
    return Finding("duplicates", passed=allow or not dups, message=message, details={"rows": dups})


def check_errors(table: EvalTable, max_gap: float = 0.0) -> Finding:
    """Errored or timed-out items per run, with the score both ways over the present items.

    The gap here leaves missing items out of both conventions, so it measures the effect of
    errors alone; ``check_item_sets`` reports the missing items.
    """
    collapsed = collapse(table)
    runs: dict[str, dict[str, Any]] = {}
    failing: list[str] = []
    for run in table.runs():
        present = [r for (rid, _), r in collapsed.items() if rid == run and r.present]
        bad = sorted(r.item_id for r in present if r.status in (Status.ERROR, Status.TIMEOUT))
        if not bad:
            continue
        strict = sum(r.strict for r in present) / len(present)
        scored = [r.lenient for r in present if r.lenient is not None]
        lenient = sum(scored) / len(scored) if scored else None
        gap = None if lenient is None else lenient - strict
        runs[run] = {
            "errored": bad,
            "n_present": len(present),
            "errors_as_failures": strict,
            "errors_left_out": lenient,
            "gap": gap,
        }
        if gap is None or abs(gap) > max_gap:
            failing.append(run)
    if not runs:
        message = "no errored or timed-out items"
    else:
        parts = []
        for run, info in runs.items():
            gap = "n/a" if info["gap"] is None else f"{info['gap']:+.4f}"
            parts.append(f"{run}: {len(info['errored'])} errored, gap {gap}")
        message = "; ".join(parts)
    return Finding(
        "errors", passed=not failing, message=message, details={"runs": runs, "failing": failing}
    )


def check_sample_counts(table: EvalTable, allow: bool = False) -> Finding:
    """Distinct sample indexes per item, compared within and across runs (missing rows skipped)."""
    samples: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in table:
        if row.status is not Status.MISSING:
            samples[(row.run_id, row.item_id)].add(row.sample_index)
    per_run: dict[str, dict[int, int]] = defaultdict(dict)
    per_item: dict[str, dict[str, int]] = defaultdict(dict)
    for (run, item), indexes in samples.items():
        n = len(indexes)
        per_run[run][n] = per_run[run].get(n, 0) + 1
        per_item[item][run] = n
    differing = {
        item: dict(sorted(c.items())) for item, c in per_item.items() if len(set(c.values())) > 1
    }
    mixed_runs = sorted(run for run, counts in per_run.items() if len(counts) > 1)
    ok = not differing and not mixed_runs
    if ok:
        values = sorted({n for counts in per_run.values() for n in counts})
        message = f"every item has {values[0] if values else 0} sample(s) in every run"
    else:
        message = (
            f"sample counts differ: {len(differing)} items differ across runs, "
            f"{len(mixed_runs)} runs mix counts"
        )
    return Finding(
        "sample_counts",
        passed=allow or ok,
        message=message,
        details={
            "per_run": {run: dict(sorted(c.items())) for run, c in sorted(per_run.items())},
            "differing_items": dict(sorted(differing.items())),
            "mixed_runs": mixed_runs,
        },
    )


def run_checks(
    table: EvalTable,
    items: Iterable[str] | None = None,
    options: CheckOptions | None = None,
) -> CheckReport:
    options = options or CheckOptions()
    reference = list(items) if items is not None else None
    return CheckReport(
        [
            check_item_sets(table, reference, options.allow_missing),
            check_duplicates(table, options.allow_duplicates),
            check_errors(table, options.max_gap),
            check_sample_counts(table, options.allow_sample_mismatch),
        ]
    )
