"""Per-item collapse of samples, per-run scores under two conventions, and item-set overlap.

The two conventions:

* **errors as failures** (``strict``): every item in the reference set counts; error, timeout
  and missing samples score 0, and an item absent from a run scores 0.
* **errors left out** (``lenient``): only ``ok`` samples count; an item with no ``ok`` sample
  is dropped from the denominator.

When a run has no error, timeout or missing rows and covers the reference set, the two agree.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .model import STATUS_PRECEDENCE, EvalTable, Row, Status
from .stats import wilson_interval


@dataclass(frozen=True)
class ItemResult:
    """All samples of one item in one run, collapsed."""

    run_id: str
    item_id: str
    n_rows: int
    n_ok: int
    n_error: int
    n_timeout: int
    n_missing: int
    strict: float
    lenient: float | None
    status: Status

    @property
    def present(self) -> bool:
        return self.n_missing < self.n_rows


def collapse(rows: Iterable[Row]) -> dict[tuple[str, str], ItemResult]:
    """Group rows by (run, item) and collapse their samples."""
    groups: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in rows:
        groups[(row.run_id, row.item_id)].append(row)
    result: dict[tuple[str, str], ItemResult] = {}
    for (run, item), group in groups.items():
        counts = {status: 0 for status in Status}
        ok_scores: list[float] = []
        for row in group:
            counts[row.status] += 1
            if row.status is Status.OK:
                ok_scores.append(row.score)  # type: ignore[arg-type]
        status = next(s for s in STATUS_PRECEDENCE if counts[s])
        result[(run, item)] = ItemResult(
            run_id=run,
            item_id=item,
            n_rows=len(group),
            n_ok=counts[Status.OK],
            n_error=counts[Status.ERROR],
            n_timeout=counts[Status.TIMEOUT],
            n_missing=counts[Status.MISSING],
            strict=sum(ok_scores) / len(group),
            lenient=sum(ok_scores) / len(ok_scores) if ok_scores else None,
            status=status,
        )
    return result


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    n_reference: int
    n_present: int
    n_ok: int
    n_error: int
    n_timeout: int
    n_missing: int
    coverage: float
    error_rate: float
    strict: float | None
    strict_ci: tuple[float, float] | None
    lenient: float | None
    lenient_n: int
    lenient_ci: tuple[float, float] | None
    gap: float | None

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key in ("strict_ci", "lenient_ci"):
            if data[key] is not None:
                data[key] = list(data[key])
        return data


def _interval(total: float, n: int, confidence: float, bounded: bool) -> tuple[float, float] | None:
    if not bounded or n == 0:
        return None
    return wilson_interval(total, n, confidence)


def summarize(
    table: EvalTable,
    items: Iterable[str] | None = None,
    confidence: float = 0.95,
) -> list[RunSummary]:
    """Summarise every run against a reference item set (default: union of all items).

    Item counts (``n_ok``, ``n_error``, ...) use each item's collapsed status. ``coverage`` is
    the share of reference items with at least one ``ok`` sample; ``error_rate`` is the share
    of present items whose status is error or timeout. Intervals are Wilson intervals and are
    left out when any score lies outside [0, 1].
    """
    reference = sorted(set(items)) if items is not None else table.items()
    collapsed = collapse(table)
    bounded = all(row.score is None or 0.0 <= row.score <= 1.0 for row in table)
    summaries: list[RunSummary] = []
    for run in table.runs():
        results = [collapsed.get((run, item)) for item in reference]
        present = [r for r in results if r is not None and r.present]
        by_status = {status: 0 for status in Status}
        for r in results:
            by_status[Status.MISSING if r is None else r.status] += 1
        n_ref = len(reference)
        strict_total = sum(r.strict for r in results if r is not None)
        lenient_scores = [r.lenient for r in results if r is not None and r.lenient is not None]
        lenient_n = len(lenient_scores)
        strict = strict_total / n_ref if n_ref else None
        lenient = sum(lenient_scores) / lenient_n if lenient_n else None
        n_bad = sum(1 for r in present if r.status in (Status.ERROR, Status.TIMEOUT))
        summaries.append(
            RunSummary(
                run_id=run,
                n_reference=n_ref,
                n_present=len(present),
                n_ok=by_status[Status.OK],
                n_error=by_status[Status.ERROR],
                n_timeout=by_status[Status.TIMEOUT],
                n_missing=by_status[Status.MISSING],
                coverage=sum(1 for r in present if r.n_ok > 0) / n_ref if n_ref else 0.0,
                error_rate=n_bad / len(present) if present else 0.0,
                strict=strict,
                strict_ci=_interval(strict_total, n_ref, confidence, bounded),
                lenient=lenient,
                lenient_n=lenient_n,
                lenient_ci=_interval(sum(lenient_scores), lenient_n, confidence, bounded),
                gap=None if strict is None or lenient is None else lenient - strict,
            )
        )
    return summaries


def overlap_matrix(table: EvalTable) -> dict[str, dict[str, int]]:
    """Number of items present (some non-missing row) in both runs, for every pair of runs."""
    present = {run: set(table.items(run, present_only=True)) for run in table.runs()}
    return {a: {b: len(present[a] & present[b]) for b in present} for a in present}
