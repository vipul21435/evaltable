"""The long per-item table: one row per (run, item, sample)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

CORE_COLUMNS: tuple[str, ...] = ("item_id", "run_id", "sample_index", "score", "status")


class Status(str, Enum):
    """Outcome of one evaluated sample."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    MISSING = "missing"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def parse(cls, value: str | Status) -> Status:
        """Parse a status name or one of its common synonyms."""
        if isinstance(value, Status):
            return value
        key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        try:
            return _STATUS_SYNONYMS[key]
        except KeyError:
            raise ValueError(f"unknown status {value!r}") from None


_STATUS_SYNONYMS: dict[str, Status] = {
    "ok": Status.OK,
    "success": Status.OK,
    "succeeded": Status.OK,
    "completed": Status.OK,
    "done": Status.OK,
    "error": Status.ERROR,
    "errored": Status.ERROR,
    "exception": Status.ERROR,
    "crash": Status.ERROR,
    "crashed": Status.ERROR,
    "timeout": Status.TIMEOUT,
    "timed_out": Status.TIMEOUT,
    "time_out": Status.TIMEOUT,
    "missing": Status.MISSING,
    "not_run": Status.MISSING,
    "absent": Status.MISSING,
}

# When one item has rows with several statuses, the item takes the first of these that occurs.
STATUS_PRECEDENCE: tuple[Status, ...] = (Status.ERROR, Status.TIMEOUT, Status.MISSING, Status.OK)


@dataclass
class Row:
    """One evaluated sample of one item in one run.

    ``score`` is required for ``ok`` rows and ignored for the other statuses, where it is
    usually ``None``.
    """

    item_id: str
    run_id: str
    score: float | None
    status: Status = Status.OK
    sample_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.item_id = str(self.item_id)
        self.run_id = str(self.run_id)
        self.status = Status.parse(self.status)
        self.sample_index = int(self.sample_index)
        if self.score is not None:
            self.score = float(self.score)
            if not math.isfinite(self.score):
                raise ValueError(f"score must be finite, got {self.score!r} for {self.item_id!r}")
        if self.status is Status.OK and self.score is None:
            raise ValueError(f"row {self.item_id!r} in run {self.run_id!r} is ok but has no score")

    def key(self) -> tuple[str, str, int]:
        return (self.run_id, self.item_id, self.sample_index)

    def sort_key(self) -> tuple[str, str, int, str, float]:
        score = -math.inf if self.score is None else self.score
        return (self.run_id, self.item_id, self.sample_index, self.status.value, score)


class EvalTable:
    """A list of :class:`Row` objects with helpers for runs and items."""

    def __init__(self, rows: Iterable[Row] = ()) -> None:
        self.rows: list[Row] = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Row]:
        return iter(self.rows)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EvalTable):
            return NotImplemented
        return self.rows == other.rows

    def __repr__(self) -> str:
        return f"EvalTable({len(self.rows)} rows, runs={self.runs()})"

    @classmethod
    def concat(cls, tables: Iterable[EvalTable]) -> EvalTable:
        """Join tables into one, in canonical row order (so input order does not matter)."""
        rows: list[Row] = []
        for table in tables:
            rows.extend(table.rows)
        return cls(rows).sorted()

    def sorted(self) -> EvalTable:
        """Rows ordered by run, item, sample index, status and score."""
        return EvalTable(sorted(self.rows, key=Row.sort_key))

    def runs(self) -> list[str]:
        return sorted({row.run_id for row in self.rows})

    def items(self, run_id: str | None = None, present_only: bool = False) -> list[str]:
        """Item ids, optionally for one run; ``present_only`` drops rows with status missing."""
        found = {
            row.item_id
            for row in self.rows
            if (run_id is None or row.run_id == run_id)
            and not (present_only and row.status is Status.MISSING)
        }
        return sorted(found)

    def for_run(self, run_id: str) -> EvalTable:
        return EvalTable(row for row in self.rows if row.run_id == run_id)

    def metadata_keys(self) -> list[str]:
        keys: set[str] = set()
        for row in self.rows:
            keys.update(row.metadata)
        return sorted(keys)

    def with_missing(self, items: Iterable[str] | None = None) -> EvalTable:
        """Add a ``missing`` row for every (run, item) pair that has no row at all.

        ``items`` is the reference item set; by default it is the union over all runs.
        """
        reference = sorted(set(items)) if items is not None else self.items()
        have = {(row.run_id, row.item_id) for row in self.rows}
        extra = [
            Row(item_id=item, run_id=run, score=None, status=Status.MISSING)
            for run in self.runs()
            for item in reference
            if (run, item) not in have
        ]
        return EvalTable(self.rows + extra).sorted()
