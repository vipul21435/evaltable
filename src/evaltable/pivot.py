"""Wide form of the table: one row per item, one column per run."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TextIO

from .model import EvalTable
from .summary import collapse

PIVOT_VALUES: tuple[str, ...] = ("score", "strict", "status")


@dataclass
class WideTable:
    """Cells keyed by (item, run). ``None`` means the run has no value for the item."""

    value: str
    items: list[str]
    runs: list[str]
    cells: dict[tuple[str, str], float | str | None]

    def get(self, item: str, run: str) -> float | str | None:
        return self.cells.get((item, run))

    def write_csv(self, handle: TextIO) -> None:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["item_id", *self.runs])
        for item in self.items:
            writer.writerow([item, *(_format(self.get(item, run)) for run in self.runs)])


def _format(value: float | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    return value


def pivot(table: EvalTable, value: str = "score") -> WideTable:
    """Items by runs.

    ``score`` is the mean over ``ok`` samples (empty when there is none), ``strict`` the
    errors-as-failures item score, ``status`` the item's collapsed status; a run with no
    row for an item shows ``missing`` under ``status`` and an empty cell otherwise.
    """
    if value not in PIVOT_VALUES:
        raise ValueError(f"unknown pivot value {value!r}; known: {', '.join(PIVOT_VALUES)}")
    collapsed = collapse(table)
    runs, items = table.runs(), table.items()
    cells: dict[tuple[str, str], float | str | None] = {}
    for item in items:
        for run in runs:
            result = collapsed.get((run, item))
            if result is None:
                cells[(item, run)] = "missing" if value == "status" else None
            elif value == "score":
                cells[(item, run)] = result.lenient
            elif value == "strict":
                cells[(item, run)] = result.strict
            else:
                cells[(item, run)] = result.status.value
    return WideTable(value, items, runs, cells)


def read_wide_csv(handle: TextIO, value: str = "score") -> WideTable:
    """Read a wide CSV written by :meth:`WideTable.write_csv`."""
    reader = csv.reader(handle)
    header = next(reader)
    runs = header[1:]
    items: list[str] = []
    cells: dict[tuple[str, str], float | str | None] = {}
    for line in reader:
        item = line[0]
        items.append(item)
        for k, run in enumerate(runs, start=1):
            text = line[k] if k < len(line) else ""
            if text == "":
                cells[(item, run)] = None
            else:
                cells[(item, run)] = text if value == "status" else float(text)
    return WideTable(value, items, runs, cells)
