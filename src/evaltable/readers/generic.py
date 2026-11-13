"""Generic CSV and JSONL readers driven by a column map.

The long table that evaltable writes is itself read by these readers with the default map, so
``read`` followed by ``write`` round-trips.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..model import EvalTable, Row, Status

# Logical field -> column name in the file. A user map overrides single entries.
DEFAULT_COLUMN_MAP: dict[str, str] = {
    "item": "item_id",
    "run": "run_id",
    "score": "score",
    "status": "status",
    "sample": "sample_index",
}

_TRUE = {"true", "t", "yes", "y", "pass", "passed", "resolved", "correct"}
_FALSE = {"false", "f", "no", "n", "fail", "failed", "unresolved", "incorrect"}
_NULL = {"", "nan", "none", "null", "na", "n/a"}


def parse_column_map(text: str | None) -> dict[str, str]:
    """Parse ``"item=instance_id,score=resolved"`` into a dict, checking the field names."""
    if not text:
        return {}
    result: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        field, sep, column = part.partition("=")
        field, column = field.strip(), column.strip()
        if not sep or not column:
            raise ValueError(f"column map entry {part!r} is not of the form field=column")
        if field not in DEFAULT_COLUMN_MAP:
            known = ", ".join(DEFAULT_COLUMN_MAP)
            raise ValueError(f"unknown field {field!r} in column map (known: {known})")
        result[field] = column
    return result


def parse_score(value: Any) -> float | None:
    """Turn a score cell into a float, or ``None`` when it is empty or not a number."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        return None if math.isnan(number) else number
    text = str(value).strip().lower()
    if text in _NULL:
        return None
    if text in _TRUE:
        return 1.0
    if text in _FALSE:
        return 0.0
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"cannot read {value!r} as a score") from None
    return None if math.isnan(number) else number


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _parse_sample(value: Any, number: int) -> int:
    """A sample index: an integer, an integral float, or a string of digits."""
    if isinstance(value, bool):
        ok = False
    elif isinstance(value, int):
        return value
    elif isinstance(value, float):
        ok = value.is_integer()
    elif isinstance(value, str):
        ok = value.strip().lstrip("-").isdigit()
    else:
        ok = False
    if not ok:
        raise ValueError(f"record {number}: sample index {value!r} is not an integer")
    return int(float(value)) if isinstance(value, float) else int(value)


def rows_from_records(
    records: Iterable[Mapping[str, Any]],
    column_map: Mapping[str, str] | None = None,
    run_id: str | None = None,
    default_run: str = "run",
    drop_blank_metadata: bool = False,
) -> list[Row]:
    """Build rows from dict records.

    ``run_id`` overrides any run column; otherwise rows without a run column take
    ``default_run``. A record with no status column is ``ok`` when it has a score and
    ``error`` when it does not. Columns not named in the map become metadata.
    """
    cmap = {**DEFAULT_COLUMN_MAP, **(column_map or {})}
    used = set(cmap.values())
    rows: list[Row] = []
    for number, record in enumerate(records, start=1):
        item = record.get(cmap["item"])
        if _is_blank(item):
            raise ValueError(f"record {number} has no item column {cmap['item']!r}")
        if run_id is not None:
            run = run_id
        else:
            raw_run = record.get(cmap["run"])
            run = default_run if _is_blank(raw_run) else str(raw_run)
        try:
            score = parse_score(record.get(cmap["score"]))
        except ValueError as exc:
            raise ValueError(f"record {number}: {exc}") from None
        raw_status = record.get(cmap["status"])
        if _is_blank(raw_status):
            status = Status.OK if score is not None else Status.ERROR
        else:
            status = Status.parse(raw_status)
        raw_sample = record.get(cmap["sample"])
        sample = 0 if _is_blank(raw_sample) else _parse_sample(raw_sample, number)
        metadata = {
            key: value
            for key, value in record.items()
            if key not in used and not (drop_blank_metadata and _is_blank(value))
        }
        try:
            rows.append(Row(str(item), run, score, status, sample, metadata))
        except ValueError as exc:
            raise ValueError(f"record {number}: {exc}") from None
    return rows


@contextmanager
def _naming(path: Path) -> Iterator[None]:
    """Prefix the file name to a ValueError raised while reading it."""
    try:
        yield
    except ValueError as exc:
        if str(exc).startswith(str(path)):
            raise
        raise ValueError(f"{path}: {exc}") from None


def read_csv(
    path: str | Path,
    column_map: Mapping[str, str] | None = None,
    run_id: str | None = None,
    delimiter: str | None = None,
) -> EvalTable:
    """Read a CSV (or TSV, by extension) file with a header row, gzipped when it ends in .gz.

    A UTF-8 byte-order mark is allowed.
    """
    path = Path(path)
    gzipped = path.suffix.lower() == ".gz"
    base = path.with_suffix("") if gzipped else path
    if delimiter is None:
        delimiter = "\t" if base.suffix.lower() == ".tsv" else ","
    opener = gzip.open if gzipped else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        with _naming(path):
            rows = rows_from_records(
                reader, column_map, run_id, default_run=base.stem, drop_blank_metadata=True
            )
    return EvalTable(rows)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield the JSON objects of a JSONL file (gzipped when it ends in .gz), skipping blanks."""
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: not valid JSON ({exc.msg})") from None
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield value


def read_jsonl(
    path: str | Path,
    column_map: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> EvalTable:
    """Read a JSONL file with one object per line."""
    path = Path(path)
    with _naming(path):
        rows = rows_from_records(iter_jsonl(path), column_map, run_id, default_run=path.stem)
    return EvalTable(rows)
