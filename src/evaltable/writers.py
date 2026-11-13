"""Writers for the long table (CSV, JSONL, and Parquet when pyarrow is installed)."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from .model import CORE_COLUMNS, EvalTable, Row

WRITE_FORMATS: tuple[str, ...] = ("csv", "jsonl", "parquet")


def _metadata_column(key: str) -> str:
    # A metadata key that collides with a core column is written with a prefix.
    return f"meta_{key}" if key in CORE_COLUMNS else key


def row_record(row: Row) -> dict[str, Any]:
    """A flat dict for one row: core columns first, then metadata keys."""
    record: dict[str, Any] = {
        "item_id": row.item_id,
        "run_id": row.run_id,
        "sample_index": row.sample_index,
        "score": row.score,
        "status": row.status.value,
    }
    for key, value in row.metadata.items():
        record[_metadata_column(key)] = value
    return record


def _format_score(score: float | None) -> str:
    if score is None:
        return ""
    return str(int(score)) if score.is_integer() else repr(score)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def write_csv(table: EvalTable, handle: TextIO) -> None:
    meta_columns = [_metadata_column(key) for key in table.metadata_keys()]
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(list(CORE_COLUMNS) + meta_columns)
    for row in table:
        record = row_record(row)
        cells = [
            row.item_id,
            row.run_id,
            str(row.sample_index),
            _format_score(row.score),
            row.status.value,
        ]
        cells.extend(_cell(record.get(column)) for column in meta_columns)
        writer.writerow(cells)


def write_jsonl(table: EvalTable, handle: TextIO) -> None:
    for row in table:
        handle.write(json.dumps(row_record(row), sort_keys=False) + "\n")


def write_parquet(table: EvalTable, path: str | Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - depends on the optional extra
        raise RuntimeError(
            "Parquet output needs pyarrow: pip install 'evaltable[parquet]'"
        ) from None
    meta_columns = [_metadata_column(key) for key in table.metadata_keys()]
    columns: dict[str, list[Any]] = {name: [] for name in list(CORE_COLUMNS) + meta_columns}
    for row in table:
        record = row_record(row)
        for name in columns:
            value = record.get(name)
            if name in meta_columns and isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            columns[name].append(value)
    pq.write_table(pa.table(columns), str(path))


def format_from_path(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if suffix == ".parquet":
        return "parquet"
    return "csv"


def write_table(table: EvalTable, path: str | Path | None, fmt: str | None = None) -> None:
    """Write to ``path`` (format from its extension unless given); ``None`` or ``-`` is stdout."""
    to_stdout = path is None or str(path) == "-"
    if fmt is None:
        fmt = "csv" if to_stdout else format_from_path(path)  # type: ignore[arg-type]
    if fmt not in WRITE_FORMATS:
        raise ValueError(f"unknown output format {fmt!r}; known: {', '.join(WRITE_FORMATS)}")
    if fmt == "parquet":
        if to_stdout:
            raise ValueError("Parquet output needs a file path")
        write_parquet(table, path)  # type: ignore[arg-type]
        return
    writer = write_csv if fmt == "csv" else write_jsonl
    if to_stdout:
        writer(table, sys.stdout)
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:  # type: ignore[arg-type]
        writer(table, handle)


def table_to_string(table: EvalTable, fmt: str = "csv") -> str:
    buffer = io.StringIO()
    (write_csv if fmt == "csv" else write_jsonl)(table, buffer)
    return buffer.getvalue()
