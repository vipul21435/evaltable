"""Readers that turn result files into an :class:`~evaltable.model.EvalTable`."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..model import EvalTable
from .generic import parse_column_map, parse_score, read_csv, read_jsonl

FORMATS: tuple[str, ...] = ("csv", "jsonl")

_EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "csv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
}


def detect_format(path: str | Path) -> str:
    """Pick a reader from the file extension."""
    suffix = Path(path).suffix.lower()
    try:
        return _EXTENSIONS[suffix]
    except KeyError:
        raise ValueError(
            f"cannot tell the format of {str(path)!r}; pass one of: {', '.join(FORMATS)}"
        ) from None


def read(
    path: str | Path,
    fmt: str = "auto",
    column_map: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> EvalTable:
    """Read one result file into a long table."""
    if fmt == "auto":
        fmt = detect_format(path)
    if fmt == "csv":
        return read_csv(path, column_map, run_id)
    if fmt == "jsonl":
        return read_jsonl(path, column_map, run_id)
    raise ValueError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")


__all__ = [
    "FORMATS",
    "detect_format",
    "parse_column_map",
    "parse_score",
    "read",
    "read_csv",
    "read_jsonl",
]
