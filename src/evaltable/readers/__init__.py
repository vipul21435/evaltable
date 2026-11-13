"""Readers that turn result files into an :class:`~evaltable.model.EvalTable`.

``read`` picks a reader from the file's extension and, for JSON and JSONL, from its content.
Each reader module documents the format's source and what it ignores.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..model import EvalTable
from .codegen import looks_like_evalplus, looks_like_humaneval, read_evalplus, read_humaneval
from .generic import parse_column_map, parse_score, read_csv, read_jsonl
from .inspectlog import looks_like_inspect, read_inspect
from .lmeval import looks_like_lm_eval, read_lm_eval
from .openai_evals import looks_like_openai_evals, read_openai_evals
from .swebench import looks_like_report, looks_like_results, read_swebench

FORMATS: tuple[str, ...] = (
    "csv",
    "jsonl",
    "swebench",
    "lm-eval",
    "inspect",
    "openai-evals",
    "humaneval",
    "evalplus",
)


@dataclass(frozen=True)
class ReadOptions:
    """Format-specific choices. Readers ignore the options that do not apply to them.

    ``metric``: lm-eval metric name, Inspect scorer name, or OpenAI Evals ``metrics`` key.
    ``filter``: lm-eval filter. ``tests``: EvalPlus ``base`` or ``plus``. ``items``: reference
    instance ids, needed for SWE-bench ``results.json`` files that list only some outcomes.
    """

    metric: str | None = None
    filter: str | None = None
    tests: str = "plus"
    items: tuple[str, ...] | None = None


def _base_suffix(path: Path) -> str:
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    return suffixes[-1] if suffixes else ""


def _first_record(path: Path) -> dict[str, Any] | None:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def detect_format(path: str | Path) -> str:
    """Pick a reader from the extension and, for JSON/JSONL, the content."""
    path = Path(path)
    suffix = _base_suffix(path)
    if suffix in (".csv", ".tsv"):
        return "csv"
    if suffix == ".eval":
        return "inspect"
    if suffix in (".jsonl", ".ndjson"):
        first = _first_record(path)
        if first is not None:
            if looks_like_openai_evals(first):
                return "openai-evals"
            if looks_like_lm_eval(first):
                return "lm-eval"
            if looks_like_humaneval(first):
                return "humaneval"
        return "jsonl"
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if looks_like_report(data) or looks_like_results(data):
            return "swebench"
        if looks_like_inspect(data):
            return "inspect"
        if looks_like_evalplus(data):
            return "evalplus"
        raise ValueError(f"cannot tell which harness wrote {str(path)!r}; pass --format")
    raise ValueError(f"cannot tell the format of {str(path)!r}; pass one of: {', '.join(FORMATS)}")


def read(
    path: str | Path,
    fmt: str = "auto",
    column_map: Mapping[str, str] | None = None,
    run_id: str | None = None,
    options: ReadOptions | None = None,
) -> EvalTable:
    """Read one result file into a long table.

    Malformed input raises ``ValueError`` naming the file.
    """
    try:
        return _read(path, fmt, column_map, run_id, options or ReadOptions())
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"{path}: unexpected structure ({type(exc).__name__}: {exc})") from None


def _read(
    path: str | Path,
    fmt: str,
    column_map: Mapping[str, str] | None,
    run_id: str | None,
    options: ReadOptions,
) -> EvalTable:
    if fmt == "auto":
        fmt = detect_format(path)
    if fmt == "csv":
        return read_csv(path, column_map, run_id)
    if fmt == "jsonl":
        return read_jsonl(path, column_map, run_id)
    if fmt == "swebench":
        return read_swebench(path, run_id, options.items)
    if fmt == "lm-eval":
        return read_lm_eval(path, run_id, options.metric, options.filter)
    if fmt == "inspect":
        return read_inspect(path, run_id, options.metric)
    if fmt == "openai-evals":
        return read_openai_evals(path, run_id, options.metric)
    if fmt == "humaneval":
        return read_humaneval(path, run_id)
    if fmt == "evalplus":
        return read_evalplus(path, run_id, options.tests)
    raise ValueError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")


def read_many(
    paths: Iterable[str | Path],
    fmt: str = "auto",
    column_map: Mapping[str, str] | None = None,
    run_ids: Iterable[str | None] | None = None,
    options: ReadOptions | None = None,
    distinct_runs: bool = False,
) -> EvalTable:
    """Read and merge files; the result does not depend on their order.

    With ``distinct_runs`` two files that give one run id rows for the same (item, sample) are
    refused, since their rows would silently merge into one run. Files that share a run id but
    not items (for example several lm-eval task files of one model) are fine.
    """
    paths = list(paths)
    runs = list(run_ids) if run_ids is not None else [None] * len(paths)
    if len(runs) != len(paths):
        raise ValueError(f"{len(runs)} run id(s) for {len(paths)} file(s)")
    tables = [read(p, fmt, column_map, r, options) for p, r in zip(paths, runs, strict=True)]
    if distinct_runs:
        owner: dict[tuple[str, str, int], str | Path] = {}
        for path, table in zip(paths, tables, strict=True):
            mine = {row.key() for row in table}
            for key in mine:
                other = owner.get(key)
                if other is not None and other != path:
                    raise ValueError(
                        f"{other} and {path} both give run {key[0]!r} rows for the same items; "
                        "pass --run once per file"
                    )
            for key in mine:
                owner.setdefault(key, path)
    return EvalTable.concat(tables)


__all__ = [
    "FORMATS",
    "ReadOptions",
    "detect_format",
    "parse_column_map",
    "parse_score",
    "read",
    "read_csv",
    "read_evalplus",
    "read_humaneval",
    "read_inspect",
    "read_jsonl",
    "read_lm_eval",
    "read_many",
    "read_openai_evals",
    "read_swebench",
]
