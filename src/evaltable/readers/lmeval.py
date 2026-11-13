"""Reader for lm-evaluation-harness per-sample logs (``--log_samples``).

Source: ``lm_eval/evaluator.py`` and ``lm_eval/loggers/evaluation_tracker.py`` in
EleutherAI/lm-evaluation-harness. With ``--log_samples`` the harness writes
``<output_path>/<model_name_sanitized>/samples_<task>_<date>.jsonl``, one line per
(document, filter) with ``doc_id``, ``doc``, ``target``, ``arguments``, ``resps``,
``filtered_resps``, ``filter``, ``metrics`` (the metric names), ``doc_hash``, ``prompt_hash``,
``target_hash``, and one key per metric holding that document's value.

Mapping: item = ``<task>/<doc_id>`` (the task is taken from the file name when it has the
harness's shape, otherwise the item is the bare ``doc_id``); run = the model directory name;
score = the chosen metric (default: the first name in ``metrics``). Every logged document is
``ok``: the harness has no per-document error state. A file with several filters gives one run
per filter (``<run>/<filter>``) unless one filter is chosen.

Ignored: ``doc``, ``target``, ``arguments``, ``resps``, ``filtered_resps``, ``prompt_hash``,
``target_hash`` and the metrics that are not chosen. Metrics whose per-document value is not a
number or boolean (for example the (prediction, reference) pairs logged for corpus-level
metrics) cannot be used as scores.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..model import EvalTable, Row
from .generic import iter_jsonl

_SAMPLE_FILE = re.compile(r"^samples_(?P<task>.+)_(?P<date>\d{4}-\d{2}-\d{2}T[\d.\-]+)$")


def looks_like_lm_eval(record: dict[str, Any]) -> bool:
    return "doc_id" in record and "filter" in record and "metrics" in record


def task_from_filename(path: Path) -> str | None:
    stem = path.name
    for suffix in (".gz", ".jsonl"):
        stem = stem.removesuffix(suffix)
    match = _SAMPLE_FILE.match(stem)
    return match.group("task") if match else None


def _metric_value(record: dict[str, Any], name: str, number: int) -> float:
    if name not in record:
        raise ValueError(f"line {number}: no metric {name!r} (has: {record.get('metrics')})")
    value = record[name]
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(
        f"line {number}: metric {name!r} is not a number per document ({type(value).__name__}); "
        "choose another with --metric"
    )


def read_lm_eval(
    path: str | Path,
    run_id: str | None = None,
    metric: str | None = None,
    filter_name: str | None = None,
) -> EvalTable:
    path = Path(path)
    records = list(iter_jsonl(path))
    task = task_from_filename(path)
    run = run_id or path.parent.name or path.stem
    filters = sorted({str(r.get("filter", "none")) for r in records})
    if filter_name is not None and filter_name not in filters:
        raise ValueError(f"filter {filter_name!r} not in file (has: {', '.join(filters)})")
    rows = []
    for number, record in enumerate(records, start=1):
        if not looks_like_lm_eval(record):
            raise ValueError(f"line {number}: not an lm-evaluation-harness sample record")
        flt = str(record.get("filter", "none"))
        if filter_name is not None and flt != filter_name:
            continue
        names = record.get("metrics") or []
        name = metric or (names[0] if names else None)
        if name is None:
            raise ValueError(f"line {number}: record lists no metrics; pass --metric")
        doc_id = str(record["doc_id"])
        item = f"{task}/{doc_id}" if task else doc_id
        this_run = run if filter_name is not None or len(filters) == 1 else f"{run}/{flt}"
        meta = {"task": task, "filter": flt, "metric": name, "doc_hash": record.get("doc_hash")}
        meta = {k: v for k, v in meta.items() if v is not None}
        rows.append(Row(item, this_run, _metric_value(record, name, number), metadata=meta))
    return EvalTable(rows)
