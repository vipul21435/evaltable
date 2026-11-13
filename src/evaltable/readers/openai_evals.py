"""Reader for OpenAI Evals record files (the JSONL written by ``LocalRecorder``).

Source: ``evals/record.py`` and ``evals/base.py`` (``RunSpec``) in openai/evals. The file starts
with a ``{"spec": {...}}`` line (``completion_fns``, ``eval_name``, ``run_id``, ...), then one
line per event with ``run_id``, ``event_id``, ``sample_id``, ``type``, ``data``,
``created_by`` and ``created_at``, and may end with a ``{"final_report": {...}}`` line.

Mapping: item = ``sample_id``; run = the completion functions joined by ``,``. By default the
score comes from ``match`` events (``data.correct``, 1 or 0); with a metric name it comes from
that key of ``metrics`` events instead. Several scoring events for one sample become sample
indexes 0, 1, ... in ``event_id`` order. A sample with an ``error`` event and no scoring event
is ``error``; a sample that has other events (for example ``sampling``) but no scoring event is
also ``error``, with ``no scoring event`` in its metadata.

Ignored: ``sampling``, ``embedding``, ``cond_logp``, ``pick_option``, ``raw`` and ``extra``
events (apart from marking the sample as seen), ``expected``/``picked`` values and the
final report.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..model import EvalTable, Row, Status
from .generic import iter_jsonl, parse_score


def looks_like_openai_evals(record: dict[str, Any]) -> bool:
    return isinstance(record.get("spec"), dict) and "eval_name" in record["spec"]


def read_openai_evals(
    path: str | Path, run_id: str | None = None, metric: str | None = None
) -> EvalTable:
    path = Path(path)
    spec: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    for record in iter_jsonl(path):
        if "spec" in record:
            spec = record["spec"]
        elif "final_report" in record:
            continue
        elif "type" in record and "sample_id" in record:
            events.append(record)
    if not spec and not events:
        raise ValueError(f"{path} has no OpenAI Evals spec or events")
    fns = spec.get("completion_fns") or []
    run = run_id or (",".join(fns) if fns else path.stem)
    meta_base = {"eval_name": spec.get("eval_name")} if spec.get("eval_name") else {}
    scored: dict[str, list[float]] = defaultdict(list)
    errors: dict[str, str] = {}
    order: dict[str, None] = {}
    for event in sorted(events, key=lambda e: e.get("event_id", 0)):
        sid = event.get("sample_id")
        if sid is None:
            continue
        sid = str(sid)
        order.setdefault(sid)
        data = event.get("data") or {}
        kind = event.get("type")
        if metric is None and kind == "match":
            scored[sid].append(1.0 if data.get("correct") else 0.0)
        elif metric is not None and kind == "metrics" and metric in data:
            value = parse_score(data[metric])
            if value is not None:
                scored[sid].append(value)
        elif kind == "error":
            errors.setdefault(sid, str(data.get("message", data.get("type", "error")))[:200])
    rows: list[Row] = []
    for sid in order:
        if scored.get(sid):
            for k, value in enumerate(scored[sid]):
                rows.append(Row(sid, run, value, Status.OK, k, dict(meta_base)))
        else:
            reason = errors.get(sid, "no scoring event")
            rows.append(Row(sid, run, None, Status.ERROR, 0, {**meta_base, "error": reason}))
    return EvalTable(rows)
