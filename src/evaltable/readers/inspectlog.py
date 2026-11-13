"""Reader for Inspect AI evaluation logs (``.json`` and ``.eval``).

Source: ``inspect_ai/log/_log.py`` (``EvalLog``, ``EvalSample``), ``inspect_ai/scorer/_metric.py``
(``value_to_float``) and ``inspect_ai/log/_recorders/eval.py`` in UKGovernmentBEIS/inspect_ai,
and the "Log Files" page of the Inspect documentation. A ``.json`` log is one ``EvalLog``
object with a ``samples`` list. An ``.eval`` log is a zip archive with ``header.json`` (the log
without samples) and one ``samples/<id>_epoch_<n>.json`` per sample.

Mapping: item = sample ``id``; run = ``eval.model``; sample index = ``epoch - 1``. The score is
the ``value`` of the chosen scorer (default: the first in ``scores``), converted the way
Inspect's ``value_to_float`` does it: ``C`` 1, ``P`` 0.5, ``I`` and ``N`` 0, numbers as they
are, ``yes``/``true`` 1, ``no``/``false`` 0, numeric strings parsed, anything else 0. A sample
with an ``error`` is ``error``; a sample with no score that hit a ``time`` or ``working`` limit
is ``timeout``; any other unscored sample (for example in a log written before scoring) is
``missing``. Ids listed in
``eval.dataset.sample_ids`` with no sample in the log become ``missing`` rows, one per epoch
in ``eval.config.epochs``.

Ignored: messages, events, model output, the store, usage, the ``answer`` and ``explanation``
of each score, and ``results`` / ``reductions`` (they are recomputed from the samples).
Recent Inspect versions compress ``.eval`` archives with zstd, which Python's ``zipfile``
reads only from 3.14; for those, convert first with
``inspect log convert LOG_OR_DIR --to json --output-dir OUT_DIR``.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any

from ..model import EvalTable, Row, Status

_LETTERS = {"C": 1.0, "P": 0.5, "I": 0.0, "N": 0.0}


def looks_like_inspect(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("eval"), dict)
        and "task" in data["eval"]
        and "model" in data["eval"]
    )


def value_to_float(value: Any) -> float:
    """Inspect's default conversion of a score value to a number."""
    if isinstance(value, str) and value in _LETTERS:
        return _LETTERS[value]
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else 0.0
    if isinstance(value, str):
        text = value.lower()
        if text in ("yes", "true"):
            return 1.0
        if text in ("no", "false"):
            return 0.0
        try:
            number = float(text)
        except ValueError:
            return 0.0
        return number if math.isfinite(number) else 0.0
    return 0.0


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.suffix.lower() == ".eval":
        try:
            with zipfile.ZipFile(path) as archive:
                header = json.loads(archive.read("header.json"))
                samples = [
                    json.loads(archive.read(name))
                    for name in sorted(archive.namelist())
                    if name.startswith("samples/") and name.endswith(".json")
                ]
        except NotImplementedError as exc:
            raise ValueError(
                f"{path}: {exc}; convert it first with "
                f"'inspect log convert {path} --to json --output-dir OUT_DIR'"
            ) from None
        except zipfile.BadZipFile:
            raise ValueError(f"{path}: not a zip archive, so not an Inspect .eval log") from None
        except KeyError:
            raise ValueError(
                f"{path}: no header.json in the archive (an evaluation still in progress?)"
            ) from None
        return header, samples
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not an Inspect evaluation log")
    return data, list(data.get("samples") or [])


def _short(text: Any, limit: int = 200) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "..."


def read_inspect(
    path: str | Path, run_id: str | None = None, scorer: str | None = None
) -> EvalTable:
    path = Path(path)
    header, samples = _load(path)
    if not looks_like_inspect(header):
        raise ValueError(f"{path} is not an Inspect evaluation log")
    spec = header["eval"]
    run = run_id or str(spec.get("model"))
    task = spec.get("task")
    rows: list[Row] = []
    seen: set[str] = set()
    scorers_found: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("id") is None:
            raise ValueError(f"{path}: a sample has no id")
        item = str(sample["id"])
        seen.add(item)
        epoch = int(sample.get("epoch") or 1)
        meta: dict[str, Any] = {"task": task, "epoch": epoch}
        scores = sample.get("scores") or {}
        limit = sample.get("limit") or {}
        if limit:
            meta["limit"] = limit.get("type")
        if sample.get("error"):
            error = sample["error"]
            message = error.get("message") if isinstance(error, dict) else error
            meta["error"] = _short(message)
            status, score = Status.ERROR, None
        elif scores and (scorer is None or scorer in scores):
            name = scorer or next(iter(scores))
            scorers_found.add(name)
            meta["scorer"] = name
            status, score = Status.OK, value_to_float(scores[name].get("value"))
        elif scores:
            meta["error"] = f"no score from {scorer}"
            status, score = Status.ERROR, None
        elif limit.get("type") in ("time", "working"):
            status, score = Status.TIMEOUT, None
        else:
            meta["reason"] = "no score"
            status, score = Status.MISSING, None
        rows.append(Row(item, run, score, status, epoch - 1, meta))
    if scorer is not None and samples and scorer not in scorers_found:
        raise ValueError(f"{path}: no sample has a score from scorer {scorer!r}")
    dataset = spec.get("dataset") or {}
    epochs = int((spec.get("config") or {}).get("epochs") or 1)
    for sid in dataset.get("sample_ids") or []:
        if str(sid) in seen:
            continue
        for epoch in range(1, epochs + 1):
            meta = {"task": task, "epoch": epoch}
            rows.append(Row(str(sid), run, None, Status.MISSING, epoch - 1, meta))
    return EvalTable(rows)
