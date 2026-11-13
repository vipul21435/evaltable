"""Readers for HumanEval and EvalPlus result files.

HumanEval. Source: ``human_eval/evaluation.py`` and ``human_eval/execution.py`` in
openai/human-eval. ``evaluate_functional_correctness samples.jsonl`` writes
``samples.jsonl_results.jsonl``: each input line plus ``result`` (``passed``, ``timed out`` or
``failed: <exception>``) and ``passed`` (bool). Mapping: item = ``task_id``; sample index =
the order of the completion among the lines of its task; ``passed`` is ok 1, ``timed out`` is
``timeout``, ``failed: ...`` is ok 0 (the completion ran and was wrong). HumanEval's own
pass@k counts a timeout as a failure, which is the errors-as-failures convention.
Ignored: ``completion`` and any other input columns, apart from keeping them out of metadata.

EvalPlus. Source: ``evalplus/evaluate.py`` and ``evalplus/eval/__init__.py`` in
evalplus/evalplus. ``evalplus.evaluate`` writes ``*_eval_results.json`` (older versions
``*.eval_results.json``) with ``date``, ``hash`` and ``eval``, a map from ``task_id`` to a list of
``{task_id, solution, base_status, plus_status, base_fail_tests, plus_fail_tests}``, statuses
being ``pass``, ``fail``, ``timeout`` or null. Mapping: with ``tests="plus"`` (default) a
solution scores 1 when both statuses are ``pass``, as EvalPlus counts it; with ``tests="base"``
only ``base_status`` counts. Any ``fail`` gives ok 0; otherwise a ``timeout`` gives
``timeout``, and a null status gives ``error``. Ignored: ``solution``, the failing test inputs,
``date`` and ``hash``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..model import EvalTable, Row, Status
from .generic import iter_jsonl


def looks_like_humaneval(record: dict[str, Any]) -> bool:
    return "task_id" in record and "passed" in record and "result" in record


def looks_like_evalplus(data: Any) -> bool:
    if not isinstance(data, dict) or not isinstance(data.get("eval"), dict):
        return False
    for results in data["eval"].values():
        return isinstance(results, list) and all("base_status" in r for r in results)
    return "hash" in data


def _humaneval_run(path: Path) -> str:
    name = path.name
    for suffix in ("_results.jsonl", ".jsonl"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.removesuffix(".jsonl")


def read_humaneval(path: str | Path, run_id: str | None = None) -> EvalTable:
    path = Path(path)
    run = run_id or _humaneval_run(path)
    counter: Counter[str] = Counter()
    rows = []
    for number, record in enumerate(iter_jsonl(path), start=1):
        if not looks_like_humaneval(record):
            raise ValueError(f"line {number}: not a HumanEval results line")
        task = str(record["task_id"])
        index = counter[task]
        counter[task] += 1
        result = str(record["result"])
        if record["passed"] is True or result == "passed":
            status, score = Status.OK, 1.0
        elif result == "timed out":
            status, score = Status.TIMEOUT, None
        else:
            status, score = Status.OK, 0.0
        meta = {"result": result[:200]}
        rows.append(Row(task, run, score, status, index, meta))
    return EvalTable(rows)


def _evalplus_run(path: Path) -> str:
    name = path.name
    for suffix in ("_eval_results.json", ".eval_results.json", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)] or path.parent.name
    return path.stem


def read_evalplus(path: str | Path, run_id: str | None = None, tests: str = "plus") -> EvalTable:
    if tests not in ("base", "plus"):
        raise ValueError(f"tests must be 'base' or 'plus', got {tests!r}")
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not looks_like_evalplus(data):
        raise ValueError(f"{path} is not an EvalPlus eval_results.json")
    run = run_id or _evalplus_run(path)
    rows = []
    for task, results in data["eval"].items():
        for index, result in enumerate(results):
            base = result.get("base_status")
            statuses = [base] if tests == "base" else [base, result.get("plus_status")]
            if all(s == "pass" for s in statuses):
                status, score = Status.OK, 1.0
            elif "fail" in statuses:
                status, score = Status.OK, 0.0
            elif "timeout" in statuses:
                status, score = Status.TIMEOUT, None
            else:
                status, score = Status.ERROR, None
            meta = {"base_status": base, "plus_status": result.get("plus_status"), "tests": tests}
            rows.append(Row(str(task), run, score, status, index, meta))
    return EvalTable(rows)
