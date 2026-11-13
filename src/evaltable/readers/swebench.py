"""Readers for SWE-bench result files.

Two shapes are read.

``results.json`` of a submission in the public SWE-bench/experiments repository
(``evaluation/<split>/<submission>/results/results.json``). It maps outcome names to lists of
instance ids. Older entries list every outcome (``no_generation``, ``generated``, ``with_logs``,
``install_fail``, ``reset_failed``, ``no_apply``, ``applied``, ``test_errored``,
``test_timeout``, ``resolved``); newer entries list only ``no_generation``, ``no_logs`` and
``resolved``, so the unresolved instances can only be recovered from the split's instance ids
(``items``). Newer files are written by ``analysis/get_results.py`` in that repository from the
entry's uploaded ``logs/`` folder: an id is under ``no_generation`` when ``logs/`` has no folder
for it or the folder has no ``patch.diff``, and under ``no_logs`` when it has ``patch.diff``
but no ``test_output.txt``. A file therefore cannot tell a patch that was never produced from
logs that were not uploaded, and in newer files a patch that failed to apply sits under
``no_logs``. Each instance takes the first outcome that applies:

==========================================  ==========  =====
outcome                                     status      score
==========================================  ==========  =====
resolved                                    ok          1
no_generation                               missing
test_timeout                                timeout
install_fail, reset_failed, test_errored    error
generated without logs, or no_logs          error
no_apply (older files only)                 ok          0
anything else in the reference set          ok          0
==========================================  ==========  =====

The run report written by ``swebench.harness.run_evaluation`` (``make_run_report`` in
``swebench/harness/reporting.py``): ``resolved_ids`` (ok, 1), ``unresolved_ids`` (ok, 0),
``error_ids`` (error), ``empty_patch_ids`` and ``incomplete_ids`` (missing). Ids in
``infra_failure_ids`` and ``ambiguous_failure_ids`` keep their status and get a metadata flag.

Ignored: the count fields, ``submitted_ids``, ``completed_ids``, ``failure_reasons`` and the
``resolved_by_repo`` / ``resolved_by_time`` breakdown files.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..model import EvalTable, Row, Status

RESULTS_KEYS = frozenset(
    {
        "no_generation",
        "generated",
        "with_logs",
        "install_fail",
        "reset_failed",
        "no_apply",
        "applied",
        "test_errored",
        "test_timeout",
        "resolved",
        "no_logs",
    }
)
REPORT_KEYS = frozenset({"resolved_ids", "unresolved_ids", "error_ids", "total_instances"})


def looks_like_results(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("resolved"), list)
        and set(data) <= RESULTS_KEYS
    )


def looks_like_report(data: Any) -> bool:
    return isinstance(data, dict) and set(data) >= REPORT_KEYS


def _default_run(path: Path) -> str:
    # evaluation/<split>/<submission>/results/results.json -> <submission>
    if path.parent.name == "results" and path.parent.parent.name:
        return path.parent.parent.name
    return path.stem


def _ids(data: dict[str, Any], key: str) -> set[str]:
    value = data.get(key) or []
    if not isinstance(value, list):
        raise ValueError(f"expected a list of instance ids under {key!r}")
    return {str(v) for v in value}


def results_rows(
    data: dict[str, Any], run_id: str, items: Iterable[str] | None = None
) -> list[Row]:
    """Rows for one ``results.json`` mapping."""
    lists = {key: _ids(data, key) for key in RESULTS_KEYS if key in data}
    if items is not None:
        reference = {str(i) for i in items}
    elif "generated" in lists:
        reference = lists["generated"] | lists.get("no_generation", set())
    else:
        raise ValueError(
            "this results.json lists only some outcomes (no 'generated' key), so unresolved "
            "instances cannot be told apart from absent ones; pass the split's instance ids "
            "with --items"
        )
    listed: set[str] = set().union(*lists.values()) if lists else set()
    empty: set[str] = set()
    errors = lists.get("install_fail", empty) | lists.get("reset_failed", empty)
    errors |= lists.get("test_errored", empty) | lists.get("no_logs", empty)
    no_logs = lists["generated"] - lists.get("with_logs", empty) if "generated" in lists else empty
    rows = []
    for iid in sorted(reference | listed):
        if iid in lists.get("resolved", empty):
            status, score, category = Status.OK, 1.0, "resolved"
        elif iid in lists.get("no_generation", empty):
            status, score, category = Status.MISSING, None, "no_generation"
        elif iid in lists.get("test_timeout", empty):
            status, score, category = Status.TIMEOUT, None, "test_timeout"
        elif iid in errors:
            category = next(
                k
                for k in ("install_fail", "reset_failed", "test_errored", "no_logs")
                if iid in lists.get(k, empty)
            )
            status, score = Status.ERROR, None
        elif iid in no_logs:
            status, score, category = Status.ERROR, None, "no_logs"
        elif iid in lists.get("no_apply", empty):
            status, score, category = Status.OK, 0.0, "no_apply"
        else:
            status, score, category = Status.OK, 0.0, "unresolved"
        rows.append(Row(iid, run_id, score, status, 0, {"swebench_outcome": category}))
    return rows


def report_rows(data: dict[str, Any], run_id: str) -> list[Row]:
    """Rows for one harness run report."""
    outcomes = [
        ("resolved_ids", Status.OK, 1.0),
        ("unresolved_ids", Status.OK, 0.0),
        ("error_ids", Status.ERROR, None),
        ("empty_patch_ids", Status.MISSING, None),
        ("incomplete_ids", Status.MISSING, None),
    ]
    infra = _ids(data, "infra_failure_ids")
    ambiguous = _ids(data, "ambiguous_failure_ids")
    rows: list[Row] = []
    seen: set[str] = set()
    for key, status, score in outcomes:
        for iid in sorted(_ids(data, key) - seen):
            seen.add(iid)
            meta: dict[str, Any] = {"swebench_outcome": key.removesuffix("_ids")}
            if iid in infra:
                meta["infra_failure"] = True
            if iid in ambiguous:
                meta["ambiguous_failure"] = True
            rows.append(Row(iid, run_id, score, status, 0, meta))
    return rows


def read_swebench(
    path: str | Path, run_id: str | None = None, items: Iterable[str] | None = None
) -> EvalTable:
    """Read a SWE-bench ``results.json`` or a harness run report (detected from the keys)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    run = run_id or _default_run(path)
    if looks_like_report(data):
        return EvalTable(report_rows(data, run))
    if looks_like_results(data):
        return EvalTable(results_rows(data, run, items))
    raise ValueError(f"{path} is neither a SWE-bench results.json nor a run report")
