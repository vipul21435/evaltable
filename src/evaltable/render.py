"""Plain-text rendering of summaries, overlap matrices and check reports."""

from __future__ import annotations

from collections.abc import Sequence

from .checks import CheckReport
from .summary import RunSummary


def grid(headers: Sequence[str], rows: Sequence[Sequence[str]], left: int = 1) -> str:
    """Aligned columns; the first ``left`` columns are left-aligned, the rest right-aligned."""
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row, strict=True)]

    def line(cells: Sequence[str]) -> str:
        parts = [
            cell.ljust(w) if k < left else cell.rjust(w)
            for k, (cell, w) in enumerate(zip(cells, widths, strict=True))
        ]
        return "  ".join(parts).rstrip()

    return "\n".join([line(headers), *(line(row) for row in rows)])


def _score(value: float | None, ci: tuple[float, float] | None) -> str:
    if value is None:
        return "n/a"
    if ci is None:
        return f"{value:.4f}"
    return f"{value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def summary_text(summaries: Sequence[RunSummary], confidence: float) -> str:
    headers = [
        "run",
        "items",
        "present",
        "ok",
        "error",
        "timeout",
        "missing",
        "coverage",
        "error rate",
        "errors as failures",
        "errors left out",
        "n left in",
        "gap",
    ]
    rows = []
    for s in summaries:
        rows.append(
            [
                s.run_id,
                str(s.n_reference),
                str(s.n_present),
                str(s.n_ok),
                str(s.n_error),
                str(s.n_timeout),
                str(s.n_missing),
                f"{s.coverage:.3f}",
                f"{s.error_rate:.3f}",
                _score(s.strict, s.strict_ci),
                _score(s.lenient, s.lenient_ci),
                str(s.lenient_n),
                "n/a" if s.gap is None else f"{s.gap:+.4f}",
            ]
        )
    level = f"{confidence * 100:g}"
    return grid(headers, rows) + f"\n(scores with {level}% Wilson intervals)"


def overlap_text(matrix: dict[str, dict[str, int]]) -> str:
    runs = list(matrix)
    rows = [[a, *(str(matrix[a][b]) for b in runs)] for a in runs]
    return "items present in both runs\n" + grid(["", *runs], rows)


def _listing(items: Sequence[str], limit: int = 10) -> str:
    shown = ", ".join(items[:limit])
    return shown if len(items) <= limit else f"{shown}, ... ({len(items) - limit} more)"


def check_text(report: CheckReport) -> str:
    rows = [[f.check, "ok" if f.passed else "FAIL", f.message] for f in report.findings]
    lines = [grid(["check", "result", "detail"], rows, left=3)]
    for finding in report.findings:
        if finding.check == "item_sets":
            for run, absent in finding.details["lacking"].items():
                lines.append(f"  {run} lacks: {_listing(absent)}")
        elif finding.check == "errors":
            for run, info in finding.details["runs"].items():
                lines.append(f"  {run} errored or timed out: {_listing(info['errored'])}")
        elif finding.check == "duplicates":
            keys = [
                f"{d['run_id']}/{d['item_id']}#{d['sample_index']}" for d in finding.details["rows"]
            ]
            if keys:
                lines.append(f"  repeated: {_listing(keys)}")
        elif finding.check == "sample_counts" and finding.details["differing_items"]:
            items = list(finding.details["differing_items"])
            lines.append(f"  differing items: {_listing(items)}")
    lines.append("comparable" if report.comparable else "not comparable")
    return "\n".join(lines)
