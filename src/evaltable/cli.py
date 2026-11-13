"""Command-line interface: ``evaltable read | merge | check | pivot | summary``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .checks import CheckOptions, run_checks
from .model import EvalTable
from .pivot import PIVOT_VALUES, pivot
from .readers import FORMATS, ReadOptions, parse_column_map, read_many
from .render import check_text, overlap_text, summary_text
from .summary import overlap_matrix, summarize
from .writers import WRITE_FORMATS, write_table


def _add_inputs(parser: argparse.ArgumentParser, many: bool = True) -> None:
    parser.add_argument("files", nargs="+" if many else 1, metavar="FILE", help="result file(s)")
    parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", *FORMATS],
        help="input format (default: from the file extension)",
    )
    parser.add_argument(
        "--map",
        default=None,
        metavar="FIELD=COLUMN,...",
        help="column map for generic CSV/JSONL, fields: item, run, score, status, sample",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        metavar="NAME",
        help="run id for an input file; give once per file, in order (default: from the file)",
    )
    parser.add_argument(
        "--items",
        default=None,
        metavar="FILE",
        help="reference item ids, one per line (default: union of items over all runs)",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="lm-eval metric, Inspect scorer or OpenAI Evals metrics key (default: the first)",
    )
    parser.add_argument("--filter", default=None, help="lm-eval filter (default: every filter)")
    parser.add_argument(
        "--tests",
        default="plus",
        choices=["base", "plus"],
        help="EvalPlus tests that must pass (default: plus, i.e. base and plus)",
    )


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", default=None, help="output file (default: stdout)")
    parser.add_argument(
        "--to", default=None, choices=WRITE_FORMATS, help="output format (default: from -o, or csv)"
    )


def _load(args: argparse.Namespace) -> EvalTable:
    files: list[str] = args.files
    runs: list[str] | None = args.run
    if runs is not None and len(runs) != len(files):
        raise ValueError(f"--run was given {len(runs)} time(s) for {len(files)} file(s)")
    column_map = parse_column_map(args.map)
    reference = _reference(args)
    options = ReadOptions(
        metric=args.metric,
        filter=args.filter,
        tests=args.tests,
        items=tuple(reference) if reference is not None else None,
    )
    return read_many(files, args.format, column_map, runs, options)


def _reference(args: argparse.Namespace) -> list[str] | None:
    # Read once: the file may be a pipe, which cannot be read a second time.
    if not hasattr(args, "reference_items"):
        if args.items is None:
            args.reference_items = None
        else:
            lines = Path(args.items).read_text(encoding="utf-8").splitlines()
            args.reference_items = [line.strip() for line in lines if line.strip()]
    return args.reference_items


def _cmd_read(args: argparse.Namespace) -> int:
    write_table(_load(args), args.output, args.to)
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    table = _load(args)
    if args.fill_missing:
        table = table.with_missing(_reference(args))
    write_table(table, args.output, args.to)
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    options = CheckOptions(
        allow_missing=args.allow_missing,
        allow_duplicates=args.allow_duplicates,
        allow_sample_mismatch=args.allow_sample_mismatch,
        max_gap=args.max_gap,
    )
    report = run_checks(_load(args), _reference(args), options)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(check_text(report))
    return 0 if report.comparable else 1


def _cmd_pivot(args: argparse.Namespace) -> int:
    wide = pivot(_load(args), args.value)
    if args.output in (None, "-"):
        wide.write_csv(sys.stdout)
    else:
        with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
            wide.write_csv(handle)
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    table = _load(args)
    summaries = summarize(table, _reference(args), args.confidence)
    matrix = overlap_matrix(table)
    if args.json:
        data = {
            "confidence": args.confidence,
            "runs": [s.as_dict() for s in summaries],
            "overlap": matrix,
        }
        print(json.dumps(data, indent=2))
    else:
        print(summary_text(summaries, args.confidence))
        print()
        print(overlap_text(matrix))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaltable",
        description="Read evaluation result files into one per-item table and check runs.",
    )
    parser.add_argument("--version", action="version", version=f"evaltable {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("read", help="read one file into the long table")
    _add_inputs(p, many=False)
    _add_output(p)
    p.set_defaults(func=_cmd_read)

    p = sub.add_parser("merge", help="join many files into one long table")
    _add_inputs(p)
    p.add_argument(
        "--fill-missing",
        action="store_true",
        help="add a 'missing' row for every run that lacks a reference item",
    )
    _add_output(p)
    p.set_defaults(func=_cmd_merge)

    p = sub.add_parser("check", help="exit 1 when runs are not comparable under the flags")
    _add_inputs(p)
    p.add_argument("--allow-missing", action="store_true", help="tolerate differing item sets")
    p.add_argument("--allow-duplicates", action="store_true", help="tolerate repeated rows")
    p.add_argument(
        "--allow-sample-mismatch", action="store_true", help="tolerate differing sample counts"
    )
    p.add_argument(
        "--max-gap",
        type=float,
        default=0.0,
        help="largest tolerated gap between the two error conventions (default: 0)",
    )
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("pivot", help="write the wide table: items by runs")
    _add_inputs(p)
    p.add_argument("--value", default="score", choices=PIVOT_VALUES, help="cell value")
    p.add_argument("-o", "--output", default=None, help="output CSV (default: stdout)")
    p.set_defaults(func=_cmd_pivot)

    p = sub.add_parser("summary", help="per-run scores both ways, coverage, overlap")
    _add_inputs(p)
    p.add_argument("--confidence", type=float, default=0.95, help="interval level (default 0.95)")
    p.add_argument("--json", action="store_true", help="print the summary as JSON")
    p.set_defaults(func=_cmd_summary)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"evaltable: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
