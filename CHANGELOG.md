# Changelog

## 0.1.0 (2026-11-13)

First release.

- Long per-item table (item, run, sample, score, status, metadata) with CSV and JSONL reading and writing,
  and Parquet writing through the `parquet` extra.
- Readers for SWE-bench `results.json` and harness run reports, lm-evaluation-harness `--log_samples` files,
  Inspect AI `.json` and `.eval` logs, OpenAI Evals record files, HumanEval and EvalPlus results, and generic
  CSV/JSONL with a column map; the format is detected from the extension and content.
- Per-run scores under two conventions (errors as failures, errors left out) with Wilson intervals, coverage,
  error rate and the pairwise overlap matrix of item sets.
- Checks for differing item sets, duplicated rows, errored items and differing sample counts; `check` exits 1
  when runs are not comparable under the flags given.
- Wide pivot of items by runs.
- Command line: `evaltable read`, `merge`, `check`, `pivot`, `summary`.
- A fetch script for public SWE-bench and Inspect result files and two examples that reproduce the README
  numbers.
