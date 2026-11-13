# evaltable

Read the result files of evaluation harnesses into one tidy per-item table (item, run, score, status) and
check the runs for the differences that make their scores incomparable.

Status: early. The table model, the checks, the summary and the command line work on generic CSV and JSONL
files. Readers for specific harness formats are not written yet (see "What works so far").

## Why this is hard

Every harness writes its results its own way, and they disagree on what happens to an item that errored, timed
out or never ran: some count it as a failure, some leave it out of the denominator, some do not record it at
all. Two headline scores can therefore differ for bookkeeping reasons alone, before any difference between the
systems being compared. Spotting that needs the per-item results of every run side by side, which is what this
table is for.

## Install

```
pip install git+https://github.com/vipul21435/evaltable
```

Python 3.10 or newer. The core has no dependencies; Parquet output needs `pip install "evaltable[parquet]"`.

## Quickstart

The two files in `examples/data/` are small synthetic runs over the same eight items: `model_a` has one
errored and one timed-out item, `model_b` never ran `item-08`.

```
$ evaltable summary examples/data/model_a.jsonl examples/data/model_b.jsonl
run      items  present  ok  error  timeout  missing  coverage  error rate       errors as failures          errors left out  n left in      gap
model_a      8        8   6      1        1        0     0.750       0.250  0.5000 [0.2152, 0.7848]  0.6667 [0.3000, 0.9032]          6  +0.1667
model_b      8        7   7      0        0        1     0.875       0.000  0.5000 [0.2152, 0.7848]  0.5714 [0.2505, 0.8418]          7  +0.0714
(scores with 95% Wilson intervals)

items present in both runs
         model_a  model_b
model_a        8        7
model_b        7        7
```

Counted with errors as failures the two runs tie at 0.5; with errors left out `model_a` looks ahead. `check`
says why, and exits 1:

```
$ evaltable check examples/data/model_a.jsonl examples/data/model_b.jsonl
check          result  detail
item_sets      FAIL    item sets differ over 8 items: model_b lacks 1
duplicates     ok      no duplicated rows
errors         FAIL    model_a: 2 errored or timed out, gap +0.1667
sample_counts  ok      every item has 1 sample(s) in every run
  model_b lacks: item-08
  model_a errored or timed out: item-05, item-07
not comparable
```

`--allow-missing`, `--allow-duplicates`, `--allow-sample-mismatch` and `--max-gap G` relax the check;
`--json` prints the report as JSON. The other commands:

```
evaltable read FILE [--map item=question_id,score=correct] [--run NAME] [-o out.csv|out.jsonl]
evaltable merge FILE... [--fill-missing] [-o merged.csv]
evaltable pivot FILE... [--value score|strict|status]
```

## The table and the two conventions

The long table has one row per (run, item, sample) with columns `item_id`, `run_id`, `sample_index`, `score`
and `status` (`ok`, `error`, `timeout`, `missing`), followed by any metadata columns. The samples of an item
are collapsed per run before scoring:

- **errors as failures**: every item in the reference set counts (by default the union of items over all
  runs). Error, timeout and missing samples score 0, and an item a run never recorded scores 0.
- **errors left out**: only `ok` samples count; an item with no `ok` sample is left out of the denominator.

The `gap` column is the second minus the first. When a run has no errored, timed-out or missing items, the two
agree (a property test checks this). Intervals are Wilson score intervals; for scores between 0 and 1 that are
not 0/1 the same formula is applied to the mean, which is an approximation, and no interval is shown when a
score falls outside [0, 1].

## What works so far

- Long table model, CSV and JSONL reading (with a column map) and writing; Parquet writing with pyarrow.
- Checks: differing item sets, duplicated rows, errored items with the score both ways, differing sample
  counts per item.
- Summary per run (both scores with Wilson intervals, coverage, error rate) and the pairwise overlap matrix.
- Wide pivot, items by runs.

Not done yet: readers for SWE-bench results and reports, lm-evaluation-harness samples, Inspect AI logs,
OpenAI Evals records and HumanEval/EvalPlus results; any run on real public data.

## Licence

MIT, see `LICENSE`.

Built by Vipul Raj Jha · https://vipul21435.github.io
