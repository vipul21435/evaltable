# evaltable

Read the result files of public evaluation harnesses into one tidy per-item table (item, run, score, status)
and check the runs for the differences that make their scores incomparable.

## Why this is hard

Every harness writes its results its own way, and they disagree on what happens to an item that errored, timed
out or never ran: some count it as a failure, some leave it out of the denominator, some do not record it at
all. Two headline scores can therefore differ for bookkeeping reasons alone, before any difference between the
systems being compared. Seeing that needs the per-item results of every run side by side, under one set of
statuses, which is what this table is for.

## Install

```
pip install git+https://github.com/vipul21435/evaltable
```

Python 3.10 or newer; the core has no dependencies. Parquet output needs pyarrow:

```
pip install "evaltable[parquet] @ git+https://github.com/vipul21435/evaltable"
```

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

The commands:

```
evaltable read FILE [--format F] [--run NAME] [-o out.csv|out.jsonl|out.parquet]
evaltable merge FILE... [--fill-missing] [--items ids.txt] [-o merged.csv]
evaltable check FILE... [--allow-missing] [--allow-duplicates] [--allow-sample-mismatch] [--max-gap G] [--json]
evaltable pivot FILE... [--value score|strict|status] [-o wide.csv]
evaltable summary FILE... [--items ids.txt] [--confidence 0.95] [--json]
```

All of them take `--format` (default: detected from the extension and content), `--run` (once per file),
`--items` (reference item ids, one per line), `--map` for generic CSV/JSONL (`item=question_id,score=correct`),
`--metric` (lm-eval metric, Inspect scorer or OpenAI Evals metrics key), `--filter` (lm-eval filter) and
`--tests base|plus` (EvalPlus). `merge`, `pivot` and `summary` refuse two files that give one run id rows
for the same items (pass `--run` once per file); `check` reads them and reports the repeated rows. `check` exits 0 when the runs are comparable under the flags given, 1 when
they are not, and 2 on bad input. The same functions are importable: `evaltable.readers.read_many`,
`evaltable.summary.summarize`, `evaltable.checks.run_checks`, `evaltable.pivot.pivot`.

## How it works

**The table.** One row per (run, item, sample) with `item_id`, `run_id`, `sample_index`, `score` and `status`
(`ok`, `error`, `timeout`, `missing`), then metadata columns. `ok` rows carry a score; the score of other rows
is ignored. `merge` sorts rows into one canonical order, so its output does not depend on the order of the
input files (a property test checks this, and that the long and wide forms agree).

**Collapsing samples.** For each run and item, with `n` rows of which the `ok` ones score `s_1..s_k`: the
errors-as-failures item score is `(s_1 + ... + s_k) / n` and the errors-left-out item score is the mean of
`s_1..s_k`, undefined when `k = 0`. The item's status is the first of error, timeout, missing, ok that any of
its rows has.

**The two conventions.** Over a reference set of `N` items (by default the union over all runs, or `--items`):

- errors as failures: the sum of item scores over all `N` items divided by `N`; an item a run never recorded
  scores 0. On a SWE-bench split this is resolved instances out of all instances; it is also how HumanEval
  and EvalPlus treat a timeout (only a pass counts).
- errors left out: the mean item score over the items with at least one `ok` sample.

`gap` is the second minus the first. The two agree when a run has no errored, timed-out or missing items
(property-tested). Coverage is the share of reference items with an `ok` sample; error rate is the share of
present items whose status is error or timeout.

**Intervals.** Wilson score intervals (Wilson, 1927): with `p = x/n` and `z` the normal quantile,
`(p + z²/2n ± z·sqrt(p(1-p)/n + z²/4n²)) / (1 + z²/n)`. They are tested against `scipy.stats.binomtest`. For
scores that are not 0/1 the same formula is applied to the mean, which is an approximation; no interval is
shown when a score lies outside [0, 1].

**Checks.** `item_sets`: reference items each run lacks. `duplicates`: repeated (run, item, sample) keys.
`errors`: errored or timed-out items per run, with both scores over the items the run has and their gap
(fails when the gap exceeds `--max-gap`, default 0). This gap leaves missing items out of both scores, so
it is smaller than the `summary` gap whenever a run lacks items: `check --allow-missing --max-gap 0.01` can
pass a run whose `summary` gap is far larger. `sample_counts`: distinct sample indexes per item,
compared within and across runs.

**Readers.** Each module in `src/evaltable/readers/` states its source and what it ignores.

| format | source read for the structure | item, run, sample | status |
|---|---|---|---|
| `swebench` results.json | SWE-bench/experiments `evaluation/<split>/<entry>/results/results.json` and `analysis/get_results.py` | instance id, entry folder | `resolved` 1; `no_generation` (no log folder or no `patch.diff`) missing; `test_timeout` timeout; `install_fail`, `reset_failed`, `test_errored`, `no_logs` (`patch.diff` without test output) error; `no_apply` (older files) and the rest 0 |
| `swebench` run report | `make_run_report` in SWE-bench `swebench/harness/reporting.py` | instance id, file name (its directory for `results.json`) | `resolved_ids` 1, `unresolved_ids` 0, `error_ids` error, `empty_patch_ids` and `incomplete_ids` missing |
| `lm-eval` | `evaluator.py` and `loggers/evaluation_tracker.py` of lm-evaluation-harness (`--log_samples`) | `<task>/<doc_id>`, model directory | all ok; score = chosen metric |
| `inspect` | `log/_log.py`, `scorer/_metric.py`, `log/_recorders/eval.py` of inspect_ai; the Log Files docs page | sample id, `eval.model`, epoch - 1 | `error` error; time/working limit without a score timeout; unscored missing; `value_to_float` for values |
| `openai-evals` | `evals/record.py` and `evals/base.py` of openai/evals | `sample_id`, completion functions, event order | `match.correct` 1/0 (or a `metrics` key); `error` event or no scoring event error |
| `humaneval` | `human_eval/evaluation.py` and `execution.py` of openai/human-eval | `task_id`, file name, completion order | `passed` 1, `failed: ...` 0, `timed out` timeout |
| `evalplus` | `evalplus/evaluate.py` and `eval/__init__.py` of evalplus | `task_id`, file name, solution order | base and plus pass 1 (`--tests base`: base only), any fail 0, timeout timeout, other null error; a file with no `plus_status` (`--base-only`) needs `--tests base` |
| `csv`, `jsonl` | this table, or any file through `--map` | mapped columns | `status` column, else ok with a score and error without |

The benchmarks behind these formats are described in Jimenez et al., "SWE-bench: Can Language Models Resolve
Real-World GitHub Issues?" (arXiv:2310.06770), Chen et al., "Evaluating Large Language Models Trained on Code"
(arXiv:2107.03374, HumanEval) and Liu et al. (2023), the EvalPlus paper on rigorous evaluation of code
generation (arXiv:2305.01210).

## Results on real public data

Both data sets are downloaded by `scripts/fetch_public_data.py` at pinned commits; none of the files are
committed here.

**SWE-bench Verified entries.** Source: the public
[SWE-bench/experiments](https://github.com/SWE-bench/experiments) repository at commit `40f164d5b8f1`. That
repository carries no licence file (GitHub reports none), so only the fetch script and the derived numbers
below are in this repository. 135 of the 182 entries under `evaluation/verified/` have a
`results/results.json` at that commit; the other 47 return 404 and are skipped. The reference set is the union
of instance ids listed by any entry, 500 ids.

What the outcome lists mean: newer files are written by `analysis/get_results.py` in that repository from the
logs uploaded with the entry. An id is under `no_generation` when the entry's `logs/` has no folder for it or
the folder has no `patch.diff`, and under `no_logs` when there is a `patch.diff` but no `test_output.txt`. So
a file cannot tell a patch that was never produced from logs that were not uploaded. The counts and gaps
below describe what was uploaded; they are not a statement about any system.

```
$ python scripts/fetch_public_data.py DATA
$ python examples/swebench_conventions.py DATA --write-items verified-ids.txt
split: verified
entries: 135 (13 list every outcome, 122 do not)
reference instances: 500
entries with ids under no_generation (no log folder or no patch.diff): 92 (1109 ids, summed over entries)
entries with ids under no_logs (patch.diff without test output): 41 (216 ids, summed over entries)
entries with ids under test_timeout: 6 (10 ids, summed over entries)
entries with ids under install_fail: 1 (5 ids, summed over entries)
entries with ids under reset_failed: 6 (47 ids, summed over entries)
entries with ids under test_errored: 0 (0 ids, summed over entries)
entries with reference ids listed under no outcome: 6 (41 ids, summed over entries)
entries with any of these: 108
gap, all entries: median 0.0031, max 0.2177
gap, entries with any of these: median 0.0046, mean 0.0122
gap >= 0.01: 32
gap >= 0.05: 5
gap >= 0.10: 2
```

So 108 of the 135 files leave at least one reference instance without a test result, and for 32 of them the score
moves by a point or more between the two conventions. The errors-as-failures score is resolved instances out
of 500, the rate `get_results.py` itself prints. `--write-items` saves the 500 reference ids, one per line;
with them any one entry's file reads the same way from the command line:

```
$ evaltable summary --items verified-ids.txt DATA/swebench/evaluation/verified/<entry>/results/results.json
```

**Inspect logs.** Source: the evaluation logs kept in the test suite of
[UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) (MIT licence) at commit
`e4435baa530d`: two `.eval` archives and two `.json` logs, read end to end and compared with the accuracy each
log reports in its own `results` block.

```
$ python examples/inspect_logs.py DATA
2025-02-11T15-17-00-05-00_popularity_dPiJifoWeEQBrfWsAopzWr.eval: task popularity, 10 items, ok 0, missing 10, error 0; errors as failures 0.0000, log's own accuracy none
2025-02-11T15-18-04-05-00_popularity_mj7khqpMM4GBCfVQozKgzB.eval: task popularity, 10 items, ok 10, missing 0, error 0; errors as failures 0.8000, log's own accuracy 0.8000
2025-05-12T20-28-13-04-00_popularity.json: task popularity, 3 items, ok 3, missing 0, error 0; errors as failures 0.6667, log's own accuracy 0.6667
2025-05-12T20-28-26-04-00_security-guide.json: task security_guide, 3 items, ok 3, missing 0, error 0; errors as failures 1.0000, log's own accuracy 1.0000
```

The first archive was written before scoring (its own results list no scores), so all ten samples are
`missing`. `check` refuses to compare it with the scored run of the same ten samples:

```
$ evaltable check --run unscored --run scored \
    DATA/inspect/2025-02-11T15-17-00-05-00_popularity_dPiJifoWeEQBrfWsAopzWr.eval \
    DATA/inspect/2025-02-11T15-18-04-05-00_popularity_mj7khqpMM4GBCfVQozKgzB.eval
check          result  detail
item_sets      FAIL    item sets differ over 10 items: unscored lacks 10
duplicates     ok      no duplicated rows
errors         ok      no errored or timed-out items
sample_counts  ok      every item has 1 sample(s) in every run
  unscored lacks: 1, 10, 2, 3, 4, 5, 6, 7, 8, 9
not comparable
```

## Limitations

- Only the SWE-bench `results.json` and Inspect readers have been run on real public files, and the four
  Inspect logs are small: 26 samples, one scorer, one epoch, no errors and no limits. The Inspect paths for
  errors, limits, several epochs and several scorers, and the SWE-bench run report, lm-evaluation-harness,
  OpenAI Evals, HumanEval and EvalPlus readers, are tested only on fixtures written from the harnesses' source
  code with synthetic values. The Open LLM Leaderboard sample files (lm-eval output) need a Hugging Face
  login, so they were not used.
- An Inspect score whose value is a dict or a list is read as ok with score 0, as Inspect's own conversion
  does, but without a warning.
- `summary` and `check` ignore items outside `--items` without saying so.
- Item ids are compared as raw text: `1` and `01` are two items, and so are the composed and decomposed
  Unicode forms of one accented id.
- The status mapping involves choices other people may make differently. A SWE-bench id under
  `no_generation` (no log folder or no `patch.diff` uploaded) is `missing`, so it is a failure in the first
  convention and left out in the second; an id under `no_logs` (`patch.diff` without test output) is `error`.
  In the 13 older files a patch that did not apply is listed under `no_apply` and read as a plain failure; in
  the 122 newer files the same case has a `patch.diff` and no test output, so it is under `no_logs` and read
  as `error`. EvalPlus and HumanEval timeouts are `timeout`.
- Newer SWE-bench `results.json` files list only `resolved`, `no_generation` and `no_logs`; reading one needs
  the split's instance ids (`--items`).
- Recent Inspect `.eval` archives use zstd compression, which Python's `zipfile` reads only from 3.14. Convert
  them first with `inspect log convert LOG_OR_DIR --to json --output-dir OUT_DIR`.
- Intervals treat the per-item mean as one Bernoulli outcome; there is no correction for several samples per
  item or for clustering. For paired comparisons and error bars use evalstats.
- Files are read whole into memory; `--map`, `--metric` and `--filter` apply to every file in one command;
  Parquet is written, not read.

## Related tools

- [evalstats](https://github.com/vipul21435/evalstats): error bars and paired comparisons for evaluations.
  Its input columns are `item`, `model` and `score`, and it reports a score that is not a finite number as an
  error. To hand it this table, rename `item_id` to `item` and `run_id` to `model`, and first either drop the
  rows whose status is not `ok` (errors left out) or give them score 0 (errors as failures); evaltable writes
  an empty score on those rows.
- [irtbench](https://github.com/vipul21435/irtbench) and [evalstop](https://github.com/vipul21435/evalstop)
  also consume this table.
- [testlogs](https://github.com/vipul21435/testlogs) parses test-runner output, one level below these result
  files.
- The harnesses whose files are read here: [SWE-bench](https://github.com/SWE-bench/SWE-bench),
  [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
  [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai), [OpenAI Evals](https://github.com/openai/evals),
  [HumanEval](https://github.com/openai/human-eval) and [EvalPlus](https://github.com/evalplus/evalplus).
- [rankstable](https://github.com/vipul21435/rankstable): how stable a leaderboard's order is to the averaging rule and the choice of benchmarks.

## References

- E. B. Wilson (1927). Probable Inference, the Law of Succession, and Statistical Inference. Journal of the
  American Statistical Association 22(158), 209-212. doi:10.1080/01621459.1927.10502953

## Licence

MIT, see `LICENSE`.

Built by Vipul Raj Jha · https://vipul21435.github.io
