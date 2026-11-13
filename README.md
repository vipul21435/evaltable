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
`--tests base|plus` (EvalPlus). `check` exits 0 when the runs are comparable under the flags given, 1 when
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
(fails when the gap exceeds `--max-gap`, default 0). `sample_counts`: distinct sample indexes per item,
compared within and across runs.

**Readers.** Each module in `src/evaltable/readers/` states its source and what it ignores.

| format | source read for the structure | item, run, sample | status |
|---|---|---|---|
| `swebench` results.json | SWE-bench/experiments `evaluation/<split>/<entry>/results/results.json` | instance id, entry folder | `resolved` 1; `no_generation` missing; `test_timeout` timeout; `install_fail`, `reset_failed`, `test_errored`, no logs error; `no_apply` and the rest 0 |
| `swebench` run report | `make_run_report` in SWE-bench `swebench/harness/reporting.py` | instance id, file name | `resolved_ids` 1, `unresolved_ids` 0, `error_ids` error, `empty_patch_ids` and `incomplete_ids` missing |
| `lm-eval` | `evaluator.py` and `loggers/evaluation_tracker.py` of lm-evaluation-harness (`--log_samples`) | `<task>/<doc_id>`, model directory | all ok; score = chosen metric |
| `inspect` | `log/_log.py`, `scorer/_metric.py`, `log/_recorders/eval.py` of inspect_ai; the Log Files docs page | sample id, `eval.model`, epoch - 1 | `error` error; time/working limit without a score timeout; unscored missing; `value_to_float` for values |
| `openai-evals` | `evals/record.py` and `evals/base.py` of openai/evals | `sample_id`, completion functions, event order | `match.correct` 1/0 (or a `metrics` key); `error` event or no scoring event error |
| `humaneval` | `human_eval/evaluation.py` and `execution.py` of openai/human-eval | `task_id`, file name, completion order | `passed` 1, `failed: ...` 0, `timed out` timeout |
| `evalplus` | `evalplus/evaluate.py` and `eval/__init__.py` of evalplus | `task_id`, file name, solution order | base and plus pass 1 (`--tests base`: base only), any fail 0, timeout timeout, null error |
| `csv`, `jsonl` | this table, or any file through `--map` | mapped columns | `status` column, else ok with a score and error without |

The benchmarks behind these formats are described in Jimenez et al., "SWE-bench: Can Language Models Resolve
Real-World GitHub Issues?" (arXiv:2310.06770), Chen et al., "Evaluating Large Language Models Trained on Code"
(arXiv:2107.03374, HumanEval) and Liu et al. (2023), the EvalPlus paper on rigorous evaluation of code
generation (arXiv:2305.01210).

## Results on real public data

Both data sets are downloaded by `scripts/fetch_public_data.py` at pinned commits; none of the files are
committed here.

**SWE-bench Verified submissions.** Source: the public
[SWE-bench/experiments](https://github.com/SWE-bench/experiments) repository at commit `40f164d5b8f1`. That
repository carries no licence file (GitHub reports none), so only the fetch script and the derived numbers
below are in this repository. 135 of the 182 entries under `evaluation/verified/` have a
`results/results.json` at that commit; the other 47 return 404 and are skipped. The reference set is the union
of instance ids listed by any entry, 500 ids.

```
$ python scripts/fetch_public_data.py DATA
$ python examples/swebench_conventions.py DATA
split: verified
submissions: 135 (13 list every outcome, 122 do not)
reference instances: 500
with missing instances (no patch): 92
with errored or timed-out instances: 51
with either: 108
gap, all submissions: median 0.0025, max 0.2177
gap, affected submissions: median 0.0045, mean 0.0120
gap >= 0.01: 32
gap >= 0.05: 5
gap >= 0.10: 2

largest 5 gaps (errors as failures -> errors left out):
  20250516_cortexa_o3: 0.6820 -> 0.8997 (gap +0.2177; missing 121, error 0, timeout 0)
  20250110_learn_by_interact_claude3.5: 0.6020 -> 0.7341 (gap +0.1321; missing 0, error 90, timeout 0)
  20260901_mini-v2.4.2_gemini-3-5-flash: 0.7180 -> 0.8141 (gap +0.0961; missing 59, error 0, timeout 0)
  20241221_codestory_midwit_claude-3-5-sonnet_swe-search: 0.6220 -> 0.6835 (gap +0.0615; missing 45, error 0, timeout 0)
  20240820_honeycomb: 0.4060 -> 0.4624 (gap +0.0564; missing 14, error 47, timeout 0)
```

So 108 of 135 entries have at least one instance without a patch or without an evaluation, and for 32 of them
the score moves by a point or more between the two conventions. The errors-as-failures column is resolved
instances out of 500. The same comparison for two entries from the command line, with the
500 ids written one per line to `verified-ids.txt`:

```
$ evaltable summary --items verified-ids.txt \
    DATA/swebench/evaluation/verified/20250516_cortexa_o3/results/results.json \
    DATA/swebench/evaluation/verified/20240820_honeycomb/results/results.json
run                  items  present   ok  error  timeout  missing  coverage  error rate       errors as failures          errors left out  n left in      gap
20240820_honeycomb     500      486  439     47        0       14     0.878       0.097  0.4060 [0.3638, 0.4496]  0.4624 [0.4163, 0.5092]        439  +0.0564
20250516_cortexa_o3    500      379  379      0        0      121     0.758       0.000  0.6820 [0.6399, 0.7213]  0.8997 [0.8654, 0.9261]        379  +0.2177
(scores with 95% Wilson intervals)

items present in both runs
                     20240820_honeycomb  20250516_cortexa_o3
20240820_honeycomb                  486                  371
20250516_cortexa_o3                 371                  379
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

- Only the SWE-bench `results.json` and Inspect readers have been run on real public files. The SWE-bench run
  report, lm-evaluation-harness, OpenAI Evals, HumanEval and EvalPlus readers are written from the harnesses'
  source code and tested on fixtures with that structure and synthetic values. The Open LLM Leaderboard
  sample files (lm-eval output) need a Hugging Face login, so they were not used.
- The status mapping involves choices other people may make differently: a SWE-bench instance with no patch
  is `missing` (so it is a failure in the first convention and left out in the second), a patch that does not
  apply is a plain failure, and EvalPlus and HumanEval timeouts are `timeout`.
- Newer SWE-bench `results.json` files list only resolved instances, instances without a patch and instances
  without logs; reading one needs the split's instance ids (`--items`).
- Recent Inspect `.eval` archives use zstd compression, which Python's `zipfile` reads only from 3.14. Convert
  them with `inspect log convert --to json` first.
- Intervals treat the per-item mean as one Bernoulli outcome; there is no correction for several samples per
  item or for clustering. For paired comparisons and error bars use evalstats.
- Files are read whole into memory; `--map`, `--metric` and `--filter` apply to every file in one command;
  Parquet is written, not read.

## Related tools

- [evalstats](https://github.com/vipul21435/evalstats): error bars and paired comparisons for evaluations;
  it takes a per-item table like this one.
- [irtbench](https://github.com/vipul21435/irtbench) and [evalstop](https://github.com/vipul21435/evalstop)
  also consume this table.
- [testlogs](https://github.com/vipul21435/testlogs) parses test-runner output, one level below these result
  files.
- The harnesses whose files are read here: [SWE-bench](https://github.com/SWE-bench/SWE-bench),
  [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
  [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai), [OpenAI Evals](https://github.com/openai/evals),
  [HumanEval](https://github.com/openai/human-eval) and [EvalPlus](https://github.com/evalplus/evalplus).

## References

- E. B. Wilson (1927). Probable Inference, the Law of Succession, and Statistical Inference. Journal of the
  American Statistical Association 22(158), 209-212. doi:10.1080/01621459.1927.10502953

## Licence

MIT, see `LICENSE`.

Built by Vipul Raj Jha · https://vipul21435.github.io
