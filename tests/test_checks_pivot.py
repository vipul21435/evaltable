import io
import random

import pytest

from evaltable.checks import CheckOptions, run_checks
from evaltable.model import EvalTable, Row, Status
from evaltable.pivot import pivot, read_wide_csv
from evaltable.summary import collapse


def clean_pair() -> EvalTable:
    rows = []
    for run in ("a", "b"):
        for k in range(5):
            rows.append(Row(f"q{k}", run, float(k % 2)))
    return EvalTable(rows)


def by_check(report):
    return {f.check: f for f in report.findings}


def test_clean_runs_are_comparable():
    report = run_checks(clean_pair())
    assert report.comparable
    assert by_check(report)["sample_counts"].message == "every item has 1 sample(s) in every run"


def test_item_sets_and_allow_missing():
    table = EvalTable([r for r in clean_pair() if not (r.run_id == "b" and r.item_id == "q4")])
    finding = by_check(run_checks(table))["item_sets"]
    assert not finding.passed
    assert finding.details["lacking"] == {"b": ["q4"]}
    assert run_checks(table, options=CheckOptions(allow_missing=True)).comparable


def test_missing_rows_count_as_absent():
    table = EvalTable(clean_pair().rows + [Row("q9", "a", None, Status.MISSING), Row("q9", "b", 1)])
    finding = by_check(run_checks(table))["item_sets"]
    assert finding.details["lacking"] == {"a": ["q9"]}


def test_reference_items():
    finding = by_check(run_checks(clean_pair(), items=["q0", "q7"]))["item_sets"]
    assert finding.details["lacking"] == {"a": ["q7"], "b": ["q7"]}


def test_duplicates():
    table = EvalTable(clean_pair().rows + [Row("q1", "a", 1.0)])
    finding = by_check(run_checks(table))["duplicates"]
    assert not finding.passed
    assert finding.details["rows"] == [
        {"run_id": "a", "item_id": "q1", "sample_index": 0, "count": 2}
    ]
    assert by_check(run_checks(table, options=CheckOptions(allow_duplicates=True)))[
        "duplicates"
    ].passed


def test_errors_gap_and_threshold():
    rows = [r for r in clean_pair() if not (r.run_id == "b" and r.item_id == "q0")]
    rows.append(Row("q0", "b", None, Status.ERROR))
    finding = by_check(run_checks(EvalTable(rows)))["errors"]
    info = finding.details["runs"]["b"]
    # b: q1, q3 pass, q2, q4 fail, q0 errored -> 2/5 as failures, 2/4 left out.
    assert info["errored"] == ["q0"]
    assert info["errors_as_failures"] == pytest.approx(0.4)
    assert info["errors_left_out"] == pytest.approx(0.5)
    assert not finding.passed
    loose = run_checks(EvalTable(rows), options=CheckOptions(max_gap=0.2))
    assert by_check(loose)["errors"].passed


def test_sample_counts():
    rows = clean_pair().rows + [Row("q0", "a", 1.0, sample_index=1)]
    finding = by_check(run_checks(EvalTable(rows)))["sample_counts"]
    assert not finding.passed
    assert finding.details["differing_items"] == {"q0": {"a": 2, "b": 1}}
    assert finding.details["mixed_runs"] == ["a"]
    assert finding.details["per_run"]["a"] == {1: 4, 2: 1}


def test_pivot_values():
    rows = clean_pair().rows + [Row("q5", "a", None, Status.TIMEOUT)]
    table = EvalTable(rows)
    wide = pivot(table)
    assert wide.runs == ["a", "b"] and wide.items[-1] == "q5"
    assert wide.get("q1", "a") == 1.0
    assert wide.get("q5", "a") is None and wide.get("q5", "b") is None
    status = pivot(table, "status")
    assert status.get("q5", "a") == "timeout" and status.get("q5", "b") == "missing"
    assert pivot(table, "strict").get("q5", "a") == 0.0
    with pytest.raises(ValueError):
        pivot(table, "nope")


def random_table(seed: int) -> EvalTable:
    rng = random.Random(seed)
    rows = []
    for run in [f"r{k}" for k in range(rng.randint(1, 4))]:
        for item in [f"i{k}" for k in range(rng.randint(1, 15))]:
            if rng.random() < 0.2:
                continue
            for sample in range(rng.randint(1, 3)):
                status = rng.choice([Status.OK] * 6 + list(Status))
                score = rng.choice([0.0, 1.0, round(rng.random(), 3)])
                ok_score = score if status is Status.OK else None
                rows.append(Row(item, run, ok_score, status, sample, {"tag": rng.randint(0, 2)}))
            if rng.random() < 0.1:
                rows.append(Row(item, run, 1.0, Status.OK, 0, {"tag": 9}))
    return EvalTable(rows)


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("value", ["score", "strict", "status"])
def test_long_and_wide_agree(seed, value):
    table = random_table(seed)
    wide = pivot(table, value)
    collapsed = collapse(table)
    for item in wide.items:
        for run in wide.runs:
            result = collapsed.get((run, item))
            cell = wide.get(item, run)
            if result is None:
                assert cell == ("missing" if value == "status" else None)
            elif value == "score":
                assert cell == result.lenient
            elif value == "strict":
                assert cell == result.strict
            else:
                assert cell == result.status.value
    buffer = io.StringIO()
    wide.write_csv(buffer)
    buffer.seek(0)
    back = read_wide_csv(buffer, value)
    assert back.items == wide.items and back.runs == wide.runs and back.cells == wide.cells


@pytest.mark.parametrize("seed", range(25))
def test_merge_does_not_depend_on_order(seed):
    table = random_table(seed)
    parts = [table.for_run(run) for run in table.runs()]
    rng = random.Random(seed)
    shuffled = parts[:]
    rng.shuffle(shuffled)
    for part in shuffled:
        rng.shuffle(part.rows)
    assert EvalTable.concat(parts) == EvalTable.concat(shuffled)
