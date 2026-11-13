import random

import pytest
from scipy.stats import binomtest

from evaltable.model import EvalTable, Row, Status
from evaltable.stats import wilson_interval
from evaltable.summary import collapse, overlap_matrix, summarize


def hand_table() -> EvalTable:
    return EvalTable(
        [
            Row("q1", "a", 1),
            Row("q2", "a", 0),
            Row("q3", "a", None, Status.ERROR),
            Row("q4", "a", 1),
            Row("q1", "b", 1, sample_index=0),
            Row("q1", "b", 0, sample_index=1),
            Row("q2", "b", None, Status.TIMEOUT),
            Row("q3", "b", 1),
        ]
    )


def test_wilson_closed_form_zero_successes():
    # With p = 0 the upper bound is z^2 / (n + z^2).
    z2 = 1.959963984540054**2
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(z2 / (10 + z2), abs=1e-12)


@pytest.mark.parametrize("k,n", [(0, 1), (1, 1), (3, 10), (50, 100), (299, 300), (7, 2294)])
@pytest.mark.parametrize("confidence", [0.9, 0.95, 0.99])
def test_wilson_matches_scipy(k, n, confidence):
    ref = binomtest(k, n).proportion_ci(confidence_level=confidence, method="wilson")
    lo, hi = wilson_interval(k, n, confidence)
    assert lo == pytest.approx(ref.low, abs=1e-10)
    assert hi == pytest.approx(ref.high, abs=1e-10)


def test_wilson_symmetry_and_edges():
    lo, hi = wilson_interval(3, 20)
    lo2, hi2 = wilson_interval(17, 20)
    assert lo == pytest.approx(1 - hi2) and hi == pytest.approx(1 - lo2)
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError):
        wilson_interval(5, 4)
    with pytest.raises(ValueError):
        wilson_interval(1, 4, confidence=1.0)


def test_collapse_samples():
    items = collapse(hand_table())
    q1 = items[("b", "q1")]
    assert (q1.n_rows, q1.n_ok, q1.strict, q1.lenient, q1.status) == (2, 2, 0.5, 0.5, Status.OK)
    q2 = items[("b", "q2")]
    assert q2.status is Status.TIMEOUT and q2.lenient is None and q2.strict == 0.0


def test_summary_hand_worked():
    by_run = {s.run_id: s for s in summarize(hand_table())}
    a, b = by_run["a"], by_run["b"]
    assert a.n_reference == 4
    assert a.strict == pytest.approx(0.5)
    assert a.lenient == pytest.approx(2 / 3)
    assert a.gap == pytest.approx(2 / 3 - 0.5)
    assert (a.coverage, a.error_rate, a.n_error) == (0.75, 0.25, 1)
    assert b.strict == pytest.approx(1.5 / 4)
    assert b.lenient == pytest.approx(0.75)
    assert (b.n_present, b.n_missing, b.n_timeout, b.n_ok) == (3, 1, 1, 2)
    assert b.coverage == 0.5 and b.error_rate == pytest.approx(1 / 3)
    lo, hi = a.strict_ci
    assert (lo, hi) == pytest.approx(wilson_interval(2, 4))
    assert b.lenient_ci == pytest.approx(wilson_interval(1.5, 2))


def test_summary_reference_items_and_unbounded_scores():
    table = EvalTable([Row("q1", "a", 30.0), Row("q2", "a", 10.0)])
    (s,) = summarize(table, items=["q1", "q2", "q3"])
    assert s.strict == pytest.approx(40 / 3)
    assert s.lenient == pytest.approx(20)
    assert s.strict_ci is None and s.lenient_ci is None
    assert s.as_dict()["strict_ci"] is None


def test_overlap_matrix():
    m = overlap_matrix(hand_table().with_missing())
    assert m == {"a": {"a": 4, "b": 3}, "b": {"a": 3, "b": 3}}


@pytest.mark.parametrize("seed", range(20))
def test_conventions_agree_without_errors(seed):
    rng = random.Random(seed)
    items = [f"i{k}" for k in range(rng.randint(1, 30))]
    rows = []
    for run in ("r1", "r2", "r3"):
        for item in items:
            for sample in range(rng.randint(1, 4)):
                score = rng.choice([0.0, 1.0, rng.random()])
                rows.append(Row(item, run, score, sample_index=sample))
    for s in summarize(EvalTable(rows)):
        assert s.strict == pytest.approx(s.lenient)
        assert s.gap == pytest.approx(0.0)
        assert s.coverage == 1.0 and s.error_rate == 0.0
