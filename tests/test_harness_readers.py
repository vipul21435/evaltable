"""Each reader on a fixture written from the format's documented structure, synthetic values."""

import json
import zipfile

import pytest

from evaltable.model import Status
from evaltable.readers import (
    ReadOptions,
    detect_format,
    read,
    read_evalplus,
    read_humaneval,
    read_inspect,
    read_lm_eval,
    read_openai_evals,
    read_swebench,
)
from evaltable.readers.inspectlog import value_to_float


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def by_item(table):
    return {(r.item_id, r.sample_index): r for r in table}


# SWE-bench ---------------------------------------------------------------------------------

OLD_RESULTS = {
    "no_generation": ["p__q-1"],
    "generated": ["p__q-2", "p__q-3", "p__q-4", "p__q-5", "p__q-6", "p__q-7", "p__q-8"],
    "with_logs": ["p__q-2", "p__q-3", "p__q-4", "p__q-5", "p__q-6", "p__q-7"],
    "install_fail": ["p__q-3"],
    "reset_failed": [],
    "no_apply": ["p__q-4"],
    "applied": ["p__q-2", "p__q-5", "p__q-6", "p__q-7"],
    "test_errored": ["p__q-5"],
    "test_timeout": ["p__q-6"],
    "resolved": ["p__q-2"],
}


def test_swebench_old_results(tmp_path):
    path = write_json(
        tmp_path / "verified" / "20240101_demo" / "results" / "results.json", OLD_RESULTS
    )
    assert detect_format(path) == "swebench"
    table = read(path)
    assert table.runs() == ["20240101_demo"]
    got = {r.item_id: (r.status, r.score, r.metadata["swebench_outcome"]) for r in table}
    assert got == {
        "p__q-1": (Status.MISSING, None, "no_generation"),
        "p__q-2": (Status.OK, 1.0, "resolved"),
        "p__q-3": (Status.ERROR, None, "install_fail"),
        "p__q-4": (Status.OK, 0.0, "no_apply"),
        "p__q-5": (Status.ERROR, None, "test_errored"),
        "p__q-6": (Status.TIMEOUT, None, "test_timeout"),
        "p__q-7": (Status.OK, 0.0, "unresolved"),
        "p__q-8": (Status.ERROR, None, "no_logs"),
    }


def test_swebench_new_results_need_items(tmp_path):
    data = {"no_generation": ["a-1"], "no_logs": ["a-2"], "resolved": ["a-3"]}
    path = write_json(tmp_path / "results.json", data)
    with pytest.raises(ValueError, match="--items"):
        read(path)
    table = read(path, options=ReadOptions(items=("a-1", "a-2", "a-3", "a-4")))
    got = {r.item_id: (r.status, r.score) for r in table}
    assert got == {
        "a-1": (Status.MISSING, None),
        "a-2": (Status.ERROR, None),
        "a-3": (Status.OK, 1.0),
        "a-4": (Status.OK, 0.0),
    }
    assert table.runs() == ["results"]


def test_swebench_report(tmp_path):
    report = {
        "total_instances": 6,
        "submitted_instances": 5,
        "completed_instances": 3,
        "resolved_instances": 1,
        "unresolved_instances": 2,
        "empty_patch_instances": 1,
        "error_instances": 1,
        "completed_ids": ["x-1", "x-2", "x-3"],
        "incomplete_ids": ["x-6"],
        "empty_patch_ids": ["x-5"],
        "submitted_ids": ["x-1", "x-2", "x-3", "x-4", "x-5"],
        "resolved_ids": ["x-1"],
        "unresolved_ids": ["x-2", "x-3"],
        "infra_failure_ids": ["x-3"],
        "ambiguous_failure_ids": [],
        "error_ids": ["x-4"],
        "schema_version": 2,
    }
    path = write_json(tmp_path / "demo-model.run1.json", report)
    table = read(path)
    got = {r.item_id: (r.status, r.score) for r in table}
    assert got == {
        "x-1": (Status.OK, 1.0),
        "x-2": (Status.OK, 0.0),
        "x-3": (Status.OK, 0.0),
        "x-4": (Status.ERROR, None),
        "x-5": (Status.MISSING, None),
        "x-6": (Status.MISSING, None),
    }
    assert by_item(table)[("x-3", 0)].metadata["infra_failure"] is True
    assert table.runs() == ["demo-model.run1"]


def test_swebench_rejects_other_json(tmp_path):
    with pytest.raises(ValueError):
        read_swebench(write_json(tmp_path / "x.json", {"a": 1}))
    with pytest.raises(ValueError):
        read_swebench(write_json(tmp_path / "y.json", {"resolved": "nope", "generated": []}))


# lm-evaluation-harness ---------------------------------------------------------------------


def lm_record(doc_id, flt, acc, **extra):
    record = {
        "doc_id": doc_id,
        "doc": {"question": f"q{doc_id}"},
        "target": "A",
        "arguments": [["ctx", " A"]],
        "resps": [[["-1.0", "False"]]],
        "filtered_resps": [["-1.0", "False"]],
        "filter": flt,
        "metrics": ["acc", "acc_norm"],
        "doc_hash": f"h{doc_id}",
        "prompt_hash": "p",
        "target_hash": "t",
        "acc": acc,
        "acc_norm": 1.0 - acc,
    }
    record.update(extra)
    return record


def test_lm_eval_samples(tmp_path):
    path = tmp_path / "org__model-7b" / "samples_arc_easy_2024-05-13T15-30-00.123456.jsonl"
    write_jsonl(path, [lm_record(0, "none", 1.0), lm_record(1, "none", 0.0)])
    assert detect_format(path) == "lm-eval"
    table = read(path)
    assert table.runs() == ["org__model-7b"]
    assert [(r.item_id, r.score) for r in table] == [("arc_easy/0", 1.0), ("arc_easy/1", 0.0)]
    assert table.rows[0].metadata == {
        "task": "arc_easy",
        "filter": "none",
        "metric": "acc",
        "doc_hash": "h0",
    }
    norm = read(path, options=ReadOptions(metric="acc_norm"))
    assert [r.score for r in norm] == [0.0, 1.0]


def test_lm_eval_filters_and_bad_metric(tmp_path):
    records = [lm_record(0, "strict", 1.0), lm_record(0, "flexible", 0.0)]
    path = write_jsonl(tmp_path / "m" / "plain.jsonl", records)
    both = read_lm_eval(path)
    assert both.runs() == ["m/flexible", "m/strict"]
    assert both.items() == ["0"]
    one = read_lm_eval(path, filter_name="strict")
    assert one.runs() == ["m"] and one.rows[0].score == 1.0
    with pytest.raises(ValueError, match="filter"):
        read_lm_eval(path, filter_name="other")
    path2 = write_jsonl(tmp_path / "m" / "b.jsonl", [lm_record(0, "none", 1.0, bleu=["a", "b"])])
    with pytest.raises(ValueError, match="not a number"):
        read_lm_eval(path2, metric="bleu")
    with pytest.raises(ValueError, match="no metric"):
        read_lm_eval(path2, metric="nope")
    with pytest.raises(ValueError, match="not an lm-evaluation-harness"):
        read_lm_eval(write_jsonl(tmp_path / "c.jsonl", [{"doc_id": 1}]))


# Inspect AI --------------------------------------------------------------------------------

INSPECT_HEADER = {
    "version": 2,
    "status": "success",
    "eval": {
        "run_id": "run-a",
        "task": "demo_task",
        "model": "provider/model-x",
        "created": "2026-01-01T00:00:00+00:00",
        "dataset": {"name": "demo", "samples": 4, "sample_ids": [1, 2, 3, 4]},
        "config": {"epochs": 2},
    },
}
INSPECT_SAMPLES = [
    {"id": 1, "epoch": 1, "scores": {"match": {"value": "C"}, "other": {"value": 0.25}}},
    {"id": 1, "epoch": 2, "scores": {"match": {"value": "I"}}},
    {"id": 2, "epoch": 1, "scores": {"match": {"value": "P"}}},
    {"id": 2, "epoch": 2, "error": {"message": "boom", "traceback": "..."}},
    {"id": 3, "epoch": 1, "limit": {"type": "time", "limit": 60}},
    {"id": 3, "epoch": 2, "scores": {}},
]


def check_inspect_table(table):
    assert table.runs() == ["provider/model-x"]
    got = {k: (r.status, r.score) for k, r in by_item(table).items()}
    assert got == {
        ("1", 0): (Status.OK, 1.0),
        ("1", 1): (Status.OK, 0.0),
        ("2", 0): (Status.OK, 0.5),
        ("2", 1): (Status.ERROR, None),
        ("3", 0): (Status.TIMEOUT, None),
        ("3", 1): (Status.MISSING, None),
        ("4", 0): (Status.MISSING, None),
        ("4", 1): (Status.MISSING, None),
    }
    assert by_item(table)[("2", 1)].metadata["error"] == "boom"


def test_inspect_json(tmp_path):
    path = write_json(tmp_path / "log.json", {**INSPECT_HEADER, "samples": INSPECT_SAMPLES})
    assert detect_format(path) == "inspect"
    table = read(path)
    check_inspect_table(table)
    other = read_inspect(path, scorer="other")
    assert by_item(other)[("1", 0)].score == 0.25
    assert by_item(other)[("1", 1)].status is Status.ERROR
    assert by_item(other)[("1", 1)].metadata["error"] == "no score from other"
    with pytest.raises(ValueError, match="from scorer"):
        read_inspect(path, scorer="missing-scorer")


def test_inspect_eval_zip(tmp_path):
    path = tmp_path / "log.eval"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("header.json", json.dumps(INSPECT_HEADER))
        archive.writestr("_journal/start.json", "{}")
        for sample in INSPECT_SAMPLES:
            name = f"samples/{sample['id']}_epoch_{sample['epoch']}.json"
            archive.writestr(name, json.dumps(sample))
    assert detect_format(path) == "inspect"
    check_inspect_table(read(path))


def test_inspect_rejects_other_json(tmp_path):
    with pytest.raises(ValueError):
        read_inspect(write_json(tmp_path / "x.json", {"eval": {"task": "t"}}))


@pytest.mark.parametrize(
    "value,expected",
    [
        ("C", 1.0),
        ("P", 0.5),
        ("I", 0.0),
        ("N", 0.0),
        (True, 1.0),
        (3, 3.0),
        ("yes", 1.0),
        ("False", 0.0),
        ("0.75", 0.75),
        ("abc", 0.0),
        ([1, 2], 0.0),
        (float("nan"), 0.0),
    ],
)
def test_inspect_value_to_float(value, expected):
    assert value_to_float(value) == expected


# OpenAI Evals ------------------------------------------------------------------------------


def oai_event(event_id, sample_id, kind, data):
    return {
        "run_id": "r1",
        "event_id": event_id,
        "sample_id": sample_id,
        "type": kind,
        "data": data,
        "created_by": "",
        "created_at": "2026-01-01 00:00:00",
    }


def test_openai_evals(tmp_path):
    records = [
        {"spec": {"completion_fns": ["model-x"], "eval_name": "demo.dev.v0", "run_id": "r1"}},
        oai_event(0, "demo.dev.0", "sampling", {"prompt": "p", "sampled": ["a"]}),
        oai_event(1, "demo.dev.0", "match", {"correct": True, "expected": "a", "picked": "a"}),
        oai_event(2, "demo.dev.1", "match", {"correct": False, "expected": "a", "picked": "b"}),
        oai_event(3, "demo.dev.1", "match", {"correct": True, "expected": "a", "picked": "a"}),
        oai_event(4, "demo.dev.2", "error", {"type": "ValueError", "message": "bad"}),
        oai_event(5, "demo.dev.3", "sampling", {"prompt": "p", "sampled": ["c"]}),
        oai_event(6, "demo.dev.4", "metrics", {"score": 0.5}),
        {"final_report": {"accuracy": 0.5}, "run_id": "r1"},
    ]
    path = write_jsonl(tmp_path / "record.jsonl", records)
    assert detect_format(path) == "openai-evals"
    table = read(path)
    assert table.runs() == ["model-x"]
    got = {k: (r.status, r.score) for k, r in by_item(table).items()}
    assert got == {
        ("demo.dev.0", 0): (Status.OK, 1.0),
        ("demo.dev.1", 0): (Status.OK, 0.0),
        ("demo.dev.1", 1): (Status.OK, 1.0),
        ("demo.dev.2", 0): (Status.ERROR, None),
        ("demo.dev.3", 0): (Status.ERROR, None),
        ("demo.dev.4", 0): (Status.ERROR, None),
    }
    assert by_item(table)[("demo.dev.2", 0)].metadata["error"] == "bad"
    metric = read_openai_evals(path, metric="score")
    assert by_item(metric)[("demo.dev.4", 0)].score == 0.5
    with pytest.raises(ValueError):
        read_openai_evals(write_jsonl(tmp_path / "empty.jsonl", [{"other": 1}]))


# HumanEval and EvalPlus --------------------------------------------------------------------


def test_humaneval_results(tmp_path):
    records = [
        {"task_id": "HumanEval/0", "completion": "x", "result": "passed", "passed": True},
        {"task_id": "HumanEval/0", "completion": "y", "result": "failed: ", "passed": False},
        {"task_id": "HumanEval/1", "completion": "z", "result": "timed out", "passed": False},
        {
            "task_id": "HumanEval/2",
            "completion": "w",
            "result": "failed: name 'f' is not defined",
            "passed": False,
        },
    ]
    path = write_jsonl(tmp_path / "samples.jsonl_results.jsonl", records)
    assert detect_format(path) == "humaneval"
    table = read(path)
    assert table.runs() == ["samples"]
    got = {k: (r.status, r.score) for k, r in by_item(table).items()}
    assert got == {
        ("HumanEval/0", 0): (Status.OK, 1.0),
        ("HumanEval/0", 1): (Status.OK, 0.0),
        ("HumanEval/1", 0): (Status.TIMEOUT, None),
        ("HumanEval/2", 0): (Status.OK, 0.0),
    }
    with pytest.raises(ValueError):
        read_humaneval(write_jsonl(tmp_path / "bad.jsonl", [{"task_id": "x"}]))


def evalplus_result(task, base, plus):
    return {
        "task_id": task,
        "solution": "def f(): pass",
        "base_status": base,
        "plus_status": plus,
        "base_fail_tests": [],
        "plus_fail_tests": [],
    }


def test_evalplus_results(tmp_path):
    data = {
        "date": "2026-01-01 00:00",
        "hash": "abc",
        "eval": {
            "HumanEval/0": [
                evalplus_result("HumanEval/0", "pass", "pass"),
                evalplus_result("HumanEval/0", "pass", "fail"),
            ],
            "HumanEval/1": [evalplus_result("HumanEval/1", "timeout", "timeout")],
            "HumanEval/2": [evalplus_result("HumanEval/2", "pass", "timeout")],
            "HumanEval/3": [evalplus_result("HumanEval/3", None, None)],
        },
    }
    path = write_json(tmp_path / "model_x_eval_results.json", data)
    assert detect_format(path) == "evalplus"
    plus = read(path)
    assert plus.runs() == ["model_x"]
    got = {k: (r.status, r.score) for k, r in by_item(plus).items()}
    assert got == {
        ("HumanEval/0", 0): (Status.OK, 1.0),
        ("HumanEval/0", 1): (Status.OK, 0.0),
        ("HumanEval/1", 0): (Status.TIMEOUT, None),
        ("HumanEval/2", 0): (Status.TIMEOUT, None),
        ("HumanEval/3", 0): (Status.ERROR, None),
    }
    base = read_evalplus(path, tests="base")
    assert by_item(base)[("HumanEval/0", 1)].score == 1.0
    assert by_item(base)[("HumanEval/2", 0)].score == 1.0
    with pytest.raises(ValueError):
        read_evalplus(path, tests="nope")
    with pytest.raises(ValueError):
        read_evalplus(write_json(tmp_path / "x.json", {"eval": {"t": [{"x": 1}]}}))


def test_detect_unknown_json_and_gzip(tmp_path):
    with pytest.raises(ValueError, match="--format"):
        detect_format(write_json(tmp_path / "odd.json", {"a": 1}))
    import gzip

    path = tmp_path / "samples.jsonl_results.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"task_id": "T/0", "result": "passed", "passed": True}) + "\n")
    assert detect_format(path) == "humaneval"
    assert read(path).rows[0].score == 1.0
    plain = write_jsonl(tmp_path / "plain.jsonl", [{"item_id": "a", "score": 1}])
    assert detect_format(plain) == "jsonl"


def test_swebench_old_results_unlisted_reference_ids_are_missing(tmp_path):
    path = write_json(tmp_path / "e" / "results" / "results.json", OLD_RESULTS)
    items = tuple(OLD_RESULTS["generated"] + OLD_RESULTS["no_generation"] + ["p__q-9"])
    table = read(path, options=ReadOptions(items=items))
    row = by_item(table)[("p__q-9", 0)]
    assert (row.status, row.metadata["swebench_outcome"]) == (Status.MISSING, "not_listed")
    # The listed ids read the same with or without --items.
    without = {r.item_id: (r.status, r.score) for r in read(path)}
    with_items = {r.item_id: (r.status, r.score) for r in table if r.item_id != "p__q-9"}
    assert without == with_items


def test_evalplus_base_only_file_needs_tests_base(tmp_path):
    data = {
        "date": "2026-01-01 00:00",
        "hash": "abc",
        "eval": {
            "Mbpp/1": [evalplus_result("Mbpp/1", "pass", None)],
            "Mbpp/2": [evalplus_result("Mbpp/2", "fail", None)],
        },
    }
    path = write_json(tmp_path / "base_only_eval_results.json", data)
    with pytest.raises(ValueError, match="--tests base"):
        read(path)
    base = read(path, options=ReadOptions(tests="base"))
    assert [(r.status, r.score) for r in base] == [(Status.OK, 1.0), (Status.OK, 0.0)]


def test_openai_evals_without_the_chosen_score_is_refused(tmp_path):
    records = [
        {"spec": {"completion_fns": ["m"], "eval_name": "graded.dev.v0", "run_id": "r1"}},
        oai_event(0, "s.0", "sampling", {"prompt": "p", "sampled": ["a"]}),
        oai_event(1, "s.0", "metrics", {"choice": "Y", "score": 1.0}),
        oai_event(2, "s.1", "metrics", {"choice": "N", "score": 0.0}),
    ]
    path = write_jsonl(tmp_path / "graded.jsonl", records)
    with pytest.raises(ValueError, match="metrics keys seen: choice, score"):
        read(path)
    with pytest.raises(ValueError, match="'scroe'"):
        read_openai_evals(path, metric="scroe")
    assert [r.score for r in read_openai_evals(path, metric="score")] == [1.0, 0.0]
    # A file in which every sample errored is still read, as errors.
    errored = write_jsonl(
        tmp_path / "errored.jsonl", records[:1] + [oai_event(0, "s.0", "error", {})]
    )
    assert [r.status for r in read(errored)] == [Status.ERROR]
