import json

import pytest

from evaltable.model import EvalTable, Row, Status
from evaltable.readers import parse_column_map, parse_score, read, read_csv, read_jsonl
from evaltable.writers import table_to_string, write_table


def make_table() -> EvalTable:
    return EvalTable(
        [
            Row("q1", "a", 1.0, Status.OK, 0, {"subset": "x"}),
            Row("q2", "a", 0.0, Status.OK, 0, {"subset": "y", "tokens": 12}),
            Row("q3", "a", None, Status.ERROR, 0, {}),
            Row("q1", "b", 0.5, Status.OK, 1, {}),
            Row("q2", "b", None, Status.TIMEOUT, 0, {}),
        ]
    ).sorted()


def test_status_parse_synonyms():
    assert Status.parse("OK") is Status.OK
    assert Status.parse("timed-out") is Status.TIMEOUT
    assert Status.parse(Status.MISSING) is Status.MISSING
    with pytest.raises(ValueError):
        Status.parse("weird")


def test_row_validation():
    with pytest.raises(ValueError):
        Row("q1", "a", None, Status.OK)
    with pytest.raises(ValueError):
        Row("q1", "a", float("nan"))
    row = Row(7, "a", 1, "error")
    assert row.item_id == "7" and row.status is Status.ERROR and row.score == 1.0


def test_items_and_runs():
    table = make_table()
    assert table.runs() == ["a", "b"]
    assert table.items() == ["q1", "q2", "q3"]
    assert table.items("b") == ["q1", "q2"]


def test_with_missing_fills_union():
    table = make_table().with_missing()
    missing = [r for r in table if r.status is Status.MISSING]
    assert [(r.run_id, r.item_id) for r in missing] == [("b", "q3")]
    assert table.with_missing() == table


def test_parse_score():
    assert parse_score(True) == 1.0
    assert parse_score("PASS") == 1.0
    assert parse_score("unresolved") == 0.0
    assert parse_score("0.25") == 0.25
    assert parse_score("") is None
    assert parse_score(float("nan")) is None
    with pytest.raises(ValueError):
        parse_score("abc")


def test_parse_column_map():
    assert parse_column_map("item=instance_id, score=resolved") == {
        "item": "instance_id",
        "score": "resolved",
    }
    with pytest.raises(ValueError):
        parse_column_map("nope=x")
    with pytest.raises(ValueError):
        parse_column_map("item")


@pytest.mark.parametrize("suffix", [".csv", ".jsonl"])
def test_round_trip(tmp_path, suffix):
    table = make_table()
    path = tmp_path / f"long{suffix}"
    write_table(table, path)
    back = read(path).sorted()
    assert [r.key() for r in back] == [r.key() for r in table]
    assert [r.score for r in back] == [r.score for r in table]
    assert [r.status for r in back] == [r.status for r in table]
    if suffix == ".jsonl":
        assert back == table
    else:
        # CSV keeps metadata as text and drops empty cells.
        assert back.rows[1].metadata == {"subset": "y", "tokens": "12"}


def test_generic_csv_with_map(tmp_path):
    path = tmp_path / "model-x.csv"
    path.write_text("instance_id,resolved,notes\nA,true,\nB,false,slow\nC,,\n")
    table = read_csv(path, {"item": "instance_id", "score": "resolved"})
    assert table.runs() == ["model-x"]
    by_item = {r.item_id: r for r in table}
    assert by_item["A"].score == 1.0 and by_item["A"].status is Status.OK
    assert by_item["B"].metadata == {"notes": "slow"}
    # No score and no status column: counted as an error, not silently dropped.
    assert by_item["C"].status is Status.ERROR


def test_generic_jsonl_errors(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"item_id": "x", "score": 1}\nnot json\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        read_jsonl(path)
    path.write_text('{"score": 1}\n')
    with pytest.raises(ValueError, match="no item column"):
        read_jsonl(path)


def test_run_override_and_metadata_collision(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"qid": "1", "score": 1, "item_id": "dup"}) + "\n")
    table = read_jsonl(path, {"item": "qid"}, run_id="forced")
    assert table.rows[0].run_id == "forced"
    text = table_to_string(table, "csv")
    assert text.splitlines()[0] == "item_id,run_id,sample_index,score,status,meta_item_id"


def test_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        read(tmp_path / "x.unknown")
    with pytest.raises(ValueError):
        read(tmp_path / "x.csv", fmt="nope")


def test_parquet_round_trip(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "long.parquet"
    write_table(make_table(), path)
    data = pq.read_table(path).to_pylist()
    assert [d["item_id"] for d in data] == [r.item_id for r in make_table()]
    assert data[2]["status"] == "error" and data[2]["score"] is None
    assert data[1]["tokens"] == 12
    with pytest.raises(ValueError):
        write_table(make_table(), "-", "parquet")
