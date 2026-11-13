import json
from pathlib import Path

import pytest

from evaltable.cli import main
from evaltable.readers import read

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "data"
A = str(EXAMPLES / "model_a.jsonl")
B = str(EXAMPLES / "model_b.jsonl")


def test_read_to_csv(capsys):
    assert main(["read", A]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "item_id,run_id,sample_index,score,status,subset"
    assert lines[5] == "item-05,model_a,0,,error,hard"


def test_read_with_run_and_output(tmp_path):
    out = tmp_path / "long.jsonl"
    assert main(["read", A, "--run", "renamed", "-o", str(out)]) == 0
    table = read(out)
    assert table.runs() == ["renamed"] and len(table) == 8


def test_merge_fill_missing(tmp_path):
    out = tmp_path / "merged.csv"
    assert main(["merge", B, A, "--fill-missing", "-o", str(out)]) == 0
    table = read(out)
    assert len(table) == 16
    missing = [(r.run_id, r.item_id) for r in table if r.status.value == "missing"]
    assert missing == [("model_b", "item-08")]


def test_merge_order_gives_same_bytes(tmp_path):
    one, two = tmp_path / "1.csv", tmp_path / "2.csv"
    assert main(["merge", A, B, "-o", str(one)]) == 0
    assert main(["merge", B, A, "-o", str(two)]) == 0
    assert one.read_bytes() == two.read_bytes()


def test_check_exit_codes(capsys):
    assert main(["check", A, B]) == 1
    out = capsys.readouterr().out
    assert "model_b lacks: item-08" in out and out.strip().endswith("not comparable")
    assert main(["check", A, B, "--allow-missing", "--max-gap", "0.2"]) == 0
    assert capsys.readouterr().out.strip().endswith("comparable")
    assert main(["check", A, A, "--run", "x", "--run", "y", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["comparable"] is False
    assert [f["check"] for f in report["findings"] if not f["passed"]] == ["errors"]


def test_check_reference_items(tmp_path, capsys):
    items = tmp_path / "items.txt"
    items.write_text("item-01\nitem-02\n\n")
    assert main(["check", B, "--items", str(items)]) == 0
    capsys.readouterr()


def test_pivot(tmp_path, capsys):
    assert main(["pivot", A, B]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "item_id,model_a,model_b"
    assert lines[5] == "item-05,,0"
    out = tmp_path / "wide.csv"
    assert main(["pivot", A, B, "--value", "status", "-o", str(out)]) == 0
    assert out.read_text().splitlines()[8] == "item-08,ok,missing"


def test_summary_text_and_json(capsys):
    assert main(["summary", A, B]) == 0
    out = capsys.readouterr().out
    assert "0.5000 [0.2152, 0.7848]" in out and "+0.1667" in out
    assert main(["summary", A, B, "--json", "--confidence", "0.9"]) == 0
    data = json.loads(capsys.readouterr().out)
    first = data["runs"][0]
    assert first["run_id"] == "model_a"
    assert first["strict"] == pytest.approx(0.5)
    assert first["lenient"] == pytest.approx(4 / 6)
    assert data["overlap"]["model_a"]["model_b"] == 7


def test_errors_exit_two(tmp_path, capsys):
    assert main(["read", str(tmp_path / "nope.jsonl")]) == 2
    assert "error" in capsys.readouterr().err
    assert main(["merge", A, B, "--run", "only-one"]) == 2
    assert main(["read", A, "--map", "bogus=x"]) == 2


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("evaltable ")


def test_items_file_read_once(tmp_path, monkeypatch, capsys):
    items = tmp_path / "items.txt"
    items.write_text("item-01\nitem-02\nitem-99\n")
    calls = []
    original = Path.read_text

    def counting(self, *args, **kwargs):
        if self == items:
            calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting)
    assert main(["summary", B, "--items", str(items), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["runs"][0]["n_reference"] == 3
    assert len(calls) == 1
