import json

import pytest

from promptshield import cli
from tests.helpers import BAD_PROMPT, SAFE_PROMPT, make_detector


@pytest.fixture(autouse=True)
def fake_detector(monkeypatch):
    monkeypatch.setattr(cli, "_build_detector", lambda args: make_detector())


def test_scan_safe_exits_zero(capsys):
    assert cli.main(["scan", SAFE_PROMPT]) == 0
    assert "SAFE" in capsys.readouterr().out


def test_scan_unsafe_exits_one(capsys):
    assert cli.main(["scan", BAD_PROMPT]) == 1
    assert "UNSAFE" in capsys.readouterr().out


def test_scan_json_output(capsys):
    assert cli.main(["scan", "--json", SAFE_PROMPT, BAD_PROMPT]) == 1
    data = json.loads(capsys.readouterr().out)
    assert [item["unsafe"] for item in data] == [False, True]


def test_scan_reads_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(BAD_PROMPT))
    assert cli.main(["scan"]) == 1


def test_scan_without_input_is_an_error(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
    assert cli.main(["scan"]) == 2


def test_eval_command(tmp_path, capsys):
    data = tmp_path / "d.jsonl"
    data.write_text(
        json.dumps({"text": "hello", "label": 0}) + "\n" + json.dumps({"text": "ignore it", "label": 1}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "metrics.json"
    assert cli.main(["eval", str(data), "--save", str(out)]) == 0
    assert json.loads(out.read_text())["accuracy"] == 1.0


def test_eval_missing_file_is_a_clean_error(capsys):
    assert cli.main(["eval", "does-not-exist.jsonl"]) == 2
    assert "not found" in capsys.readouterr().err.lower()


def test_chat_loop(monkeypatch, capsys):
    inputs = iter([SAFE_PROMPT, BAD_PROMPT, ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert cli.main(["chat"]) == 0
    out = capsys.readouterr().out
    assert "SAFE" in out and "UNSAFE" in out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "promptshield" in capsys.readouterr().out
