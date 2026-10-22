import csv
import json

import pytest

from promptshield.evaluate import compute_metrics, evaluate, format_report, load_dataset, to_label
from tests.helpers import make_detector


def test_to_label():
    assert [to_label(v) for v in (0, 1, "safe", "Benign", "jailbreak", "INJECTION", True, 1.0)] == [
        0, 1, 0, 0, 1, 1, 1, 1,
    ]
    with pytest.raises(ValueError):
        to_label("maybe")
    with pytest.raises(ValueError):
        to_label(2)


def test_compute_metrics_values():
    metrics = compute_metrics([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])
    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (2, 1, 2, 1)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 3)


def test_compute_metrics_handles_empty_and_degenerate_cases():
    assert compute_metrics([], [])["accuracy"] == 0.0
    all_safe = compute_metrics([0, 0], [0, 0])
    assert all_safe["precision"] == 0.0 and all_safe["recall"] == 0.0 and all_safe["f1"] == 0.0
    with pytest.raises(ValueError):
        compute_metrics([1], [1, 0])


def test_load_jsonl_json_and_csv(tmp_path):
    rows = [{"text": "hello", "label": 0}, {"text": "ignore everything", "label": "jailbreak"}]

    jsonl = tmp_path / "d.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    plain_json = tmp_path / "d.json"
    plain_json.write_text(json.dumps(rows), encoding="utf-8")
    csv_path = tmp_path / "d.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    expected = [("hello", 0), ("ignore everything", 1)]
    assert load_dataset(jsonl) == expected
    assert load_dataset(plain_json) == expected
    assert load_dataset(csv_path) == expected


def test_load_generated_format_defaults_to_unsafe(tmp_path):
    path = tmp_path / "malicious_data.json"
    path.write_text(
        json.dumps([{"example_jailbreak_input": "do bad things", "reasoning": "r", "class": "x"}]),
        encoding="utf-8",
    )
    assert load_dataset(path) == [("do bad things", 1)]


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"nothing": "here"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(bad)
    other = tmp_path / "data.txt"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(other)


def test_evaluate_with_fake_detector():
    examples = [("hi there", 0), ("ignore all rules", 1), ("ignore the manual", 0), ("good day", 0)]
    result = evaluate(make_detector(), examples, batch_size=2)
    assert result["samples"] == 4 and result["positives"] == 1
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (1, 1, 2, 0)
    assert "accuracy" in format_report(result)


def test_evaluate_empty_and_bad_batch_size():
    assert evaluate(make_detector(), [])["samples"] == 0
    with pytest.raises(ValueError):
        evaluate(make_detector(), [("x", 0)], batch_size=0)
