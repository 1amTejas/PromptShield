"""Evaluate a detector on a labelled dataset (``promptshield eval``).

Supported inputs:
  * ``.jsonl``: one JSON object per line
  * ``.json``: a list of objects (this includes the generated
    ``malicious_data.json``, whose rows are all treated as unsafe)
  * ``.csv``: with a header row
  * a directory saved with ``datasets.Dataset.save_to_disk`` (what
    ``promptshield.training.prepare_data`` writes)

Each row needs the text under ``text`` (or ``prompt`` / ``example_jailbreak_input``)
and a ``label`` (0/1, or safe/benign vs. jailbreak/injection/malicious).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

from .detector import Detector

_NEGATIVE = {"0", "safe", "benign", "clean", "normal", "false", "no"}
_POSITIVE = {"1", "unsafe", "jailbreak", "injection", "malicious", "attack", "true", "yes"}
_TEXT_KEYS = ("text", "prompt", "example_jailbreak_input")

Example = Tuple[str, int]


def to_label(value: Any) -> int:
    """Normalise a dataset label to 0 (safe) or 1 (unsafe)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if int(value) in (0, 1) and float(value) == int(value):
            return int(value)
        raise ValueError(f"Unrecognised label: {value!r}")
    key = str(value).strip().lower()
    if key in _NEGATIVE:
        return 0
    if key in _POSITIVE:
        return 1
    raise ValueError(f"Unrecognised label: {value!r}")


def _row_to_example(row: Dict[str, Any], index: int) -> Example:
    text = next((row[key] for key in _TEXT_KEYS if row.get(key) is not None), None)
    if not isinstance(text, str):
        raise ValueError(f"Row {index} has no text field (looked for {', '.join(_TEXT_KEYS)})")
    if "label" in row:
        label = to_label(row["label"])
    elif "example_jailbreak_input" in row:
        label = 1  # generated data contains only malicious examples
    else:
        raise ValueError(f"Row {index} has no 'label' field")
    return text, label


def load_dataset(path: Union[str, Path]) -> List[Example]:
    """Load ``(text, label)`` pairs from a file or a saved Hugging Face dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.is_dir():
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError(
                "Reading a saved dataset needs: pip install 'promptshield[train]'"
            ) from exc
        rows: List[Dict[str, Any]] = list(load_from_disk(str(path)))
    elif path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
    elif path.suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix or path}")

    return [_row_to_example(row, index) for index, row in enumerate(rows)]


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    """Confusion matrix plus accuracy, precision, recall, F1 and false-positive rate."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": ratio(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall),
        "false_positive_rate": ratio(fp, fp + tn),
    }


def evaluate(
    detector: Detector, examples: Sequence[Example], batch_size: int = 32
) -> Dict[str, Any]:
    """Run ``detector`` over ``examples`` and return metrics plus timing."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    texts = [text for text, _ in examples]
    labels = [label for _, label in examples]
    predictions: List[int] = []
    start = time.perf_counter()
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        predictions.extend(int(item.unsafe) for item in detector.predict(batch))
    elapsed = time.perf_counter() - start

    result = compute_metrics(labels, predictions)
    result.update(
        samples=len(texts),
        positives=sum(labels),
        seconds=round(elapsed, 3),
        ms_per_sample=round(1000 * elapsed / len(texts), 3) if texts else 0.0,
    )
    return result


def format_report(result: Dict[str, Any]) -> str:
    """Human-readable summary of an :func:`evaluate` result."""
    return "\n".join(
        [
            f"samples: {result['samples']} ({result['positives']} unsafe)",
            f"accuracy:  {result['accuracy']:.4f}",
            f"precision: {result['precision']:.4f}",
            f"recall:    {result['recall']:.4f}",
            f"f1:        {result['f1']:.4f}",
            f"false-positive rate: {result['false_positive_rate']:.4f}",
            f"confusion: tp={result['tp']} fp={result['fp']} tn={result['tn']} fn={result['fn']}",
            f"speed: {result['ms_per_sample']} ms/sample ({result['seconds']} s total)",
        ]
    )
