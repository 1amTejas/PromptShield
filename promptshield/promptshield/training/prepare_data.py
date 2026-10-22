"""Build the train/test datasets used by ``promptshield.training.train``.

* Train: the public dataset's train split, plus (optionally) generated
  jailbreaks from ``promptshield.datagen``.
* Test: the public dataset's own test split only. Generated examples are derived
  from training seeds, so they never enter the test set.
* Exact duplicates are removed, and any training text that also appears in the
  test set is dropped to avoid leakage.

Usage::

    python -m promptshield.training.prepare_data --generated data/malicious_data.json --out data
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..datagen import SEED_DATASET, load_examples, normalise
from ..evaluate import to_label

Row = Tuple[str, int]


def assemble(
    train_rows: Sequence[Row],
    test_rows: Sequence[Row],
    generated_texts: Sequence[str] = (),
    seed: int = 42,
) -> Tuple[List[Row], List[Row]]:
    """Merge, de-duplicate and shuffle. Returns ``(train, test)``."""
    test: List[Row] = []
    test_keys = set()
    for text, label in test_rows:
        key = normalise(text)
        if key not in test_keys:
            test_keys.add(key)
            test.append((text, label))

    train: List[Row] = []
    train_keys = set()
    candidates = list(train_rows) + [(text, 1) for text in generated_texts]
    for text, label in candidates:
        key = normalise(text)
        if key in train_keys or key in test_keys:
            continue
        train_keys.add(key)
        train.append((text, label))

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _describe(name: str, rows: Sequence[Row]) -> str:
    positives = sum(label for _, label in rows)
    share = positives / len(rows) if rows else 0.0
    return f"{name}: {len(rows)} rows, {positives} unsafe ({share:.0%})"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare train/test datasets for detector training.")
    parser.add_argument("--generated", help="path to malicious_data.json from promptshield.datagen")
    parser.add_argument("--dataset", default=SEED_DATASET, help="Hugging Face dataset with prompt/type columns")
    parser.add_argument("--out", default="data", help="directory for train_data/ and test_data/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from datasets import Dataset, load_dataset

    raw = load_dataset(args.dataset)

    def rows(split: str) -> List[Row]:
        return [(item["prompt"], to_label(item["type"])) for item in raw[split]]

    generated = (
        [example.example_jailbreak_input for example in load_examples(args.generated)]
        if args.generated
        else []
    )
    train, test = assemble(rows("train"), rows("test"), generated, args.seed)

    print(_describe("train", train))
    print(_describe("test", test))
    unsafe_share = sum(label for _, label in train) / max(1, len(train))
    if unsafe_share > 0.75 or unsafe_share < 0.25:
        print(
            "warning: the training set is heavily imbalanced. Add more benign prompts, "
            "or the detector may over-flag harmless input (check the false-positive rate)."
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("train_data", train), ("test_data", test)):
        Dataset.from_dict(
            {"text": [text for text, _ in data], "label": [label for _, label in data]}
        ).save_to_disk(str(out / name))
    print(f"saved to {out}/train_data and {out}/test_data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
