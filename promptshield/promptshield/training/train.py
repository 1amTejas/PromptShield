"""Fine-tune a sequence classifier as the PromptShield detector.

Usage::

    python -m promptshield.training.prepare_data --out data
    python -m promptshield.training.train --data-dir data --output-dir outputs/detector
    promptshield eval data/test_data --model outputs/detector/final

The saved model uses the labels ``safe`` / ``jailbreak``; ``jailbreak`` is in the
default unsafe-label set, so it works with ``Detector`` out of the box.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..evaluate import compute_metrics

ID2LABEL = {0: "safe", 1: "jailbreak"}
LABEL2ID = {"safe": 0, "jailbreak": 1}


def metrics_from_predictions(eval_pred: Any) -> Dict[str, float]:
    """``Trainer`` metric callback: accuracy, precision, recall, F1."""
    import numpy as np

    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.asarray(predictions)
    if predictions.ndim > 1:  # raw logits
        predictions = predictions.argmax(axis=-1)
    result = compute_metrics([int(x) for x in labels], [int(x) for x in predictions])
    return {key: result[key] for key in ("accuracy", "precision", "recall", "f1")}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a prompt-injection detector.")
    parser.add_argument("--model", default="microsoft/deberta-v3-small", help="base model to fine-tune")
    parser.add_argument("--data-dir", default="data", help="directory with train_data/ and test_data/")
    parser.add_argument("--output-dir", default="outputs/detector")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="mixed precision (CUDA only)")
    args = parser.parse_args(argv)

    import torch
    from datasets import load_from_disk
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    data_dir = Path(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_data = load_from_disk(str(data_dir / "train_data")).map(tokenize, batched=True)
    test_data = load_from_disk(str(data_dir / "test_data")).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    def keep_predictions_only(logits: Any, labels: Any) -> Any:
        """Store class ids, not full logits, to keep evaluation memory low."""
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.argmax(logits, dim=-1)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=args.seed,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        report_to="none",
        push_to_hub=False,
    )

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics_from_predictions,
        preprocess_logits_for_metrics=keep_predictions_only,
    )
    # Newer transformers renamed Trainer(tokenizer=...) to Trainer(processing_class=...).
    tokenizer_arg = (
        "processing_class"
        if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else "tokenizer"
    )
    trainer_kwargs[tokenizer_arg] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    metrics = trainer.evaluate()
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    with open(Path(args.output_dir) / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"model saved to {final_dir}")
    print(f"use it with: PROMPTSHIELD_DETECTOR_MODEL={final_dir} promptshield scan 'some text'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
