"""Prompt-injection detector.

Wraps a Hugging Face sequence-classification model. It runs in-process by
default, or talks to a ``promptshield serve`` instance over HTTP when a ``url``
is configured.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import requests

from .config import load_config
from .errors import DetectorError


@dataclass(frozen=True)
class Detection:
    """Verdict for a single prompt."""

    unsafe: bool
    score: float  # probability the model puts on unsafe labels (0..1)
    label: str  # top label reported by the model


def is_unsafe_label(label: Any, unsafe_labels: Iterable[str]) -> bool:
    """Case-insensitive membership test against the unsafe label set."""
    wanted = {str(item).strip().lower() for item in unsafe_labels}
    return str(label).strip().lower() in wanted


class Detector:
    """Classify prompts as safe or unsafe.

    Args:
        model: Hugging Face model id or local path (default from config).
        url: Base URL of a ``promptshield serve`` instance. If set (and no
            ``pipeline`` is given) predictions are made remotely, and the
            server's own threshold applies. Pass ``""`` to force local mode.
        threshold: Flag a prompt when the unsafe probability reaches this value.
        max_length: Tokens kept per prompt (longer prompts are truncated).
        unsafe_labels: Labels that count as unsafe.
        device: ``"cpu"``, ``"cuda"``, ... (default: cuda if available).
        timeout: Seconds to wait for the remote server.
        pipeline: A callable with the Hugging Face text-classification
            pipeline interface. Mainly for tests and custom backends.
        config: A config object from :func:`promptshield.config.load_config`.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        url: Optional[str] = None,
        threshold: Optional[float] = None,
        max_length: Optional[int] = None,
        unsafe_labels: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        timeout: Optional[float] = None,
        pipeline: Optional[Callable[..., Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = (config if config is not None else load_config()).detector
        self.model = model or cfg.model
        self.url = url if url is not None else cfg.url
        self.threshold = float(cfg.threshold if threshold is None else threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.max_length = int(max_length or cfg.max_length)
        self.unsafe_labels = list(unsafe_labels or cfg.unsafe_labels)
        self.timeout = float(timeout or cfg.timeout)
        self.device = device
        self._pipeline = pipeline
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ API
    def predict(self, texts: Union[str, Sequence[str]]) -> List[Detection]:
        """Return one :class:`Detection` per input text, in order."""
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("texts must be strings")
        if not texts:
            return []
        if self.url and self._pipeline is None:
            return self._predict_remote(texts)
        return self._predict_local(texts)

    def predict_one(self, text: str) -> Detection:
        """Convenience wrapper for a single prompt."""
        return self.predict(text)[0]

    def load(self) -> None:
        """Load the model now instead of on first use (no-op in remote mode)."""
        if self.url and self._pipeline is None:
            return
        self._get_pipeline()

    # ---------------------------------------------------------------- local
    def _get_pipeline(self) -> Callable[..., Any]:
        with self._load_lock:
            if self._pipeline is None:
                try:
                    import torch
                    from transformers import pipeline
                except ImportError as exc:
                    raise DetectorError(
                        "Local detection needs the 'local' extra: "
                        "pip install 'promptshield[local]'"
                    ) from exc
                device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                try:
                    self._pipeline = pipeline(
                        "text-classification",
                        model=self.model,
                        truncation=True,
                        max_length=self.max_length,
                        device=torch.device(device),
                    )
                except Exception as exc:  # bad model id, no network, ...
                    raise DetectorError(
                        f"Could not load detector model '{self.model}': {exc}"
                    ) from exc
            return self._pipeline

    def _predict_local(self, texts: List[str]) -> List[Detection]:
        pipe = self._get_pipeline()
        try:
            raw = pipe(texts, top_k=None)  # scores for every label
        except Exception as exc:
            raise DetectorError(f"Detector inference failed: {exc}") from exc
        if raw and isinstance(raw[0], dict):  # single-input output shape
            raw = [raw]
        if len(raw) != len(texts):
            raise DetectorError(
                f"Detector returned {len(raw)} results for {len(texts)} inputs"
            )
        return [self._to_detection(scores) for scores in raw]

    def _to_detection(self, scores: Sequence[Dict[str, Any]]) -> Detection:
        top = max(scores, key=lambda item: item["score"])
        unsafe_score = sum(
            item["score"]
            for item in scores
            if is_unsafe_label(item["label"], self.unsafe_labels)
        )
        unsafe_score = min(1.0, max(0.0, float(unsafe_score)))
        return Detection(
            unsafe=unsafe_score >= self.threshold,
            score=unsafe_score,
            label=str(top["label"]),
        )

    # --------------------------------------------------------------- remote
    def _predict_remote(self, texts: List[str]) -> List[Detection]:
        endpoint = str(self.url).rstrip("/") + "/detect"
        try:
            response = requests.post(endpoint, json={"texts": texts}, timeout=self.timeout)
            response.raise_for_status()
            detections = [
                Detection(
                    unsafe=bool(item["unsafe"]),
                    score=float(item["score"]),
                    label=str(item["label"]),
                )
                for item in response.json()["results"]
            ]
        except Exception as exc:
            raise DetectorError(f"Detection server at {endpoint} failed: {exc}") from exc
        if len(detections) != len(texts):
            raise DetectorError(
                f"Detection server returned {len(detections)} results for {len(texts)} inputs"
            )
        return detections
