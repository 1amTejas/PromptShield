"""Guard: put PromptShield in front of any LLM call."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from .classifier import AttackClassifier, AttackType
from .config import load_config
from .detector import Detection, Detector
from .errors import DetectorError
from .sanitizer import Sanitizer

_ERROR_POLICIES = {"raise", "block", "allow"}


@dataclass
class GuardResult:
    """What happened to one prompt.

    ``response`` is what your application should show: the LLM's answer, or the
    blocked message. It is ``None`` when the guard has no LLM attached.
    """

    response: Optional[str]
    blocked: bool
    unsafe: bool  # detector verdict on the ORIGINAL prompt
    score: float
    label: str
    reason: str  # safe | blocked | sanitized | sanitization_failed | detector_error_*
    attack_type: Optional[AttackType] = None
    sanitized_prompt: Optional[str] = None

    @property
    def safe(self) -> bool:
        return not self.unsafe

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["attack_type"] = self.attack_type.value if self.attack_type else None
        data["safe"] = self.safe
        return data


class Guard:
    """Detect, optionally classify and sanitize, then forward to your LLM.

    Args:
        llm: What to call with safe prompts. Any object with ``complete(prompt)``
            or ``forward(prompt)``, or a plain callable ``fn(prompt) -> str``.
            ``None`` gives an analysis-only guard (no response is produced).
        detector: A :class:`Detector` (default: built from config).
        classifier: An :class:`AttackClassifier` (built lazily if needed).
        sanitizer: A :class:`Sanitizer` (built lazily if needed).
        classify: Label flagged prompts with an attack type by default.
        sanitize: Try to repair flagged prompts instead of blocking them.
        blocked_message: Text returned for blocked prompts.
        on_detector_error: ``"raise"`` (default), ``"block"`` (fail closed) or
            ``"allow"`` (fail open) when the detector itself fails.
    """

    def __init__(
        self,
        llm: Any = None,
        *,
        detector: Optional[Detector] = None,
        classifier: Optional[AttackClassifier] = None,
        sanitizer: Optional[Sanitizer] = None,
        classify: bool = False,
        sanitize: bool = False,
        blocked_message: Optional[str] = None,
        on_detector_error: str = "raise",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if on_detector_error not in _ERROR_POLICIES:
            raise ValueError(f"on_detector_error must be one of {sorted(_ERROR_POLICIES)}")
        cfg = config if config is not None else load_config()
        self.llm = llm
        self.detector = detector if detector is not None else Detector(config=cfg)
        self._classifier = classifier
        self._sanitizer = sanitizer
        self._config = cfg
        self.classify = classify
        self.sanitize = sanitize
        self.blocked_message = blocked_message or cfg.guard.blocked_message
        self.on_detector_error = on_detector_error

    # Built lazily so plain detection never needs an LLM API key.
    @property
    def classifier(self) -> AttackClassifier:
        if self._classifier is None:
            self._classifier = AttackClassifier()
        return self._classifier

    @property
    def sanitizer(self) -> Sanitizer:
        if self._sanitizer is None:
            self._sanitizer = Sanitizer(self.detector, config=self._config)
        return self._sanitizer

    # ------------------------------------------------------------------ API
    def scan(self, prompt: str) -> Detection:
        """Run only the detector. No LLM is called."""
        return self.detector.predict_one(prompt)

    def protect(
        self,
        prompt: str,
        *,
        classify: Optional[bool] = None,
        sanitize: Optional[bool] = None,
    ) -> GuardResult:
        """Check ``prompt`` and, if it is safe (or repaired), call the LLM."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        classify = self.classify if classify is None else classify
        sanitize = self.sanitize if sanitize is None else sanitize

        detection, error_reason = self._detect(prompt)

        if not detection.unsafe:
            return GuardResult(
                response=self._ask_llm(prompt),
                blocked=False,
                unsafe=False,
                score=detection.score,
                label=detection.label,
                reason=error_reason or "safe",
            )

        attack_type = self.classifier.classify(prompt) if classify and not error_reason else None

        reason = error_reason or "blocked"
        if sanitize and not error_reason:
            outcome = self.sanitizer.sanitize(prompt)
            if outcome.clean:
                return GuardResult(
                    response=self._ask_llm(outcome.text),
                    blocked=False,
                    unsafe=True,
                    score=detection.score,
                    label=detection.label,
                    reason="sanitized",
                    attack_type=attack_type,
                    sanitized_prompt=outcome.text,
                )
            reason = "sanitization_failed"

        return GuardResult(
            response=self.blocked_message if self.llm is not None else None,
            blocked=True,
            unsafe=True,
            score=detection.score,
            label=detection.label,
            reason=reason,
            attack_type=attack_type,
        )

    __call__ = protect

    # ------------------------------------------------------------- internals
    def _detect(self, prompt: str) -> Tuple[Detection, Optional[str]]:
        try:
            return self.detector.predict_one(prompt), None
        except DetectorError:
            if self.on_detector_error == "raise":
                raise
            if self.on_detector_error == "allow":
                return Detection(unsafe=False, score=0.0, label="error"), "detector_error_allowed"
            return Detection(unsafe=True, score=1.0, label="error"), "detector_error_blocked"

    def _ask_llm(self, prompt: Optional[str]) -> Optional[str]:
        llm = self.llm
        if llm is None:
            return None
        for name in ("complete", "forward"):
            method = getattr(llm, name, None)
            if callable(method):
                return method(prompt)
        if callable(llm):
            return llm(prompt)
        raise TypeError("llm must be callable or provide complete(prompt) / forward(prompt)")
