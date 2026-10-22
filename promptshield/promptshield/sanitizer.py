"""Iterative sanitizer: rewrite a flagged prompt until the detector accepts it."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import load_config
from .detector import Detector
from .errors import LLMError
from .llm import make_llm
from .prompts import NO_SAFE_CONTENT, SANITIZER_SYSTEM, wrap_untrusted


@dataclass
class SanitizeResult:
    """Outcome of a sanitisation attempt."""

    text: Optional[str]  # final safe text, or None if it could not be made safe
    clean: bool
    rounds: int
    history: List[str] = field(default_factory=list)  # every rewrite, in order


class Sanitizer:
    """Rewrite a prompt with a helper LLM, re-checking each rewrite.

    Each round asks the helper to strip the malicious parts and then runs the
    detector on the result. The loop ends when a rewrite passes, when the
    helper says nothing legitimate is left, when the helper fails, or after
    ``max_rounds``. In every failure case ``clean`` is ``False`` (fail closed).
    """

    def __init__(
        self,
        detector: Detector,
        llm: Any = None,
        *,
        provider: Optional[str] = None,
        max_rounds: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config if config is not None else load_config()
        self.detector = detector
        self._llm = llm
        self._provider = provider
        self.max_rounds = int(max_rounds or cfg.guard.max_sanitize_rounds)
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = make_llm(self._provider)
        return self._llm

    def sanitize(self, text: str) -> SanitizeResult:
        llm = self.llm  # built outside the loop so a missing API key is not hidden
        history: List[str] = []
        current = text
        for round_number in range(1, self.max_rounds + 1):
            try:
                rewritten = llm.complete(wrap_untrusted(current), system=SANITIZER_SYSTEM)
            except LLMError:
                return SanitizeResult(None, False, round_number, history)
            rewritten = rewritten.strip()
            if not rewritten or NO_SAFE_CONTENT in rewritten:
                return SanitizeResult(None, False, round_number, history)
            history.append(rewritten)
            if not self.detector.predict_one(rewritten).unsafe:
                return SanitizeResult(rewritten, True, round_number, history)
            current = rewritten
        return SanitizeResult(None, False, self.max_rounds, history)
