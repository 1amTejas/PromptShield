"""Threat taxonomy: label a flagged prompt with the kind of attack it is."""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from .errors import LLMError
from .llm import make_llm
from .prompts import CLASSIFIER_SYSTEM, wrap_untrusted


class AttackType(str, Enum):
    """The five attack surfaces PromptShield distinguishes."""

    CONTEXT_MANIPULATION = "context_manipulation"
    IGNORE_PROMPT_TEMPLATE = "ignore_prompt_template"
    OTHER_LANGUAGES = "other_languages"
    FAKE_COMPLETION = "fake_completion"
    SOCIAL_ENGINEERING = "social_engineering"
    UNKNOWN = "unknown"

    @property
    def title(self) -> str:
        return self.value.replace("_", " ").capitalize()


def parse_attack_type(text: Any) -> AttackType:
    """Map free-form model output to an :class:`AttackType`.

    If several category names appear, the earliest one wins. Anything that does
    not match returns ``AttackType.UNKNOWN`` rather than raising.
    """
    lowered = re.sub(r"[_\-\s]+", " ", str(text).lower())
    best = AttackType.UNKNOWN
    best_position: Optional[int] = None
    for attack in AttackType:
        if attack is AttackType.UNKNOWN:
            continue
        position = lowered.find(attack.value.replace("_", " "))
        if position != -1 and (best_position is None or position < best_position):
            best, best_position = attack, position
    return best


class AttackClassifier:
    """Ask a helper LLM which attack category a flagged prompt belongs to.

    The prompt is fenced as untrusted data, and the reply is validated against
    the known categories, so a prompt cannot dictate the classifier's output.
    """

    def __init__(self, llm: Any = None, *, provider: Optional[str] = None) -> None:
        self._llm = llm
        self._provider = provider

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = make_llm(self._provider)
        return self._llm

    def classify(self, text: str) -> AttackType:
        llm = self.llm  # built outside the try so a missing API key is not hidden
        try:
            reply = llm.complete(wrap_untrusted(text), system=CLASSIFIER_SYSTEM)
        except LLMError:
            return AttackType.UNKNOWN
        return parse_attack_type(reply)
