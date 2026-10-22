"""PromptShield: a lightweight prompt-injection defense layer for LLM apps."""
from ._version import __version__
from .classifier import AttackClassifier, AttackType
from .detector import Detection, Detector
from .errors import DetectorError, LLMError, PromptShieldError
from .guard import Guard, GuardResult
from .sanitizer import Sanitizer, SanitizeResult

__author__ = "1amTejas"

__all__ = [
    "AttackClassifier",
    "AttackType",
    "Detection",
    "Detector",
    "DetectorError",
    "Guard",
    "GuardResult",
    "LLMError",
    "PromptShieldError",
    "Sanitizer",
    "SanitizeResult",
    "__version__",
]
