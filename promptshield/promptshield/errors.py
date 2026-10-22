"""Exception types used across PromptShield."""


class PromptShieldError(Exception):
    """Base class for all PromptShield errors."""


class DetectorError(PromptShieldError):
    """The detector could not produce a verdict (model missing, server down, ...)."""


class LLMError(PromptShieldError):
    """A call to an LLM provider failed after all retries."""
