"""Thin wrappers around the OpenAI and Anthropic SDKs.

Every wrapper exposes ``complete(prompt, system=None) -> str`` and raises
:class:`~promptshield.errors.LLMError` once retries are exhausted. (The original
prototype swallowed errors and returned ``None``, which let failed
sanitisation silently pass through.)
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from .config import load_config
from .errors import LLMError


def _load_env() -> None:
    """Load a local ``.env`` file if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


class BaseLLM:
    """Retry loop shared by the provider wrappers."""

    def __init__(
        self,
        *,
        system: Optional[str] = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self.system = system
        self.max_retries = max(0, int(max_retries))
        self.retry_delay = retry_delay

    def _request(self, prompt: str, system: Optional[str]) -> str:
        raise NotImplementedError

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Send ``prompt`` and return the model's text reply."""
        system = system if system is not None else self.system
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._request(prompt, system)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
        raise LLMError(
            f"{type(self).__name__} failed after {self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    __call__ = complete


class OpenAILLM(BaseLLM):
    """Chat-completions wrapper. Needs ``OPENAI_API_KEY``."""

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        api_key: Optional[str] = None,
        client: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = (config if config is not None else load_config()).llm.openai
        super().__init__(
            system=system,
            max_retries=cfg.max_retries if max_retries is None else max_retries,
        )
        self.model = model or cfg.model
        self.temperature = cfg.temperature if temperature is None else temperature
        self.max_tokens = int(max_tokens or cfg.max_tokens)
        if client is None:
            try:
                import openai
            except ImportError as exc:
                raise LLMError(
                    "The OpenAI SDK is not installed: pip install 'promptshield[openai]'"
                ) from exc
            _load_env()
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise LLMError(
                    "OPENAI_API_KEY is not set (use your environment or a .env file)."
                )
            client = openai.OpenAI(api_key=key)
        self._client = client

    def _request(self, prompt: str, system: Optional[str]) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        completion = self._client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content
        if not content:
            raise LLMError("OpenAI returned an empty response")
        return content


class AnthropicLLM(BaseLLM):
    """Messages-API wrapper. Needs ``ANTHROPIC_API_KEY``."""

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        api_key: Optional[str] = None,
        client: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = (config if config is not None else load_config()).llm.anthropic
        super().__init__(
            system=system,
            max_retries=cfg.max_retries if max_retries is None else max_retries,
        )
        self.model = model or cfg.model
        self.temperature = cfg.temperature if temperature is None else temperature
        self.max_tokens = int(max_tokens or cfg.max_tokens)
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMError(
                    "The Anthropic SDK is not installed: pip install 'promptshield[anthropic]'"
                ) from exc
            _load_env()
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError(
                    "ANTHROPIC_API_KEY is not set (use your environment or a .env file)."
                )
            client = anthropic.Anthropic(api_key=key)
        self._client = client

    def _request(self, prompt: str, system: Optional[str]) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        message = self._client.messages.create(**kwargs)
        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        if not text:
            raise LLMError("Anthropic returned an empty response")
        return text


def make_llm(provider: Optional[str] = None, **kwargs: Any) -> BaseLLM:
    """Build a wrapper for ``provider`` ("openai" or "anthropic").

    Without a provider, the ``helper.provider`` config value is used.
    """
    name = (provider or load_config().helper.provider).lower()
    if name == "openai":
        return OpenAILLM(**kwargs)
    if name == "anthropic":
        return AnthropicLLM(**kwargs)
    raise ValueError(f"Unknown LLM provider '{name}' (expected 'openai' or 'anthropic')")
