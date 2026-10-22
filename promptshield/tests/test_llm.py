from types import SimpleNamespace

import pytest

from promptshield.errors import LLMError
from promptshield.llm import AnthropicLLM, OpenAILLM, make_llm


def _openai_client(responder):
    completions = SimpleNamespace(create=responder)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _openai_reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_openai_builds_messages_with_system_prompt():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return _openai_reply("hello")

    llm = OpenAILLM(client=_openai_client(create), system="be nice", model="m")
    assert llm.complete("hi") == "hello"
    assert seen["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
    ]
    assert seen["model"] == "m"


def test_per_call_system_prompt_overrides_default():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return _openai_reply("x")

    llm = OpenAILLM(client=_openai_client(create), system="default")
    llm.complete("hi", system="override")
    assert seen["messages"][0]["content"] == "override"


def test_retries_then_succeeds():
    attempts = []

    def create(**kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("rate limited")
        return _openai_reply("finally")

    llm = OpenAILLM(client=_openai_client(create), max_retries=2)
    llm.retry_delay = 0
    assert llm.complete("hi") == "finally"
    assert len(attempts) == 3


def test_raises_llm_error_after_retries():
    def create(**kwargs):
        raise RuntimeError("always fails")

    llm = OpenAILLM(client=_openai_client(create), max_retries=1)
    llm.retry_delay = 0
    with pytest.raises(LLMError):
        llm.complete("hi")


def test_empty_response_is_an_error():
    llm = OpenAILLM(client=_openai_client(lambda **kw: _openai_reply("")), max_retries=0)
    with pytest.raises(LLMError):
        llm.complete("hi")


def test_anthropic_extracts_text_blocks_and_passes_system():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Hello "),
                SimpleNamespace(type="tool_use", text="ignored"),
                SimpleNamespace(type="text", text="world"),
            ]
        )

    llm = AnthropicLLM(client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert llm.complete("hi", system="sys") == "Hello world"
    assert seen["system"] == "sys"
    assert seen["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_omits_system_when_absent():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])

    AnthropicLLM(client=SimpleNamespace(messages=SimpleNamespace(create=create))).complete("hi")
    assert "system" not in seen


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("promptshield.llm._load_env", lambda: None)
    pytest.importorskip("openai")
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAILLM()


def test_make_llm_rejects_unknown_provider():
    with pytest.raises(ValueError):
        make_llm("mystery")
