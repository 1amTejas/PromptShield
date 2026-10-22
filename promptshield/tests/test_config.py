import pytest

from promptshield.config import DotDict, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.detector.model
    assert cfg.detector.threshold == 0.5
    assert cfg.guard.max_sanitize_rounds >= 1
    assert cfg.llm.anthropic.model


def test_dotdict_attribute_access_and_missing_key():
    data = DotDict({"a": {"b": 1}})
    assert data.a.b == 1
    data.c = 2
    assert data["c"] == 2
    with pytest.raises(AttributeError):
        _ = data.missing


def test_env_override(monkeypatch):
    monkeypatch.setenv("PROMPTSHIELD_DETECTOR_MODEL", "my/model")
    monkeypatch.setenv("PROMPTSHIELD_ANTHROPIC_MODEL", "some-claude")
    cfg = load_config()
    assert cfg.detector.model == "my/model"
    assert cfg.llm.anthropic.model == "some-claude"


def test_custom_file_is_deep_merged(tmp_path, monkeypatch):
    custom = tmp_path / "custom.yaml"
    custom.write_text("detector:\n  threshold: 0.9\n", encoding="utf-8")
    monkeypatch.setenv("PROMPTSHIELD_CONFIG", str(custom))
    cfg = load_config()
    assert cfg.detector.threshold == 0.9
    assert cfg.detector.max_length == 512  # untouched default survives the merge
