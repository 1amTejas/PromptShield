"""Configuration loading.

The packaged ``configs/default.yaml`` provides defaults. Override them with a
custom YAML file (``PROMPTSHIELD_CONFIG`` or the ``path`` argument) and/or the
environment variables listed in ``ENV_OVERRIDES``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default.yaml"

# environment variable -> key path inside the config
ENV_OVERRIDES: Dict[str, Tuple[str, ...]] = {
    "PROMPTSHIELD_DETECTOR_MODEL": ("detector", "model"),
    "PROMPTSHIELD_DETECTOR_URL": ("detector", "url"),
    "PROMPTSHIELD_HELPER_PROVIDER": ("helper", "provider"),
    "PROMPTSHIELD_OPENAI_MODEL": ("llm", "openai", "model"),
    "PROMPTSHIELD_ANTHROPIC_MODEL": ("llm", "anthropic", "model"),
}


class DotDict(dict):
    """A dict with attribute access. Nested dicts are converted recursively."""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = DotDict(value) if isinstance(value, dict) else value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _set_path(data: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    node = data
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[Union[str, Path]] = None) -> DotDict:
    """Load defaults, then the custom file (if any), then environment overrides."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    custom = path or os.getenv("PROMPTSHIELD_CONFIG")
    if custom:
        with open(custom, "r", encoding="utf-8") as handle:
            _deep_merge(data, yaml.safe_load(handle) or {})

    for env_name, key_path in ENV_OVERRIDES.items():
        value = os.getenv(env_name)
        if value:
            _set_path(data, key_path, value)

    return DotDict(data)
