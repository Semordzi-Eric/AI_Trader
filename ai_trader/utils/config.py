"""YAML configuration loader with simple `inherits` support and dotted access."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import copy
import yaml


class Config(dict):
    """Dict subclass with attribute-style access and dotted lookup."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Resolve a dotted path like 'rl.reward.pnl_weight'."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | Path) -> Config:
    """Load YAML config; if it has an `inherits` key, merge with the parent.

    Inheritance is one level deep, which is enough for our research vs. live split.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    parent_name = raw.pop("inherits", None)
    if parent_name:
        parent_path = path.parent / parent_name
        with parent_path.open("r", encoding="utf-8") as fh:
            parent_raw: dict = yaml.safe_load(fh) or {}
        raw = _deep_merge(parent_raw, raw)

    return Config(raw)
