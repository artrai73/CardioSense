"""Configuration loading.

A single entry point, :func:`load_config`, reads a modality YAML from
``configs/``, merges it on top of the shared ``configs/paths.yaml``, and returns
a :class:`Config` object supporting both dictionary and attribute access::

    cfg = load_config("clinical")
    cfg.dataset.target_column          # 'num'
    cfg["split"]["test_size"]          # 0.15
    cfg.get("training.batch_size", 64) # dotted lookup with default

Command-line style overrides are supported so that notebooks can tweak one value
without editing YAML::

    cfg = load_config("ecg", overrides={"training.epochs": 2, "model.name": "resnet1d"})
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .paths import PATHS, resolve_path

__all__ = ["Config", "load_config", "save_config"]

_BASE_CONFIG = "paths.yaml"
_MODALITY_FILES = {
    "clinical": "clinical_config.yaml",
    "ecg": "ecg_config.yaml",
    "xray": "xray_config.yaml",
}


class Config(dict):
    """A ``dict`` whose nested mappings are also :class:`Config` objects.

    Attribute access is provided for readability; the object remains a plain
    ``dict`` so it serialises to JSON without a custom encoder.
    """

    def __init__(self, mapping: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        for key, value in (mapping or {}).items():
            self[key] = self._wrap(value)

    # -- construction ---------------------------------------------------------
    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, Mapping):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(item) for item in value]
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))

    # -- attribute access -----------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"Config has no key {name!r}. Available top-level keys: {sorted(self)}"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    # -- dotted helpers -------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        """Fetch ``a.b.c`` style keys, returning *default* when absent."""
        if "." not in key:
            return super().get(key, default)
        node: Any = self
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key: str, value: Any) -> None:
        """Set an ``a.b.c`` style key, creating intermediate maps as needed."""
        parts = key.split(".")
        node: Config = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Config):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = value

    def to_dict(self) -> dict:
        """Return a plain nested ``dict`` (useful for JSON dumps)."""
        out: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, Config):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, Config) else v for v in value]
            else:
                out[key] = value
        return out

    def resolved(self, key: str) -> Path:
        """Return the value at *key* resolved to an absolute path."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"No path configured at {key!r}")
        return resolve_path(value, base=PATHS.data if str(value).startswith("data") else PATHS.root)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config({json.dumps(self.to_dict(), indent=2, default=str)})"


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge *override* into *base* without mutating either."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Expected it under {PATHS.configs}. Did you clone the full repo?"
        )
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level.")
    return data


def load_config(
    name_or_path: str | Path,
    overrides: Mapping[str, Any] | None = None,
    include_base: bool = True,
) -> Config:
    """Load a CardioSense configuration.

    Args:
        name_or_path: ``"clinical"`` / ``"ecg"`` / ``"xray"``, a filename inside
            ``configs/``, or an explicit path to a YAML file.
        overrides: Optional dotted-key overrides applied last, e.g.
            ``{"training.epochs": 2}``.
        include_base: Merge ``configs/paths.yaml`` underneath. Turn off only when
            loading a standalone file.

    Returns:
        A fully merged :class:`Config`.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    key = str(name_or_path)
    if key in _MODALITY_FILES:
        config_path = PATHS.configs / _MODALITY_FILES[key]
    else:
        candidate = Path(key)
        config_path = candidate if candidate.is_absolute() else PATHS.configs / candidate
        if not config_path.exists() and candidate.exists():
            config_path = candidate.resolve()

    merged: dict = {}
    if include_base:
        merged = _read_yaml(PATHS.configs / _BASE_CONFIG)
    merged = _deep_merge(merged, _read_yaml(config_path))

    config = Config(merged)
    config.set("_config_path", str(config_path))

    for dotted_key, value in (overrides or {}).items():
        config.set(dotted_key, value)

    return config


def save_config(config: Config | Mapping[str, Any], path: Path | str) -> Path:
    """Write *config* to *path* as JSON (used for model metadata sidecars)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict() if isinstance(config, Config) else dict(config)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return target


def iter_modalities() -> Iterable[str]:
    """Yield the three Phase 1 modality names."""
    return tuple(_MODALITY_FILES)
