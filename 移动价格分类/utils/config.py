from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "defaults" in cfg:
        base_path = path.parent / "default.yaml"
        base = load_config(base_path)
        cfg = _deep_merge(base, cfg)
        cfg.pop("defaults", None)

    if overrides:
        for kv in overrides:
            key, value = kv.split("=", 1)
            _set_nested(cfg, key, yaml.safe_load(value))
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_nested(cfg: dict, key: str, value: Any) -> None:
    keys = key.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value