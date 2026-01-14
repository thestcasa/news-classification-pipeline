from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

@dataclass(frozen=True)
class Paths:
    dev_csv: str
    eval_csv: str
    cv_out_dir: str
    model_out: str
    submission_out: str

@dataclass(frozen=True)
class CV:
    k: int
    seed: int

@dataclass(frozen=True)
class Text:
    max_features: int
    ngram_min: int
    ngram_max: int
    min_df: int
    title_repeat: int

@dataclass(frozen=True)
class Model:
    type: str          # "logreg" | "linearsvc"
    C: float
    max_iter: int
    class_weight: str | None  # "balanced" or None

@dataclass(frozen=True)
class Config:
    paths: Paths
    cv: CV
    text: Text
    model: Model

def _deep_update(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """
    --set section.key=value
    Example: --set model.type=logreg --set text.max_features=200000
    """
    out: dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Bad override '{p}' (expected section.key=value)")
        lhs, rhs = p.split("=", 1)
        if "." not in lhs:
            raise ValueError(f"Bad override '{p}' (expected section.key=value)")
        section, key = lhs.split(".", 1)

        r = rhs.strip()
        if r.lower() in ("none", "null"):
            val: Any = None
        else:
            try:
                if r.isdigit() or (r.startswith("-") and r[1:].isdigit()):
                    val = int(r)
                else:
                    val = float(r)
            except ValueError:
                val = r

        out.setdefault(section, {})
        out[section][key] = val
    return out

def load_config(path: str | Path, *, overrides: dict[str, Any] | None = None) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if overrides:
        cfg = _deep_update(cfg, overrides)

    cw = cfg["model"].get("class_weight", "balanced")
    cw_norm = None if cw in (None, "none", "null") else str(cw)

    return Config(
        paths=Paths(**cfg["paths"]),
        cv=CV(**cfg["cv"]),
        text=Text(**cfg["text"]),
        model=Model(
            type=str(cfg["model"]["type"]),
            C=float(cfg["model"]["C"]),
            max_iter=int(cfg["model"].get("max_iter", 2000)),
            class_weight=cw_norm,
        )
    )
