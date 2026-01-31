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
    missing_article_token: str | None = "__MISSING_ARTICLE__"
    max_df: float = 0.95
    lowercase: bool = True
    strip_accents: str | None = "unicode"
    sublinear_tf: bool = True
    char_enabled: bool = True
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    char_min_df: int = 3
    char_max_features: int | None = None
    char_analyzer: str = "char_wb"
    title_char: bool = False
    title_char_ngram_min: int = 2
    title_char_ngram_max: int = 4
    title_char_min_df: int = 5
    title_char_max_features: int = 20000

@dataclass(frozen=True)
class Model:
    type: str          # "logreg" | "linearsvc"
    C: float
    max_iter: int
    class_weight: str | None  # "balanced" or None
    class_weight_power: float = 1.0
    logreg_solver: str = "liblinear"
    logreg_n_jobs: int = 1
    linearsvc_dual: bool = False
    ridge_alpha: float = 1.0

@dataclass(frozen=True)
class Source:
    min_count: int | None = 5
    min_frac: float | None = None

@dataclass(frozen=True)
class Dedup:
    drop_same_label: bool = False

@dataclass(frozen=True)
class Config:
    paths: Paths
    cv: CV
    text: Text
    model: Model
    source: Source
    dedup: Dedup

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
        r_low = r.lower()
        if r_low in ("none", "null"):
            val: Any = None
        elif r_low in ("true", "false"):
            val = r_low == "true"
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
            class_weight_power=float(cfg["model"].get("class_weight_power", 1.0)),
            logreg_solver=str(cfg["model"].get("logreg_solver", "liblinear")),
            logreg_n_jobs=int(cfg["model"].get("logreg_n_jobs", 1)),
            linearsvc_dual=bool(cfg["model"].get("linearsvc_dual", False)),
            ridge_alpha=float(cfg["model"].get("ridge_alpha", 1.0)),
        ),
        source=Source(**cfg.get("source", {})),
        dedup=Dedup(**cfg.get("dedup", {})),
    )
