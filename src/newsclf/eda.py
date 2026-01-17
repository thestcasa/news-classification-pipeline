#!/usr/bin/env python3
"""
Standalone EDA for the DS/ML Lab News Classification project.

Fixed input files (repo-relative):
  - data/raw/development.csv
  - data/raw/evaluation.csv

Fixed schemas (no column robustness):
  development.csv columns:
    ["Id","source","title","article","page_rank","timestamp","label"]
  evaluation.csv columns:
    ["Id","source","title","article","page_rank","timestamp"]

Usage:
  python src/eda.py
  python src/eda.py --plots
  python src/eda.py --output-dir reports/eda --plots
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# Fixed schema & fixed file paths
# -----------------------------
DEV_COLS = ["Id", "source", "title", "article", "page_rank", "timestamp", "label"]
EVAL_COLS = ["Id", "source", "title", "article", "page_rank", "timestamp"]

ID_COL = "Id"

DEV_PATH = Path("data/raw/development.csv")
EVAL_PATH = Path("data/raw/evaluation.csv")


# -----------------------------
# Formatting helpers
# -----------------------------
SEP = "=" * 96
SUBSEP = "-" * 96


def section(title: str) -> None:
    print("\n" + SEP)
    print(title)
    print(SEP)


def subsection(title: str) -> None:
    print("\n" + title)
    print(SUBSEP)


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


def fmt_int(x: int) -> str:
    return f"{x:,}"


def shorten(s: str, max_len: int = 160) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def stable_hash_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).apply(
        lambda x: hashlib.md5(x.encode("utf-8", errors="ignore")).hexdigest()
    )


def try_parse_datetime(ts: pd.Series) -> pd.Series:
    placeholder = "0000-00-00 00:00:00"
    ts_clean = ts.astype(str)
    ts_clean = ts_clean.mask(ts_clean.eq(placeholder), other=np.nan)
    return pd.to_datetime(ts_clean, errors="coerce")


@dataclass
class DatasetProfile:
    name: str
    path: Path
    n_rows: int
    n_cols: int
    columns: List[str]


# -----------------------------
# Schema checks
# -----------------------------
def enforce_schema(df: pd.DataFrame, expected_cols: List[str], name: str) -> None:
    cols = list(df.columns)
    if cols != expected_cols:
        missing = [c for c in expected_cols if c not in cols]
        extra = [c for c in cols if c not in expected_cols]
        msg = [f"[SCHEMA ERROR] {name} columns do not match expected schema."]
        msg.append(f"  Current:  {cols}")
        msg.append(f"  Expected: {expected_cols}")
        if missing:
            msg.append(f"  Missing:  {missing}")
        if extra:
            msg.append(f"  Extra:    {extra}")
        raise ValueError("\n".join(msg))


def infer_profile(name: str, path: Path, df: pd.DataFrame) -> DatasetProfile:
    return DatasetProfile(
        name=name,
        path=path,
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=list(df.columns),
    )


# -----------------------------
# Core EDA routines
# -----------------------------
def print_basic_info(df: pd.DataFrame, prof: DatasetProfile) -> None:
    subsection("Basic structure")
    print(f"File: {prof.path}")
    print(f"Shape: {fmt_int(prof.n_rows)} rows × {fmt_int(prof.n_cols)} columns")
    print("Columns / dtypes:")
    dtypes = df.dtypes.astype(str)
    for c in df.columns:
        print(f"  - {c:12s} {dtypes[c]}")

    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"Approx. memory usage (deep): {mem_mb:.2f} MB")

    sample = df.head(min(50000, len(df)))
    h = hashlib.md5(pd.util.hash_pandas_object(sample, index=True).values.tobytes()).hexdigest()
    print(f"Fingerprint (md5 over head rows): {h}")


def print_missingness(df: pd.DataFrame) -> None:
    subsection("Missingness & empty strings")
    na = df.isna().sum()
    na_pct = (na / len(df)).replace([np.inf, np.nan], 0.0)

    empty_counts = {}
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            s = df[c].astype("string")
            empty = s.fillna("").str.strip().eq("")
            empty_counts[c] = int(empty.sum())

    rows = []
    for c in df.columns:
        rows.append(
            {
                "col": c,
                "na": int(na[c]),
                "na_%": float(na_pct[c]),
                "empty_str": int(empty_counts.get(c, 0)),
                "empty_%": (empty_counts.get(c, 0) / len(df)) if len(df) else 0.0,
            }
        )
    out = pd.DataFrame(rows).sort_values(["na", "empty_str"], ascending=False)
    print(out.to_string(index=False, formatters={"na_%": fmt_pct, "empty_%": fmt_pct}))


def print_id_checks(df: pd.DataFrame) -> None:
    subsection("ID sanity checks")
    ids = df[ID_COL]
    print(f"ID column: {ID_COL} (dtype={ids.dtype})")
    print(f"Missing IDs: {fmt_int(int(ids.isna().sum()))} ({fmt_pct(float(ids.isna().mean()))})")

    dup = ids.duplicated().sum()
    print(f"Duplicate IDs: {fmt_int(int(dup))} ({fmt_pct(float(dup / len(df)))})")

    if pd.api.types.is_numeric_dtype(ids):
        print(f"ID min/max: {int(np.nanmin(ids))} / {int(np.nanmax(ids))}")
    else:
        nun = ids.nunique(dropna=True)
        print(f"Unique IDs: {fmt_int(int(nun))}")


def print_duplicates(df: pd.DataFrame) -> None:
    subsection("Row & content duplicates")
    dup_rows = df.duplicated().sum()
    print(f"Fully duplicated rows: {fmt_int(int(dup_rows))} ({fmt_pct(float(dup_rows / len(df)))})")

    dup_content = df.duplicated(subset=["title", "article"]).sum()
    print(f"Duplicate pairs on ['title','article']: {fmt_int(int(dup_content))} ({fmt_pct(float(dup_content / len(df)))})")

    dup_title = df.duplicated(subset=["title"]).sum()
    print(f"Duplicate titles: {fmt_int(int(dup_title))} ({fmt_pct(float(dup_title / len(df)))})")


def print_categorical_summary(df: pd.DataFrame, col: str, topn: int = 10) -> None:
    nun = df[col].nunique(dropna=True)
    print(f"\n[{col}] unique values: {fmt_int(int(nun))}")
    vc = df[col].value_counts(dropna=False).head(topn)
    out = pd.DataFrame({"value": vc.index.astype(str), "count": vc.values})
    out["%"] = out["count"] / len(df)
    print(out.to_string(index=False, formatters={"%": fmt_pct}))


def print_numeric_summary(df: pd.DataFrame, col: str) -> None:
    s = pd.to_numeric(df[col], errors="coerce")
    desc = s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(f"\n[{col}] numeric summary (coerced)")
    print(desc.to_string())


def print_timestamp_summary(df: pd.DataFrame) -> None:
    subsection("Timestamp quality & distribution")
    ts_raw = df["timestamp"]

    placeholder = "0000-00-00 00:00:00"
    is_placeholder = ts_raw.astype(str).eq(placeholder)
    print(f"Placeholder '{placeholder}': {fmt_int(int(is_placeholder.sum()))} ({fmt_pct(float(is_placeholder.mean()))})")

    ts = try_parse_datetime(ts_raw)
    invalid = ts.isna()
    print(f"Unparseable/NaT timestamps: {fmt_int(int(invalid.sum()))} ({fmt_pct(float(invalid.mean()))})")

    if (~invalid).any():
        tmin, tmax = ts.min(), ts.max()
        print(f"Valid timestamp range: {tmin}  →  {tmax}")

        years = ts.dropna().dt.year.value_counts().sort_index()
        print("\nCounts by year (valid only):")
        out = pd.DataFrame({"year": years.index.astype(int), "count": years.values})
        out["%"] = out["count"] / len(df)
        print(out.to_string(index=False, formatters={"%": fmt_pct}))

        hours = ts.dropna().dt.hour.value_counts().sort_index()
        if len(hours) > 0:
            print("\nCounts by hour (valid only):")
            out_h = pd.DataFrame({"hour": hours.index.astype(int), "count": hours.values})
            out_h["%"] = out_h["count"] / max(1, int((~invalid).sum()))
            print(out_h.to_string(index=False, formatters={"%": fmt_pct}))


def print_text_summary(df: pd.DataFrame, col: str, sample_k: int = 3, seed: int = 0) -> None:
    subsection(f"Text field analysis: {col}")
    s = df[col].astype("string").fillna("")
    char_len = s.str.len()
    word_len = s.str.split().str.len()

    desc = pd.DataFrame(
        {
            "chars": char_len.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]),
            "words": word_len.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]),
        }
    )
    print(desc.to_string())

    patterns = {
        "contains_html_tag": r"<[^>]+>",
        "contains_url": r"https?://|www\.",
        "contains_amp_entity": r"&[a-zA-Z]+;",
        "contains_non_ascii": r"[^\x00-\x7F]",
    }
    print("\nPattern prevalence (share of rows):")
    rows = []
    for name, pat in patterns.items():
        m = s.str.contains(pat, regex=True, na=False)
        rows.append({"pattern": name, "count": int(m.sum()), "share": float(m.mean())})
    out = pd.DataFrame(rows).sort_values("share", ascending=False)
    print(out.to_string(index=False, formatters={"share": fmt_pct}))

    if len(df) > 0:
        idx_short = char_len.nsmallest(min(3, len(df))).index.tolist()
        idx_long = char_len.nlargest(min(3, len(df))).index.tolist()

        print("\nShortest examples:")
        for i in idx_short:
            print(f"  - ({int(char_len.loc[i])} chars) {shorten(s.loc[i])}")

        print("\nLongest examples:")
        for i in idx_long:
            print(f"  - ({int(char_len.loc[i])} chars) {shorten(s.loc[i])}")

    if sample_k > 0:
        rng = np.random.default_rng(seed)
        pick = rng.choice(df.index.to_numpy(), size=min(sample_k, len(df)), replace=False)
        print(f"\nRandom sample ({min(sample_k, len(df))} rows):")
        for i in pick:
            print(f"  - {shorten(s.loc[i], max_len=220)}")


def print_source_pagerank_checks(df: pd.DataFrame, topn: int = 10) -> None:
    subsection("Source / page_rank consistency")
    src = df["source"].astype("string")
    pr = pd.to_numeric(df["page_rank"], errors="coerce")

    print(f"Sources: {fmt_int(int(src.nunique(dropna=True)))} unique")
    vc = src.value_counts(dropna=False).head(topn)
    out = pd.DataFrame({"source": vc.index.astype(str), "count": vc.values})
    out["%"] = out["count"] / len(df)
    print("\nTop sources:")
    print(out.to_string(index=False, formatters={"%": fmt_pct}))

    grp = pd.DataFrame({"source": src, "page_rank": pr}).dropna()
    nun_pr = grp.groupby("source")["page_rank"].nunique()
    varying = nun_pr[nun_pr > 1].sort_values(ascending=False)
    print(f"\nSources with non-constant page_rank: {fmt_int(int((nun_pr > 1).sum()))}")
    if len(varying) > 0:
        print("Examples (source -> #distinct page_rank):")
        for k, v in varying.head(10).items():
            print(f"  - {k}: {int(v)}")


def print_label_summary(df: pd.DataFrame) -> None:
    subsection("Label distribution & imbalance indicators (development only)")
    y = df["label"]
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.notna().mean() > 0.95:
        y = y_num.astype("Int64")

    vc = y.value_counts(dropna=False).sort_index()
    out = pd.DataFrame({"label": vc.index.astype(str), "count": vc.values})
    out["%"] = out["count"] / len(df)
    print(out.to_string(index=False, formatters={"%": fmt_pct}))

    K = int(y.nunique(dropna=True))
    if K > 1:
        p = float(vc.max() / len(df))
        f1_major = 2.0 * p / (p + 1.0)
        macro_f1_majority = f1_major / K
        print(f"\nMajority share p: {p:.4f}")
        print(f"Majority-only baseline Macro-F1 (theoretical): {macro_f1_majority:.4f} (K={K})")


def jensen_shannon(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compare_splits(dev: pd.DataFrame, eva: pd.DataFrame) -> None:
    section("Cross-split checks (development vs evaluation)")

    subsection("Schema alignment")
    common = [c for c in EVAL_COLS if c in DEV_COLS]
    only_dev = [c for c in DEV_COLS if c not in EVAL_COLS]
    only_eva = [c for c in EVAL_COLS if c not in DEV_COLS]
    print(f"Common columns: {common}")
    print(f"Only in development: {only_dev}")
    print(f"Only in evaluation:  {only_eva}")

    subsection("ID overlap")
    a = set(dev["Id"].dropna().astype(int).tolist())
    b = set(eva["Id"].dropna().astype(int).tolist())
    inter = a.intersection(b)
    print(f"Overlapping IDs: {fmt_int(len(inter))}")
    if len(inter) > 0:
        print("[WARN] There should usually be no ID overlap between splits. Investigate data leakage risk.")

    subsection("Source overlap & unseen sources")
    src_dev = set([s.strip() for s in dev["source"].dropna().astype(str).tolist()])
    src_eva = set([s.strip() for s in eva["source"].dropna().astype(str).tolist()])
    unseen = src_eva - src_dev
    print(f"Dev sources: {fmt_int(len(src_dev))}")
    print(f"Eval sources: {fmt_int(len(src_eva))}")
    print(f"Eval sources unseen in dev: {fmt_int(len(unseen))} ({fmt_pct(len(unseen)/max(1,len(src_eva)))})")
    if len(unseen) > 0:
        print("Examples of unseen eval sources:")
        for s in sorted(list(unseen))[:15]:
            print(f"  - {s}")

    subsection("Drift diagnostics (Jensen–Shannon divergence)")
    drift_rows = []

    d1 = pd.to_numeric(dev["page_rank"], errors="coerce").dropna()
    d2 = pd.to_numeric(eva["page_rank"], errors="coerce").dropna()
    bins = sorted(set(d1.unique()).union(set(d2.unique())))
    p = np.array([(d1 == b).mean() for b in bins])
    q = np.array([(d2 == b).mean() for b in bins])
    drift_rows.append({"feature": "page_rank (categorical)", "JS_div": jensen_shannon(p, q)})

    t1 = try_parse_datetime(dev["timestamp"]).dropna()
    t2 = try_parse_datetime(eva["timestamp"]).dropna()
    if len(t1) > 0 and len(t2) > 0:
        y1 = t1.dt.year
        y2 = t2.dt.year
        years = sorted(set(y1.unique()).union(set(y2.unique())))
        p = np.array([(y1 == y).mean() for y in years])
        q = np.array([(y2 == y).mean() for y in years])
        drift_rows.append({"feature": "timestamp.year (valid only)", "JS_div": jensen_shannon(p, q)})

    for c in ["title", "article"]:
        w1 = dev[c].astype("string").fillna("").str.split().str.len()
        w2 = eva[c].astype("string").fillna("").str.split().str.len()
        bins = [0, 5, 10, 20, 40, 80, 160, 320, 10_000]
        p = np.histogram(w1, bins=bins, density=True)[0]
        q = np.histogram(w2, bins=bins, density=True)[0]
        drift_rows.append({"feature": f"{c}.word_len (binned)", "JS_div": jensen_shannon(p, q)})

    out = pd.DataFrame(drift_rows).sort_values("JS_div", ascending=False)
    print(out.to_string(index=False, formatters={"JS_div": lambda x: f"{x:.6f}"}))

    subsection("Exact overlap on content hashes (potential leakage / reprints)")
    for k in ["title", "article"]:
        h1 = set(stable_hash_series(dev[k]).tolist())
        h2 = set(stable_hash_series(eva[k]).tolist())
        inter = len(h1.intersection(h2))
        print(f"Exact overlap on {k} hashes: {fmt_int(inter)}")
        if inter > 0:
            print(
                f"[WARN] Found exact {k} overlap between dev and eval. "
                "This can inflate validation if not handled carefully (dedup/group split)."
            )


def save_plots(dev: pd.DataFrame, eva: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_hist(series: pd.Series, title: str, fname: str, bins: int = 50) -> None:
        plt.figure()
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            plt.close()
            return
        plt.hist(s, bins=bins)
        plt.title(title)
        plt.xlabel(series.name)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    def save_bar(vc: pd.Series, title: str, fname: str) -> None:
        plt.figure(figsize=(10, 4))
        vc = vc.head(15)
        plt.bar(vc.index.astype(str), vc.values)
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    save_hist(dev["page_rank"], "Development: page_rank distribution", "dev_page_rank.png", bins=10)
    save_hist(eva["page_rank"], "Evaluation: page_rank distribution", "eval_page_rank.png", bins=10)

    save_bar(dev["source"].value_counts(), "Development: top sources", "dev_top_sources.png")
    save_bar(eva["source"].value_counts(), "Evaluation: top sources", "eval_top_sources.png")

    save_bar(dev["label"].value_counts().sort_index(), "Development: label distribution", "dev_label_dist.png")

    for c in ["title", "article"]:
        w = dev[c].astype("string").fillna("").str.split().str.len()
        save_hist(w.rename(f"{c}_words"), f"Development: {c} word length", f"dev_{c}_wordlen.png", bins=60)

        w = eva[c].astype("string").fillna("").str.split().str.len()
        save_hist(w.rename(f"{c}_words"), f"Evaluation: {c} word length", f"eval_{c}_wordlen.png", bins=60)

    t = try_parse_datetime(dev["timestamp"]).dropna()
    if len(t) > 0:
        save_bar(t.dt.year.value_counts().sort_index(), "Development: timestamp years (valid)", "dev_years.png")

    t = try_parse_datetime(eva["timestamp"]).dropna()
    if len(t) > 0:
        save_bar(t.dt.year.value_counts().sort_index(), "Evaluation: timestamp years (valid)", "eval_years.png")

    print(f"[OK] Saved plots to: {out_dir}")


def find_repo_root(start: Path) -> Optional[Path]:
    for p in [start] + list(start.parents):
        if (p / "data" / "raw").exists():
            return p
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detailed EDA on development/evaluation CSVs in data/raw.")
    parser.add_argument("--output-dir", type=str, default="reports/eda", help="Where to save plots (if --plots).")
    parser.add_argument("--plots", action="store_true", help="Save a small set of EDA plots as PNG files.")
    parser.add_argument("--max-cats", type=int, default=10, help="Top-N categories to print for categorical columns.")
    parser.add_argument("--text-sample", type=int, default=2, help="Number of random text samples to print per text column.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    return parser.parse_args()


def run_eda(repo_root: Path, output_dir: Path, make_plots: bool, max_cats: int, text_sample: int, seed: int) -> None:
    dev_path = (repo_root / DEV_PATH).resolve()
    eva_path = (repo_root / EVAL_PATH).resolve()

    if not dev_path.exists():
        raise FileNotFoundError(f"Missing: {dev_path}")
    if not eva_path.exists():
        raise FileNotFoundError(f"Missing: {eva_path}")

    dev = pd.read_csv(dev_path)
    eva = pd.read_csv(eva_path)

    enforce_schema(dev, DEV_COLS, "development.csv")
    enforce_schema(eva, EVAL_COLS, "evaluation.csv")

    section("Loaded datasets")
    print("Schema checks: OK (fixed columns; evaluation has no label).")

    dev_prof = infer_profile("development", dev_path, dev)
    eva_prof = infer_profile("evaluation", eva_path, eva)

    # --- Development report (with labels) ---
    section("EDA report: DEVELOPMENT")
    print_basic_info(dev, dev_prof)
    print_missingness(dev)
    print_id_checks(dev)
    print_duplicates(dev)

    subsection("Categorical / high-cardinality fields")
    print_categorical_summary(dev, "source", topn=max_cats)

    subsection("Numeric fields")
    print_numeric_summary(dev, "page_rank")

    print_timestamp_summary(dev)

    print_text_summary(dev, "title", sample_k=text_sample, seed=seed)
    print_text_summary(dev, "article", sample_k=text_sample, seed=seed)

    print_source_pagerank_checks(dev, topn=max_cats)
    print_label_summary(dev)

    subsection("Potential gotchas detected (heuristics)")
    gotchas = []

    placeholder_share = dev["timestamp"].astype(str).eq("0000-00-00 00:00:00").mean()
    if placeholder_share > 0.01:
        gotchas.append(
            f"timestamp contains placeholder '0000-00-00 00:00:00' in {fmt_pct(placeholder_share)} of rows "
            "(treat as missing; avoid misleading time features)."
        )

    for c in ["article", "title"]:
        s = dev[c].astype("string").fillna("")
        html = s.str.contains(r"<[^>]+>", regex=True).mean()
        url = s.str.contains(r"https?://|www\.", regex=True).mean()
        if html > 0.05:
            gotchas.append(f"{c} contains HTML-like tags in {fmt_pct(html)} of rows (consider cleaning).")
        if url > 0.05:
            gotchas.append(f"{c} contains URLs in {fmt_pct(url)} of rows (consider stripping or normalizing).")

    grp = dev[["source", "page_rank"]].copy()
    grp["page_rank"] = pd.to_numeric(grp["page_rank"], errors="coerce")
    nun = grp.dropna().groupby("source")["page_rank"].nunique()
    if (nun > 1).mean() > 0.001:
        gotchas.append(
            "page_rank is mostly constant per source (feature may act as a proxy for source; "
            "beware of overfitting to publishers)."
        )

    if gotchas:
        for g in gotchas:
            print(f"  - {g}")
    else:
        print("  - No obvious gotchas flagged by heuristics (still inspect manually).")

    # --- Evaluation report (no labels) ---
    section("EDA report: EVALUATION")
    print_basic_info(eva, eva_prof)
    print_missingness(eva)
    print_id_checks(eva)
    print_duplicates(eva)

    subsection("Categorical / high-cardinality fields")
    print_categorical_summary(eva, "source", topn=max_cats)

    subsection("Numeric fields")
    print_numeric_summary(eva, "page_rank")

    print_timestamp_summary(eva)

    print_text_summary(eva, "title", sample_k=text_sample, seed=seed)
    print_text_summary(eva, "article", sample_k=text_sample, seed=seed)

    print_source_pagerank_checks(eva, topn=max_cats)

    subsection("Potential gotchas detected (heuristics)")
    gotchas = []

    placeholder_share = eva["timestamp"].astype(str).eq("0000-00-00 00:00:00").mean()
    if placeholder_share > 0.01:
        gotchas.append(
            f"timestamp contains placeholder '0000-00-00 00:00:00' in {fmt_pct(placeholder_share)} of rows "
            "(treat as missing; avoid misleading time features)."
        )

    for c in ["article", "title"]:
        s = eva[c].astype("string").fillna("")
        html = s.str.contains(r"<[^>]+>", regex=True).mean()
        url = s.str.contains(r"https?://|www\.", regex=True).mean()
        if html > 0.05:
            gotchas.append(f"{c} contains HTML-like tags in {fmt_pct(html)} of rows (consider cleaning).")
        if url > 0.05:
            gotchas.append(f"{c} contains URLs in {fmt_pct(url)} of rows (consider stripping or normalizing).")

    grp = eva[["source", "page_rank"]].copy()
    grp["page_rank"] = pd.to_numeric(grp["page_rank"], errors="coerce")
    nun = grp.dropna().groupby("source")["page_rank"].nunique()
    if (nun > 1).mean() > 0.001:
        gotchas.append(
            "page_rank is mostly constant per source (feature may act as a proxy for source; "
            "beware of overfitting to publishers)."
        )

    if gotchas:
        for g in gotchas:
            print(f"  - {g}")
    else:
        print("  - No obvious gotchas flagged by heuristics (still inspect manually).")

    # Cross-split checks + plots
    compare_splits(dev, eva)

    if make_plots:
        section("Saving plots")
        save_plots(dev, eva, output_dir)


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = find_repo_root(script_dir) or find_repo_root(Path.cwd())
    if repo_root is None:
        raise RuntimeError("Could not locate repo root containing data/raw from current location.")

    output_dir = (repo_root / args.output_dir).resolve()

    run_eda(
        repo_root=repo_root,
        output_dir=output_dir,
        make_plots=bool(args.plots),
        max_cats=int(args.max_cats),
        text_sample=int(args.text_sample),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 60)
    main()
