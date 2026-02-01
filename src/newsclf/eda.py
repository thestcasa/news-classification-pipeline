from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve


DEV_COLS = ["Id", "source", "title", "article", "page_rank", "timestamp", "label"]
EVAL_COLS = ["Id", "source", "title", "article", "page_rank", "timestamp"]

ID_COL = "Id"
TEXT_COLS = ["title", "article"]

DEV_PATH = Path("data/raw/development.csv")
EVAL_PATH = Path("data/raw/evaluation.csv")

TS_PLACEHOLDERS = {"0000-00-00 00:00:00", "0000-00-00"}
MISSING_SENTINELS = {"\\N"}  

SEP = "=" * 96
SUBSEP = "-" * 96


# Console helpers

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


def ratio_str(count: int, total: int) -> str:
    pct = (count / total) if total else 0.0
    return f"{fmt_int(int(count))}/{fmt_int(int(total))} ({fmt_pct(pct)})"


# Small data utilities

def as_str_series(x: pd.Series) -> pd.Series:
    # Keep behavior consistent: strings, but don't explode on NaN
    return x.astype("string")


def stable_hash_series(s: pd.Series) -> pd.Series:
    # NOTE: intentionally MD5 (fast, stable for EDA), not for security.
    return s.fillna("").astype(str).apply(
        lambda x: hashlib.md5(x.encode("utf-8", errors="ignore")).hexdigest()
    )


def fingerprint_head(df: pd.DataFrame, n: int = 50_000) -> str:
    sample = df.head(min(n, len(df)))
    h = hashlib.md5(pd.util.hash_pandas_object(sample, index=True).values.tobytes()).hexdigest()
    return h


def try_parse_datetime(ts: pd.Series) -> pd.Series:
    ts_str = ts.astype(str)
    ts_str = ts_str.mask(ts_str.isin(TS_PLACEHOLDERS), other=np.nan)
    return pd.to_datetime(ts_str, errors="coerce")


def missing_like_text(s: pd.Series) -> pd.Series:
    ss = as_str_series(s)
    norm = ss.fillna("").str.strip()
    return ss.isna() | norm.eq("") | norm.isin(list(MISSING_SENTINELS))


@dataclass
class DatasetProfile:
    name: str
    path: Path
    n_rows: int
    n_cols: int
    columns: List[str]


def infer_profile(name: str, path: Path, df: pd.DataFrame) -> DatasetProfile:
    return DatasetProfile(
        name=name,
        path=path,
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=list(df.columns),
    )


def enforce_schema(df: pd.DataFrame, expected_cols: List[str], name: str) -> None:
    cols = list(df.columns)
    if cols == expected_cols:
        return

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


# Core EDA printers

def print_basic_info(df: pd.DataFrame, prof: DatasetProfile) -> None:
    subsection("Basic structure")
    print(f"File:  {prof.path}")
    print(f"Shape: {fmt_int(prof.n_rows)} rows × {fmt_int(prof.n_cols)} columns")

    print("Columns / dtypes:")
    dtypes = df.dtypes.astype(str)
    for c in df.columns:
        print(f"  - {c:12s} {dtypes[c]}")

    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"Approx. memory usage (deep): {mem_mb:.2f} MB")

    print(f"Fingerprint (md5 over head rows): {fingerprint_head(df)}")


def print_missingness(df: pd.DataFrame) -> None:
    subsection("Missingness & empty strings")

    na = df.isna().sum()
    na_pct = (na / max(1, len(df))).replace([np.inf, np.nan], 0.0)

    placeholder_by_col = {
        "timestamp": TS_PLACEHOLDERS,
    }

    rows = []
    for c in df.columns:
        if c == "timestamp":
            ts_missing = try_parse_datetime(df[c]).isna()
            empty = as_str_series(df[c]).fillna("").str.strip().eq("")
            placeholder = as_str_series(df[c]).fillna("").str.strip().isin(list(TS_PLACEHOLDERS | MISSING_SENTINELS))
            missing_like = ts_missing | empty | placeholder
            rows.append(
                {
                    "col": c,
                    "na": int(na[c]),
                    "na_%": float(na_pct[c]),
                    "empty_str": int(empty.sum()),
                    "empty_%": float(empty.mean()) if len(df) else 0.0,
                    "placeholder": int(placeholder.sum()),
                    "placeholder_%": float(placeholder.mean()) if len(df) else 0.0,
                    "missing_like": int(missing_like.sum()),
                    "missing_like_%": float(missing_like.mean()) if len(df) else 0.0,
                }
            )
            continue

        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            s = as_str_series(df[c])
            norm = s.fillna("").str.strip()
            placeholders = set(placeholder_by_col.get(c, set())) | MISSING_SENTINELS
            empty = norm.eq("")
            placeholder = norm.isin(list(placeholders))
            missing_like = s.isna() | empty | placeholder
            rows.append(
                {
                    "col": c,
                    "na": int(na[c]),
                    "na_%": float(na_pct[c]),
                    "empty_str": int(empty.sum()),
                    "empty_%": float(empty.mean()) if len(df) else 0.0,
                    "placeholder": int(placeholder.sum()),
                    "placeholder_%": float(placeholder.mean()) if len(df) else 0.0,
                    "missing_like": int(missing_like.sum()),
                    "missing_like_%": float(missing_like.mean()) if len(df) else 0.0,
                }
            )
        else:
            rows.append(
                {
                    "col": c,
                    "na": int(na[c]),
                    "na_%": float(na_pct[c]),
                    "empty_str": 0,
                    "empty_%": 0.0,
                    "placeholder": 0,
                    "placeholder_%": 0.0,
                    "missing_like": int(na[c]),
                    "missing_like_%": float(na_pct[c]),
                }
            )

    out = pd.DataFrame(rows).sort_values(["na", "empty_str"], ascending=False)
    print(
        out.to_string(
            index=False,
            formatters={
                "na_%": fmt_pct,
                "empty_%": fmt_pct,
                "placeholder_%": fmt_pct,
                "missing_like_%": fmt_pct,
            },
        )
    )


def print_text_missingness(df: pd.DataFrame, col: str) -> None:
    subsection(f"Missing/empty check: {col}")
    miss = missing_like_text(df[col])
    na = as_str_series(df[col]).isna().sum()
    empty = as_str_series(df[col]).fillna("").str.strip().eq("").sum()
    placeholder = as_str_series(df[col]).fillna("").str.strip().isin(list(MISSING_SENTINELS)).sum()

    n = max(1, len(df))
    print(f"{col} missing (NA): {fmt_int(int(na))} ({fmt_pct(float(na / n))})")
    print(f"{col} empty/blank: {fmt_int(int(empty))} ({fmt_pct(float(empty / n))})")
    print(f"{col} placeholder '\\\\N': {fmt_int(int(placeholder))} ({fmt_pct(float(placeholder / n))})")
    print(f"{col} missing/empty/\\\\N: {fmt_int(int(miss.sum()))} ({fmt_pct(float(miss.mean()))})")


def print_missing_article_label_distribution(df: pd.DataFrame) -> None:
    subsection("Missing articles by label (share of all missing articles)")
    miss = missing_like_text(df["article"])
    total_missing = int(miss.sum())
    if total_missing == 0:
        print("No missing/empty/\\N articles found.")
        return

    counts = df.loc[miss, "label"].value_counts().sort_index()
    out = pd.DataFrame({"label": counts.index.astype(str), "missing_count": counts.values})
    out["share_of_missing"] = out["missing_count"] / total_missing
    print(out.sort_values("label").to_string(index=False, formatters={"share_of_missing": fmt_pct}))


def print_missing_timestamp_label_distribution(df: pd.DataFrame) -> None:
    subsection("Missing timestamps by label (share of all missing timestamps)")
    ts = try_parse_datetime(df["timestamp"])
    miss = ts.isna()
    total_missing = int(miss.sum())
    if total_missing == 0:
        print("No missing/invalid timestamps found.")
        return

    counts = df.loc[miss, "label"].value_counts().sort_index()
    out = pd.DataFrame({"label": counts.index.astype(str), "missing_count": counts.values})
    out["share_of_missing"] = out["missing_count"] / total_missing
    print(out.sort_values("label").to_string(index=False, formatters={"share_of_missing": fmt_pct}))


def print_label_timestamp_missingness(df: pd.DataFrame) -> None:
    subsection("Per-label missingness: timestamp")
    s = as_str_series(df["timestamp"]).fillna("").str.strip()
    placeholders = s.isin(list(TS_PLACEHOLDERS))
    missing_like = try_parse_datetime(df["timestamp"]).isna()

    tmp = pd.DataFrame({"label": df["label"], "placeholder": placeholders, "missing_like": missing_like})
    grp = tmp.groupby("label")
    totals = grp.size().rename("total")
    counts = grp[["placeholder", "missing_like"]].sum()
    out = counts.join(totals).reset_index().sort_values("label")

    out_fmt = pd.DataFrame(
        {
            "label": out["label"],
            "placeholder": [ratio_str(x, t) for x, t in zip(out["placeholder"], out["total"])],
            "missing_like": [ratio_str(x, t) for x, t in zip(out["missing_like"], out["total"])],
        }
    )
    print(out_fmt.to_string(index=False))


def print_label_text_missingness(df: pd.DataFrame, col: str) -> None:
    subsection(f"Per-label missingness: {col}")
    s = as_str_series(df[col])
    norm = s.fillna("").str.strip()

    tmp = pd.DataFrame(
        {
            "label": df["label"],
            "na": s.isna(),
            "empty": norm.eq(""),
            "placeholder": norm.isin(list(MISSING_SENTINELS)),
        }
    )
    tmp["missing_like"] = tmp["na"] | tmp["empty"] | tmp["placeholder"]

    grp = tmp.groupby("label")
    totals = grp.size().rename("total")
    counts = grp[["na", "empty", "placeholder", "missing_like"]].sum()
    out = counts.join(totals).reset_index().sort_values("label")

    out_fmt = pd.DataFrame(
        {
            "label": out["label"],
            "na": [ratio_str(x, t) for x, t in zip(out["na"], out["total"])],
            "empty": [ratio_str(x, t) for x, t in zip(out["empty"], out["total"])],
            "placeholder": [ratio_str(x, t) for x, t in zip(out["placeholder"], out["total"])],
            "missing_like": [ratio_str(x, t) for x, t in zip(out["missing_like"], out["total"])],
        }
    )
    print(out_fmt.to_string(index=False))


def print_id_checks(df: pd.DataFrame) -> None:
    subsection("ID sanity checks")
    ids = df[ID_COL]
    print(f"ID column: {ID_COL} (dtype={ids.dtype})")

    miss = int(ids.isna().sum())
    print(f"Missing IDs: {fmt_int(miss)} ({fmt_pct(float(ids.isna().mean()))})")

    dup = int(ids.duplicated().sum())
    print(f"Duplicate IDs: {fmt_int(dup)} ({fmt_pct(float(dup / max(1, len(df))))})")

    if pd.api.types.is_numeric_dtype(ids):
        print(f"ID min/max: {int(np.nanmin(ids))} / {int(np.nanmax(ids))}")
    else:
        nun = int(ids.nunique(dropna=True))
        print(f"Unique IDs: {fmt_int(nun)}")


def print_duplicates(df: pd.DataFrame) -> None:
    subsection("Row & content duplicates")

    dup_rows = int(df.duplicated().sum())
    print(f"Fully duplicated rows: {fmt_int(dup_rows)} ({fmt_pct(float(dup_rows / max(1, len(df))))})")

    dup_content = int(df.duplicated(subset=["title", "article"]).sum())
    print(
        "Duplicate pairs on ['title','article']: "
        f"{fmt_int(dup_content)} ({fmt_pct(float(dup_content / max(1, len(df))))})"
    )

    dup_title = int(df.duplicated(subset=["title"]).sum())
    print(f"Duplicate titles: {fmt_int(dup_title)} ({fmt_pct(float(dup_title / max(1, len(df))))})")


def print_categorical_summary(df: pd.DataFrame, col: str, topn: int = 10) -> None:
    nun = int(df[col].nunique(dropna=True))
    print(f"\n[{col}] unique values: {fmt_int(nun)}")

    vc = df[col].value_counts(dropna=False).head(topn)
    out = pd.DataFrame({"value": vc.index.astype(str), "count": vc.values})
    out["%"] = out["count"] / max(1, len(df))
    print(out.to_string(index=False, formatters={"%": fmt_pct}))


def print_numeric_summary(df: pd.DataFrame, col: str) -> None:
    s = pd.to_numeric(df[col], errors="coerce")
    desc = s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(f"\n[{col}] numeric summary (coerced)")
    print(desc.to_string())


def print_timestamp_summary(df: pd.DataFrame) -> None:
    subsection("Timestamp quality & distribution")

    ts_raw = as_str_series(df["timestamp"])
    is_placeholder = ts_raw.fillna("").isin(list(TS_PLACEHOLDERS))
    print(f"Placeholder '0000-00-00 00:00:00': {fmt_int(int(is_placeholder.sum()))} ({fmt_pct(float(is_placeholder.mean()))})")

    ts = try_parse_datetime(df["timestamp"])
    invalid = ts.isna()
    print(f"Unparseable/NaT timestamps: {fmt_int(int(invalid.sum()))} ({fmt_pct(float(invalid.mean()))})")

    if (~invalid).any():
        tmin, tmax = ts.min(), ts.max()
        print(f"Valid timestamp range: {tmin}  →  {tmax}")

        years = ts.dropna().dt.year.value_counts().sort_index()
        print("\nCounts by year (valid only):")
        out = pd.DataFrame({"year": years.index.astype(int), "count": years.values})
        out["%"] = out["count"] / max(1, len(df))
        print(out.to_string(index=False, formatters={"%": fmt_pct}))

        hours = ts.dropna().dt.hour.value_counts().sort_index()
        if len(hours) > 0:
            print("\nCounts by hour (valid only):")
            out_h = pd.DataFrame({"hour": hours.index.astype(int), "count": hours.values})
            out_h["%"] = out_h["count"] / max(1, int((~invalid).sum()))
            print(out_h.to_string(index=False, formatters={"%": fmt_pct}))


def print_text_summary(df: pd.DataFrame, col: str, sample_k: int = 3, seed: int = 0) -> None:
    subsection(f"Text field analysis: {col}")

    s = as_str_series(df[col]).fillna("")
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

    if sample_k > 0 and len(df) > 0:
        rng = np.random.default_rng(seed)
        pick = rng.choice(df.index.to_numpy(), size=min(sample_k, len(df)), replace=False)
        print(f"\nRandom sample ({min(sample_k, len(df))} rows):")
        for i in pick:
            print(f"  - {shorten(s.loc[i], max_len=220)}")


def print_source_pagerank_checks(df: pd.DataFrame, topn: int = 10) -> None:
    subsection("Source / page_rank consistency")

    src = as_str_series(df["source"])
    pr = pd.to_numeric(df["page_rank"], errors="coerce")

    print(f"Sources: {fmt_int(int(src.nunique(dropna=True)))} unique")

    vc = src.value_counts(dropna=False).head(topn)
    out = pd.DataFrame({"source": vc.index.astype(str), "count": vc.values})
    out["%"] = out["count"] / max(1, len(df))
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
    out["%"] = out["count"] / max(1, len(df))
    print(out.to_string(index=False, formatters={"%": fmt_pct}))

    K = int(y.nunique(dropna=True))
    if K > 1:
        p = float(vc.max() / max(1, len(df)))
        f1_major = 2.0 * p / (p + 1.0)
        macro_f1_majority = f1_major / K
        print(f"\nMajority share p: {p:.4f}")
        print(f"Majority-only baseline Macro-F1 (theoretical): {macro_f1_majority:.4f} (K={K})")


def _percentile(s: pd.Series, q: float) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    return float(np.percentile(s.to_numpy(), q))


def print_label_text_lengths(df: pd.DataFrame, col: str) -> None:
    subsection(f"Per-label text length stats: {col}")

    s = as_str_series(df[col]).fillna("")
    tmp = pd.DataFrame(
        {
            "label": df["label"],
            "chars": s.str.len(),
            "words": s.str.split().str.len(),
        }
    )
    out = (
        tmp.groupby("label")
        .agg(
            count=("chars", "size"),
            chars_mean=("chars", "mean"),
            chars_p50=("chars", lambda x: _percentile(x, 50)),
            chars_p90=("chars", lambda x: _percentile(x, 90)),
            words_mean=("words", "mean"),
            words_p50=("words", lambda x: _percentile(x, 50)),
            words_p90=("words", lambda x: _percentile(x, 90)),
        )
        .reset_index()
        .sort_values("label")
    )
    float_cols = [c for c in out.columns if c not in {"label", "count"}]
    out[float_cols] = out[float_cols].round(2)
    print(out.to_string(index=False))


def print_label_pattern_prevalence(df: pd.DataFrame, col: str) -> None:
    subsection(f"Per-label URL/HTML prevalence: {col}")

    s = as_str_series(df[col]).fillna("")
    tmp = pd.DataFrame(
        {
            "label": df["label"],
            "has_html": s.str.contains(r"<[^>]+>", regex=True, na=False),
            "has_url": s.str.contains(r"https?://|www\.", regex=True, na=False),
        }
    )
    out = tmp.groupby("label")[["has_html", "has_url"]].mean().reset_index().sort_values("label")
    print(out.to_string(index=False, formatters={"has_html": fmt_pct, "has_url": fmt_pct}))


def print_label_top_sources(df: pd.DataFrame, topn: int = 5) -> None:
    subsection("Top sources per label")

    src = as_str_series(df["source"]).fillna("")
    tmp = pd.DataFrame({"label": df["label"], "source": src})

    counts = tmp.groupby(["label", "source"]).size().reset_index(name="count")
    label_counts = tmp["label"].value_counts()
    counts["label_count"] = counts["label"].map(label_counts)
    counts["share"] = counts["count"] / counts["label_count"]

    rows = []
    for label, g in counts.groupby("label", sort=True):
        rows.append(g.sort_values("count", ascending=False).head(topn))
    out = pd.concat(rows, ignore_index=True) if rows else counts.head(0)
    out = out.sort_values(["label", "count"], ascending=[True, False])

    print(out[["label", "source", "count", "share"]].to_string(index=False, formatters={"share": fmt_pct}))


def print_label_time_distribution(df: pd.DataFrame, max_rows: int = 60) -> None:
    subsection("Timestamp distribution by label (valid only)")

    ts = try_parse_datetime(df["timestamp"])
    valid = ts.notna()
    if not valid.any():
        print("No valid timestamps available.")
        return

    tmp = pd.DataFrame(
        {
            "label": df.loc[valid, "label"],
            "year": ts[valid].dt.year.astype(int),
            "month": ts[valid].dt.month.astype(int),
        }
    )
    label_counts = tmp["label"].value_counts()

    year_counts = tmp.groupby(["label", "year"]).size().reset_index(name="count")
    year_counts["share"] = year_counts["count"] / year_counts["label"].map(label_counts)
    if len(year_counts) > max_rows:
        print(f"[INFO] Too many label-year rows; showing top {max_rows} by count.")
        year_counts = year_counts.sort_values("count", ascending=False).head(max_rows)
    else:
        year_counts = year_counts.sort_values(["label", "year"])
    print("\nCounts by year:")
    print(year_counts.to_string(index=False, formatters={"share": fmt_pct}))

    month_counts = tmp.groupby(["label", "month"]).size().reset_index(name="count")
    month_counts["share"] = month_counts["count"] / month_counts["label"].map(label_counts)
    if len(month_counts) > max_rows:
        print(f"\n[INFO] Too many label-month rows; showing top {max_rows} by count.")
        month_counts = month_counts.sort_values("count", ascending=False).head(max_rows)
    else:
        month_counts = month_counts.sort_values(["label", "month"])
    print("\nCounts by month:")
    print(month_counts.to_string(index=False, formatters={"share": fmt_pct}))


# Duplicate analysis 

def _normalize_for_dup(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.str.replace(r"[^\w\s]", "", regex=True)
    return s


def content_hash(df: pd.DataFrame, near_dup: bool = False) -> pd.Series:
    if near_dup:
        title = _normalize_for_dup(df["title"])
        article = _normalize_for_dup(df["article"])
    else:
        title = df["title"].fillna("").astype(str)
        article = df["article"].fillna("").astype(str)
    combo = title + "\n" + article
    return stable_hash_series(combo)


def print_label_duplicate_checks(df: pd.DataFrame, near_dup: bool = False) -> None:
    subsection("Label-conditional duplicate checks")

    tmp = df[["label", "title", "article"]].copy()
    tmp["content_hash"] = content_hash(tmp, near_dup=near_dup)

    group_stats = tmp.groupby("content_hash")["label"].agg(nunique="nunique", size="size")
    dup_groups = group_stats[group_stats["size"] > 1]
    same_label = dup_groups[dup_groups["nunique"] == 1]
    cross_label = dup_groups[dup_groups["nunique"] > 1]

    print(f"Duplicate content groups (size>1): {fmt_int(len(dup_groups))}")
    print(f"  - same-label groups: {fmt_int(len(same_label))}")
    print(f"  - cross-label groups: {fmt_int(len(cross_label))}")

    if len(dup_groups) > 0:
        dup_rows = int(dup_groups["size"].sum())
        same_rows = int(same_label["size"].sum())
        cross_rows = int(cross_label["size"].sum())
        print(f"Duplicate rows (total): {fmt_int(dup_rows)} ({fmt_pct(dup_rows / max(1, len(df)))})")
        print(f"  - same-label rows: {fmt_int(same_rows)} ({fmt_pct(same_rows / max(1, len(df)))})")
        print(f"  - cross-label rows: {fmt_int(cross_rows)} ({fmt_pct(cross_rows / max(1, len(df)))})")

    if len(cross_label) == 0:
        print("Cross-label duplicate groups: 0")
    else:
        print(f"Cross-label duplicate groups: {fmt_int(len(cross_label))}")

        cross_rows = tmp[tmp["content_hash"].isin(cross_label.index)]
        label_counts = tmp["label"].astype(str).value_counts()
        cross_counts = cross_rows["label"].astype(str).value_counts()

        out = pd.DataFrame({"label": cross_counts.index.astype(str), "cross_dup_rows": cross_counts.values})
        out["share_of_label"] = out["cross_dup_rows"] / out["label"].map(label_counts)
        print("\nCross-label duplicate rows by label:")
        print(out.to_string(index=False, formatters={"share_of_label": fmt_pct}))

        print("\nExamples (cross-label duplicates):")
        for key in cross_label.sort_values("size", ascending=False).head(5).index.tolist():
            grp = tmp[tmp["content_hash"] == key]
            labels = grp["label"].astype(str).value_counts().to_dict()
            title = shorten(grp["title"].iloc[0])
            print(f"  - labels={labels} | title='{title}'")

    if not near_dup:
        print("\nTip: use --near-dup to run a normalized duplicate check (lowercase + punctuation strip).")


# Optional: OOF diagnostics 

def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _find_score_columns(df: pd.DataFrame, score_prefix: Optional[str]) -> Tuple[Dict[str, str], Optional[str]]:
    prefixes = [score_prefix] if score_prefix else ["score_", "prob_", "logit_"]
    for prefix in prefixes:
        if not prefix:
            continue
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            return {c[len(prefix):]: c for c in cols}, prefix
    return {}, None


def print_oof_pred_counts(df: pd.DataFrame, label_col: str, pred_col: str) -> None:
    subsection("Out-of-fold predicted vs true class counts")

    y_true = df[label_col].astype(str)
    y_pred = df[pred_col].astype(str)

    true_counts = y_true.value_counts().sort_index()
    pred_counts = y_pred.value_counts().reindex(true_counts.index, fill_value=0)

    out = pd.DataFrame(
        {
            "label": true_counts.index.astype(str),
            "true_count": true_counts.values,
            "pred_count": pred_counts.values,
        }
    )
    out["pred/true"] = out["pred_count"] / out["true_count"].replace(0, np.nan)
    out["true_%"] = out["true_count"] / out["true_count"].sum()
    out["pred_%"] = out["pred_count"] / out["pred_count"].sum()
    out = out.sort_values("label")

    print(
        out.to_string(
            index=False,
            formatters={
                "pred/true": lambda x: f"{x:.2f}" if pd.notna(x) else "nan",
                "true_%": fmt_pct,
                "pred_%": fmt_pct,
            },
        )
    )


def _best_f1_from_pr(precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
    if len(thresholds) == 0:
        return float("nan"), float("nan")
    f1 = 2.0 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    if np.all(np.isnan(f1)):
        return float("nan"), float("nan")
    i = int(np.nanargmax(f1))
    return float(f1[i]), float(thresholds[i])


def print_pr_curves(
    df: pd.DataFrame,
    label_col: str,
    pred_col: str,
    score_cols: Dict[str, str],
    output_dir: Path,
    make_plots: bool,
) -> List[str]:
    subsection("One-vs-rest PR curves (requires score columns)")

    y_true = df[label_col].astype(str)
    y_pred = df[pred_col].astype(str)
    labels = sorted(y_true.unique().tolist())
    labels_with_scores = [lab for lab in labels if lab in score_cols]

    if not labels_with_scores:
        print("No matching score columns for labels; skipping PR curves.")
        return []

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    rows = []
    for lab in labels_with_scores:
        scores = pd.to_numeric(df[score_cols[lab]], errors="coerce").to_numpy()
        valid = ~np.isnan(scores)
        if valid.sum() == 0:
            continue

        y_bin = (y_true == lab).astype(int).to_numpy()
        if y_bin[valid].sum() == 0 or y_bin[valid].sum() == valid.sum():
            print(f"[WARN] label {lab} has only one class in y_true; skipping PR curve.")
            continue

        precision, recall, thresholds = precision_recall_curve(y_bin[valid], scores[valid])
        ap = average_precision_score(y_bin[valid], scores[valid])
        best_f1, best_thr = _best_f1_from_pr(precision, recall, thresholds)

        rows.append(
            {
                "label": lab,
                "ap": float(ap),
                "f1_pred": float(report.get(lab, {}).get("f1-score", 0.0)),
                "best_f1": float(best_f1),
                "best_thr": float(best_thr),
            }
        )

        if make_plots:
            plt.figure()
            plt.plot(recall, precision, lw=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR curve (label={lab})")
            plt.tight_layout()
            plt.savefig(output_dir / f"oof_pr_curve_label_{lab}.png", dpi=150)
            plt.close()

    out = pd.DataFrame(rows).sort_values("label")
    if len(out) == 0:
        print("No valid score rows; skipping PR curve summary.")
        return []

    out[["ap", "f1_pred", "best_f1"]] = out[["ap", "f1_pred", "best_f1"]].round(4)
    out["best_thr"] = out["best_thr"].round(6)
    print(out.to_string(index=False))
    return out["label"].tolist()


def plot_score_distributions(
    df: pd.DataFrame,
    label_col: str,
    score_cols: Dict[str, str],
    weak_labels: List[str],
    output_dir: Path,
    make_plots: bool,
) -> None:
    subsection("Score separation for weak classes (requires score columns)")

    if not weak_labels:
        print("No weak labels provided; skipping score distributions.")
        return
    if not score_cols:
        print("No score columns provided; skipping score distributions.")
        return

    y_true = df[label_col].astype(str)
    for lab in weak_labels:
        if lab not in score_cols:
            print(f"[WARN] Missing score column for label {lab}; skipping.")
            continue

        scores = pd.to_numeric(df[score_cols[lab]], errors="coerce")
        valid = scores.notna()
        scores = scores[valid]
        y_bin = y_true[valid] == lab

        pos = scores[y_bin]
        neg = scores[~y_bin]
        if len(pos) == 0 or len(neg) == 0:
            print(f"[WARN] label {lab} has empty pos/neg scores; skipping.")
            continue

        print(
            f"Label {lab}: pos_mean={pos.mean():.4f} neg_mean={neg.mean():.4f} "
            f"pos_p90={np.percentile(pos, 90):.4f} neg_p90={np.percentile(neg, 90):.4f}"
        )

        if make_plots:
            plt.figure()
            plt.hist(neg, bins=40, alpha=0.6, label="neg", density=True)
            plt.hist(pos, bins=40, alpha=0.6, label="pos", density=True)
            plt.title(f"Score distribution (label={lab})")
            plt.xlabel("Score")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"oof_score_dist_label_{lab}.png", dpi=150)
            plt.close()


def run_oof_diagnostics(
    oof_path: Path,
    output_dir: Path,
    make_plots: bool,
    weak_k: int,
    score_prefix: Optional[str],
) -> None:
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing: {oof_path}")

    df = pd.read_csv(oof_path)

    label_col = _first_existing(df, ["label", "true", "y_true", "gold"])
    pred_col = _first_existing(df, ["pred", "y_pred", "prediction"])
    if label_col is None or pred_col is None:
        raise ValueError("OOF predictions must include label/true and pred/y_pred columns.")

    print(f"Using OOF labels column: {label_col}")
    print(f"Using OOF preds column:  {pred_col}")

    print_oof_pred_counts(df, label_col, pred_col)

    score_cols, used_prefix = _find_score_columns(df, score_prefix)
    if used_prefix:
        print(f"Using score columns with prefix: '{used_prefix}'")
    else:
        print("No score columns detected (prefix: score_, prob_, logit_).")

    if not score_cols:
        return

    labels_order = print_pr_curves(df, label_col, pred_col, score_cols, output_dir, make_plots)
    if not labels_order:
        return

    report = classification_report(
        df[label_col].astype(str),
        df[pred_col].astype(str),
        labels=labels_order,
        output_dict=True,
        zero_division=0,
    )
    f1_by_label = {lab: report.get(lab, {}).get("f1-score", 0.0) for lab in labels_order}
    weak_labels = [k for k, _ in sorted(f1_by_label.items(), key=lambda x: x[1])][: max(1, int(weak_k))]
    plot_score_distributions(df, label_col, score_cols, weak_labels, output_dir, make_plots)


# Cross-split checks (dev vs eval) + drift metric

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
    a = pd.to_numeric(dev["Id"], errors="coerce").dropna().astype(int)
    b = pd.to_numeric(eva["Id"], errors="coerce").dropna().astype(int)
    inter = set(a.tolist()).intersection(set(b.tolist()))
    print(f"Overlapping IDs: {fmt_int(len(inter))}")
    if len(inter) > 0:
        print("[WARN] There should usually be no ID overlap between splits. Investigate data leakage risk.")

    subsection("Source overlap & unseen sources")
    src_dev = set(as_str_series(dev["source"]).dropna().astype(str).map(str.strip).tolist())
    src_eva = set(as_str_series(eva["source"]).dropna().astype(str).map(str.strip).tolist())
    unseen = src_eva - src_dev
    print(f"Dev sources: {fmt_int(len(src_dev))}")
    print(f"Eval sources: {fmt_int(len(src_eva))}")
    print(f"Eval sources unseen in dev: {fmt_int(len(unseen))} ({fmt_pct(len(unseen)/max(1,len(src_eva)))})")
    if unseen:
        print("Examples of unseen eval sources:")
        for s in sorted(unseen)[:15]:
            print(f"  - {s}")

    subsection("Drift diagnostics (Jensen–Shannon divergence)")
    drift_rows = []

    d1 = pd.to_numeric(dev["page_rank"], errors="coerce").dropna()
    d2 = pd.to_numeric(eva["page_rank"], errors="coerce").dropna()
    bins = sorted(set(d1.unique()).union(set(d2.unique())))
    p = np.array([(d1 == b).mean() for b in bins]) if bins else np.array([1.0])
    q = np.array([(d2 == b).mean() for b in bins]) if bins else np.array([1.0])
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
        w1 = as_str_series(dev[c]).fillna("").str.split().str.len()
        w2 = as_str_series(eva[c]).fillna("").str.split().str.len()
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


# Plot saving

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

    for c in TEXT_COLS:
        w = as_str_series(dev[c]).fillna("").str.split().str.len()
        save_hist(w.rename(f"{c}_words"), f"Development: {c} word length", f"dev_{c}_wordlen.png", bins=60)

        w = as_str_series(eva[c]).fillna("").str.split().str.len()
        save_hist(w.rename(f"{c}_words"), f"Evaluation: {c} word length", f"eval_{c}_wordlen.png", bins=60)

    t = try_parse_datetime(dev["timestamp"]).dropna()
    if len(t) > 0:
        save_bar(t.dt.year.value_counts().sort_index(), "Development: timestamp years (valid)", "dev_years.png")

    t = try_parse_datetime(eva["timestamp"]).dropna()
    if len(t) > 0:
        save_bar(t.dt.year.value_counts().sort_index(), "Evaluation: timestamp years (valid)", "eval_years.png")

    print(f"[OK] Saved plots to: {out_dir}")


# Heuristics (gotchas)

def print_gotchas(df: pd.DataFrame) -> None:
    subsection("Potential gotchas detected (heuristics)")

    gotchas: List[str] = []

    placeholder_share = as_str_series(df["timestamp"]).fillna("").eq("0000-00-00 00:00:00").mean()
    if placeholder_share > 0.01:
        gotchas.append(
            f"timestamp contains placeholder '0000-00-00 00:00:00' in {fmt_pct(placeholder_share)} of rows "
            "(treat as missing; avoid misleading time features)."
        )

    for c in TEXT_COLS:
        s = as_str_series(df[c]).fillna("")
        html = s.str.contains(r"<[^>]+>", regex=True).mean()
        url = s.str.contains(r"https?://|www\.", regex=True).mean()
        if html > 0.05:
            gotchas.append(f"{c} contains HTML-like tags in {fmt_pct(html)} of rows (consider cleaning).")
        if url > 0.05:
            gotchas.append(f"{c} contains URLs in {fmt_pct(url)} of rows (consider stripping or normalizing).")

    grp = df[["source", "page_rank"]].copy()
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


# Repo utilities / CLI

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
    parser.add_argument("--near-dup", action="store_true", help="Also run a normalized (near) duplicate check on text.")
    parser.add_argument("--max-time-rows", type=int, default=60, help="Max rows to print for label/time tables.")
    parser.add_argument("--oof-preds", type=str, default=None, help="CSV with out-of-fold preds (label + pred + score_* columns).")
    parser.add_argument("--score-prefix", type=str, default=None, help="Prefix for score columns (default: score_/prob_/logit_).")
    parser.add_argument("--weak-k", type=int, default=2, help="How many weak labels to analyze with score plots.")
    return parser.parse_args()


# Main runner

def run_eda(
    repo_root: Path,
    output_dir: Path,
    make_plots: bool,
    max_cats: int,
    text_sample: int,
    seed: int,
    near_dup: bool,
    max_time_rows: int,
    oof_preds: Optional[str],
    score_prefix: Optional[str],
    weak_k: int,
) -> None:
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

    # --- DEVELOPMENT ---
    section("EDA report: DEVELOPMENT")
    print_basic_info(dev, dev_prof)
    print_missingness(dev)
    print_text_missingness(dev, "article")
    print_missing_article_label_distribution(dev)
    print_label_text_missingness(dev, "article")
    print_missing_timestamp_label_distribution(dev)
    print_label_timestamp_missingness(dev)
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
    print_label_text_lengths(dev, "title")
    print_label_text_lengths(dev, "article")
    print_label_pattern_prevalence(dev, "title")
    print_label_pattern_prevalence(dev, "article")
    print_label_top_sources(dev, topn=min(5, max_cats))
    print_label_time_distribution(dev, max_rows=max_time_rows)
    print_label_duplicate_checks(dev, near_dup=near_dup)
    print_gotchas(dev)

    # --- EVALUATION ---
    section("EDA report: EVALUATION")
    print_basic_info(eva, eva_prof)
    print_missingness(eva)
    print_text_missingness(eva, "article")
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
    print_gotchas(eva)

    # Cross-split checks + optional plots
    compare_splits(dev, eva)

    if oof_preds:
        section("Model-aware diagnostics (optional)")
        oof_path = (repo_root / oof_preds).resolve()
        run_oof_diagnostics(
            oof_path=oof_path,
            output_dir=output_dir,
            make_plots=make_plots,
            weak_k=weak_k,
            score_prefix=score_prefix,
        )

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
        near_dup=bool(args.near_dup),
        max_time_rows=int(args.max_time_rows),
        oof_preds=args.oof_preds,
        score_prefix=args.score_prefix,
        weak_k=int(args.weak_k),
    )


if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 60)
    main()
