from __future__ import annotations
from pathlib import Path
import pandas as pd

import re
import html as _html
import unicodedata
import numpy as np


REQ_DEV = {"id", "source", "title", "article", "page_rank", "timestamp", "label"}
REQ_EVAL = {"id", "source", "title", "article", "page_rank", "timestamp"}

_RE_TAGS = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_RE_WS = re.compile(r"\s+")

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and standardize Id -> id.
    Keeps everything else unchanged.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # standardize Id -> id
    if "Id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"Id": "id"})
    if "ID" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"ID": "id"})

    return df

def read_development(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardize(df)
    missing = REQ_DEV - set(df.columns)
    if missing:
        raise ValueError(f"development.csv missing columns: {sorted(missing)}. Found: {list(df.columns)}")
    return df

def read_evaluation(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardize(df)
    missing = REQ_EVAL - set(df.columns)
    if missing:
        raise ValueError(f"evaluation.csv missing columns: {sorted(missing)}. Found: {list(df.columns)}")
    return df

def write_submission(ids: pd.Series, pred, out_path: str | Path) -> None:
    out = pd.DataFrame({"Id": ids.astype(int), "Predicted": pred.astype(int)})
    out.to_csv(out_path, index=False)


def _canon_text_series(
    s: pd.Series,
    *,
    url_token: str = " __URL__ ",
    lowercase: bool = True,
    strip_accents: bool = True,
    placeholders: tuple[str, ...] = ("\\N",),
) -> pd.Series:
    """
    Canonicalize text in a way that matches your modeling pipeline reasonably well:
      - treat placeholders (e.g. \\N) as missing
      - HTML unescape
      - strip HTML tags
      - replace URLs with token
      - NFKC normalize
      - collapse whitespace
      - casefold (optional)
      - strip accents (optional, consistent with TfidfVectorizer(strip_accents="unicode"))
    """
    s = s.astype("string").fillna("")
    s = s.where(~s.str.strip().isin(placeholders), "")

    # Pandas has no vectorized html.unescape/unicodedata.normalize, so we map.
    s = s.map(_html.unescape)

    s = s.str.replace(_RE_TAGS, " ", regex=True)
    s = s.str.replace(_RE_URL, url_token, regex=True)

    s = s.map(lambda x: unicodedata.normalize("NFKC", x))

    # strip accents (unicode) to align with strip_accents="unicode"
    if strip_accents:
        # simple accent stripping via NFKD + removing combining marks
        def _strip_acc(x: str) -> str:
            x = unicodedata.normalize("NFKD", x)
            return "".join(ch for ch in x if not unicodedata.combining(ch))
        s = s.map(_strip_acc)

    s = s.str.replace(_RE_WS, " ", regex=True).str.strip()

    if lowercase:
        # align with TfidfVectorizer's default lowercasing
        s = s.str.lower()

    return s

def drop_dev_rows_overlapping_eval(
    dev_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    on: tuple[str, str] = ("title", "article"),
    url_token: str = " __URL__ ",
    lowercase: bool = True,
    strip_accents: bool = True,
    placeholders: tuple[str, ...] = ("\\N",),
) -> tuple[pd.DataFrame, dict]:
    """
    Remove from development ALL rows whose (on[0], on[1]) pair appears >=1 time in evaluation.
    Equality is checked after canonicalization (see _canon_text_series).

    Returns (filtered_dev_df, report_dict).
    """
    c1, c2 = on
    for c in (c1, c2):
        if c not in dev_df.columns:
            raise ValueError(f"dev_df missing column '{c}'")
        if c not in eval_df.columns:
            raise ValueError(f"eval_df missing column '{c}'")

    # Canonicalize
    dev_c1 = _canon_text_series(dev_df[c1], url_token=url_token, lowercase=lowercase,
                               strip_accents=strip_accents, placeholders=placeholders)
    dev_c2 = _canon_text_series(dev_df[c2], url_token=url_token, lowercase=lowercase,
                               strip_accents=strip_accents, placeholders=placeholders)

    eval_c1 = _canon_text_series(eval_df[c1], url_token=url_token, lowercase=lowercase,
                                strip_accents=strip_accents, placeholders=placeholders)
    eval_c2 = _canon_text_series(eval_df[c2], url_token=url_token, lowercase=lowercase,
                                strip_accents=strip_accents, placeholders=placeholders)

    # Row-wise hashes for (c1,c2) keys
    dev_key = pd.util.hash_pandas_object(pd.DataFrame({c1: dev_c1, c2: dev_c2}), index=False).to_numpy(np.uint64)
    eval_key = pd.util.hash_pandas_object(pd.DataFrame({c1: eval_c1, c2: eval_c2}), index=False).to_numpy(np.uint64)

    eval_key_u = np.unique(eval_key)
    leak_mask = np.isin(dev_key, eval_key_u)

    removed = int(leak_mask.sum())
    kept = int((~leak_mask).sum())

    out = dev_df.loc[~leak_mask].copy()

    report = {
        "on": on,
        "lowercase": lowercase,
        "strip_accents": strip_accents,
        "url_token": url_token,
        "n_dev_before": int(len(dev_df)),
        "n_eval": int(len(eval_df)),
        "n_removed_from_dev": removed,
        "n_dev_after": kept,
        "pct_removed_from_dev": 100.0 * removed / max(1, len(dev_df)),
    }
    return out, report
