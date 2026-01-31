from __future__ import annotations

import numpy as np
import pandas as pd
import html as _html
import unicodedata
import re

from sklearn.base import BaseEstimator, TransformerMixin

_RE_TAGS = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)
_RE_WS = re.compile(r"\s+")
_RE_ADV_PREFIX = re.compile(
    r"^\s*(?:adv\s*[:\-]|advertisement\s*[:\-]?|sponsored\s*(?:content)?\s*[:\-]?)",
    re.IGNORECASE,
)
_RE_AMP_ENTITY = re.compile(r"&(?:nbsp|amp|lt|gt|quot|#\d+);", re.IGNORECASE)
_RE_NON_ASCII = re.compile(r"[^\x00-\x7F]")
_TS_MISSING_TOKENS = {
    "",
    "NA",
    "N/A",
    "null",
    "None",
    "0000-00-00 00:00:00",
    "0000-00-00",
}


def _parse_timestamp(raw: pd.Series) -> pd.Series:
    raw = raw.astype("string").str.strip()
    raw = raw.replace({tok: pd.NA for tok in _TS_MISSING_TOKENS})
    return pd.to_datetime(raw, errors="coerce", utc=True)

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Lightweight normalization for noisy news feeds:
      - HTML entity unescape (e.g., &#39;, &quot;)
      - strip HTML tags
      - replace URLs with a placeholder token
      - unicode normalization + whitespace cleanup
    Keeps content words intact (good for TF-IDF + linear models).
    """

    def __init__(self, url_token: str = " __URL__ "):
        self.url_token = url_token

    def fit(self, X, y=None):
        return self
    
    def _clean_one(self,  s:str) -> str:
        if s is None:
            return ""
        s = str(s)
        s = _html.unescape(s)
        s = _RE_TAGS.sub(" ", s)                 # strip HTML tags
        s = _RE_URL.sub(self.url_token, s)
        s = unicodedata.normalize("NFKC", s)
        # drop non-printing chars (keep whitespace/punct)
        s = "".join(ch if (ch.isprintable() or ch.isspace()) else " " for ch in s)
        s = _RE_WS.sub(" ", s).strip()
        return s
    
    def transform(self, X):
        # X is expected to be an array-like of strings (output of TextJoiner)
        ser = pd.Series(X, dtype="object")
        return ser.map(self._clean_one).to_numpy(dtype=object)


class TextJoiner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        title_col="title",
        article_col="article",
        title_repeat: int = 5,
        missing_article_token: str | None = "__MISSING_ARTICLE__",
    ):
        self.title_col = title_col
        self.article_col = article_col
        self.title_repeat = title_repeat
        self.missing_article_token = missing_article_token

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        title = X[self.title_col].fillna("").astype(str)
        article = X[self.article_col].fillna("").astype(str)
        article_norm = article.str.strip()
        is_missing = article_norm.eq("") | article_norm.eq("\\N")
        article = article.where(~is_missing, "")
        # Vectorized title emphasis (avoids Python loops)
        title_rep = (title + " ") * int(self.title_repeat)
        out = title_rep + article
        if self.missing_article_token:
            token = f" {self.missing_article_token} "
            out = out + np.where(is_missing, token, "")
        out = out.to_numpy(dtype=object)
        return out


class AdPrefixStripper(BaseEstimator, TransformerMixin):
    def __init__(self, title_col: str = "title", article_col: str = "article"):
        self.title_col = title_col
        self.article_col = article_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in (self.title_col, self.article_col):
            if col not in X.columns:
                continue
            s = X[col].astype("string").fillna("")
            X[col] = s.str.replace(_RE_ADV_PREFIX, "", regex=True)
        return X


class TextColumn(BaseEstimator, TransformerMixin):
    def __init__(self, col: str, missing_token: str | None = None):
        self.col = col
        self.missing_token = missing_token

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = X[self.col].fillna("").astype(str)
        if self.missing_token:
            s_norm = s.str.strip()
            s = s.where(~s_norm.isin(["", "\\N"]), other=self.missing_token)
        return s.to_numpy(dtype=object)


class TextPatternFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, title_col: str = "title", article_col: str = "article"):
        self.title_col = title_col
        self.article_col = article_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        title = self._prep(X[self.title_col])
        article = self._prep(X[self.article_col])

        title_chars = title.str.len().to_numpy(dtype=np.float32)
        article_chars = article.str.len().to_numpy(dtype=np.float32)
        title_words = title.str.split().str.len().to_numpy(dtype=np.float32)
        article_words = article.str.split().str.len().to_numpy(dtype=np.float32)

        title_adv = title.str.contains(_RE_ADV_PREFIX, regex=True, na=False)
        article_adv = article.str.contains(_RE_ADV_PREFIX, regex=True, na=False)
        any_adv = (title_adv | article_adv).to_numpy(dtype=np.float32)

        title_url = title.str.contains(_RE_URL, regex=True, na=False)
        article_url = article.str.contains(_RE_URL, regex=True, na=False)
        any_url = (title_url | article_url).to_numpy(dtype=np.float32)

        title_html = title.str.contains(_RE_TAGS, regex=True, na=False) | title.str.contains(
            _RE_AMP_ENTITY, regex=True, na=False
        )
        article_html = article.str.contains(_RE_TAGS, regex=True, na=False) | article.str.contains(
            _RE_AMP_ENTITY, regex=True, na=False
        )
        any_html = (title_html | article_html).to_numpy(dtype=np.float32)

        title_non_ascii = title.str.contains(_RE_NON_ASCII, regex=True, na=False)
        article_non_ascii = article.str.contains(_RE_NON_ASCII, regex=True, na=False)
        any_non_ascii = (title_non_ascii | article_non_ascii).to_numpy(dtype=np.float32)

        feats = np.vstack(
            [
                title_chars,
                article_chars,
                title_words,
                article_words,
                title_adv.to_numpy(dtype=np.float32),
                article_adv.to_numpy(dtype=np.float32),
                any_adv,
                title_url.to_numpy(dtype=np.float32),
                article_url.to_numpy(dtype=np.float32),
                any_url,
                title_html.to_numpy(dtype=np.float32),
                article_html.to_numpy(dtype=np.float32),
                any_html,
                any_non_ascii,
            ]
        ).T
        return feats

    @staticmethod
    def _prep(s: pd.Series) -> pd.Series:
        s = s.astype("string").fillna("")
        s = s.where(~s.str.strip().eq("\\N"), other="")
        return s


class SourceBinner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        col: str = "source",
        min_count: int | None = 5,
        min_frac: float | None = None,
        other_token: str = "__OTHER__",
        missing_token: str = "__MISSING__",
    ):
        self.col = col
        self.min_count = min_count
        self.min_frac = min_frac
        self.other_token = other_token
        self.missing_token = missing_token
        self._keep: set[str] | None = None

    def fit(self, X, y=None):
        s = self._normalize(X)
        counts = s.value_counts()
        thr = 1
        if self.min_frac is not None:
            thr = max(thr, int(np.ceil(float(self.min_frac) * len(s))))
        if self.min_count is not None:
            thr = max(thr, int(self.min_count))
        self._keep = set(counts[counts >= thr].index.astype(str).tolist())
        return self

    def transform(self, X):
        s = self._normalize(X)
        keep = self._keep or set()
        out = s.where(s.isin(keep), other=self.other_token)
        return out.to_numpy(dtype=object).reshape(-1, 1)

    def _normalize(self, X) -> pd.Series:
        s = X[self.col].astype("string")
        s_norm = s.fillna("").str.strip()
        s_norm = s_norm.replace({"\\N": ""})
        s_norm = s_norm.where(s_norm.ne(""), other=self.missing_token)
        return s_norm.astype(str)

class NumericColumn(BaseEstimator, TransformerMixin):
    
    def __init__(self, col: str):
         self.col = col

    def fit(self, X, y=None):
         return self

    def transform(self, X):
        v = pd.to_numeric(X[self.col], errors="coerce").fillna(0.0).to_numpy()
        return v.reshape(-1, 1)

class TimestampFeatures(BaseEstimator, TransformerMixin):
    """
    Extract simple, robust datetime features:
    - year, month, dayofweek, hour
    - plus cyclic encodings for hour and dayofweek (helps linear models)
    """
    def __init__(self, col: str = "timestamp", include_missing: bool = True):
        self.col = col
        self.include_missing = include_missing
        # learned fill values from training data (per-CV-fold)
        self._fill_year = 1970
        self._fill_month = 1
        self._fill_dow = 0
        self._fill_hour = 0
        self._fill_doy = 1

    def fit(self, X, y=None):
        ts = self._parse_ts(X)
        valid = ts.notna()
        if valid.any():
            self._fill_year = int(ts.dt.year[valid].median())
            self._fill_month = int(ts.dt.month[valid].median())
            self._fill_dow = int(ts.dt.dayofweek[valid].median())
            self._fill_hour = int(ts.dt.hour[valid].median())
            self._fill_doy = int(ts.dt.dayofyear[valid].median())
        return self
        
    def _parse_ts(self, X):
        raw = X[self.col]
        return _parse_timestamp(raw)


    def transform(self, X):
        ts = self._parse_ts(X)
        is_missing = ts.isna().astype(float).to_numpy().reshape(-1, 1)
        
        year = ts.dt.year.fillna(self._fill_year).astype(int).to_numpy()
        month = ts.dt.month.fillna(self._fill_month).astype(int).to_numpy()
        dow = ts.dt.dayofweek.fillna(self._fill_dow).astype(int).to_numpy()
        hour = ts.dt.hour.fillna(self._fill_hour).astype(int).to_numpy()
        doy = ts.dt.dayofyear.fillna(self._fill_doy).astype(int).to_numpy()

        # cyclic
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        dow_sin = np.sin(2 * np.pi * dow / 7.0)
        dow_cos = np.cos(2 * np.pi * dow / 7.0)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12.0)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12.0)
        doy_sin = np.sin(2 * np.pi * (doy - 1) / 365.0)
        doy_cos = np.cos(2 * np.pi * (doy - 1) / 365.0)

        out = np.vstack([
            year, month, doy, dow,
            hour_sin, hour_cos, dow_sin, dow_cos,
            month_sin, month_cos, doy_sin, doy_cos
        ]).T.astype(np.float32)
        if self.include_missing:
            return np.hstack([is_missing.astype(np.float32), out]).astype(np.float32)
        return out.astype(np.float32)


class TimestampMissingIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, col: str = "timestamp"):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ts = _parse_timestamp(X[self.col])
        missing = ts.isna().astype(float).to_numpy().reshape(-1, 1)
        return missing


class MissingIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, col: str, placeholders: list[str] | None = None):
        self.col = col
        self.placeholders = placeholders if placeholders is not None else ["\\N"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = X[self.col].astype("string")
        s_norm = s.fillna("").str.strip()
        placeholder = s_norm.isin(self.placeholders)
        missing = s.isna() | s_norm.eq("") | placeholder
        return missing.astype(float).to_numpy().reshape(-1, 1)
