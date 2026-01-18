from __future__ import annotations

import numpy as np
import pandas as pd
import html as _html
import unicodedata
import re

from sklearn.base import BaseEstimator, TransformerMixin

_RE_TAGS = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_RE_WS = re.compile(r"\s+")

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
    def __init__(self, title_col="title", article_col="article", title_repeat: int = 5):
        self.title_col = title_col
        self.article_col = article_col
        self.title_repeat = title_repeat

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        title = X[self.title_col].fillna("").astype(str)
        article = X[self.article_col].fillna("").astype(str)
        # Vectorized title emphasis (avoids Python loops)
        title_rep = (title + " ") * int(self.title_repeat)        
        out = (title_rep + article).to_numpy(dtype=object)
        return out

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
    def __init__(self, col: str = "timestamp"):
        self.col = col
        # learned fill values from training data (per-CV-fold)
        self._fill_year = 1970
        self._fill_month = 1
        self._fill_dow = 0
        self._fill_hour = 0

    def fit(self, X, y=None):
        ts = self._parse_ts(X)
        valid = ts.notna()
        if valid.any():
            self._fill_year = int(ts.dt.year[valid].median())
            self._fill_month = int(ts.dt.month[valid].median())
            self._fill_dow = int(ts.dt.dayofweek[valid].median())
            self._fill_hour = int(ts.dt.hour[valid].median())
        return self
        
    def _parse_ts(self, X):
        raw = X[self.col]
        # normalize to string, but keep missing as NA
        raw = raw.astype("string")
        raw = raw.replace("0000-00-00 00:00:00", pd.NA)
        # IMPORTANT: do NOT force a single format; dataset contains seconds-only timestamps
        return pd.to_datetime(raw, errors="coerce", utc=True)


    def transform(self, X):
        ts = self._parse_ts(X)
        is_missing = ts.isna().astype(float).to_numpy().reshape(-1, 1)
        
        year = ts.dt.year.fillna(self._fill_year).astype(int).to_numpy()
        month = ts.dt.month.fillna(self._fill_month).astype(int).to_numpy()
        dow = ts.dt.dayofweek.fillna(self._fill_dow).astype(int).to_numpy()
        hour = ts.dt.hour.fillna(self._fill_hour).astype(int).to_numpy()
        doy = ts.dt.dayofyear.fillna(1).astype(int).to_numpy()

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
        return np.hstack([is_missing.astype(np.float32), out]).astype(np.float32)
