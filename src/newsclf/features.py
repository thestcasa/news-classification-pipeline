from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TextJoiner(BaseEstimator, TransformerMixin):
    def __init__(self, title_col="title", article_col="article", title_repeat: int = 3):
        self.title_col = title_col
        self.article_col = article_col
        self.title_repeat = title_repeat

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        title = X[self.title_col].fillna("").astype(str).to_numpy()
        article = X[self.article_col].fillna("").astype(str).to_numpy()
        rep = np.array([(" ".join([t] * self.title_repeat)) for t in title], dtype=object)
        return np.array([f"{t} {a}" for t, a in zip(rep, article)], dtype=object)

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

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ts = pd.to_datetime(X[self.col], errors="coerce", utc=True, format="%Y-%m-%d %H:%M:%S.%f")

        year = ts.dt.year.fillna(1970).astype(int).to_numpy()
        month = ts.dt.month.fillna(1).astype(int).to_numpy()
        dow = ts.dt.dayofweek.fillna(0).astype(int).to_numpy()
        hour = ts.dt.hour.fillna(0).astype(int).to_numpy()

        # cyclic
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        dow_sin = np.sin(2 * np.pi * dow / 7.0)
        dow_cos = np.cos(2 * np.pi * dow / 7.0)

        out = np.vstack([year, month, dow, hour, hour_sin, hour_cos, dow_sin, dow_cos]).T
        return out.astype(float)
