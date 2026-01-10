from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TextJoiner(BaseEstimator, TransformerMixin):
    """
    Join title + article; repeat title to weight it (cheap + effective).
    """
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

class DatetimeFeaturizer(BaseEstimator, TransformerMixin):
    """
    Extract cyclic features from timestamp (hour, day-of-week, month).
    Output: dense numpy (n, 6)
    """
    def __init__(self, col="timestamp"):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ts = pd.to_datetime(X[self.col], errors="coerce", utc=True)
        hour = ts.dt.hour.fillna(0).astype(int).to_numpy()
        dow = ts.dt.dayofweek.fillna(0).astype(int).to_numpy()
        month = ts.dt.month.fillna(1).astype(int).to_numpy()

        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        dow_sin = np.sin(2 * np.pi * dow / 7.0)
        dow_cos = np.cos(2 * np.pi * dow / 7.0)
        mon_sin = np.sin(2 * np.pi * month / 12.0)
        mon_cos = np.cos(2 * np.pi * month / 12.0)

        return np.vstack([hour_sin, hour_cos, dow_sin, dow_cos, mon_sin, mon_cos]).T

class NumericColumn(BaseEstimator, TransformerMixin):
    def __init__(self, col: str):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        v = pd.to_numeric(X[self.col], errors="coerce").fillna(0.0).to_numpy()
        return v.reshape(-1, 1)
