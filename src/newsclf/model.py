from __future__ import annotations
from joblib import Memory

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC

from .features import (
    TextCleaner,
    TextJoiner,
    AdPrefixStripper,
    TextColumn,
    TextPatternFeatures,
    SourceBinner,
    NumericColumn,
    TimestampFeatures,
    TimestampMissingIndicator,
    MissingIndicator,
)


import numpy as np

def compute_balanced_class_weight(y: np.ndarray, power: float = 1.0) -> dict[int, float]:
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_samples = float(len(y))
    n_classes = float(len(classes))
    weights = n_samples / (n_classes * counts.astype(float))
    if power != 1.0:
        weights = np.power(weights, float(power))
    return {int(c): float(w) for c, w in zip(classes, weights)}

def build_pipeline(
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    min_df: int,
    title_repeat: int,
    max_df: float = 0.95,
    lowercase: bool = True,
    strip_accents: str | None = "unicode",
    sublinear_tf: bool = True,
    stop_words: str | list[str] | None = None,
    char_enabled: bool = True,
    char_ngram_min: int = 3,
    char_ngram_max: int = 5,
    char_min_df: int = 3,
    char_max_features: int | None = None,
    char_analyzer: str = "char_wb",
    title_char: bool = False,
    title_char_ngram_min: int = 2,
    title_char_ngram_max: int = 4,
    title_char_min_df: int = 5,
    title_char_max_features: int = 20000,
    source_min_count: int | None = 5,
    source_min_frac: float | None = None,
    model_type: str,
    C: float,
    max_iter: int,
    class_weight: str | None,
    logreg_solver: str = "liblinear",
    logreg_n_jobs: int = 1,
    linearsvc_dual: bool = False,
    ridge_alpha: float = 1.0,
    cache_dir: str | None = None,
    missing_article_token: str | None = "__MISSING_ARTICLE__",
):
    
    # PREPROCESSING
    # text -> tfidf
    tfidf_union = [
        ("word", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,  # es: (1,2)
            min_df=min_df,
            max_df=max_df,
            strip_accents=strip_accents,
            lowercase=lowercase,
            sublinear_tf=sublinear_tf,
            stop_words=stop_words,
        )),
    ]
    if char_enabled:
        char_max = int(char_max_features) if char_max_features is not None else max(20000, max_features // 4)
        tfidf_union.append(
            ("char", TfidfVectorizer(
                analyzer=char_analyzer,
                ngram_range=(int(char_ngram_min), int(char_ngram_max)),  # buon default
                min_df=int(char_min_df),
                max_df=max_df,
                max_features=char_max,  # limita costo
                strip_accents=strip_accents,
                lowercase=lowercase,
                sublinear_tf=sublinear_tf,
            ))
        )
    text_pipe = Pipeline(steps=[
        ("ad_strip", AdPrefixStripper()),
        ("join", TextJoiner(title_repeat=title_repeat, missing_article_token=missing_article_token)),
        ("clean", TextCleaner()),
        ("tfidf_union", FeatureUnion(transformer_list=tfidf_union)),
        
    ])

    text_meta_pipe = Pipeline(steps=[
        ("meta", TextPatternFeatures()),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    title_char_pipe = None
    if title_char:
        title_char_pipe = Pipeline(steps=[
            ("col", TextColumn("title")),
            ("clean", TextCleaner()),
            ("tfidf", TfidfVectorizer(
                analyzer=char_analyzer,
                ngram_range=(int(title_char_ngram_min), int(title_char_ngram_max)),
                min_df=int(title_char_min_df),
                max_df=max_df,
                max_features=int(title_char_max_features),
                strip_accents=strip_accents,
                lowercase=lowercase,
                sublinear_tf=sublinear_tf,
            )),
        ])

    # numeric -> scaler (with_mean=False keeps sparse compatibility)
    page_rank_pipe = Pipeline(steps=[
        ("num", NumericColumn("page_rank")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    # timestamp -> numeric features + scaler
    time_pipe = Pipeline(steps=[
        ("time", TimestampFeatures("timestamp", include_missing=False)),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    # timestamp missingness -> binary
    timestamp_missing_pipe = Pipeline(steps=[
        ("missing", TimestampMissingIndicator("timestamp")),
    ])

    # article missingness -> binary + scaler
    article_missing_pipe = Pipeline(steps=[
        ("missing", MissingIndicator("article", placeholders=["\\N"])),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    source_pipe = Pipeline(steps=[
        ("bucket", SourceBinner(
            col="source",
            min_count=source_min_count,
            min_frac=source_min_frac,
        )),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = [
        ("text", text_pipe, ["title", "article"]),
    ]
    if title_char_pipe is not None:
        transformers.append(("title_char", title_char_pipe, ["title"]))
    transformers += [
        ("text_meta", text_meta_pipe, ["title", "article"]),
        ("source", source_pipe, ["source"]),
        ("page_rank", page_rank_pipe, ["page_rank"]),
        ("time", time_pipe, ["timestamp"]),
        ("timestamp_missing", timestamp_missing_pipe, ["timestamp"]),
        ("article_missing", article_missing_pipe, ["article"]),
    ]

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


    # CLASSIFIER
    if model_type == "logreg":
        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            n_jobs=logreg_n_jobs,
            solver=logreg_solver,

        )
    elif model_type == "linearsvc":
        clf = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            dual=linearsvc_dual,
        )

    elif model_type == "ridge":
        # strong/fast linear baseline for high-dimenisonal sparse text
        clf = RidgeClassifier(
            alpha=ridge_alpha,
            class_weight=class_weight,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # memory cache should make grid-search faster (avoids recomputing transforms)
    mem = Memory(location=cache_dir, verbose=0) if cache_dir else None

    return Pipeline(steps=[("pre", pre), ("clf", clf)], memory=mem, verbose=True)
