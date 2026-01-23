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
    TextColumn,
    NumericColumn,
    TimestampFeatures,
    TimestampMissingIndicator,
    MissingIndicator,
)


import numpy as np

def build_pipeline(
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    min_df: int,
    title_repeat: int,
    title_char: bool = False,
    title_char_ngram_min: int = 2,
    title_char_ngram_max: int = 4,
    title_char_min_df: int = 5,
    title_char_max_features: int = 20000,
    model_type: str,
    C: float,
    max_iter: int,
    class_weight: str | None,
    cache_dir: str | None = None,
    missing_article_token: str | None = "__MISSING_ARTICLE__",
):
    
    # PREPROCESSING
    # text -> tfidf
    text_pipe = Pipeline(steps=[
        ("join", TextJoiner(title_repeat=title_repeat, missing_article_token=missing_article_token)),
        ("clean", TextCleaner()),
        ("tfidf_union", FeatureUnion(transformer_list=[
            ("word", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,  # es: (1,2)
                min_df=min_df,
                max_df=0.95,
                strip_accents="unicode",
                lowercase=True,
                sublinear_tf=True,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),  # buon default
                min_df=3,
                max_df=0.95,
                max_features=max(20000, max_features // 4),  # limita costo
                strip_accents="unicode",
                lowercase=True,
                sublinear_tf=True,
            )),
        ])),
        
    ])

    title_char_pipe = None
    if title_char:
        title_char_pipe = Pipeline(steps=[
            ("col", TextColumn("title")),
            ("clean", TextCleaner()),
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(int(title_char_ngram_min), int(title_char_ngram_max)),
                min_df=int(title_char_min_df),
                max_df=0.95,
                max_features=int(title_char_max_features),
                strip_accents="unicode",
                lowercase=True,
                sublinear_tf=True,
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

    transformers = [
        ("text", text_pipe, ["title", "article"]),
    ]
    if title_char_pipe is not None:
        transformers.append(("title_char", title_char_pipe, ["title"]))
    transformers += [
        ("source", OneHotEncoder(handle_unknown="ignore"), ["source"]),
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
            n_jobs=1,      
            solver="liblinear",

        )
    elif model_type == "linearsvc":
        clf = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            dual=False,
        )

    elif model_type == "ridge":
        # strong/fast linear baseline for high-dimenisonal sparse text
        clf = RidgeClassifier(
            alpha=1.0,
            class_weight=class_weight,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # memory cache should make grid-search faster (avoids recomputing transforms)
    mem = Memory(location=cache_dir, verbose=0) if cache_dir else None

    return Pipeline(steps=[("pre", pre), ("clf", clf)], memory=mem, verbose=True)
