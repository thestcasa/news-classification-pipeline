from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from .features import TextJoiner, DatetimeFeaturizer, NumericColumn

def build_pipeline(
    *,
    max_features: int = 200000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    title_repeat: int = 3,
    model_type: str = "logreg",
    C: float = 4.0,
    max_iter: int = 2000,
    class_weight: str | None = "balanced",
):
    text_pipe = Pipeline(steps=[
        ("join", TextJoiner(title_repeat=title_repeat)),
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            strip_accents="unicode",
            lowercase=True,
        )),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("text", text_pipe, ["title", "article"]),
            ("source", OneHotEncoder(handle_unknown="ignore"), ["source"]),
            ("page_rank", Pipeline([
                ("num", NumericColumn("page_rank")),
                ("sc", StandardScaler(with_mean=False)),
            ]), ["page_rank"]),
            ("time", DatetimeFeaturizer("timestamp"), ["timestamp"]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    if model_type == "logreg":
        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            n_jobs=-1,
        )
    elif model_type == "linearsvc":
        base = LinearSVC(C=C, class_weight=class_weight)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline(steps=[("pre", pre), ("clf", clf)])
