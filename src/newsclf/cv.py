from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from .io import read_development
from .model import build_pipeline

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="data/raw/development.csv")
    ap.add_argument("--out_dir", default="reports/cv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)

    ap.add_argument("--max_features", type=int, default=200000)
    ap.add_argument("--ngram_min", type=int, default=1)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--title_repeat", type=int, default=3)

    ap.add_argument("--model_type", choices=["logreg", "linearsvc"], default="logreg")
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--class_weight", default="balanced")  # "balanced" or "none"
    return ap.parse_args()

def main():
    args = parse_args()
    df = read_development(args.dev)

    X = df.drop(columns=["label"])
    y = df["label"].astype(int).to_numpy()

    cw = None if args.class_weight == "none" else args.class_weight

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    per_class_reports = []
    cm_sum = None

    labels_sorted = np.sort(np.unique(y))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr = X.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y[va_idx]

        pipe = build_pipeline(
            max_features=args.max_features,
            ngram_range=(args.ngram_min, args.ngram_max),
            min_df=args.min_df,
            title_repeat=args.title_repeat,
            model_type=args.model_type,
            C=args.C,
            max_iter=args.max_iter,
            class_weight=cw,
        )

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)

        macro = f1_score(y_va, pred, average="macro")
        micro = f1_score(y_va, pred, average="micro")
        weighted = f1_score(y_va, pred, average="weighted")

        fold_rows.append({
            "fold": fold,
            "macro_f1": float(macro),
            "micro_f1": float(micro),
            "weighted_f1": float(weighted),
            "n_train": int(len(tr_idx)),
            "n_valid": int(len(va_idx)),
        })

        # per-class f1 for this fold
        rep = classification_report(
            y_va, pred,
            labels=labels_sorted.tolist(),
            output_dict=True,
            zero_division=0,
        )
        # keep only per-class rows (keys are class labels as strings)
        for lab in labels_sorted:
            k = str(int(lab))
            per_class_reports.append({
                "fold": fold,
                "label": int(lab),
                "precision": float(rep[k]["precision"]),
                "recall": float(rep[k]["recall"]),
                "f1": float(rep[k]["f1-score"]),
                "support": int(rep[k]["support"]),
            })

        cm = confusion_matrix(y_va, pred, labels=labels_sorted)
        cm_sum = cm if cm_sum is None else (cm_sum + cm)

        print(f"[fold {fold}/{args.k}] macro_f1={macro:.5f}")

    folds_df = pd.DataFrame(fold_rows).sort_values("fold")
    folds_df.to_csv(out_dir / "folds.csv", index=False)

    per_class_df = pd.DataFrame(per_class_reports)
    # mean per class across folds (precision/recall/f1)
    per_class_mean = (
        per_class_df
        .groupby("label")[["precision", "recall", "f1"]]
        .mean()
        .reset_index()
        .sort_values("label")
    )
    per_class_mean.to_csv(out_dir / "per_class_mean.csv", index=False)

    # aggregated confusion matrix
    cm_df = pd.DataFrame(cm_sum, index=labels_sorted, columns=labels_sorted)
    cm_df.index.name = "true"
    cm_df.columns.name = "pred"
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    summary = {
        "k": int(args.k),
        "seed": int(args.seed),
        "model_type": args.model_type,
        "C": float(args.C),
        "max_iter": int(args.max_iter),
        "class_weight": None if cw is None else str(cw),
        "text": {
            "max_features": int(args.max_features),
            "ngram_range": [int(args.ngram_min), int(args.ngram_max)],
            "min_df": int(args.min_df),
            "title_repeat": int(args.title_repeat),
        },
        "macro_f1_mean": float(folds_df["macro_f1"].mean()),
        "macro_f1_std": float(folds_df["macro_f1"].std(ddof=1)) if len(folds_df) > 1 else 0.0,
        "micro_f1_mean": float(folds_df["micro_f1"].mean()),
        "weighted_f1_mean": float(folds_df["weighted_f1"].mean()),
    }
    with open(out_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nCV done.")
    print(f"macro_f1 mean={summary['macro_f1_mean']:.5f} std={summary['macro_f1_std']:.5f}")
    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    main()
