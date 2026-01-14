from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from .config import load_config, parse_overrides
from .io import read_development
from .model import build_pipeline
from .plots import plot_confusion_matrix, plot_folds_macro, plot_per_class_f1

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--set", action="append", default=[])
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=parse_overrides(args.set))

    df = read_development(cfg.paths.dev_csv)
    X = df.drop(columns=["label"])
    y = df["label"].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=cfg.cv.k, shuffle=True, random_state=cfg.cv.seed)
    out_dir = Path(cfg.paths.cv_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_sorted = np.sort(np.unique(y))

    fold_rows = []
    per_class_rows = []
    cm_sum = None

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        pipe = build_pipeline(
            max_features=cfg.text.max_features,
            ngram_range=(cfg.text.ngram_min, cfg.text.ngram_max),
            min_df=cfg.text.min_df,
            title_repeat=cfg.text.title_repeat,
            model_type=cfg.model.type,
            C=cfg.model.C,
            max_iter=cfg.model.max_iter,
            class_weight=cfg.model.class_weight
        )

        print(f"[fold {fold}/{cfg.cv.k}] start fit", flush=True)
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

        rep = classification_report(
            y_va, pred,
            labels=labels_sorted.tolist(),
            output_dict=True,
            zero_division=0
        )
        for lab in labels_sorted:
            k = str(int(lab))
            per_class_rows.append({
                "fold": fold,
                "label": int(lab),
                "precision": float(rep[k]["precision"]),
                "recall": float(rep[k]["recall"]),
                "f1": float(rep[k]["f1-score"]),
                "support": int(rep[k]["support"]),
            })

        cm = confusion_matrix(y_va, pred, labels=labels_sorted)
        cm_sum = cm if cm_sum is None else (cm_sum + cm)

        print(f"[fold {fold}/{cfg.cv.k}] macro_f1={macro:.5f}", flush=True)

    folds_df = pd.DataFrame(fold_rows).sort_values("fold")
    folds_df.to_csv(out_dir / "folds.csv", index=False)

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_mean = (
        per_class_df.groupby("label")[["precision", "recall", "f1"]]
        .mean().reset_index().sort_values("label")
    )
    per_class_mean.to_csv(out_dir / "per_class_mean.csv", index=False)

    cm_df = pd.DataFrame(cm_sum, index=labels_sorted, columns=labels_sorted)
    cm_df.index.name = "true"
    cm_df.columns.name = "pred"
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    summary = {
        "config": str(args.config),
        "overrides": args.set,
        "k": cfg.cv.k,
        "seed": cfg.cv.seed,
        "model": {
            "type": cfg.model.type,
            "C": cfg.model.C,
            "max_iter": cfg.model.max_iter,
            "class_weight": cfg.model.class_weight
        },
        "text": {
            "max_features": cfg.text.max_features,
            "ngram_range": [cfg.text.ngram_min, cfg.text.ngram_max],
            "min_df": cfg.text.min_df,
            "title_repeat": cfg.text.title_repeat
        },
        "macro_f1_mean": float(folds_df["macro_f1"].mean()),
        "macro_f1_std": float(folds_df["macro_f1"].std(ddof=1)) if len(folds_df) > 1 else 0.0
    }
    with open(out_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # plots
    plot_folds_macro(out_dir / "folds.csv", out_dir / "folds_macro_f1.png")
    plot_per_class_f1(out_dir / "per_class_mean.csv", out_dir / "per_class_f1.png")
    plot_confusion_matrix(out_dir / "confusion_matrix.csv", out_dir / "confusion_matrix.png")

    print("\nCV done.")
    print(f"macro_f1 mean={summary['macro_f1_mean']:.5f} std={summary['macro_f1_std']:.5f}")
    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    main()
