from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from newsclf.config import load_config, parse_overrides  # noqa: E402
from newsclf.io import (
    read_development,
    read_evaluation,
    write_submission,
    drop_dev_rows_overlapping_eval,
    drop_cross_label_duplicates,
)  # noqa: E402
from newsclf.model import build_pipeline, compute_balanced_class_weight  # noqa: E402
from newsclf.plots import plot_confusion_matrix, plot_folds_macro, plot_per_class_f1  # noqa: E402


def _now_tag() -> str:
    # filesystem-friendly timestamp
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(obj: Any, path: Path) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        _ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def _print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _summarize_labels(y: np.ndarray) -> pd.DataFrame:
    s = pd.Series(y).value_counts().sort_index()
    out = pd.DataFrame({"label": s.index.astype(int), "count": s.values})
    out["share"] = out["count"] / float(out["count"].sum())
    return out


def _resolve_class_weight(cfg, y: np.ndarray):
    cw = cfg.model.class_weight
    if cw is None:
        return None
    if cw != "balanced":
        return cw
    power = float(cfg.model.class_weight_power)
    if power == 1.0:
        return "balanced"
    return compute_balanced_class_weight(y, power=power)


def run_cv(
    *,
    cfg,
    out_dir: Path,
    cache_dir: str | None,
) -> None:
    _print_header("STAGE 1/2 — Cross-validation on development set")
    t0 = time.perf_counter()

    df_dev = read_development(cfg.paths.dev_csv)

    eval_df = read_evaluation(cfg.paths.eval_csv)
    df_dev, leak_report = drop_dev_rows_overlapping_eval(
        df_dev,
        eval_df,
        on=("title", "article"),
        lowercase=True,
        strip_accents=True,
        drop=False,
    )

    print("[leakage]", leak_report)
    df_dev, dup_report = drop_cross_label_duplicates(df_dev, on=("title", "article"))
    print("[dedup]", dup_report)
    # -------------------------------------------------

    X = df_dev.drop(columns=["label"])
    y = df_dev["label"].astype(int).to_numpy()

    print(f"[cv] dev_csv={cfg.paths.dev_csv}")
    print(f"[cv] n_samples={len(df_dev):,}  n_features_raw={X.shape[1]}")
    print(f"[cv] k={cfg.cv.k} seed={cfg.cv.seed}")
    print(
        f"[cv] model={cfg.model.type}  C={cfg.model.C}  max_iter={cfg.model.max_iter}  "
        f"class_weight={cfg.model.class_weight}  class_weight_power={cfg.model.class_weight_power}"
    )
    print(f"[cv] source: min_count={cfg.source.min_count}  min_frac={cfg.source.min_frac}")
    print(
        f"[cv] text: max_features={cfg.text.max_features}  ngram=({cfg.text.ngram_min},{cfg.text.ngram_max})  "
        f"min_df={cfg.text.min_df}  max_df={cfg.text.max_df}  title_repeat={cfg.text.title_repeat}  "
        f"missing_article_token={cfg.text.missing_article_token}  "
        f"stop_words={cfg.text.stop_words}  "
        f"char_enabled={cfg.text.char_enabled}  char_analyzer={cfg.text.char_analyzer}  "
        f"char_ngram=({cfg.text.char_ngram_min},{cfg.text.char_ngram_max})  char_min_df={cfg.text.char_min_df}  "
        f"char_max_features={cfg.text.char_max_features}  "
        f"title_char={cfg.text.title_char}  title_char_ngram=({cfg.text.title_char_ngram_min},{cfg.text.title_char_ngram_max})  "
        f"title_char_min_df={cfg.text.title_char_min_df}  title_char_max_features={cfg.text.title_char_max_features}"
    )

    lab_df = _summarize_labels(y)
    print("\n[cv] label distribution:")
    print(lab_df.to_string(index=False))

    _ensure_dir(out_dir)

    labels_sorted = np.sort(np.unique(y))
    skf = StratifiedKFold(n_splits=cfg.cv.k, shuffle=True, random_state=cfg.cv.seed)

    fold_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    cm_sum: np.ndarray | None = None
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        fold_t0 = time.perf_counter()
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        cw = _resolve_class_weight(cfg, y_tr)
        
        pipe = build_pipeline(
            max_features=cfg.text.max_features,
            ngram_range=(cfg.text.ngram_min, cfg.text.ngram_max),
            min_df=cfg.text.min_df,
            title_repeat=cfg.text.title_repeat,
            missing_article_token=cfg.text.missing_article_token,
            max_df=cfg.text.max_df,
            lowercase=cfg.text.lowercase,
            strip_accents=cfg.text.strip_accents,
            sublinear_tf=cfg.text.sublinear_tf,
            stop_words=cfg.text.stop_words,
            char_enabled=cfg.text.char_enabled,
            char_analyzer=cfg.text.char_analyzer,
            char_ngram_min=cfg.text.char_ngram_min,
            char_ngram_max=cfg.text.char_ngram_max,
            char_min_df=cfg.text.char_min_df,
            char_max_features=cfg.text.char_max_features,
            title_char=cfg.text.title_char,
            title_char_ngram_min=cfg.text.title_char_ngram_min,
            title_char_ngram_max=cfg.text.title_char_ngram_max,
            title_char_min_df=cfg.text.title_char_min_df,
            title_char_max_features=cfg.text.title_char_max_features,
            source_min_count=cfg.source.min_count,
            source_min_frac=cfg.source.min_frac,
            model_type=cfg.model.type,
            C=cfg.model.C,
            max_iter=cfg.model.max_iter,
            class_weight=cw,
            logreg_solver=cfg.model.logreg_solver,
            logreg_n_jobs=cfg.model.logreg_n_jobs,
            linearsvc_dual=cfg.model.linearsvc_dual,
            ridge_alpha=cfg.model.ridge_alpha,
            cache_dir=cache_dir,
        )

        print(f"\n[cv][fold {fold}/{cfg.cv.k}] fit start  (n_train={len(tr_idx):,} n_valid={len(va_idx):,})", flush=True)

        from time import perf_counter
        import os

        debug_pre = os.getenv("NEWSCLF_DEBUG_PRE", "0") == "1"
        t = perf_counter()
        pipe.fit(X_tr, y_tr)  
        print(f"[debug] pipe.fit: {perf_counter()-t:.2f}s")
        pre = pipe.named_steps["pre"]
        if debug_pre:
            t = perf_counter()
            Xtr = pre.transform(X_tr)
            print(f"[debug] pre.transform(train): {perf_counter()-t:.2f}s")
            print(f"[debug] Xtr shape={Xtr.shape}, nnz={Xtr.nnz:,}, dtype={Xtr.dtype}")
            print(f"[debug] density={Xtr.nnz/(Xtr.shape[0]*Xtr.shape[1]):.6e}")

        t = perf_counter()
        pred = pipe.predict(X_va)
        print(f"[debug] pipe.predict: {perf_counter()-t:.2f}s")
        if debug_pre:
            t = perf_counter()
            Xva = pre.transform(X_va)
            print(f"[debug] pre.transform(valid): {perf_counter()-t:.2f}s")
            print(f"[debug] Xva shape={Xva.shape}, nnz={Xva.nnz:,}, dtype={Xva.dtype}")

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
            "seconds": float(time.perf_counter() - fold_t0),
        })

        rep = classification_report(
            y_va, pred,
            labels=labels_sorted.tolist(),
            output_dict=True,
            zero_division=0,
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
        y_true_all.append(y_va)
        y_pred_all.append(pred)

        print(f"[cv][fold {fold}/{cfg.cv.k}] macro_f1={macro:.5f}  fold_time={fold_rows[-1]['seconds']:.1f}s", flush=True)

    _ensure_dir(out_dir)
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

    if y_true_all:
        y_true_cat = np.concatenate(y_true_all)
        y_pred_cat = np.concatenate(y_pred_all)
        true_counts = pd.Series(y_true_cat).value_counts().sort_index()
        pred_counts = pd.Series(y_pred_cat).value_counts().reindex(true_counts.index, fill_value=0)
        counts_df = pd.DataFrame(
            {
                "label": true_counts.index.astype(int),
                "true_count": true_counts.values,
                "pred_count": pred_counts.values,
            }
        )
        counts_df["pred/true"] = counts_df["pred_count"] / counts_df["true_count"].replace(0, np.nan)
        counts_df.to_csv(out_dir / "oof_true_pred_counts.csv", index=False)
        print("\n[cv] OOF true vs predicted counts:")
        print(counts_df.to_string(index=False))

    summary = {
        "stage": "cv",
        "k": cfg.cv.k,
        "seed": cfg.cv.seed,
        "model": asdict(cfg.model),
        "text": asdict(cfg.text),
        "macro_f1_mean": float(folds_df["macro_f1"].mean()),
        "macro_f1_std": float(folds_df["macro_f1"].std(ddof=1)) if len(folds_df) > 1 else 0.0,
        "total_seconds": float(time.perf_counter() - t0),
    }
    _save_json(summary, out_dir / "cv_summary.json")

    plot_folds_macro(out_dir / "folds.csv", out_dir / "folds_macro_f1.png")
    plot_per_class_f1(out_dir / "per_class_mean.csv", out_dir / "per_class_f1.png")
    plot_confusion_matrix(out_dir / "confusion_matrix.csv", out_dir / "confusion_matrix.png")

    print("\n[cv] done.")
    print(f"[cv] macro_f1 mean={summary['macro_f1_mean']:.5f} std={summary['macro_f1_std']:.5f}")
    print(f"[cv] outputs saved in: {out_dir}")
    print(f"[cv] total_time={summary['total_seconds']:.1f}s")


def train_and_test(
    *,
    cfg,
    out_dir: Path,
    cache_dir: str | None,
) -> None:
    _print_header("STAGE 2/2 — Train final model on development set, then predict evaluation set")
    t0 = time.perf_counter()

    _ensure_dir(out_dir)

    df_dev = read_development(cfg.paths.dev_csv)

    df_eval = read_evaluation(cfg.paths.eval_csv)
    df_dev, leak_report = drop_dev_rows_overlapping_eval(
        df_dev,
        df_eval,
        on=("title", "article"),
        lowercase=True,
        strip_accents=True,
        drop=False,
    )
    print("[leakage]", leak_report)
    df_dev, dup_report = drop_cross_label_duplicates(df_dev, on=("title", "article"))
    print("[dedup]", dup_report)

    X_dev = df_dev.drop(columns=["label"])
    y_dev = df_dev["label"].astype(int).to_numpy()
    cw = _resolve_class_weight(cfg, y_dev)

    pipe = build_pipeline(
        max_features=cfg.text.max_features,
        ngram_range=(cfg.text.ngram_min, cfg.text.ngram_max),
        min_df=cfg.text.min_df,
        title_repeat=cfg.text.title_repeat,
        missing_article_token=cfg.text.missing_article_token,
        max_df=cfg.text.max_df,
        lowercase=cfg.text.lowercase,
        strip_accents=cfg.text.strip_accents,
        sublinear_tf=cfg.text.sublinear_tf,
        stop_words=cfg.text.stop_words,
        char_enabled=cfg.text.char_enabled,
        char_analyzer=cfg.text.char_analyzer,
        char_ngram_min=cfg.text.char_ngram_min,
        char_ngram_max=cfg.text.char_ngram_max,
        char_min_df=cfg.text.char_min_df,
        char_max_features=cfg.text.char_max_features,
        title_char=cfg.text.title_char,
        title_char_ngram_min=cfg.text.title_char_ngram_min,
        title_char_ngram_max=cfg.text.title_char_ngram_max,
        title_char_min_df=cfg.text.title_char_min_df,
        title_char_max_features=cfg.text.title_char_max_features,
        source_min_count=cfg.source.min_count,
        source_min_frac=cfg.source.min_frac,
        model_type=cfg.model.type,
        C=cfg.model.C,
        max_iter=cfg.model.max_iter,
        class_weight=cw,
        logreg_solver=cfg.model.logreg_solver,
        logreg_n_jobs=cfg.model.logreg_n_jobs,
        linearsvc_dual=cfg.model.linearsvc_dual,
        ridge_alpha=cfg.model.ridge_alpha,
        cache_dir=cache_dir,
    )

    print(f"[test] training on FULL development set (n={len(df_dev):,})...", flush=True)
    tr_t0 = time.perf_counter()
    pipe.fit(X_dev, y_dev)
    train_seconds = float(time.perf_counter() - tr_t0)
    print(f"[test] training done. train_time={train_seconds:.1f}s", flush=True)

 
    model_path = out_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"[test] saved model to: {model_path}")

    print(f"[test] predicting evaluation.csv (n={len(df_eval):,})...", flush=True)
    pr_t0 = time.perf_counter()
    pred = pipe.predict(df_eval)
    pred_seconds = float(time.perf_counter() - pr_t0)
    print(f"[test] prediction done. predict_time={pred_seconds:.1f}s", flush=True)

    sub_path = out_dir / "submission.csv"
    write_submission(df_eval["id"], pred, sub_path)
    print(f"[test] saved submission to: {sub_path}")

    counts = pd.Series(pred.astype(int)).value_counts().sort_index()
    counts_df = pd.DataFrame({"label": counts.index.astype(int), "count": counts.values})
    counts_df.to_csv(out_dir / "pred_label_counts.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(counts_df["label"].astype(str), counts_df["count"])
    plt.xlabel("Predicted label")
    plt.ylabel("Count")
    plt.title("Evaluation-set predicted label distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "pred_label_counts.png", dpi=180)
    plt.close()

    summary = {
        "stage": "final_train_and_predict",
        "model": asdict(cfg.model),
        "text": asdict(cfg.text),
        "n_dev": int(len(df_dev)),
        "n_eval": int(len(df_eval)),
        "train_seconds": train_seconds,
        "predict_seconds": pred_seconds,
        "total_seconds": float(time.perf_counter() - t0),
        "model_path": str(model_path),
        "submission_path": str(sub_path),
    }
    _save_json(summary, out_dir / "test_summary.json")

    print(f"[test] outputs saved in: {out_dir}")
    print(f"[test] total_time={summary['total_seconds']:.1f}s")


def parse_args():
    ap = argparse.ArgumentParser(description="News classification single entry point: CV + final train + evaluation prediction.")
    ap.add_argument("--config", default="configs/default.json", help="Path to JSON config.")
    ap.add_argument("--set", action="append", default=[], help="Overrides: section.key=value (repeatable).")
    ap.add_argument("--run_dir", default=None, help="Optional root directory for outputs. If not set, uses config paths.")
    ap.add_argument("--skip_cv", action="store_true", help="Skip CV stage.")
    ap.add_argument("--skip_test", action="store_true", help="Skip final train+predict stage.")
    ap.add_argument("--cache_dir", default=None, help="Optional joblib cache dir passed to build_pipeline(cache_dir=...).")
    return ap.parse_args()


def main():
    args = parse_args()
    overrides = parse_overrides(args.set)
    cfg = load_config(args.config, overrides=overrides)

    _print_header("NEWSCLF — MAIN ENTRY POINT")
    print(f"[main] config={args.config}")
    print(f"[main] overrides={args.set if args.set else '[]'}")
    print(f"[main] dev_csv={cfg.paths.dev_csv}")
    print(f"[main] eval_csv={cfg.paths.eval_csv}")


    if args.run_dir:
        run_root = Path(args.run_dir)
        eval_dir = run_root / "evaluation"
        test_dir = run_root / "testing"
    else:
        eval_dir = Path(cfg.paths.cv_out_dir)
        test_dir = Path(cfg.paths.submission_out).resolve().parent

    _ensure_dir(eval_dir)
    _ensure_dir(test_dir)

    cfg_dict = asdict(cfg)
    cfg_dict["_meta"] = {
        "config_path": str(Path(args.config).resolve()),
        "overrides": args.set,
        "run_tag": _now_tag(),
    }
    _save_json(cfg_dict, eval_dir / "config_used.json")
    _save_json(cfg_dict, test_dir / "config_used.json")
    _copy_if_exists(Path(args.config), eval_dir / "config_original.json")
    _copy_if_exists(Path(args.config), test_dir / "config_original.json")

    cache_dir = args.cache_dir
    if cache_dir:
        _ensure_dir(Path(cache_dir))

    if not args.skip_cv:
        run_cv(cfg=cfg, out_dir=eval_dir, cache_dir=cache_dir)
    else:
        print("[main] skip_cv=True -> skipping CV stage.")

    if not args.skip_test:
        train_and_test(cfg=cfg, out_dir=test_dir, cache_dir=cache_dir)
    else:
        print("[main] skip_test=True -> skipping final train+predict stage.")

    _print_header("DONE")
    print(f"[main] evaluation outputs: {eval_dir}")
    print(f"[main] testing outputs:    {test_dir}")


if __name__ == "__main__":
    main()
