from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_folds_macro(folds_csv: str | Path, out_png: str | Path) -> None:
    df = pd.read_csv(folds_csv)
    plt.figure(figsize=(7, 4))
    plt.plot(df["fold"], df["macro_f1"], marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Macro F1")
    plt.title("Macro F1 across folds")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_per_class_f1(per_class_csv: str | Path, out_png: str | Path) -> None:
    df = pd.read_csv(per_class_csv).sort_values("label")
    plt.figure(figsize=(8, 4))
    plt.bar(df["label"].astype(str), df["f1"])
    plt.xlabel("Label")
    plt.ylabel("F1 (mean across folds)")
    plt.title("Per-class F1 (mean)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_confusion_matrix(cm_csv: str | Path, out_png: str | Path) -> None:
    df = pd.read_csv(cm_csv, index_col=0)
    cm = df.to_numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, aspect="auto")
    plt.colorbar()
    plt.xticks(np.arange(cm.shape[1]), df.columns.astype(str), rotation=45, ha="right")
    plt.yticks(np.arange(cm.shape[0]), df.index.astype(str))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (aggregated)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
