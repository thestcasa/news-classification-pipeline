from __future__ import annotations
import argparse
from pathlib import Path
import pickle

from .config import load_config, parse_overrides
from .io import read_development
from .model import build_pipeline

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

    print("Training final model on full development set...", flush=True)
    pipe.fit(X, y)

    out_path = Path(cfg.paths.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)

    print(f"Saved final model to: {out_path}")

if __name__ == "__main__":
    main()
