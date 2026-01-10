from __future__ import annotations
import argparse
from pathlib import Path
import pickle

from .io import read_development
from .model import build_pipeline

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="data/raw/development.csv")
    ap.add_argument("--model_out", default="models/model.pkl")

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
    pipe.fit(X, y)

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)

    print(f"Saved final model to: {out_path}")

if __name__ == "__main__":
    main()
