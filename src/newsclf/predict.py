from __future__ import annotations
import argparse
from pathlib import Path
import pickle

from .io import read_evaluation, write_submission

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/raw/evaluation.csv")
    ap.add_argument("--model_in", default="models/model.pkl")
    ap.add_argument("--out", default="models/submission.csv")
    return ap.parse_args()

def main():
    args = parse_args()
    df = read_evaluation(args.eval)

    with open(args.model_in, "rb") as f:
        pipe = pickle.load(f)

    pred = pipe.predict(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(df["Id"], pred, out_path)

    print(f"Saved submission to: {out_path}")

if __name__ == "__main__":
    main()
