from __future__ import annotations
import argparse
from pathlib import Path
import pickle

from .config import load_config, parse_overrides
from .io import read_evaluation, write_submission

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--set", action="append", default=[])
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=parse_overrides(args.set))

    df = read_evaluation(cfg.paths.eval_csv)

    with open(cfg.paths.model_out, "rb") as f:
        pipe = pickle.load(f)

    print("Predicting evaluation set...", flush=True)
    pred = pipe.predict(df)

    out_path = Path(cfg.paths.submission_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(df["id"], pred, out_path)

    print(f"Saved submission to: {out_path}")

if __name__ == "__main__":
    main()
