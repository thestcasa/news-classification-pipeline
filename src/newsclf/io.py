from __future__ import annotations
from pathlib import Path
import pandas as pd

REQ_COLS_DEV = {"Id", "source", "title", "article", "page_rank", "timestamp", "label"}
REQ_COLS_EVAL = {"Id", "source", "title", "article", "page_rank", "timestamp"}

def read_development(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQ_COLS_DEV - set(df.columns)
    if missing:
        raise ValueError(f"development.csv missing columns: {sorted(missing)}")
    return df

def read_evaluation(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQ_COLS_EVAL - set(df.columns)
    if missing:
        raise ValueError(f"evaluation.csv missing columns: {sorted(missing)}")
    return df

def write_submission(ids: pd.Series, pred, out_path: str | Path) -> None:
    out = pd.DataFrame({"Id": ids.astype(int), "Predicted": pred.astype(int)})
    out.to_csv(out_path, index=False)
