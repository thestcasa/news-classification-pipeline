from __future__ import annotations
from pathlib import Path
import pandas as pd

REQ_DEV = {"id", "source", "title", "article", "page_rank", "timestamp", "label"}
REQ_EVAL = {"id", "source", "title", "article", "page_rank", "timestamp"}

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and standardize Id -> id.
    Keeps everything else unchanged.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # standardize Id -> id
    if "Id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"Id": "id"})
    if "ID" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"ID": "id"})

    return df

def read_development(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardize(df)
    missing = REQ_DEV - set(df.columns)
    if missing:
        raise ValueError(f"development.csv missing columns: {sorted(missing)}. Found: {list(df.columns)}")
    return df

def read_evaluation(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardize(df)
    missing = REQ_EVAL - set(df.columns)
    if missing:
        raise ValueError(f"evaluation.csv missing columns: {sorted(missing)}. Found: {list(df.columns)}")
    return df

def write_submission(ids: pd.Series, pred, out_path: str | Path) -> None:
    out = pd.DataFrame({"Id": ids.astype(int), "Predicted": pred.astype(int)})
    out.to_csv(out_path, index=False)
