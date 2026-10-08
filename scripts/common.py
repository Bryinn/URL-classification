"""Shared helpers for the phishing-websites pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

ROOT = Path(__file__).resolve().parents[1]
RAW_ARFF = ROOT / "phishing+websites" / "Training Dataset.arff"
OUT_DIR = ROOT / "outputs"
CLEAN_CSV = OUT_DIR / "clean_dataset.csv"
LEAKAGE_JSON = OUT_DIR / "leakage_report.json"
SELECTION_JSON = OUT_DIR / "selected_features.json"

TARGET = "Result"
LABEL = "is_phishing"  # 1 = phishing (Result == -1), 0 = legitimate (Result == 1)
GROUP = "group_id"
SPLIT = "split"
META_COLS = [LABEL, GROUP, SPLIT]
SEED = 42


def load_raw() -> pd.DataFrame:
    data, _ = arff.loadarff(RAW_ARFF)
    df = pd.DataFrame(data)
    return df.apply(lambda c: c.str.decode("utf-8").astype(np.int8))


def row_groups(features: pd.DataFrame) -> pd.Series:
    """Group id = hash of the full original feature vector.

    ~47% of rows are exact duplicates; identical rows must stay on the same side
    of any train/test or CV split, otherwise the model is scored on memorised samples.
    """
    keys = features.astype(str).agg(",".join, axis=1)
    return keys.map(lambda s: int(hashlib.md5(s.encode()).hexdigest()[:12], 16)).rename(GROUP)


def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV)


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
