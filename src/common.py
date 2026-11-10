"""Shared constants and helpers for the models track (Sameer's part).

DRAFT of the values that CONVENTIONS.md (task S1) must fix. If the team's
CONVENTIONS.md differs from anything here, CONVENTIONS.md wins: change this file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CLASSES = ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]   # fixed class order (0..4)
N_CLASSES = len(CLASSES)
FEATURES = ["id", "dlc"] + [f"d{i}" for i in range(8)]  # the 10 model inputs
SEEDS = [42, 43, 44, 45, 46]
PROTOCOLS = ["P1", "P2"]
MODELS = ["LR", "DT", "RF"]
TEST_SHARE = 0.2

_ALIASES = {
    "normal": 0, "r": 0, "benign": 0,
    "dos": 1, "fuzzy": 2, "fuzzing": 2,
    "rpm": 3, "rpm spoofing": 3, "gear": 4, "gear spoofing": 4,
}


def labels_to_int(labels) -> np.ndarray:
    """Convert a label column (class names or 0-4 integers) to int8 codes."""
    s = pd.Series(labels)
    if pd.api.types.is_numeric_dtype(s):
        arr = s.to_numpy()
        if arr.min() < 0 or arr.max() >= N_CLASSES:
            raise ValueError("integer labels must be in 0..4")
        return arr.astype(np.int8)
    mapped = s.astype(str).str.strip().str.lower().map(_ALIASES)
    if mapped.isna().any():
        bad = sorted(set(s[mapped.isna()].astype(str)))[:10]
        raise ValueError(f"unknown labels in the label column: {bad}")
    return mapped.to_numpy().astype(np.int8)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    """Hash of everything that defines the experiment (not of hardware settings)."""
    core = {"models": cfg["models"], "seeds": cfg["seeds"], "classes": cfg["classes"]}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:12]


def load_frames(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read data/frames.parquet (or .csv) -> X float32 (N,10), y int8 (N,)."""
    path = Path(path)
    cols = FEATURES + ["label"]
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=cols)
    else:
        df = pd.read_csv(path, usecols=cols)
    if df[cols].isna().any().any():
        raise ValueError("frames table contains NaN in the model columns")
    y = labels_to_int(df["label"])
    X = df[FEATURES].to_numpy(dtype=np.float32)
    return X, y
