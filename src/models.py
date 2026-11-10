"""Task M2 - model builders. All hyperparameters come from config.yaml, which is
fixed BEFORE any real result is seen and is identical for P1 and P2 (Step 6).

LR : StandardScaler (fitted on the TRAINING side only, because it lives inside the
     pipeline that is fitted on training data) + LogisticRegression, balanced weights.
DT : DecisionTreeClassifier on raw integers, balanced weights.
RF : RandomForestClassifier on raw integers, balanced weights.
random_state = seed for all three.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_model(name: str, seed: int, cfg: dict):
    if name not in ("LR", "DT", "RF"):
        raise ValueError(f"unknown model {name!r}; use LR, DT or RF")
    hp = dict(cfg["models"][name])
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **hp))
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed, **hp)
    if name == "RF":
        n_jobs = cfg.get("runtime", {}).get("n_jobs", -1)
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **hp)
    raise ValueError(f"unknown model {name!r}; use LR, DT or RF")
