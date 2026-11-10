"""Synthetic CAN-like data for tests, the timing test (M3) and the runner dry run (M4).

NOT real data and NOT a replacement for the real splits: it only has the same shape
as the real table (10 numeric columns, 5 classes, heavy repetition of signatures),
so code can be tested without touching the HCRL files (ground rule 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import FEATURES

SHARES = np.array([0.85, 0.05, 0.04, 0.03, 0.03])   # Normal, DoS, Fuzzy, RPM, Gear (made up)


def make_synthetic_arrays(n: int, seed: int = 0, dos_sigs: int = 25, normal_sigs: int = 3000,
                          rpm_sigs: int = 200, gear_sigs: int = 120):
    """Return X float32 (n,10) and y int8 (n,). Normal/RPM/Gear repeat a pool of
    signatures, DoS uses id 0 with a few payloads (real DoS may be ONE signature),
    Fuzzy is random id and random bytes (almost all unique)."""
    rng = np.random.default_rng(seed)
    y = rng.choice(5, size=n, p=SHARES).astype(np.int8)
    X = np.zeros((n, 10), dtype=np.float32)

    def pool(k, ids):
        p = np.zeros((k, 10), dtype=np.float32)
        p[:, 0] = rng.choice(ids, size=k)
        p[:, 1] = 8
        p[:, 2:] = rng.integers(0, 256, size=(k, 8))
        return p

    # Normal: many ids, Zipf-like popularity of signatures
    pn = pool(normal_sigs, np.arange(0x100, 0x7FF))
    w = 1.0 / np.arange(1, normal_sigs + 1); w /= w.sum()
    m = y == 0; X[m] = pn[rng.choice(normal_sigs, size=m.sum(), p=w)]
    # DoS: id 0, a few payload variants
    pd_ = np.zeros((dos_sigs, 10), dtype=np.float32); pd_[:, 1] = 8
    pd_[:, 2] = rng.integers(0, 256, size=dos_sigs)
    m = y == 1; X[m] = pd_[rng.integers(0, dos_sigs, size=m.sum())]
    # Fuzzy: random id and bytes
    m = y == 2; k = m.sum()
    X[m, 0] = rng.integers(0, 0x800, size=k); X[m, 1] = 8
    X[m, 2:] = rng.integers(0, 256, size=(k, 8))
    # RPM spoof on id 0x316, Gear spoof on id 0x43F
    pr = pool(rpm_sigs, [0x316]); m = y == 3; X[m] = pr[rng.integers(0, rpm_sigs, size=m.sum())]
    pg = pool(gear_sigs, [0x43F]); m = y == 4; X[m] = pg[rng.integers(0, gear_sigs, size=m.sum())]
    return X, y


def make_synthetic_frames(n: int, seed: int = 0, **kw) -> pd.DataFrame:
    X, y = make_synthetic_arrays(n, seed, **kw)
    df = pd.DataFrame(X.astype(np.int32), columns=FEATURES)
    df["label"] = y
    return df


def signature_ids(X: np.ndarray) -> np.ndarray:
    """Group id per distinct (id, dlc, d0..d7) row (exact, no hashing)."""
    df = pd.DataFrame(X.astype(np.int32))
    return df.groupby(list(df.columns), sort=False).ngroup().to_numpy()


def dummy_p1_test_idx(y, seed: int, test_share: float = 0.2) -> np.ndarray:
    idx = np.arange(len(y))
    _, test = train_test_split(idx, test_size=test_share, stratify=y, random_state=seed)
    return np.sort(test)


def dummy_p2_test_idx(sig, seed: int, test_share: float = 0.2) -> np.ndarray:
    """Shuffle signature groups with the seed, add groups to test until 20% of FRAMES.
    DUMMY for dry runs only: the real P2 comes from Raahim's src/splits.py."""
    sig = np.asarray(sig)
    counts = np.bincount(sig)
    order = np.random.default_rng(seed).permutation(len(counts))
    csum = np.cumsum(counts[order])
    k = int(np.searchsorted(csum, test_share * len(sig))) + 1
    in_test = np.zeros(len(counts), dtype=bool); in_test[order[:k]] = True
    return np.flatnonzero(in_test[sig])


def write_dummy_splits(X, y, seeds, out_dir, template="{protocol}_seed{seed}.npy"):
    from pathlib import Path
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    sig = signature_ids(X)
    for s in seeds:
        np.save(out / template.format(protocol="P1", seed=s), dummy_p1_test_idx(y, s))
        np.save(out / template.format(protocol="P2", seed=s), dummy_p2_test_idx(sig, s))
    return out
