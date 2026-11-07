"""R1 (v2): the two splitting techniques of the redirected study.

Technique A  (baseline)  : random shuffled 80/20 split of frames.
Technique B  (robust)    : chronological block split INSIDE each recording: train frames are earlier,
                           test frames later, and the cut never falls inside an attack burst.

Every function returns (train_idx, test_idx): sorted integer row positions into the frame table.
Technique B also returns a per-file info table (cut time, bursts, shares) for the validity report (R2/R3).

Signature = (id, dlc, d0..d7); the timestamp is never part of it and never a model input.
The old signature-group split is kept ONLY as `signature_group_split`, a diagnostic for Normal/Fuzzy
(its status is an open question for the supervisor). It is not one of the two techniques.
"""
import warnings

import numpy as np
import pandas as pd

SIG_COLS = ["id", "dlc"] + [f"d{i}" for i in range(8)]


# --------------------------------------------------------------------------- helpers
def _check(n, frac, name="test_size"):
    if not 0.0 < frac < 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    if n < 2:
        raise ValueError("need at least 2 rows")


def make_signature_ids(df):
    """Exact signature id per row (same rule as audit.py)."""
    return df.groupby(SIG_COLS, sort=False, observed=True).ngroup().to_numpy()


# --------------------------------------------------------------------------- Technique A
def random_split(labels, seed, test_size=0.2, stratified=False):
    """Technique A: shuffled random split.

    stratified=False : plain shuffle, exactly round(test_size * N) test frames (supervisor's wording:
                       'standard, shuffled random train-test split').
    stratified=True  : exactly round(test_size * N_c) test frames per class (decision D12 in the
                       Master Context Pack; the overlap numbers already measured there use this).
    The seed fixes the draw.
    """
    labels = np.asarray(labels)
    n = len(labels)
    _check(n, test_size)
    rng = np.random.default_rng(seed)
    if stratified:
        parts = []
        for c in np.unique(labels):
            idx = np.flatnonzero(labels == c)
            k = int(round(test_size * len(idx)))
            parts.append(rng.choice(idx, size=k, replace=False))
        test_idx = np.sort(np.concatenate(parts))
    else:
        k = int(round(test_size * n))
        test_idx = np.sort(rng.choice(n, size=k, replace=False))
    is_test = np.zeros(n, dtype=bool)
    is_test[test_idx] = True
    return np.flatnonzero(~is_test), test_idx


# --------------------------------------------------------------------------- Technique B
def _burst_intervals(t_attack, burst_gap):
    """Bursts = runs of attack frames; a gap larger than burst_gap seconds starts a new burst."""
    ta = np.sort(t_attack)
    new = np.flatnonzero(np.diff(ta) > burst_gap) + 1
    starts = ta[np.r_[0, new]]
    ends = ta[np.r_[new - 1, len(ta) - 1]]
    return starts, ends


def _cut_time(t0, t_end, starts, ends, train_fraction):
    """80% point, moved so that no burst is cut. Returns (tau, moved, quiet_length)."""
    tau0 = t0 + train_fraction * (t_end - t0)
    if len(starts) == 0:                      # no attack frames (ambient file): plain 80% of duration
        return tau0, False, float("nan")
    q_lo = np.r_[t0, ends]                    # quiet stretch k lies between end of burst k-1 ...
    q_hi = np.r_[starts, t_end]               # ... and start of burst k
    i = int(np.searchsorted(starts, tau0, side="right")) - 1     # last burst starting at or before tau0
    if i >= 0 and tau0 <= ends[i]:            # tau0 is inside burst i -> quiet stretch right after it
        moved, j = True, i + 1
    else:                                     # tau0 is in a quiet stretch: the one containing it
        moved, j = False, int(np.searchsorted(ends, tau0, side="left"))
    lo, hi = q_lo[j], q_hi[j]
    return 0.5 * (lo + hi), moved, float(hi - lo)


def chronological_split(file_ids, t, is_attack, train_fraction=0.8, burst_gap=1.0, silent_gap=1.0):
    """Technique B: per-file chronological block split (timestamps are comparable only inside a file).

    file_ids  : one id per row (use df['src_file'].cat.codes for speed)
    t         : timestamps (seconds)
    is_attack : bool per row (flag T / label != Normal)
    Procedure per file: find bursts (attack runs, gap > burst_gap s starts a new one); 80% point of the
    file's duration; if it falls inside a burst move it to the quiet stretch after that burst, otherwise
    use the quiet stretch that contains it; cut at the MIDPOINT of that stretch.
    Train = t <= cut, test = t > cut. No frames are dropped. Deterministic: no seed.
    Run it on the SANITISED table (attack-free tails removed) or the test side holds no attacks.
    A warning is raised if a file still contains a silent gap longer than `silent_gap` seconds
    (i.e. looks unsanitised). Hafiz's pilot.plan_file would silently DROP the rows after such a gap;
    this function keeps every row, so the two only agree on a sanitised table.
    """
    file_ids = np.asarray(file_ids)
    t = np.asarray(t, dtype=np.float64)
    is_attack = np.asarray(is_attack, dtype=bool)
    n = len(t)
    _check(n, train_fraction, "train_fraction")
    is_train = np.zeros(n, dtype=bool)
    rows = []
    for f in pd.unique(file_ids):
        idx = np.flatnonzero(file_ids == f)
        tf = t[idx]
        t0, t_end = float(tf.min()), float(tf.max())
        max_gap = float(np.diff(np.sort(tf)).max()) if len(tf) > 1 else 0.0
        if max_gap > silent_gap:
            warnings.warn(f"{f}: gap of {max_gap:.1f} s inside the file; table may be unsanitised "
                          f"(attack-free tail still present)", stacklevel=2)
        ta = tf[is_attack[idx]]
        if len(ta):
            starts, ends = _burst_intervals(ta, burst_gap)
        else:
            starts = ends = np.array([])
        tau, moved, quiet = _cut_time(t0, t_end, starts, ends, train_fraction)
        tr = tf <= tau
        is_train[idx] = tr
        rows.append(dict(src_file=f, n_frames=len(idx), t0=t0, t_end=t_end,
                         n_bursts=len(starts), max_gap_s=max_gap, cut_time=tau, cut_rel_s=tau - t0,
                         cut_moved_out_of_burst=moved, quiet_at_cut_s=quiet,
                         n_train=int(tr.sum()), n_test=int((~tr).sum()),
                         test_share=float((~tr).mean())))
    return np.flatnonzero(is_train), np.flatnonzero(~is_train), pd.DataFrame(rows)


# --------------------------------------------------------------------------- diagnostics (reused by R2/R3)
def share_in_test(train_idx, test_idx):
    return len(test_idx) / (len(train_idx) + len(test_idx))


def class_table(labels, train_idx, test_idx):
    """Frames per class on each side and the class's test share (validity rule: every class on both sides)."""
    labels = np.asarray(labels)
    out = []
    for c in pd.unique(labels):
        ntr, nte = int((labels[train_idx] == c).sum()), int((labels[test_idx] == c).sum())
        out.append(dict(label=c, train=ntr, test=nte, test_share=nte / (ntr + nte) if ntr + nte else np.nan))
    return pd.DataFrame(out)


def seen_mask(sig, train_idx, test_idx):
    """True for test frames whose exact signature also occurs in the training side."""
    sig = np.asarray(sig)
    return np.isin(sig[test_idx], np.unique(sig[train_idx]))


def seen_share_by_class(labels, sig, train_idx, test_idx):
    """Share of each class's test frames that are 'seen' (memorisable)."""
    labels = np.asarray(labels)
    seen = seen_mask(sig, train_idx, test_idx)
    lt = labels[test_idx]
    return {c: float(seen[lt == c].mean()) for c in pd.unique(labels) if (lt == c).any()}


def overlap_share(sig, train_idx, test_idx):
    return float(seen_mask(sig, train_idx, test_idx).mean())


# --------------------------------------------------------------------------- old group split (diagnostic only)
def signature_group_split(sig, seed, test_size=0.2):
    """DIAGNOSTIC ONLY (not a main technique): all frames of a signature go to one side.
    Cannot be valid for DoS/RPM/Gear (one signature each); meaningful only for Normal.
    Moves whole signatures to test until the test side holds ~test_size of the FRAMES (cut at the closest point).
    """
    sig = np.asarray(sig)
    n = len(sig)
    _check(n, test_size)
    uniq, inverse, counts = np.unique(sig, return_inverse=True, return_counts=True)
    if len(uniq) < 2:
        raise ValueError("needs at least 2 distinct signatures")
    order = np.random.default_rng(seed).permutation(len(uniq))
    cum = np.cumsum(counts[order])
    k = min(int(np.argmin(np.abs(cum - test_size * n))) + 1, len(uniq) - 1)
    group_is_test = np.zeros(len(uniq), dtype=bool)
    group_is_test[order[:k]] = True
    is_test = group_is_test[inverse]
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)
