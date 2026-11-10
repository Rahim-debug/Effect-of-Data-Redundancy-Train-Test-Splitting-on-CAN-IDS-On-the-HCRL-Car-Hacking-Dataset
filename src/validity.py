"""R2 (+ R3 runner): validity report for Technique A (random) and Technique B (chronological).

For every split and every class it reports: train / test frames, test share, train / test signatures,
seen share (share of that class's test frames whose exact signature also occurs in train), and, for
Technique A, the share expected from the random-split formula  L = sum m (1 - 0.2^(m-1)) / N_c.

Validity rule (PROVISIONAL, decision D17 - the supervisor has not fixed the tolerance yet):
  a split is valid only if EVERY class has frames on both sides and its test share is inside
  [tol_low, tol_high] (default 10% - 30%). Nothing is dropped or forced: invalid splits are flagged.

Usage (project root):
    python src/validity.py --parquet data/derived/frames_clean.parquet --out results --save-splits data/splits
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from splits import random_split, chronological_split, make_signature_ids
except ImportError:
    from src.splits import random_split, chronological_split, make_signature_ids

CLASSES = ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]
SEEDS = (42, 43, 44, 45, 46)
TOL_LOW, TOL_HIGH = 0.10, 0.30
FEATS = ["id", "dlc"] + [f"d{i}" for i in range(8)]

# Hafiz's temporal_audit_classes.csv, used only as a cross-check of Technique B
HAFIZ_B = {"Normal": (9317710, 2344847, 0.9145155313), "DoS": (478898, 108623, 1.0),
           "Fuzzy": (415726, 76121, 0.0), "RPM": (519813, 135084, 1.0), "Gear": (478762, 118490, 1.0)}

# same file, column seen_share_random_expected (his formula on the sanitised table)
HAFIZ_EXPECTED_RANDOM = {"Normal": 0.9709823417, "DoS": 1.0, "Fuzzy": 0.0, "RPM": 1.0, "Gear": 1.0}


def expected_seen_random(labels, sig, n_classes, n_sig, test_size=0.2):
    """Per class: expected share of test frames with a training twin under a random split."""
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        cnt = np.bincount(sig[labels == c], minlength=n_sig)
        cnt = cnt[cnt > 0].astype(np.float64)
        if len(cnt):
            out[c] = float(np.sum(cnt * (1.0 - test_size ** (cnt - 1.0))) / cnt.sum())
    return out


def split_report(technique, seed, labels, sig, train_idx, test_idx, n_sig,
                 class_names=CLASSES, tol_low=TOL_LOW, tol_high=TOL_HIGH, expected_random=None):
    """One row per class for one split."""
    sig_tr, sig_te = sig[train_idx], sig[test_idx]
    lab_tr, lab_te = labels[train_idx], labels[test_idx]
    in_train = np.zeros(n_sig, dtype=bool)
    in_train[sig_tr] = True
    seen = in_train[sig_te]
    rows = []
    for c, name in enumerate(class_names):
        mtr, mte = lab_tr == c, lab_te == c
        ntr, nte = int(mtr.sum()), int(mte.sum())
        share = nte / (ntr + nte) if (ntr + nte) else np.nan
        p_tr = np.zeros(n_sig, dtype=bool); p_tr[sig_tr[mtr]] = True
        p_te = np.zeros(n_sig, dtype=bool); p_te[sig_te[mte]] = True
        both = ntr > 0 and nte > 0
        in_tol = bool(both and tol_low <= share <= tol_high)
        rows.append(dict(technique=technique, seed=seed, label=name,
                         train_frames=ntr, test_frames=nte, test_share=share,
                         train_signatures=int(p_tr.sum()), test_signatures=int(p_te.sum()),
                         seen_share=float(seen[mte].mean()) if nte else np.nan,
                         unseen_test_frames=int((~seen[mte]).sum()) if nte else 0,
                         expected_seen_random=(float(expected_random[c]) if expected_random is not None else np.nan),
                         on_both_sides=both, share_in_tolerance=in_tol, class_valid=in_tol))
    return rows


def summarise(report):
    """One row per split: valid yes/no and which classes fail, with the reason."""
    out = []
    for (tech, seed), g in report.groupby(["technique", "seed"], sort=False, dropna=False):
        bad = []
        for _, r in g[~g.class_valid].iterrows():
            why = "missing on one side" if not r.on_both_sides else f"test share {r.test_share:.3f} outside tolerance"
            bad.append(f"{r.label} ({why})")
        out.append(dict(technique=tech, seed=seed, n_train=int(g.train_frames.sum()), n_test=int(g.test_frames.sum()),
                        test_share=g.test_frames.sum() / (g.train_frames.sum() + g.test_frames.sum()),
                        valid=bool(g.class_valid.all()), failing="; ".join(bad)))
    return pd.DataFrame(out)


def _load_signatures(df):
    try:
        from checks import sig_ids                    # Hafiz's bit-packed version (fast, exact)
        return sig_ids(df)
    except ImportError:
        return make_signature_ids(df)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="frames_clean.parquet (SANITISED table)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--a-mode", choices=["stratified", "plain"], default="stratified",
                    help="Technique A: exact per-class draw (D12, default) or plain shuffle")
    ap.add_argument("--tol-low", type=float, default=TOL_LOW)
    ap.add_argument("--tol-high", type=float, default=TOL_HIGH)
    ap.add_argument("--save-splits", default=None, help="folder for test-index .npy files (train = the rest)")
    ap.add_argument("--no-compare", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    df = pd.read_parquet(a.parquet, columns=["src_file", "t", "flag", "label"] + FEATS)
    for col in ("src_file", "flag", "label"):
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")
    names = list(df["label"].cat.categories)
    labels = df["label"].cat.codes.to_numpy().astype(np.int64)
    files = df["src_file"].cat.categories
    file_codes = df["src_file"].cat.codes.to_numpy()
    is_attack = df["flag"].cat.codes.to_numpy() == df["flag"].cat.categories.get_loc("T")
    t = df["t"].to_numpy()
    sig = _load_signatures(df).astype(np.int64)
    del df
    n_sig = int(sig.max()) + 1
    print(f"loaded {len(labels):,} rows, {n_sig:,} signatures in {time.time() - t0:.1f}s")

    exp_random = expected_seen_random(labels, sig, len(names), n_sig)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    save = Path(a.save_splits) if a.save_splits else None
    if save:
        save.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        tr, te = random_split(labels, seed, stratified=(a.a_mode == "stratified"))
        rows += split_report("A", seed, labels, sig, tr, te, n_sig, names, a.tol_low, a.tol_high, exp_random)
        if save:
            np.save(save / f"technique_A_seed{seed}_test_idx.npy", te.astype(np.int64))
    trB, teB, info = chronological_split(file_codes, t, is_attack)
    rows += split_report("B", -1, labels, sig, trB, teB, n_sig, names, a.tol_low, a.tol_high)
    if save:
        np.save(save / "technique_B_test_idx.npy", teB.astype(np.int64))

    report = pd.DataFrame(rows)
    summary = summarise(report)
    report.to_csv(out / "validity_report.csv", index=False)
    summary.to_csv(out / "validity_summary.csv", index=False)

    pd.set_option("display.width", 250)
    print("\nSUMMARY (valid = every class on both sides and test share in "
          f"[{a.tol_low:.2f}, {a.tol_high:.2f}])\n", summary.round(4).to_string(index=False))
    A = report[report.technique == "A"]
    cmp_ = (A.groupby("label", sort=False).agg(test_share_mean=("test_share", "mean"), seen_mean=("seen_share", "mean"),
                                                seen_sd=("seen_share", "std"), expected=("expected_seen_random", "first"))
            .reindex(names))
    Bc = report[report.technique == "B"].set_index("label").reindex(names)
    cmp_["B_test_share"] = Bc["test_share"]; cmp_["B_seen"] = Bc["seen_share"]
    print("\nA (mean over seeds) vs B, per class\n", cmp_.round(5).to_string())
    print(f"\nsaved {out / 'validity_report.csv'} and {out / 'validity_summary.csv'}"
          + (f"; test indices in {save}" if save else "") + f"  (total {time.time() - t0:.1f}s)")

    if not a.no_compare:
        bad = []
        for _, r in report[report.technique == "B"].iterrows():
            e = HAFIZ_B.get(r.label)
            if e is None or (int(r.train_frames), int(r.test_frames)) != e[:2] or abs(r.seen_share - e[2]) > 1e-6:
                bad.append((r.label, int(r.train_frames), int(r.test_frames), round(r.seen_share, 6), e))
        print("\nCHECK B vs Hafiz's temporal audit (counts and seen share):",
              "PASSED" if not bad else "MISMATCH")
        for b in bad:
            print("  ", b)
        badx = [(n, round(float(e), 8), HAFIZ_EXPECTED_RANDOM[n]) for n, e in zip(names, exp_random)
                if n in HAFIZ_EXPECTED_RANDOM and abs(e - HAFIZ_EXPECTED_RANDOM[n]) > 1e-6]
        print("CHECK random-split formula vs Hafiz's seen_share_random_expected:",
              "PASSED" if not badx else "MISMATCH")
        for b in badx:
            print("  ", b, "(class, mine, his)")


if __name__ == "__main__":
    main()
