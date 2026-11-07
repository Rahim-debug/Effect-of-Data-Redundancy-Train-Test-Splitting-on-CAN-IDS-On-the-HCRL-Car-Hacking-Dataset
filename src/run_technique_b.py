"""Run Technique B (chronological block split) on the sanitised frame table and check it against
Hafiz's temporal_audit_classes.csv.

Usage (from the project folder):
    python src/run_technique_b.py --parquet data/derived/frames_clean.parquet --out results

Reads only 4 columns (src_file, t, flag, label), so it needs little RAM.
Writes: results/split_b_files.csv, results/split_b_classes.csv, splits/technique_B_test_idx.npy
(train = every row not in the test file).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from splits import chronological_split, class_table
except ImportError:                                   # when run from the project root as a package
    from src.splits import chronological_split, class_table

CLASSES = ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]

# Hafiz's temporal_audit_classes.csv (train / test frames per class, sanitised table)
EXPECTED = {"Normal": (9317710, 2344847), "DoS": (478898, 108623), "Fuzzy": (415726, 76121),
            "RPM": (519813, 135084), "Gear": (478762, 118490)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="path to frames_clean.parquet (SANITISED table)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--no-compare", action="store_true", help="skip the comparison with Hafiz's numbers")
    a = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(a.parquet, columns=["src_file", "t", "flag", "label"])
    print(f"loaded {len(df):,} rows in {time.time() - t0:.1f}s")
    for col in ("src_file", "flag", "label"):
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")
    files = df["src_file"].cat.categories
    file_codes = df["src_file"].cat.codes.to_numpy()
    is_attack = df["flag"].cat.codes.to_numpy() == df["flag"].cat.categories.get_loc("T")
    lab_codes = df["label"].cat.codes.to_numpy()
    lab_names = list(df["label"].cat.categories)
    t = df["t"].to_numpy()
    del df

    train_idx, test_idx, info = chronological_split(file_codes, t, is_attack)
    info["src_file"] = [files[c] for c in info["src_file"]]
    cls = class_table(lab_codes, train_idx, test_idx)
    cls["label"] = [lab_names[c] for c in cls["label"]]
    cls = cls.set_index("label").reindex([c for c in CLASSES if c in lab_names]).reset_index()

    pd.set_option("display.width", 220)
    print("\nPER FILE\n", info[["src_file", "n_frames", "n_bursts", "cut_rel_s", "cut_moved_out_of_burst",
                               "quiet_at_cut_s", "max_gap_s", "n_train", "n_test", "test_share"]]
          .round(3).to_string(index=False))
    print("\nPER CLASS\n", cls.round(4).to_string(index=False))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sd = Path(a.splits_dir); sd.mkdir(parents=True, exist_ok=True)
    info.to_csv(out / "split_b_files.csv", index=False)
    cls.to_csv(out / "split_b_classes.csv", index=False)
    np.save(sd / "technique_B_test_idx.npy", test_idx.astype(np.int64))
    print(f"\nsaved results/split_b_*.csv and {sd / 'technique_B_test_idx.npy'}  (done in {time.time() - t0:.1f}s)")

    if not a.no_compare:
        bad = []
        for _, r in cls.iterrows():
            exp = EXPECTED.get(r["label"])
            if exp is None or (int(r["train"]), int(r["test"])) != exp:
                bad.append((r["label"], (int(r["train"]), int(r["test"])), exp))
        if bad:
            print("\nMISMATCH with Hafiz's temporal_audit_classes.csv (label, mine (train,test), his):")
            for b in bad:
                print("  ", b)
            print("Most likely causes: the table is not frames_clean.parquet (unsanitised?), or rows were reordered.")
        else:
            print("\nCHECK PASSED: every class's train and test frame count equals Hafiz's temporal audit.")


if __name__ == "__main__":
    main()
