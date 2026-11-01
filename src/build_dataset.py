"""Raw files -> data/processed/frames.parquet (the single frame table used by everything else)."""
import numpy as np
import pandas as pd
from config import RAW, PROCESSED, LABELS
from parse_raw import ATTACK_FILES, NORMAL_FILE, parse_attack_csv, parse_normal_txt


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    parts = []

    print(f"parsing {NORMAL_FILE} ...")
    parts.append(parse_normal_txt(RAW / NORMAL_FILE))
    print(f"  {len(parts[-1]):,} frames")
    for fname, (source, label) in ATTACK_FILES.items():
        print(f"parsing {fname} ...")
        parts.append(parse_attack_csv(RAW / fname, source, label))
        print(f"  {len(parts[-1]):,} frames")

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)      # time order, as in the proposal

    # exact signature id: one integer per distinct (id, dlc, data bytes). No hashing, no collisions.
    df["sig_id"] = df.groupby(["id", "dlc", "data64"], sort=False).ngroup().astype(np.int32)

    df["label"] = pd.Categorical(df["label"], categories=LABELS)
    df["source"] = df["source"].astype("category")
    df["flag"] = df["flag"].astype("category")
    df.to_parquet(PROCESSED / "frames.parquet", index=False)

    print("\n== sanity checks ==")
    print(f"frames N = {len(df):,}   distinct signatures U = {df['sig_id'].nunique():,}")
    print("\nsource x flag:\n", pd.crosstab(df["source"], df["flag"]))
    print("\nlabel counts:\n", df["label"].value_counts().reindex(LABELS))
    print("\nDLC counts:\n", df["dlc"].value_counts().sort_index())
    print("\ntimestamp range per source:\n", df.groupby("source", observed=True)["ts"].agg(["min", "max"]))
    print("\ndtypes:\n", df.dtypes)


if __name__ == "__main__":
    main()
