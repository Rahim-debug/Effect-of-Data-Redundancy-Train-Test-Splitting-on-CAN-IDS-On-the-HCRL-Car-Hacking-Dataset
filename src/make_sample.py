"""Cut a small class-balanced sample for developing code. NEVER report results from it.
Usage: python src/make_sample.py [frames_per_class=100000]
"""
import sys
import pandas as pd
from config import PROCESSED, SAMPLE, SEEDS


def main(per_class=100_000):
    SAMPLE.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED / "frames.parquet")
    parts = [g.sample(min(len(g), per_class), random_state=SEEDS[0])
             for _, g in df.groupby("label", observed=True)]
    s = pd.concat(parts).sort_values("ts", kind="stable").reset_index(drop=True)
    s.to_parquet(SAMPLE / "frames_sample.parquet", index=False)
    print(f"sample: {len(s):,} frames, {s['sig_id'].nunique():,} signatures")
    print(s["label"].value_counts())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
