import sys, os
import numpy as np, pandas as pd
from checks import sig_ids

CLASSES = ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]
COLS = ["id", "dlc"] + [f"d{i}" for i in range(8)] + ["label"]

def run(df, seeds=(42, 43, 44, 45, 46), test_frac=0.2):
    sig = sig_ids(df); lab = df["label"].cat.codes.to_numpy().astype(np.int64); U = int(sig.max()) + 1; rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed); test = np.zeros(len(df), dtype=bool)
        for c in range(len(CLASSES)):
            idx = np.flatnonzero(lab == c); test[rng.choice(idx, int(round(test_frac * len(idx))), replace=False)] = True
        in_train = np.zeros(U, dtype=bool); in_train[sig[~test]] = True
        for c, name in enumerate(CLASSES):
            m = (lab == c) & test
            rows.append(dict(seed=seed, label=name, test_frames=int(m.sum()), seen_share=float(in_train[sig[m]].mean())))
    out = pd.DataFrame(rows)
    return out, out.groupby("label", sort=False).agg(test_frames=("test_frames", "mean"), seen_mean=("seen_share", "mean"), seen_sd=("seen_share", "std")).reindex(CLASSES)

if __name__ == "__main__":
    base = sys.argv[1]; df = pd.read_parquet(f"{base}/data/derived/frames_clean.parquet", columns=COLS)
    per_seed, summ = run(df); os.makedirs(f"{base}/results", exist_ok=True)
    per_seed.to_csv(f"{base}/results/random_overlap_per_seed.csv", index=False); summ.to_csv(f"{base}/results/random_overlap_summary.csv")
    pd.set_option("display.width", 200); print(summ.round(5).to_string())
