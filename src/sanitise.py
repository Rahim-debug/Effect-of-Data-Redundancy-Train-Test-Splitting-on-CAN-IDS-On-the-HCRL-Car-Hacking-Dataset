import sys, os
import numpy as np, pandas as pd

SILENT = 1.0
CLASSES = ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]

def sanitise(df):
    src = df["src_file"].astype(str).to_numpy(); t = df["t"].to_numpy()
    keep = np.ones(len(df), dtype=bool); log = []
    for f in pd.unique(src):
        idx = np.flatnonzero(src == f); dt = np.diff(t[idx])
        r = int(np.argmax(dt)) if len(dt) else 0
        has_gap = len(dt) > 0 and dt[r] > SILENT
        cut = r + 1 if has_gap else len(idx)
        keep[idx[cut:]] = False
        log.append(dict(file=f, rows=len(idx), kept=cut, dropped=len(idx) - cut, gap_s=float(dt[r]) if len(dt) else 0.0))
    return df[keep].reset_index(drop=True), pd.DataFrame(log)

if __name__ == "__main__":
    base = sys.argv[1]
    df = pd.read_parquet(f"{base}/data/derived/frames.parquet")
    clean, log = sanitise(df)
    clean.to_parquet(f"{base}/data/derived/frames_clean.parquet", index=False)
    os.makedirs(f"{base}/results", exist_ok=True); log.to_csv(f"{base}/results/sanitise_log.csv", index=False)
    pd.set_option("display.width", 200)
    print(log.to_string(index=False))
    cnt = clean["label"].value_counts().reindex(CLASSES); N = len(clean); piN = cnt["Normal"] / N
    print("\nclean frames:", f"{N:,}", " removed:", f"{len(df) - N:,}")
    print(cnt.to_string()); print(f"\nNormal share = {piN:.4f}   all-Normal baseline macro-F1 = {2*piN/(5*(1+piN)):.4f}")
