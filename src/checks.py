import sys
import numpy as np, pandas as pd

CLASSES = ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]
COLS = ["src_file", "t", "id", "dlc"] + [f"d{i}" for i in range(8)] + ["label"]

def sig_ids(df):
    k1 = (df["id"].to_numpy().astype(np.uint16) << np.uint16(4)) | df["dlc"].to_numpy().astype(np.uint16)
    k2 = np.zeros(len(df), dtype=np.uint64)
    for i in range(8): k2 = (k2 << np.uint64(8)) | df[f"d{i}"].to_numpy().astype(np.uint64)
    return pd.DataFrame({"a": k1, "b": k2}).groupby(["a", "b"], sort=False).ngroup().to_numpy().astype(np.int64)

def run(df, tau=0.2):
    sig = sig_ids(df); lab = df["label"].cat.codes.to_numpy().astype(np.int64); U = sig.max() + 1
    print("== 1. Classes with <= 3 signatures: the signatures themselves")
    for c, name in enumerate(CLASSES):
        m = lab == c; first = np.unique(sig[m])
        if len(first) <= 3:
            for s in first:
                r = df[m].iloc[int(np.argmax(sig[m] == s))]
                print(f"{name}: id=0x{int(r['id']):03x} dlc={int(r['dlc'])} data=" + " ".join(f"{int(r[f'd{i}']):02x}" for i in range(8)),
                      f" frames={int((sig[m] == s).sum()):,}")
    print("\n== 2. Expected leakage under P1: share of test frames whose signature also occurs in train, L = sum (m/N_c)(1 - tau^(m-1)), tau =", tau)
    rows = []
    for c, name in enumerate(CLASSES):
        cnt = np.bincount(sig[lab == c], minlength=U); cnt = cnt[cnt > 0].astype(float)
        rows.append((name, int(cnt.sum()), len(cnt), float(np.sum(cnt * (1 - tau ** (cnt - 1))) / cnt.sum())))
    print(pd.DataFrame(rows, columns=["label", "N_c", "U_c", "L_expected"]).round(4).to_string(index=False))
    print("\n== 3. Segments after the big gap vs the ambient file")
    src = df["src_file"].astype(str).to_numpy(); t = df["t"].to_numpy(); segs = {}
    for f in pd.unique(src):
        idx = np.where(src == f)[0]
        if f.lower().startswith("normal"): segs["ambient"] = idx
        else:
            r = int(np.argmax(np.diff(t[idx]))); segs[f.split("_")[0] + "_post_gap"] = idx[r + 1:]
    names = list(segs); S = {k: set(np.unique(sig[v]).tolist()) for k, v in segs.items()}
    print(pd.DataFrame({"segment": names, "frames": [len(segs[k]) for k in names], "signatures": [len(S[k]) for k in names]}).to_string(index=False))
    J = pd.DataFrame([[len(S[a] & S[b]) / len(S[a] | S[b]) for b in names] for a in names], index=names, columns=names)
    print("\nJaccard overlap of signature sets (1 = identical sets):\n", J.round(3).to_string())
    n = min(len(segs[k]) for k in names if k != "ambient")
    eq = {k: float((sig[segs[k][:n]] == sig[segs[names[1]][:n]]).mean()) for k in names[1:]}
    print("\nFrame-by-frame equal signature, first", n, "frames of each post-gap segment vs", names[1] + ":", {k: round(v, 4) for k, v in eq.items()})

if __name__ == "__main__":
    run(pd.read_parquet(sys.argv[1], columns=COLS))
