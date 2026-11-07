import sys
import numpy as np, pandas as pd

CLASSES = ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]
COLS = ["id", "dlc"] + [f"d{i}" for i in range(8)] + ["label"]

def audit(df):
    assert list(df["label"].cat.categories) == CLASSES
    lab = df["label"].cat.codes.to_numpy().astype(np.int64)
    k1 = (df["id"].to_numpy().astype(np.uint16) << np.uint16(4)) | df["dlc"].to_numpy().astype(np.uint16)
    k2 = np.zeros(len(df), dtype=np.uint64)
    for i in range(8):
        k2 = (k2 << np.uint64(8)) | df[f"d{i}"].to_numpy().astype(np.uint64)
    sig = pd.DataFrame({"a": k1, "b": k2}).groupby(["a", "b"], sort=False).ngroup().to_numpy().astype(np.int64)
    N, U = len(sig), int(sig.max()) + 1
    pairs = np.unique(sig * 8 + lab)
    ps, pl = pairs // 8, pairs % 8
    conflict = np.bincount(ps, minlength=U) > 1
    rows = []
    for c, name in enumerate(CLASSES):
        m = lab == c
        Nc = int(m.sum()); Uc = int((pl == c).sum())
        bc = np.bincount(sig[m], minlength=U); cnt = np.sort(bc[bc > 0])[::-1]
        rows.append(dict(label=name, N_c=Nc, share=Nc / N, U_c=Uc, rho_c=1 - Uc / Nc,
                         top_sig_share=cnt[0] / Nc, sigs_for_50pct=int(np.searchsorted(np.cumsum(cnt), 0.5 * Nc) + 1),
                         conflicting_frames=int(conflict[sig[m]].sum()), p_no_test_sig=0.8 ** Uc))
    out = pd.DataFrame(rows)
    piN = out.loc[0, "share"]; base = 2 * piN / (len(CLASSES) * (1 + piN))
    allc = np.sort(np.bincount(sig))[::-1]
    tot = dict(label="ALL", N_c=N, share=1.0, U_c=U, rho_c=1 - U / N, top_sig_share=allc[0] / N,
               sigs_for_50pct=int(np.searchsorted(np.cumsum(allc), 0.5 * N) + 1),
               conflicting_frames=int(conflict[sig].sum()), p_no_test_sig=np.nan)
    out = pd.concat([out, pd.DataFrame([tot])], ignore_index=True)
    assert out.loc[out.label != "ALL", "N_c"].sum() == N and U <= N
    return out, int(conflict.sum()) / U, base

if __name__ == "__main__":
    df = pd.read_parquet(sys.argv[1], columns=COLS)
    out, kappa, base = audit(df)
    out["kappa_global"] = np.nan; out.loc[out.label == "ALL", "kappa_global"] = kappa
    out["trivial_macroF1"] = np.nan; out.loc[out.label == "ALL", "trivial_macroF1"] = base
    out.to_csv(sys.argv[2], index=False)
    pd.set_option("display.width", 250); print(out.round(4).to_string(index=False))
    print(f"\nkappa = {kappa:.6f}   trivial all-Normal macro-F1 = {base:.4f}")
