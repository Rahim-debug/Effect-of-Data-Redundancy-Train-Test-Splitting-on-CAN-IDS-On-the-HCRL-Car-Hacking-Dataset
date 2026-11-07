import sys, os
import numpy as np, pandas as pd
from checks import sig_ids

CLASSES = ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]
COLS = ["src_file", "t", "id", "dlc"] + [f"d{i}" for i in range(8)] + ["flag", "label"]
FRAC, BURST_GAP, SILENT = 0.8, 1.0, 1.0

def plan_file(t, isT):
    n = len(t); dt = np.diff(t)
    r = int(np.argmax(dt)) if n > 1 else 0
    E = r + 1 if (n > 1 and dt[r] > SILENT) else n
    ta, tT = t[:E], t[:E][isT[:E]]
    if len(tT):
        br = np.flatnonzero(np.diff(tT) > BURST_GAP)
        starts, ends = np.r_[tT[0], tT[br + 1]], np.r_[tT[br], tT[-1]]
    else:
        starts = ends = np.array([])
    tau = ta[0] + FRAC * (ta[-1] - ta[0]); quiet = np.nan
    if len(starts):
        inside = np.flatnonzero((starts <= tau) & (tau <= ends))
        if len(inside):
            k = inside[0]; lo = ends[k]; hi = starts[k + 1] if k + 1 < len(starts) else ta[-1]
        else:
            k = np.searchsorted(starts, tau) - 1; lo = ends[k] if k >= 0 else ta[0]; hi = starts[k + 1] if k + 1 < len(starts) else ta[-1]
        tau, quiet = 0.5 * (lo + hi), hi - lo
    return dict(E=E, tau=tau, quiet=quiet, nb=len(starts), med_burst=float(np.median(ends - starts)) if len(starts) else np.nan,
                med_quiet=float(np.median(starts[1:] - ends[:-1])) if len(starts) > 1 else np.nan)

def run(df, base=None):
    sig = sig_ids(df); lab = df["label"].cat.codes.to_numpy().astype(np.int64); U = int(sig.max()) + 1
    src = df["src_file"].astype(str).to_numpy(); t = df["t"].to_numpy(); isT = (df["flag"].astype(str) == "T").to_numpy()
    n = len(df); train = np.zeros(n, bool); test = np.zeros(n, bool); active = np.zeros(n, bool); rows = []
    for f in pd.unique(src):
        idx = np.flatnonzero(src == f); p = plan_file(t[idx], isT[idx]); E = p["E"]
        a = idx[:E]; active[a] = True; train[a[t[a] <= p["tau"]]] = True; test[a[t[a] > p["tau"]]] = True
        nv = int(0.8 * len(idx))
        rows.append(dict(file=f, bursts=p["nb"], median_burst_s=p["med_burst"], median_quiet_s=p["med_quiet"], rows=len(idx), rows_after_gap=len(idx) - E,
                         cut_s_from_start=p["tau"] - t[idx][0], quiet_at_cut_s=p["quiet"], T_total=int(isT[idx].sum()),
                         T_test=int(isT[a][t[a] > p["tau"]].sum()), T_test_if_no_sanitising=int(isT[idx][nv:].sum())))
    ft = pd.DataFrame(rows); ft["T_test_share"] = ft.T_test / ft.T_total
    in_train = np.zeros(U, bool); in_train[sig[train]] = True; cr = []
    for c, name in enumerate(CLASSES):
        m = lab == c; a_c, tr_c, te_c = int((m & active).sum()), int((m & train).sum()), int((m & test).sum())
        cnt = np.bincount(sig[m & active], minlength=U).astype(float); cnt = cnt[cnt > 0]
        cr.append(dict(label=name, active_frames=a_c, train=tr_c, test=te_c, test_share=te_c / max(a_c, 1),
                       seen_share_temporal=float(in_train[sig[m & test]].mean()) if te_c else np.nan,
                       seen_share_random_expected=float(np.sum(cnt * (1 - 0.2 ** (cnt - 1))) / cnt.sum()) if len(cnt) else np.nan))
    ct = pd.DataFrame(cr)
    if base:
        os.makedirs(f"{base}/results", exist_ok=True); ft.to_csv(f"{base}/results/temporal_audit_files.csv", index=False); ct.to_csv(f"{base}/results/temporal_audit_classes.csv", index=False)
    return ft, ct

if __name__ == "__main__":
    base = sys.argv[1]; df = pd.read_parquet(f"{base}/data/derived/frames.parquet", columns=COLS)
    ft, ct = run(df, base); pd.set_option("display.width", 250)
    print("PER FILE\n", ft.round(3).to_string(index=False)); print("\nPER CLASS (active frames only; temporal 80/20 inside each file)\n", ct.round(4).to_string(index=False))
