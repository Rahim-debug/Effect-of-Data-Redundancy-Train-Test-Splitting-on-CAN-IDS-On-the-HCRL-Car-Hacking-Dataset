import sys, os, time, argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from checks import sig_ids
from temporal_audit import plan_file

CLASSES = ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]
FEATS = ["id", "dlc"] + [f"d{i}" for i in range(8)]
COLS = ["src_file", "t", "flag", "label"] + FEATS

def make_model(name, seed):
    if name == "lr": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, class_weight="balanced", random_state=seed))
    if name == "dt": return DecisionTreeClassifier(class_weight="balanced", random_state=seed)
    if name == "rf": return RandomForestClassifier(n_estimators=50, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)

def split_A(lab, seed, test_frac=0.2):
    rng = np.random.default_rng(seed); test = np.zeros(len(lab), dtype=bool)
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(lab == c); test[rng.choice(idx, int(round(test_frac * len(idx))), replace=False)] = True
    return ~test, test

def split_B(src, t, isT):
    train = np.zeros(len(t), dtype=bool); test = np.zeros(len(t), dtype=bool)
    for f in pd.unique(src):
        idx = np.flatnonzero(src == f); p = plan_file(t[idx], isT[idx]); a = idx[:p["E"]]
        train[a[t[a] <= p["tau"]]] = True; test[a[t[a] > p["tau"]]] = True
    return train, test

def metrics(y, p, seen):
    labs = list(range(len(CLASSES))); out = {}; f1s = []
    for c in labs:
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        n_c = np.sum(y == c); out[f"recall_{CLASSES[c]}"] = tp / n_c if n_c else np.nan
        for tag, m in (("seen", seen), ("unseen", ~seen)):
            n = np.sum((y == c) & m); out[f"recall_{tag}_{CLASSES[c]}"] = np.sum((y == c) & m & (p == c)) / n if n else np.nan
    out["macro_f1"] = float(np.mean(f1s)); out["accuracy"] = float(np.mean(y == p)); return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--seeds", default="42"); ap.add_argument("--models", default="lookup,dt,lr,rf"); a = ap.parse_args()
    df = pd.read_parquet(f"{a.base}/data/derived/frames_clean.parquet", columns=COLS).iloc[::a.stride].reset_index(drop=True)
    sig = sig_ids(df); lab = df["label"].cat.codes.to_numpy().astype(np.int64); U = int(sig.max()) + 1
    X = df[FEATS].to_numpy(dtype=np.float32); src = df["src_file"].astype(str).to_numpy(); t = df["t"].to_numpy(); isT = (df["flag"].astype(str) == "T").to_numpy()
    print(f"sample: {len(df):,} rows (stride {a.stride});", {n: int((lab == i).sum()) for i, n in enumerate(CLASSES)})
    rows = []; seeds = [int(s) for s in a.seeds.split(",")]
    for proto in ("A", "B"):
        for seed in seeds:
            tr, te = split_A(lab, seed) if proto == "A" else split_B(src, t, isT)
            in_train = np.zeros(U, dtype=bool); in_train[sig[tr]] = True; seen = in_train[sig[te]]
            for name in a.models.split(","):
                t0 = time.time()
                if name == "lookup":
                    lut = np.full(U, -1, dtype=np.int64); lut[sig[tr]] = lab[tr]; p = lut[sig[te]]; p[p < 0] = 0
                else:
                    m = make_model(name, seed).fit(X[tr], lab[tr]); p = m.predict(X[te])
                r = dict(protocol=proto, seed=seed, model=name, n_train=int(tr.sum()), n_test=int(te.sum()), seen_share=float(seen.mean()), fit_s=round(time.time() - t0, 1))
                r.update(metrics(lab[te], p, seen)); rows.append(r); print(proto, seed, name, "macro-F1 =", round(r["macro_f1"], 4), f"({r['fit_s']}s)")
    out = pd.DataFrame(rows); os.makedirs(f"{a.base}/results", exist_ok=True); out.to_csv(f"{a.base}/results/pilot_runs.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nmacro-F1 and per-class recall\n", out[["protocol", "model", "macro_f1", "accuracy"] + [f"recall_{c}" for c in CLASSES]].round(4).to_string(index=False))
    print("\nrecall on UNSEEN-signature test frames\n", out[["protocol", "model"] + [f"recall_unseen_{c}" for c in CLASSES]].round(4).to_string(index=False))

if __name__ == "__main__":
    main()
