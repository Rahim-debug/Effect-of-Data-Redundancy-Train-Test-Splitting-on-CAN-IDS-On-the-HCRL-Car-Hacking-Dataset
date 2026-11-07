import os, sys, collections
import numpy as np, pandas as pd

ATTACKS = {"dos_dataset.csv": "DoS", "fuzzy_dataset.csv": "Fuzzy",
           "rpm_dataset.csv": "RPM", "gear_dataset.csv": "Gear"}
NORMAL = "normal_run_data.txt"
BCOLS = [f"d{i}" for i in range(8)]

def n_lines(path):
    n, last = 0, b"\n"
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""):
            n += ch.count(b"\n"); last = ch[-1:]
    return n + (0 if last == b"\n" else 1)

def hex_to_int(s):
    mp = {}
    for x in s.dropna().unique():
        try: mp[x] = int(x, 16)
        except ValueError: mp[x] = -1
    return s.map(mp).fillna(-1).to_numpy(dtype=np.int64)

def make_frame(src, t, ids, dlc, by, flag, label):
    df = pd.DataFrame({"src_file": pd.Categorical([src] * len(t)), "t": np.asarray(t, dtype=np.float64),
                       "id": np.asarray(ids, dtype=np.uint16), "dlc": np.asarray(dlc, dtype=np.uint8)})
    by = np.asarray(by, dtype=np.uint8)
    for i, c in enumerate(BCOLS): df[c] = by[:, i]
    df["flag"] = pd.Categorical(flag, categories=["R", "T"])
    df["label"] = pd.Categorical(label, categories=["Normal", "DoS", "Fuzzy", "RPM", "Gear"])
    return df

def parse_attack(path, attack, chunksize=500_000):
    parts = []
    for ch in pd.read_csv(path, header=None, names=list(range(12)), dtype=str, chunksize=chunksize):
        n = len(ch)
        dlc = pd.to_numeric(ch[2]).to_numpy().astype(np.int64)
        assert dlc.min() >= 0 and dlc.max() <= 8, "DLC outside 0..8"
        arr = ch.to_numpy(dtype=object)
        flag = arr[np.arange(n), 3 + dlc]
        assert np.isin(flag, ["R", "T"]).all(), "flag not R/T: field count does not match DLC"
        ids = hex_to_int(ch[1]); assert (ids >= 0).all(), "bad CAN id"
        pos = np.arange(8)[None, :] < dlc[:, None]
        by = np.zeros((n, 8), dtype=np.uint8)
        for i in range(8):
            v = hex_to_int(ch[3 + i])
            assert ((v[pos[:, i]] >= 0) & (v[pos[:, i]] <= 255)).all(), "bad data byte"
            by[:, i] = np.where(pos[:, i], v, 0)
        label = np.where(flag == "T", attack, "Normal")
        parts.append(make_frame(os.path.basename(path), ch[0].astype(float), ids, dlc, by, flag, label))
    return pd.concat(parts, ignore_index=True)

def parse_normal(path):
    t, ids, dlc, by, extra, blank = [], [], [], [], collections.Counter(), 0
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            tok = line.split()
            if not tok: blank += 1; continue
            if len(tok) < 7 and not line.endswith("\n"):      # cut-off last line of the file
                print(f"skipped truncated last line {ln}: {line!r}"); blank += 1; continue
            assert len(tok) >= 7 and tok[0] == "Timestamp:" and tok[2] == "ID:" and tok[5] == "DLC:", f"unexpected format, line {ln}: {line!r}"
            d = int(tok[6]); assert 0 <= d <= 8 and len(tok) == 7 + d, f"field count != DLC, line {ln}"
            extra[tok[4]] += 1
            t.append(float(tok[1])); ids.append(int(tok[3], 16)); dlc.append(d)
            by.append([int(x, 16) for x in tok[7:]] + [0] * (8 - d))
    n = len(t)
    return make_frame(os.path.basename(path), t, ids, dlc, by, ["R"] * n, ["Normal"] * n), blank, extra

def main(raw, out):
    files = {f.lower(): os.path.join(raw, f) for f in os.listdir(raw)}
    missing = [k for k in list(ATTACKS) + [NORMAL] if k not in files]
    assert not missing, f"missing files in {raw}: {missing}"
    frames, report = [], []
    df, blank, extra = parse_normal(files[NORMAL]); expected = n_lines(files[NORMAL]) - blank
    print(f"{NORMAL}: field after ID takes values {dict(extra)}")
    frames.append(df); report.append((NORMAL, expected, len(df)))
    for name, cls in ATTACKS.items():
        df = parse_attack(files[name], cls); expected = n_lines(files[name])
        frames.append(df); report.append((name, expected, len(df))); print("parsed", name, len(df))
    all_ = pd.concat(frames, ignore_index=True)
    rep = pd.DataFrame(report, columns=["file", "lines_in_file", "rows_parsed"])
    rep["match"] = rep.lines_in_file == rep.rows_parsed
    print(rep.to_string(index=False))
    assert rep.match.all(), "row count != line count"
    assert not all_.isna().any().any(), "NaN found"
    print("\nlabel counts:\n", all_.label.value_counts().reindex(["Normal", "DoS", "Fuzzy", "RPM", "Gear"]).to_string())
    print("\nflag per file:\n", pd.crosstab(all_.src_file, all_.flag).to_string())
    print("\nDLC counts:\n", all_.dlc.value_counts().sort_index().to_string())
    print(f"\nID range: {all_.id.min():#05x} to {all_.id.max():#05x}   unique IDs: {all_.id.nunique()}")
    all_.to_parquet(out, index=False); rep.to_csv(os.path.splitext(out)[0] + "_report.csv", index=False)
    print("\nwrote", out, len(all_), "rows")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
