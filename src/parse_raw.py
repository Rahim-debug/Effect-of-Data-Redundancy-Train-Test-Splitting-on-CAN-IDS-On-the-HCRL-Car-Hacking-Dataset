"""Parse the five raw HCRL Car-Hacking files into one common frame table.

Attack CSVs (no header):  ts, ID(hex), DLC, byte0..byte(DLC-1) (hex), flag (R/T)
    -> the flag sits right after the last data byte, so its column depends on DLC.
Normal file (text):       Timestamp: <ts>  ID: <hex>  000  DLC: <n>  <bytes in hex>
Verify these formats against the first lines of your files (see README / Day 1 step 2).
"""
import numpy as np
import pandas as pd

ATTACK_FILES = {                       # file name -> (source, label given to T frames)
    "DoS_dataset.csv":   ("dos",   "DoS"),
    "Fuzzy_dataset.csv": ("fuzzy", "Fuzzy"),
    "RPM_dataset.csv":   ("rpm",   "RPM"),
    "gear_dataset.csv":  ("gear",  "Gear"),
}
NORMAL_FILE = "normal_run_data.txt"


def hex_to_int(s: pd.Series) -> pd.Series:
    """Hex strings -> numbers. Each distinct string is converted once (fast). NaN stays NaN."""
    mapping = {v: int(v, 16) for v in s.dropna().unique()}
    return s.map(mapping)


def _assemble(ts_s, id_s, dlc_s, data_cols, flag, label_if_t, source) -> pd.DataFrame:
    n = len(ts_s)
    dlc = pd.to_numeric(dlc_s).to_numpy().astype(np.int64)
    assert dlc.min() >= 0 and dlc.max() <= 8, f"{source}: DLC outside 0..8"

    cols = {}
    data64 = np.zeros(n, dtype=np.uint64)
    for i, s in enumerate(data_cols):
        s = s.where(dlc > i)                       # bytes beyond the DLC do not exist -> NaN
        b = hex_to_int(s).fillna(0).astype(np.uint8).to_numpy()
        cols[f"d{i}"] = b
        data64 = (data64 << np.uint64(8)) | b.astype(np.uint64)

    flag = np.asarray(flag, dtype=object)
    bad = set(np.unique(flag)) - {"R", "T"}
    assert not bad, f"{source}: unexpected flag values {bad}"

    return pd.DataFrame({
        "ts": pd.to_numeric(ts_s).to_numpy(np.float64),
        "id": hex_to_int(id_s).astype(np.uint16).to_numpy(),
        "dlc": dlc.astype(np.uint8),
        **cols,
        "data64": data64,
        "label": np.where(flag == "T", label_if_t, "Normal"),
        "source": source,
        "flag": flag,
    })


def parse_attack_csv(path, source, label, chunksize=2_000_000) -> pd.DataFrame:
    parts = []
    for ch in pd.read_csv(path, header=None, names=range(12), dtype=str, chunksize=chunksize):
        dlc = pd.to_numeric(ch[2]).to_numpy().astype(np.int64)
        assert dlc.min() >= 0 and dlc.max() <= 8, f"{path}: bad DLC"
        flag = ch.to_numpy(dtype=object)[np.arange(len(ch)), 3 + dlc]   # column right after the data
        parts.append(_assemble(ch[0], ch[1], ch[2], [ch[3 + i] for i in range(8)],
                               flag, label, source))
    return pd.concat(parts, ignore_index=True)


def parse_normal_txt(path) -> pd.DataFrame:
    ch = pd.read_csv(path, sep=r"\s+", header=None, names=range(15), dtype=str)
    assert (ch[0] == "Timestamp:").all() and (ch[2] == "ID:").all() and (ch[5] == "DLC:").all(), \
        "normal file does not match the expected layout: inspect its first lines"
    flag = np.full(len(ch), "R", dtype=object)
    return _assemble(ch[1], ch[3], ch[6], [ch[7 + i] for i in range(8)], flag, "Normal", "normal")
