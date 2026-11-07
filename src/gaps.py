import sys
import numpy as np, pandas as pd

TRIM = {"DoS_dataset.csv": 2328, "Fuzzy_dataset.csv": 2466, "gear_dataset.csv": 1949, "RPM_dataset.csv": 1952}

def gap_check(df, topk=10):
    rows, summ = [], []
    for f, d in df.groupby("src_file", observed=True, sort=False):
        t = d["t"].to_numpy(); flag = d["flag"].to_numpy().astype(str)
        dt = np.diff(t); t0 = t[0]
        for r in np.argsort(dt)[::-1][:topk]:
            rows.append(dict(src_file=f, gap_s=dt[r], starts_at_s=t[r] - t0, row=int(r)))
        s = dict(src_file=f, frames=len(d), duration_s=t[-1] - t0, median_dt_ms=1000 * np.median(dt),
                 time_goes_backwards=int((dt < 0).sum()), largest_gap_s=dt.max())
        if f in TRIM:
            after = (t - t0) > TRIM[f]
            s.update(trim_mark_s=TRIM[f], frames_after_mark=int(after.sum()), pct_after=100 * after.mean(),
                     T_after_mark=int(((flag == "T") & after).sum()), T_total=int((flag == "T").sum()))
        summ.append(s)
    return pd.DataFrame(rows), pd.DataFrame(summ)

if __name__ == "__main__":
    df = pd.read_parquet(sys.argv[1], columns=["src_file", "t", "flag"])
    gaps, summ = gap_check(df)
    gaps.to_csv(sys.argv[2], index=False); summ.to_csv(sys.argv[2].replace(".csv", "_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False)); print(); print(gaps.round(3).to_string(index=False))
