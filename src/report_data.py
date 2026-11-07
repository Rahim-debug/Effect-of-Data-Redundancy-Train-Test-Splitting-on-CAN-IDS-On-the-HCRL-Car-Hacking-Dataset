import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

CLASSES = ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]
COL = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
FILES = {"normal_run_data.txt": "Normal", "DoS_dataset.csv": "DoS", "Fuzzy_dataset.csv": "Fuzzy",
         "RPM_dataset.csv": "RPM", "gear_dataset.csv": "Gear"}
INK, INK2, GRID, SURF, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#c9c8c2"

def setup():
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True})

def page(title, sub=None):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.06, 0.95, title, fontsize=16, fontweight="bold", va="top")
    if sub: fig.text(0.06, 0.905, sub, fontsize=10, color=INK2, va="top", wrap=True)
    return fig

def table(fig, rect, df, fs=9, scale=1.55):
    ax = fig.add_axes(rect); ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=list(df.columns), loc="upper center", cellLoc="right")
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, scale)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_linewidth(0.5)
        cell.set_facecolor("#efeeea" if r == 0 else SURF)
        if r == 0: cell.set_text_props(fontweight="bold")
        if c == 0: cell.set_text_props(ha="left"); cell._loc = "left"

def hbar(ax, labels, vals, colors, fmt, title):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.5, color=colors, edgecolor=SURF, linewidth=1.5)
    for yi, v in zip(y, vals): ax.text(v, yi, "  " + fmt(v), va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.35); ax.set_title(title, loc="left", fontsize=11, fontweight="bold")

def save(fig, pdf, figdir, name):
    pdf.savefig(fig); fig.savefig(f"{figdir}/{name}.png", dpi=150); plt.close(fig)

def main(base):
    d, r = f"{base}/data/derived", f"{base}/results"; figdir = f"{r}/figs"; os.makedirs(figdir, exist_ok=True)
    setup()
    df = pd.read_parquet(f"{d}/frames.parquet", columns=["src_file", "t", "dlc", "flag", "label"])
    N = len(df); cls = df["label"].value_counts().reindex(CLASSES)
    flag = pd.crosstab(df["src_file"].astype(str), df["flag"].astype(str)).reindex(list(FILES))
    dlc = df["dlc"].value_counts().sort_index(); dlc_lab = pd.crosstab(df["dlc"], df["label"].astype(str))
    piN = cls["Normal"] / N; base_f1 = 2 * piN / (5 * (1 + piN))
    with PdfPages(f"{r}/report_data.pdf") as pdf:
        fig = page("HCRL Car-Hacking: dataset and parsing summary",
                   f"{N:,} frames from 5 files; Normal share = {piN:.4f}; all-Normal baseline macro-F1 = 2*pi_N / (C*(1+pi_N)) = {base_f1:.4f}")
        try: rep = pd.read_csv(f"{d}/frames_report.csv")
        except Exception: rep = pd.DataFrame({"file": list(FILES), "lines_in_file": ["n/a"] * 5, "rows_parsed": [int(flag.loc[f].sum()) for f in FILES], "match": ["n/a"] * 5})
        rep = rep.copy()
        for c in ["lines_in_file", "rows_parsed"]: rep[c] = rep[c].map(lambda v: f"{v:,}" if isinstance(v, (int, np.integer)) else v)
        table(fig, [0.06, 0.45, 0.88, 0.40], rep, fs=10, scale=1.8)
        fig.text(0.06, 0.40, "Parsing rules\n- attack CSV: time, id, dlc, data bytes, flag (R = normal, T = attack of that file); short frames padded with 0 to 8 bytes\n"
                 "- ambient text file: all frames Normal; its last line was cut off in the source and is dropped (988,871 usable frames)\n"
                 "- row count equals line count in every file; no NaN; time is never a model input",
                 fontsize=10, color=INK2, va="top", linespacing=1.6)
        save(fig, pdf, figdir, "p1_summary")
        fig = page("Class balance", "Normal is the large majority, so accuracy alone is misleading; macro-F1 and per-class recall are reported instead.")
        ax = fig.add_axes([0.14, 0.52, 0.80, 0.32])
        hbar(ax, CLASSES, cls.values, [COL[c] for c in CLASSES], lambda v: f"{int(v):,}  ({100*v/N:.1f}%)", "Frames per class")
        ax = fig.add_axes([0.14, 0.08, 0.80, 0.34]); names = list(FILES); y = np.arange(len(names))[::-1]
        Rv, Tv = flag["R"].values, flag["T"].values
        ax.barh(y, Rv, height=0.5, color=NEUTRAL, edgecolor=SURF, linewidth=1.5)
        ax.barh(y, Tv, left=Rv, height=0.5, color=[COL[FILES[f]] for f in names], edgecolor=SURF, linewidth=1.5)
        for yi, a, b in zip(y, Rv, Tv): ax.text(a + b, yi, f"  R {int(a):,}   T {int(b):,}", va="center", fontsize=9)
        ax.set_yticks(y); ax.set_yticklabels(names); ax.grid(axis="y", visible=False); ax.set_xlim(0, (Rv + Tv).max() * 1.5)
        ax.set_title("Frames per file: R (grey) = normal, T (colour) = attack frames of that file", loc="left", fontsize=11, fontweight="bold")
        save(fig, pdf, figdir, "p2_classes")
        fig = page("Data length code (DLC)", "All frames with DLC other than 8 are Normal, so DLC by itself marks some frames as 'not an attack'.")
        ax = fig.add_axes([0.14, 0.50, 0.80, 0.34])
        hbar(ax, [f"DLC {k}" for k in dlc.index], dlc.values, ["#2a78d6"] * len(dlc), lambda v: f"{int(v):,}", "Frames per DLC")
        t = dlc_lab.copy(); t.insert(0, "DLC", t.index); t = t.reset_index(drop=True)
        for c in t.columns[1:]: t[c] = t[c].map(lambda v: f"{int(v):,}")
        table(fig, [0.06, 0.12, 0.88, 0.30], t, fs=10, scale=1.7); save(fig, pdf, figdir, "p3_dlc")
        try:
            summ = pd.read_csv(f"{r}/gap_table_summary.csv").set_index("src_file"); gaps = pd.read_csv(f"{r}/gap_table.csv")
            top = gaps.sort_values("gap_s", ascending=False).groupby("src_file").head(1).set_index("src_file")
            fig = page("Silent gaps and frames after them (H4)", "Each attack file ends with a long silent gap, then attack-free frames. They are kept, as in the proposal, and reported as a limitation.")
            ax = fig.add_axes([0.14, 0.15, 0.80, 0.65]); names = list(FILES); y = np.arange(len(names))[::-1]; xmax = 0
            for yi, f in zip(y, names):
                dur = summ.loc[f, "duration_s"]; xmax = max(xmax, dur); g, s0 = top.loc[f, "gap_s"], top.loc[f, "starts_at_s"]
                if g > 1:
                    ax.barh(yi, s0, height=0.45, color=INK2, edgecolor=SURF, linewidth=1.5)
                    ax.barh(yi, g, left=s0, height=0.45, color=SURF, edgecolor=INK2, hatch="////", linewidth=0.8)
                    ax.barh(yi, dur - s0 - g, left=s0 + g, height=0.45, color="#6da7ec", edgecolor=SURF, linewidth=1.5)
                    note = f"  gap {g:,.0f} s (starts at {s0:,.0f} s)\n  {int(summ.loc[f, 'frames_after_mark']):,} frames after the mark ({summ.loc[f, 'pct_after']:.1f}%)"
                else:
                    ax.barh(yi, dur, height=0.45, color=INK2, edgecolor=SURF, linewidth=1.5); note = f"  no gap (largest {g*1000:.1f} ms)"
                ax.text(dur, yi, note, va="center", fontsize=9)
            ax.set_yticks(y); ax.set_yticklabels(names); ax.grid(axis="y", visible=False); ax.set_xlim(0, xmax * 1.6)
            ax.set_xlabel("Seconds since first frame of the file")
            ax.legend(handles=[Patch(color=INK2, label="before the gap (attacks and normal frames)"), Patch(facecolor=SURF, edgecolor=INK2, hatch="////", label="silent gap"),
                               Patch(color="#6da7ec", label="after the gap (attack-free)")], loc="upper right", frameon=False)
            save(fig, pdf, figdir, "p4_gaps")
            fig = page("Gap tables", "Summary per file and the three largest inter-arrival gaps")
            s = summ.reset_index()[["src_file", "frames", "duration_s", "median_dt_ms", "largest_gap_s", "frames_after_mark", "T_after_mark"]].copy()
            s["frames"] = s["frames"].map(lambda v: f"{int(v):,}")
            for c in ["duration_s", "median_dt_ms", "largest_gap_s"]: s[c] = s[c].map(lambda v: f"{v:,.3f}")
            for c in ["frames_after_mark", "T_after_mark"]: s[c] = s[c].map(lambda v: "-" if pd.isna(v) else f"{int(v):,}")
            table(fig, [0.06, 0.50, 0.88, 0.36], s, fs=9, scale=1.7)
            g3 = gaps.sort_values(["src_file", "gap_s"], ascending=[True, False]).groupby("src_file").head(3).copy()
            for c in ["gap_s", "starts_at_s"]: g3[c] = g3[c].map(lambda v: f"{v:,.3f}")
            table(fig, [0.06, 0.04, 0.60, 0.42], g3[["src_file", "gap_s", "starts_at_s", "row"]], fs=8, scale=1.3)
            save(fig, pdf, figdir, "p5_gap_tables")
        except FileNotFoundError:
            print("gap_table.csv not found: run gaps.py first; gap pages skipped")
    print("saved", f"{r}/report_data.pdf")

if __name__ == "__main__":
    main(sys.argv[1])
