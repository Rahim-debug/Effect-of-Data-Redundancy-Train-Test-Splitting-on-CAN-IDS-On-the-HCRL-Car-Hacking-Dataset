import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from audit import audit, CLASSES, COLS as AUDIT_COLS
from checks import sig_ids

COL = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

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

def table(fig, rect, df, fs=9, scale=1.55, colw=None):
    ax = fig.add_axes(rect); ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=list(df.columns), loc="upper center", cellLoc="right", colWidths=colw)
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, scale)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_linewidth(0.5); cell.set_facecolor("#efeeea" if r == 0 else SURF)
        if r == 0: cell.set_text_props(fontweight="bold")
        if c == 0: cell.set_text_props(ha="left"); cell._loc = "left"

def hbar01(ax, vals, fmt, title):
    y = np.arange(len(CLASSES))[::-1]
    ax.barh(y, vals, height=0.5, color=[COL[c] for c in CLASSES], edgecolor=SURF, linewidth=1.5)
    for yi, v in zip(y, vals): ax.text(v, yi, "  " + fmt(v), va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(CLASSES); ax.grid(axis="y", visible=False); ax.set_xlim(0, 1.25)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")

def save(fig, pdf, figdir, name):
    pdf.savefig(fig); fig.savefig(f"{figdir}/{name}.png", dpi=150); plt.close(fig)

def main(base, tau=0.2):
    d, r = f"{base}/data/derived", f"{base}/results"; figdir = f"{r}/figs"; os.makedirs(figdir, exist_ok=True); setup()
    df = pd.read_parquet(f"{d}/frames.parquet", columns=AUDIT_COLS + ["src_file"])
    out, kappa, base_f1 = audit(df)
    sig = sig_ids(df); lab = df["label"].cat.codes.to_numpy().astype(np.int64); U = int(sig.max()) + 1
    L = []
    for c in range(len(CLASSES)):
        cnt = np.bincount(sig[lab == c], minlength=U); cnt = cnt[cnt > 0].astype(float)
        L.append(float(np.sum(cnt * (1 - tau ** (cnt - 1))) / cnt.sum()))
    cl = out[out.label != "ALL"].reset_index(drop=True); allr = out[out.label == "ALL"].iloc[0]
    with PdfPages(f"{r}/report_audit.pdf") as pdf:
        fig = page("Signature and redundancy audit (H3)",
                   f"Signature = (id, dlc, d0..d7).  N = {int(allr.N_c):,} frames, U = {int(allr.U_c):,} unique signatures, redundancy rho = 1 - U/N = {allr.rho_c:.4f}, "
                   f"label-conflict rate kappa = {kappa:.6f}, all-Normal baseline macro-F1 = {base_f1:.4f}")
        t = cl[["label", "N_c", "share", "U_c", "rho_c", "top_sig_share", "sigs_for_50pct", "conflicting_frames", "p_no_test_sig"]].copy()
        t["N_c"] = t["N_c"].map(lambda v: f"{int(v):,}"); t["U_c"] = t["U_c"].map(lambda v: f"{int(v):,}")
        t["sigs_for_50pct"] = t["sigs_for_50pct"].map(lambda v: f"{int(v):,}")
        for c in ["share", "rho_c", "top_sig_share", "p_no_test_sig"]: t[c] = t[c].map(lambda v: f"{v:.4f}")
        t["conflicting_frames"] = t["conflicting_frames"].map(lambda v: f"{int(v):,}")
        table(fig, [0.06, 0.52, 0.88, 0.34], t, fs=9, scale=1.9)
        fig.text(0.06, 0.45, "Columns: N_c frames; U_c unique signatures; rho_c = 1 - U_c/N_c; top_sig_share = share of the class held by its most common signature;\n"
                 "sigs_for_50pct = signatures needed to cover half the class; p_no_test_sig = 0.8^U_c (chance a class has no test signature under P2).",
                 fontsize=9, color=INK2, va="top", linespacing=1.6)
        save(fig, pdf, figdir, "a1_summary")
        fig = page("Frames versus unique signatures", "One axis (counts, log scale). The gap between the two dots is the redundancy.")
        ax = fig.add_axes([0.14, 0.12, 0.78, 0.70]); y = np.arange(len(CLASSES))[::-1]
        for yi, n_, u_ in zip(y, cl.N_c, cl.U_c):
            ax.plot([u_, n_], [yi, yi], color=GRID, linewidth=2, zorder=1)
            ax.scatter([n_], [yi], s=150, color=INK2, zorder=2, edgecolor=SURF, linewidth=1.5)
            ax.scatter([u_], [yi], s=55, color="#2a78d6", zorder=3, edgecolor=SURF, linewidth=1.0)
            ax.text(n_ * 1.3, yi + 0.12, f"{int(n_):,} frames", fontsize=9, va="bottom"); ax.text(u_ * 1.3, yi - 0.12, f"{int(u_):,} unique", fontsize=9, va="top")
        ax.set_xscale("log"); ax.set_xlim(0.5, cl.N_c.max() * 15); ax.set_ylim(-0.7, len(CLASSES) - 0.3); ax.set_yticks(y); ax.set_yticklabels(CLASSES); ax.grid(axis="y", visible=False)
        ax.scatter([], [], s=110, color=INK2, label="frames (N_c)"); ax.scatter([], [], s=55, color="#2a78d6", label="unique signatures (U_c)")
        ax.legend(loc="lower right", frameon=False); ax.set_xlabel("count (log scale)")
        save(fig, pdf, figdir, "a2_frames_vs_signatures")
        fig = page("Redundancy and expected leakage under the random split (P1)",
                   f"Leakage L = share of test frames whose exact signature also occurs in training: L = sum_s (m_s/N_c)(1 - tau^(m_s - 1)), tau = {tau}.")
        hbar01(fig.add_axes([0.14, 0.52, 0.78, 0.30]), cl.rho_c.values, lambda v: f"{v:.4f}", "Redundancy rho_c = 1 - U_c / N_c")
        hbar01(fig.add_axes([0.14, 0.10, 0.78, 0.30]), np.array(L), lambda v: f"{v:.4f}", "Expected leakage L under P1 (0 = no identical twin in training, 1 = every test frame has one)")
        save(fig, pdf, figdir, "a3_redundancy_leakage")
        fig = page("Can the group split (P2) evaluate each class?",
                   "A class with one signature is a single block: it goes wholly to training or wholly to test, so it is missing from one side in every seed.")
        f = pd.DataFrame({"class": CLASSES, "U_c": [f"{int(u):,}" for u in cl.U_c],
                          "status under P2": ["NOT EVALUABLE (single block)" if u == 1 else ("AT RISK (fewer than 10 signatures)" if u < 10 else "evaluable") for u in cl.U_c],
                          "top_sig_share": [f"{v:.4f}" for v in cl.top_sig_share], "L under P1": [f"{v:.4f}" for v in L]})
        table(fig, [0.06, 0.50, 0.88, 0.34], f, fs=10, scale=2.0, colw=[0.14, 0.10, 0.42, 0.17, 0.17])
        fig.text(0.06, 0.38, "Reading guide\n- evaluable: more than one signature, so both sides can hold the class (validity still checked per seed)\n"
                 "- not evaluable classes are reported as a limitation, not forced; comparison is then made on classes evaluable in both protocols\n"
                 "- kappa = 0 means no signature appears under two labels", fontsize=10, color=INK2, va="top", linespacing=1.7)
        save(fig, pdf, figdir, "a4_p2_feasibility")
    out.assign(L_expected=L + [np.nan]).to_csv(f"{r}/audit_tables_with_leakage.csv", index=False)
    print("saved", f"{r}/report_audit.pdf")

if __name__ == "__main__":
    main(sys.argv[1])
