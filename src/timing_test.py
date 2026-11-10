"""Task M3 - timing and memory test on SYNTHETIC data of real size (no real labels).

Fits LR, DT and RF on synthetic n x 10 data for each size in --sizes, each in its own
subprocess so memory numbers are clean, then extrapolates fit time and RAM to the
real training size (about 14M rows) with a log-log fit, and recommends RF settings
(n_estimators, max_samples) that fit a time and RAM budget.

  python -m src.timing_test --sizes 1000000 4000000 --target 14000000

Caveat (say it in the report): the synthetic data only has the right SHAPE. Tree size
and time depend on how many distinct signatures the real data has, so re-run the RF
part with the real number of signatures U from the audit (H3) if the answer is close
to the budget.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BYTES_PER_NODE = 104        # sklearn tree node struct (64 B) + value array (5 classes x 8 B)
WORKSPACE_BYTES = 40        # per-sample temporary arrays while one tree is being built
SAFETY = 1.25               # multiply time estimates, threads never scale perfectly


# ---------------------------------------------------------------- worker (one fit)
def _worker(model: str, n: int, n_est: int, max_samples, seed: int) -> dict:
    import psutil
    from .common import load_config
    from .models import build_model
    from .synthetic import make_synthetic_arrays

    cfg = load_config(ROOT / "config.yaml")
    cfg["runtime"] = {"n_jobs": 1}                      # per-tree cost, one core
    if model == "RF":
        cfg["models"]["RF"]["n_estimators"] = n_est
        cfg["models"]["RF"]["max_samples"] = max_samples
    X, y = make_synthetic_arrays(n, seed=seed)
    gc.collect()
    proc = psutil.Process()
    before = proc.memory_info().rss
    peak = [before]
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.05)

    t = threading.Thread(target=watch, daemon=True); t.start()
    m = build_model(model, seed, cfg)
    t0 = time.perf_counter(); m.fit(X, y); fit_s = time.perf_counter() - t0
    peak[0] = max(peak[0], proc.memory_info().rss)
    stop.set(); t.join()
    n_pred = min(n, 500_000)
    t0 = time.perf_counter(); m.predict(X[:n_pred]); pred_s = time.perf_counter() - t0
    nodes = 0.0
    if model == "DT":
        nodes = float(m.tree_.node_count)
    elif model == "RF":
        nodes = float(np.mean([e.tree_.node_count for e in m.estimators_]))
    return {"model": model, "n_rows": n, "n_estimators": n_est if model == "RF" else "",
            "max_samples": max_samples if model == "RF" else "", "fit_s": round(fit_s, 3),
            "predict_s": round(pred_s, 3), "n_predicted": n_pred,
            "rss_before_mb": round(before / 2**20, 1), "rss_peak_fit_mb": round(peak[0] / 2**20, 1),
            "nodes_per_tree": round(nodes, 1)}


def _run_worker(model, n, n_est, seed):
    cmd = [sys.executable, "-m", "src.timing_test", "--worker", model, "--n", str(n),
           "--n-est", str(n_est), "--seed", str(seed)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"worker failed for {model} n={n}:\n{out.stderr[-2000:]}")


# ---------------------------------------------------------------- extrapolation
def fit_power(ns, vals):
    """vals ~ a * n**b by least squares in log-log; one point -> assume linear."""
    ns, vals = np.asarray(ns, float), np.maximum(np.asarray(vals, float), 1e-9)
    if len(ns) < 2:
        return vals[0] / ns[0], 1.0
    b, loga = np.polyfit(np.log(ns), np.log(vals), 1)
    return math.exp(loga), float(b)


def fit_linear(ns, vals):
    """vals ~ c0 + c1*n (RAM: a fixed interpreter/library part plus a part per row)."""
    ns, vals = np.asarray(ns, float), np.asarray(vals, float)
    if len(ns) < 2:
        return 0.0, vals[0] / ns[0]
    c1, c0 = np.polyfit(ns, vals, 1)
    return float(c0), float(c1)


def predict_linear(c, n):
    return max(c[0] + c[1] * float(n), 0.0)


def predict_power(ab, n):
    return ab[0] * float(n) ** ab[1]


def unique_path(path: Path) -> Path:
    """Ground rule 6: never overwrite a results file."""
    if not path.exists():
        return path
    return path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")


def recommend_rf(rows, target, cores, ram_gb, budget_min, n_meas):
    rf = [r for r in rows if r["model"] == "RF"]
    ns = [r["n_rows"] for r in rf]
    t_tree = fit_power(ns, [r["fit_s"] / n_meas for r in rf])
    nodes = fit_power(ns, [r["nodes_per_tree"] for r in rf])
    peak = fit_linear(ns, [r["rss_peak_fit_mb"] for r in rf])
    peak_target_mb = predict_linear(peak, target)
    base_mb = max(peak_target_mb - n_meas * predict_power(nodes, target) * BYTES_PER_NODE / 2**20, 0)
    grid = [(100, None), (100, 0.5), (100, 0.25), (50, 0.25), (100, 0.1), (50, 0.1), (25, 0.1)]
    table, chosen = [], None
    for n_est, frac in grid:
        m = target * (frac or 1.0)
        wall_min = SAFETY * math.ceil(n_est / min(cores, n_est)) * predict_power(t_tree, m) / 60
        ram_mb = base_mb + n_est * predict_power(nodes, m) * BYTES_PER_NODE / 2**20 \
            + (min(cores, n_est) - 1) * m * WORKSPACE_BYTES / 2**20
        ok = wall_min <= budget_min and ram_mb / 1024 <= ram_gb
        table.append((n_est, frac, wall_min, ram_mb / 1024, ok))
        if ok and chosen is None:
            chosen = (n_est, frac)
    return table, chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1_000_000, 4_000_000])
    ap.add_argument("--target", type=int, default=14_000_000, help="real training rows")
    ap.add_argument("--models", nargs="+", default=["LR", "DT", "RF"])
    ap.add_argument("--rf-trees", type=int, default=8, help="trees fitted in the RF measurement")
    ap.add_argument("--cores", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--ram-gb", type=float, default=None, help="RAM you allow the run to use")
    ap.add_argument("--budget-min", type=float, default=60, help="max minutes for ONE RF fit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--worker"); ap.add_argument("--n", type=int); ap.add_argument("--n-est", type=int, default=8)
    a = ap.parse_args(argv)

    if a.worker:                                    # child process
        print("RESULT " + json.dumps(_worker(a.worker, a.n, a.n_est, None, a.seed)))
        return 0

    import psutil
    total_gb = psutil.virtual_memory().total / 2**30
    ram_gb = a.ram_gb or 0.7 * total_gb
    print(f"machine: {platform.platform()} | cores={a.cores} | RAM total={total_gb:.1f} GB, budget={ram_gb:.1f} GB")
    rows = []
    for model in a.models:
        for n in sorted(a.sizes):
            print(f"  fitting {model} on {n:,} synthetic rows ...", flush=True)
            r = _run_worker(model, n, a.rf_trees, a.seed)
            r.update(date=datetime.now().isoformat(timespec="seconds"), cores_measured=1)
            rows.append(r)
            print(f"    fit {r['fit_s']}s, peak RAM {r['rss_peak_fit_mb']} MB, nodes/tree {r['nodes_per_tree']}")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    raw = unique_path(out / "timing_table.csv")
    fields = ["date", "model", "n_rows", "n_estimators", "max_samples", "fit_s", "predict_s", "n_predicted",
              "rss_before_mb", "rss_peak_fit_mb", "nodes_per_tree", "cores_measured"]
    with open(raw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    lines = [f"# Timing and memory test ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"Machine: cores={a.cores}, RAM total {total_gb:.1f} GB, RAM budget {ram_gb:.1f} GB. "
             f"Synthetic data, sizes {', '.join(f'{s:,}' for s in sorted(a.sizes))} rows x 10 columns. "
             f"Extrapolated to {a.target:,} training rows (time: log-log fit, RAM: linear fit; rough).", "",
             "| model | est. fit time (1 core) | est. peak RAM |", "|---|---|---|"]
    ext = []
    for model in a.models:
        mr = [r for r in rows if r["model"] == model]
        ns = [r["n_rows"] for r in mr]
        if model == "RF":
            t = predict_power(fit_power(ns, [r["fit_s"] / a.rf_trees for r in mr]), a.target) * 100
            label = "100 full trees, 1 core"
        else:
            t = predict_power(fit_power(ns, [r["fit_s"] for r in mr]), a.target); label = "1 core"
        ram = predict_linear(fit_linear(ns, [r["rss_peak_fit_mb"] for r in mr]), a.target) / 1024
        ext.append({"model": model, "target_rows": a.target, "est_fit_minutes": round(t / 60, 1), "est_peak_ram_gb": round(ram, 2)})
        lines.append(f"| {model} | {t/60:.1f} min ({label}) | {ram:.1f} GB |")
    if "RF" in a.models:
        table, chosen = recommend_rf([r for r in rows if r["model"] == "RF"], a.target, a.cores, ram_gb,
                                     a.budget_min, a.rf_trees)
        lines += ["", f"## Random Forest candidates (budget {a.budget_min:.0f} min per fit, {ram_gb:.0f} GB RAM)", "",
                  "| n_estimators | max_samples | est. wall time (min) | est. RAM (GB) | fits budget |", "|---|---|---|---|---|"]
        for n_est, frac, wall, gb, ok in table:
            lines.append(f"| {n_est} | {frac if frac else 'null (full)'} | {wall:.0f} | {gb:.1f} | {'yes' if ok else 'no'} |")
        lines += ["", f"**Recommendation:** " + (
            f"n_estimators={chosen[0]}, max_samples={chosen[1] if chosen[1] else 'null'} "
            f"(first candidate inside both budgets). Put these in config.yaml, then set frozen: true. "
            f"Use the same values for P1 and P2." if chosen else
            "no candidate fits; allow more time or RAM, or train RF on fewer seeds first. Tell the team."),
            "", "Caveat: synthetic data only has the right shape. Re-check with the real number of signatures from the audit (H3)."]
    (unique_path(out / "timing_extrapolation.csv")).write_text(
        "model,target_rows,est_fit_minutes,est_peak_ram_gb\n" + "\n".join(
            f"{e['model']},{e['target_rows']},{e['est_fit_minutes']},{e['est_peak_ram_gb']}" for e in ext) + "\n")
    rec = unique_path(out / "timing_recommendation.md"); rec.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines)); print(f"\nwrote {raw.name}, {rec.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
