"""R4: results layout. Turns per-run confusion counts into the tables of the reporting plan (section 6.11).

INPUT CONTRACT  (what the model runner must write; one row per technique x seed x model x class):
    technique  A | B
    seed       42..46   (A: seeds the split AND the model; B: the split is fixed, the seed only changes the model)
    model      Lookup | DecisionTree | LogisticRegression | RandomForest
    label      Normal | DoS | Fuzzy | RPM | Gear          (always all five, even if a count is 0)
    tp, fp, fn                 class counts on the whole test side (true class = label for tp, fn)
    tp_seen, fn_seen           the same for test frames whose exact signature occurs in training
    tp_unseen, fn_unseen       the same for the remaining test frames
  (tp = tp_seen + tp_unseen and tp + fn = test frames of the class are checked.)
  An empty template is written by:  python src/summarise_results.py --make-template results/results_table_template.csv

OUTPUT (python src/summarise_results.py --runs results/runs.csv --out results):
    summary_overall.csv    per technique x model: macro-F1, accuracy (mean, SD over seeds) + all-Normal baselines
    summary_per_class.csv  per technique x model x class: recall, F1, recall on seen / unseen frames, seen share
    summary_effects.csv    per model x class: memorisation effect (A and B), temporal effect, lookup recall
    summary_A_minus_B.csv  per model: difference of the mean macro-F1 and accuracy (descriptive)

Conventions: macro-F1 = mean of the five F1 values, F1 = 2TP/(2TP+FP+FN); recall is NaN (undefined, not 0) when
the denominator is 0; SD = sample SD (ddof = 1) over seeds; statistics are descriptive, no significance tests (D26).
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]
MODELS = ["Lookup", "DecisionTree", "LogisticRegression", "RandomForest"]
TECHNIQUES = ["A", "B"]
SEEDS = (42, 43, 44, 45, 46)
COUNTS = ["tp", "fp", "fn", "tp_seen", "fn_seen", "tp_unseen", "fn_unseen"]
RUN_COLS = ["technique", "seed", "model", "label"] + COUNTS
KEYS = ["technique", "model"]


def make_template(path):
    rows = [(t, s, m, c) for t in TECHNIQUES for s in SEEDS for m in MODELS for c in CLASSES]
    df = pd.DataFrame(rows, columns=["technique", "seed", "model", "label"])
    for c in COUNTS:
        df[c] = np.nan
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _check_runs(runs):
    missing = [c for c in RUN_COLS if c not in runs.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if runs[COUNTS].isna().any().any():
        raise ValueError("empty counts: fill the template completely before summarising")
    per_run = runs.groupby(["technique", "seed", "model"]).label.nunique()
    if (per_run != len(CLASSES)).any() or set(runs.label) != set(CLASSES):
        raise ValueError("every technique/seed/model needs exactly the five classes")
    if ((runs.tp != runs.tp_seen + runs.tp_unseen) | (runs.tp_seen < 0) | (runs.tp_unseen < 0)).any():
        raise ValueError("tp must equal tp_seen + tp_unseen")
    if ((runs[COUNTS] < 0).any()).any():
        raise ValueError("negative count")


def per_class_runs(runs):
    """One row per run and class with recall, F1 and seen/unseen recall (NaN when undefined)."""
    _check_runs(runs)
    r = runs.copy()
    n_c = r.tp + r.fn
    den = 2 * r.tp + r.fp + r.fn
    if (den == 0).any():
        warnings.warn("a class has no test frames and no predictions: its F1 is set to 0 (sklearn zero_division=0)")
    r["f1"] = np.where(den > 0, 2 * r.tp / np.maximum(den, 1), 0.0)
    r["recall"] = np.where(n_c > 0, r.tp / np.maximum(n_c, 1), np.nan)
    ns, nu = r.tp_seen + r.fn_seen, r.tp_unseen + r.fn_unseen
    r["recall_seen"] = np.where(ns > 0, r.tp_seen / np.maximum(ns, 1), np.nan)
    r["recall_unseen"] = np.where(nu > 0, r.tp_unseen / np.maximum(nu, 1), np.nan)
    r["seen_share"] = np.where(n_c > 0, ns / np.maximum(n_c, 1), np.nan)
    r["test_frames"] = n_c
    return r


def run_metrics(runs):
    """One row per technique x seed x model: macro-F1, accuracy and the all-Normal reference values."""
    r = per_class_runs(runs)
    g = r.groupby(["technique", "seed", "model"])
    out = g.agg(macro_f1=("f1", "mean"), correct=("tp", "sum"), n_test=("test_frames", "sum")).reset_index()
    out["accuracy"] = out.correct / out.n_test
    normal = r[r.label == "Normal"].set_index(["technique", "seed", "model"]).test_frames
    out["pi_normal_test"] = out.set_index(["technique", "seed", "model"]).index.map(normal).to_numpy() / out.n_test
    c = len(CLASSES)
    out["baseline_macro_f1"] = 2 * out.pi_normal_test / (c * (1 + out.pi_normal_test))   # D7, from this test set
    out["baseline_accuracy"] = out.pi_normal_test
    return out.drop(columns="correct")


def _mean_sd(df, keys, cols):
    g = df.groupby(keys, sort=False)
    mean, sd, n = g[cols].mean(), g[cols].std(ddof=1), g[cols].count()
    out = mean.add_suffix("_mean").join(sd.add_suffix("_sd"))
    out["n_seeds"] = n[cols[0]]
    return out.reset_index()


def summarise(runs):
    pc = per_class_runs(runs)
    rm = run_metrics(runs)
    overall = _mean_sd(rm, KEYS, ["macro_f1", "accuracy"])
    base = rm.groupby(KEYS, sort=False)[["baseline_macro_f1", "baseline_accuracy"]].mean().reset_index()
    overall = overall.merge(base, on=KEYS)
    per_class = _mean_sd(pc, KEYS + ["label"], ["recall", "f1", "recall_seen", "recall_unseen", "seen_share"])
    # effects: section 6.9 (descriptive; seen and unseen frames are different subpopulations)
    piv = per_class.pivot(index=["model", "label"], columns="technique",
                          values=["recall_seen_mean", "recall_unseen_mean", "recall_mean"])
    eff = pd.DataFrame(index=piv.index)
    eff["memorisation_effect_A"] = piv["recall_seen_mean"]["A"] - piv["recall_unseen_mean"]["A"]
    eff["memorisation_effect_B"] = piv["recall_seen_mean"]["B"] - piv["recall_unseen_mean"]["B"]
    eff["temporal_effect"] = piv["recall_unseen_mean"]["A"] - piv["recall_unseen_mean"]["B"]
    eff = eff.reset_index()
    look = per_class[per_class.model == "Lookup"].pivot(index="label", columns="technique", values="recall_mean")
    eff["lookup_recall_A"] = eff.label.map(look["A"])
    eff["lookup_recall_B"] = eff.label.map(look["B"])
    mo = overall.pivot(index="model", columns="technique", values=["macro_f1_mean", "accuracy_mean"])
    diff = pd.DataFrame({"macro_f1_A_minus_B": mo["macro_f1_mean"]["A"] - mo["macro_f1_mean"]["B"],
                         "accuracy_A_minus_B": mo["accuracy_mean"]["A"] - mo["accuracy_mean"]["B"]}).reset_index()
    rank = {v: i for i, v in enumerate(TECHNIQUES + MODELS + CLASSES)}

    def order(d, keys):
        """Sort by the fixed technique / model / class order (not alphabetically)."""
        return d.sort_values(keys, key=lambda col: col.map(rank), kind="stable").reset_index(drop=True)

    return (order(overall, ["technique", "model"]), order(per_class, ["technique", "model", "label"]),
            order(eff, ["model", "label"]), order(diff, ["model"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-template", metavar="CSV")
    ap.add_argument("--runs", help="per-run counts (see the input contract in the module docstring)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    if a.make_template:
        df = make_template(a.make_template)
        print(f"wrote empty template with {len(df)} rows to {a.make_template}")
    if a.runs:
        runs = pd.read_csv(a.runs)
        overall, per_class, eff, diff = summarise(runs)
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        for name, d in [("summary_overall", overall), ("summary_per_class", per_class),
                        ("summary_effects", eff), ("summary_A_minus_B", diff)]:
            d.to_csv(out / f"{name}.csv", index=False)
        pd.set_option("display.width", 250)
        print(overall.round(4).to_string(index=False))
        print("\n", per_class.round(4).to_string(index=False))
        print("\n", eff.round(4).to_string(index=False))
        print("\n", diff.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
