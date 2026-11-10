import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.summarise_results import (make_template, run_metrics, per_class_runs, summarise, main,   # noqa
                                   CLASSES, MODELS, RUN_COLS, COUNTS)

N_TEST = {"Normal": 830, "DoS": 40, "Fuzzy": 35, "RPM": 50, "Gear": 45}          # test frames per class


def one_run(technique="A", seed=42, model="DecisionTree", hit=None, seen=None):
    """Counts for one run. hit[c] = recall of class c; seen[c] = share of the class's test frames that are seen.
    Errors are all sent to 'Normal' / taken from 'Normal' so that fp stays consistent."""
    hit = hit or {c: 1.0 for c in CLASSES}
    seen = seen or {c: 0.5 for c in CLASSES}
    rows, fp_normal = [], 0
    for c in CLASSES:
        n = N_TEST[c]
        ns = int(n * seen[c]); nu = n - ns
        tps, tpu = int(round(ns * hit[c])), int(round(nu * hit[c]))
        rows.append(dict(technique=technique, seed=seed, model=model, label=c, tp=tps + tpu, fp=0, fn=n - tps - tpu,
                         tp_seen=tps, fn_seen=ns - tps, tp_unseen=tpu, fn_unseen=nu - tpu))
        if c != "Normal":
            fp_normal += n - tps - tpu
    df = pd.DataFrame(rows)
    df.loc[df.label == "Normal", "fp"] = fp_normal                 # missed attack frames are predicted Normal
    return df


def all_runs():
    parts = []
    for t in ("A", "B"):
        for s in (42, 43):
            for m in MODELS:
                parts.append(one_run(t, s, m))
    return pd.concat(parts, ignore_index=True)


def test_template_has_every_combination_and_empty_counts(tmp_path):
    df = make_template(tmp_path / "t.csv")
    assert len(df) == 2 * 5 * 4 * 5 and list(df.columns) == RUN_COLS
    assert df[COUNTS].isna().all().all()
    with pytest.raises(ValueError, match="empty counts"):
        per_class_runs(df)


def test_perfect_model_has_macro_f1_and_accuracy_one():
    rm = run_metrics(one_run())
    assert rm.macro_f1.iloc[0] == pytest.approx(1.0) and rm.accuracy.iloc[0] == pytest.approx(1.0)


def test_macro_f1_and_accuracy_match_hand_calculation():
    hit = {"Normal": 1.0, "DoS": 0.5, "Fuzzy": 1.0, "RPM": 0.0, "Gear": 1.0}
    r = one_run(hit=hit, seen={c: 1.0 for c in CLASSES})
    tp = dict(zip(r.label, r.tp)); fp = dict(zip(r.label, r.fp)); fn = dict(zip(r.label, r.fn))
    f1 = {c: 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c]) for c in CLASSES}
    rm = run_metrics(r).iloc[0]
    assert rm.macro_f1 == pytest.approx(np.mean(list(f1.values())))
    assert rm.accuracy == pytest.approx(sum(tp.values()) / sum(N_TEST.values()))


def test_all_normal_predictor_equals_the_baseline_formula():
    rows = []
    for c in CLASSES:
        n = N_TEST[c]
        rows.append(dict(technique="A", seed=42, model="Lookup", label=c, tp=n if c == "Normal" else 0,
                         fp=sum(N_TEST.values()) - n if c == "Normal" else 0, fn=0 if c == "Normal" else n,
                         tp_seen=n if c == "Normal" else 0, fn_seen=0 if c == "Normal" else n, tp_unseen=0, fn_unseen=0))
    rm = run_metrics(pd.DataFrame(rows)).iloc[0]
    assert rm.macro_f1 == pytest.approx(rm.baseline_macro_f1)
    assert rm.accuracy == pytest.approx(rm.baseline_accuracy)
    pi = N_TEST["Normal"] / sum(N_TEST.values())
    assert rm.baseline_macro_f1 == pytest.approx(2 * pi / (5 * (1 + pi)))


def test_seen_and_unseen_recall_and_undefined_cases():
    seen = {"Normal": 0.9, "DoS": 1.0, "Fuzzy": 0.0, "RPM": 1.0, "Gear": 1.0}      # DoS: no unseen, Fuzzy: no seen
    hit = {c: 0.8 for c in CLASSES}
    pc = per_class_runs(one_run(hit=hit, seen=seen)).set_index("label")
    assert np.isnan(pc.loc["DoS", "recall_unseen"]) and np.isnan(pc.loc["Fuzzy", "recall_seen"])
    assert pc.loc["DoS", "seen_share"] == 1.0 and pc.loc["Fuzzy", "seen_share"] == 0.0
    assert pc.loc["Normal", "recall_seen"] == pytest.approx(0.8, abs=0.01)


def test_inconsistent_seen_unseen_counts_are_rejected():
    r = one_run()
    r.loc[0, "tp_seen"] += 1
    with pytest.raises(ValueError, match="tp must equal"):
        per_class_runs(r)


def test_missing_class_is_rejected():
    with pytest.raises(ValueError, match="five classes"):
        run_metrics(one_run().query("label != 'Gear'"))


def test_mean_and_sample_sd_over_seeds():
    parts = [one_run("A", 42, "DecisionTree", hit={**{c: 1.0 for c in CLASSES}, "DoS": 0.5}),
             one_run("A", 43, "DecisionTree", hit={**{c: 1.0 for c in CLASSES}, "DoS": 1.0})]
    runs = pd.concat(parts, ignore_index=True)
    over, per_class, eff, diff = None, None, None, None
    rm = run_metrics(runs).macro_f1.to_numpy()
    pc = per_class_runs(runs)
    from src.summarise_results import _mean_sd
    s = _mean_sd(run_metrics(runs), ["technique", "model"], ["macro_f1", "accuracy"]).iloc[0]
    assert s.macro_f1_mean == pytest.approx(rm.mean()) and s.macro_f1_sd == pytest.approx(rm.std(ddof=1))
    assert s.n_seeds == 2


def test_effects_formulas():
    sA = {"Normal": 0.9, "DoS": 1.0, "Fuzzy": 0.0, "RPM": 1.0, "Gear": 1.0}
    parts = []
    for t, hit_seen, hit_unseen in (("A", 0.99, 0.90), ("B", 0.98, 0.70)):
        for m in MODELS:
            r = one_run(t, 42, m, seen={"Normal": 0.5, "DoS": 1.0, "Fuzzy": 0.0, "RPM": 1.0, "Gear": 1.0})
            # give Normal different recall on seen and unseen frames
            idx = r.index[r.label == "Normal"][0]
            ns = r.loc[idx, "tp_seen"] + r.loc[idx, "fn_seen"]; nu = r.loc[idx, "tp_unseen"] + r.loc[idx, "fn_unseen"]
            r.loc[idx, ["tp_seen", "fn_seen"]] = [round(ns * hit_seen), ns - round(ns * hit_seen)]
            r.loc[idx, ["tp_unseen", "fn_unseen"]] = [round(nu * hit_unseen), nu - round(nu * hit_unseen)]
            r.loc[idx, "tp"] = r.loc[idx, "tp_seen"] + r.loc[idx, "tp_unseen"]
            r.loc[idx, "fn"] = N_TEST["Normal"] - r.loc[idx, "tp"]
            parts.append(r)
    runs = pd.concat(parts, ignore_index=True)
    over, pcs, eff, diff = summarise(runs)
    e = eff[(eff.model == "DecisionTree") & (eff.label == "Normal")].iloc[0]
    pc = pcs.set_index(["technique", "model", "label"])
    rsA = pc.loc[("A", "DecisionTree", "Normal"), "recall_seen_mean"]; ruA = pc.loc[("A", "DecisionTree", "Normal"), "recall_unseen_mean"]
    rsB = pc.loc[("B", "DecisionTree", "Normal"), "recall_seen_mean"]; ruB = pc.loc[("B", "DecisionTree", "Normal"), "recall_unseen_mean"]
    assert e.memorisation_effect_A == pytest.approx(rsA - ruA)
    assert e.memorisation_effect_B == pytest.approx(rsB - ruB)
    assert e.temporal_effect == pytest.approx(ruA - ruB) and e.temporal_effect > 0
    # classes without unseen (or without seen) frames: effects are undefined, not 0
    d = eff[(eff.model == "DecisionTree") & (eff.label == "DoS")].iloc[0]
    assert np.isnan(d.memorisation_effect_A) and np.isnan(d.temporal_effect)
    assert len(diff) == 4 and set(over.technique) == {"A", "B"}


def test_summary_tables_have_expected_shape_and_order():
    over, pcs, eff, diff = summarise(all_runs())
    assert len(over) == 2 * 4 and len(pcs) == 2 * 4 * 5 and len(eff) == 4 * 5 and len(diff) == 4
    assert list(over.technique[:4]) == ["A"] * 4 and list(over.model[:4]) == MODELS
    assert list(pcs.label[:5]) == CLASSES


def test_cli_end_to_end(tmp_path):
    runs_path = tmp_path / "runs.csv"
    all_runs().to_csv(runs_path, index=False)
    main(["--runs", str(runs_path), "--out", str(tmp_path / "res")])
    for name in ("summary_overall", "summary_per_class", "summary_effects", "summary_A_minus_B"):
        assert (tmp_path / "res" / f"{name}.csv").exists()
