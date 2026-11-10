import copy
import csv
import tempfile
from pathlib import Path

import numpy as np

from src.common import load_config, labels_to_int
from src.run_experiments import RUN_COLUMNS, check_split, main, run_experiments
from src.synthetic import make_synthetic_arrays, write_dummy_splits

CFG = load_config("config.yaml")
CFG["models"]["RF"]["n_estimators"] = 5


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_dry_run_columns_resume_and_invalid_handling():
    X, y = make_synthetic_arrays(20000, seed=3)
    d = Path(tempfile.mkdtemp())
    write_dummy_splits(X, y, [42, 43], d)
    out = d / "runs.csv"
    run_experiments(X, y, CFG, d, out, ["LR", "DT", "RF"], ["P1", "P2"], [42, 43], log=lambda *_: None)
    rows = _rows(out)
    assert len(rows) == 3 * 2 * 2 and list(rows[0].keys()) == RUN_COLUMNS
    ok = [r for r in rows if r["status"] == "ok"]
    assert ok and all(0 <= float(r["macro_f1"]) <= 1 and int(r["n_train"]) + int(r["n_test"]) == 20000 for r in ok)
    assert all(r["date"] and r["seed"] and r["protocol"] for r in rows)       # ground rule 6
    n_before = len(rows)
    run_experiments(X, y, CFG, d, out, ["LR", "DT", "RF"], ["P1", "P2"], [42, 43], log=lambda *_: None)
    assert len(_rows(out)) == n_before                                         # resume: nothing re-run


def test_invalid_split_is_recorded_not_forced():
    X, y = make_synthetic_arrays(20000, seed=4)
    d = Path(tempfile.mkdtemp())
    write_dummy_splits(X, y, [42], d)
    bad = np.flatnonzero(y != 1)                       # test side has no DoS at all
    np.save(d / "P2_seed42.npy", bad[: len(bad) // 5])
    out = d / "runs.csv"
    run_experiments(X, y, CFG, d, out, ["DT"], ["P2"], [42], log=lambda *_: None)
    r = _rows(out)[0]
    assert r["status"] == "skipped_invalid" and "DoS" in r["missing_test"] and r["macro_f1"] == ""


def test_check_split_counts():
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 50)
    test = np.arange(0, len(y), 5)
    res = check_split(y, test, CFG["validity"])
    assert res["test_mask"].sum() == len(test) and isinstance(res["valid"], bool)


def test_real_run_refused_until_frozen_and_tolerance_fixed():
    assert main(["--frames", "does_not_exist.parquet"]) == 2


def test_label_strings_are_mapped():
    codes = labels_to_int(["Normal", "DoS", "Fuzzy", "RPM", "Gear", "normal"])
    assert codes.tolist() == [0, 1, 2, 3, 4, 0]
