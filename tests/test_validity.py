import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from src.validity import split_report, summarise, expected_seen_random, main     # noqa
from src.splits import random_split, chronological_split, seen_share_by_class     # noqa
from test_splits import make_file, stack                                          # noqa

CLASSES = ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]


def table():
    """Synthetic sanitised table with Hafiz's schema: DoS/RPM/Gear = one signature each, Fuzzy = all unique."""
    t, lab, f = stack(make_file("normal_run_data.txt", 1.478e9, 0, 0, 0), make_file("DoS_dataset.csv", 1.479e9, 30, 4.5, 3.0, "DoS"),
                      make_file("Fuzzy_dataset.csv", 1.480e9, 30, 5.2, 3.0, "Fuzzy"), make_file("RPM_dataset.csv", 1.481e9, 30, 3.5, 3.0, "RPM"),
                      make_file("gear_dataset.csv", 1.482e9, 30, 3.4, 3.0, "Gear"))
    n = len(t)
    i = np.arange(n)
    idv = np.where(lab == "Normal", 0x100 + i % 3, np.where(lab == "DoS", 0, np.where(lab == "RPM", 0x316,
                  np.where(lab == "Gear", 0x43F, i % 2048))))
    d = np.zeros((n, 8), dtype=np.uint8)
    d[:, 0] = np.where(lab == "Normal", i % 50, np.where(lab == "Fuzzy", (i // 7) % 256, 0))
    d[:, 1] = np.where(lab == "Fuzzy", (i // 1792) % 256, 0)
    d[:, 2] = np.where(lab == "Fuzzy", (i // 458752) % 256, 0)
    d[:, 7] = np.where(lab == "Fuzzy", 1, np.where(lab == "RPM", 2, np.where(lab == "Gear", 3, 0)))
    df = pd.DataFrame({"src_file": pd.Categorical(f), "t": t,
                       "flag": pd.Categorical(np.where(lab == "Normal", "R", "T"), categories=["R", "T"]),
                       "label": pd.Categorical(lab, categories=CLASSES), "id": idv.astype(np.uint16),
                       "dlc": np.full(n, 8, dtype=np.uint8)})
    for k in range(8):
        df[f"d{k}"] = d[:, k]
    return df


@pytest.fixture(scope="module")
def prepared():
    df = table()
    from src.splits import make_signature_ids
    sig = make_signature_ids(df).astype(np.int64)
    labels = df["label"].cat.codes.to_numpy().astype(np.int64)
    return df, labels, sig, int(sig.max()) + 1


def rep(prepared, tr, te, tech="X", seed=0, **kw):
    df, labels, sig, U = prepared
    return pd.DataFrame(split_report(tech, seed, labels, sig, tr, te, U, **kw))


def test_report_has_one_row_per_class_and_counts_add_up(prepared):
    df, labels, sig, U = prepared
    tr, te = random_split(labels, 42, stratified=True)
    r = rep(prepared, tr, te)
    assert list(r.label) == CLASSES
    for c, name in enumerate(CLASSES):
        row = r[r.label == name].iloc[0]
        assert row.train_frames + row.test_frames == (labels == c).sum()


def test_both_techniques_valid_on_well_behaved_data(prepared):
    df, labels, sig, U = prepared
    trA, teA = random_split(labels, 42, stratified=True)
    trB, teB, _ = chronological_split(df["src_file"].cat.codes.to_numpy(), df["t"].to_numpy(),
                                      (df["flag"] == "T").to_numpy())
    r = pd.concat([rep(prepared, trA, teA, "A", 42), rep(prepared, trB, teB, "B", -1)])
    s = summarise(r)
    assert s.valid.all() and (s.failing == "").all()


def test_single_signature_classes_are_always_seen_and_fuzzy_never(prepared):
    df, labels, sig, U = prepared
    trB, teB, _ = chronological_split(df["src_file"].cat.codes.to_numpy(), df["t"].to_numpy(),
                                      (df["flag"] == "T").to_numpy())
    r = rep(prepared, trB, teB).set_index("label")
    assert r.loc[["DoS", "RPM", "Gear"], "seen_share"].eq(1.0).all()
    assert r.loc["Fuzzy", "seen_share"] == 0.0
    assert r.loc[["DoS", "RPM", "Gear"], "train_signatures"].eq(1).all()
    assert r.loc[["DoS", "RPM", "Gear"], "test_signatures"].eq(1).all()


def test_seen_share_agrees_with_splits_helper(prepared):
    df, labels, sig, U = prepared
    tr, te = random_split(labels, 43, stratified=True)
    r = rep(prepared, tr, te).set_index("label")
    helper = seen_share_by_class(labels, sig, tr, te)
    for c, name in enumerate(CLASSES):
        assert r.loc[name, "seen_share"] == pytest.approx(helper[c])


def test_random_split_seen_share_close_to_formula(prepared):
    df, labels, sig, U = prepared
    exp = expected_seen_random(labels, sig, 5, U)
    tr, te = random_split(labels, 42, stratified=True)
    r = rep(prepared, tr, te, expected_random=exp).set_index("label")
    for name in CLASSES:
        assert r.loc[name, "seen_share"] == pytest.approx(r.loc[name, "expected_seen_random"], abs=0.03)


def test_expected_formula_extremes():
    labels = np.array([0] * 1000 + [1] * 1000)
    sig = np.concatenate([np.zeros(1000, int), np.arange(1, 1001)])
    e = expected_seen_random(labels, sig, 2, 1001)
    assert e[0] == pytest.approx(1.0, abs=1e-9) and e[1] == 0.0       # one signature -> always seen; all unique -> never


def test_class_missing_from_test_is_invalid(prepared):
    df, labels, sig, U = prepared
    te = np.flatnonzero(labels != 4)[::5]                     # test side has no Gear frames
    tr = np.setdiff1d(np.arange(len(labels)), te)
    r = rep(prepared, tr, te)
    gear = r[r.label == "Gear"].iloc[0]
    assert not gear.on_both_sides and not gear.class_valid
    s = summarise(r.assign(technique="X", seed=0))
    assert not s.valid.iloc[0] and "Gear (missing on one side)" in s.failing.iloc[0]


def test_class_with_share_outside_tolerance_is_invalid(prepared):
    df, labels, sig, U = prepared
    idx_dos = np.flatnonzero(labels == 1)
    te = np.concatenate([np.flatnonzero(labels != 1)[::5], idx_dos[: int(0.6 * len(idx_dos))]])   # DoS test share 60%
    te.sort()
    tr = np.setdiff1d(np.arange(len(labels)), te)
    r = rep(prepared, tr, te)
    assert not r[r.label == "DoS"].iloc[0].class_valid
    s = summarise(r.assign(technique="X", seed=0))
    assert "DoS (test share" in s.failing.iloc[0]


def test_tolerance_is_configurable(prepared):
    df, labels, sig, U = prepared
    tr, te = random_split(labels, 42, stratified=True)
    assert rep(prepared, tr, te).class_valid.all()
    assert not rep(prepared, tr, te, tol_low=0.25, tol_high=0.30).class_valid.any()


def test_cli_end_to_end(tmp_path):
    df = table()
    p = tmp_path / "frames_clean.parquet"
    df.to_parquet(p, index=False)
    main(["--parquet", str(p), "--out", str(tmp_path / "res"), "--save-splits", str(tmp_path / "sp"),
          "--seeds", "42,43", "--no-compare"])
    rep_ = pd.read_csv(tmp_path / "res" / "validity_report.csv")
    summ = pd.read_csv(tmp_path / "res" / "validity_summary.csv")
    assert len(rep_) == 3 * 5 and set(rep_.technique) == {"A", "B"}
    assert len(summ) == 3 and summ.valid.all()
    assert (tmp_path / "sp" / "technique_A_seed42_test_idx.npy").exists()
    assert (tmp_path / "sp" / "technique_B_test_idx.npy").exists()
