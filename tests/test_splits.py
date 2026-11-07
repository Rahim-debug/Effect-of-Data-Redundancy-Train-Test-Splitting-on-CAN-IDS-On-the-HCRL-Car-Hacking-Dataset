import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.splits import (random_split, chronological_split, signature_group_split, class_table,   # noqa
                        share_in_test, seen_share_by_class, seen_mask, overlap_share)

SEEDS = [42, 43, 44, 45, 46]
CLASSES = ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]


# ------------------------------------------------------------------ synthetic recordings
def make_file(name, t0, n_bursts, burst_len, quiet_len, attack=None, step=0.01, first_quiet=None):
    """One recording: regular (R) frames every `step` s all the time; injected (T) frames inside bursts.
    Layout: quiet, burst, quiet, burst, ..., quiet (so the file starts and ends in a quiet stretch)."""
    first_quiet = quiet_len if first_quiet is None else first_quiet
    starts = [first_quiet + k * (burst_len + quiet_len) for k in range(n_bursts)]
    total = starts[-1] + burst_len + quiet_len if n_bursts else 300.0
    tr = np.arange(0, total + 1e-9, step)
    rows = [(t0 + x, "Normal", name) for x in tr]
    if attack:
        for s in starts:
            for x in np.arange(s, s + burst_len + 1e-9, step):
                rows.append((t0 + x + step / 2, attack, name))
    rows.sort(key=lambda r: r[0])
    t = np.array([r[0] for r in rows]); lab = np.array([r[1] for r in rows]); f = np.array([r[2] for r in rows])
    return t, lab, f


def stack(*files):
    return tuple(np.concatenate(x) for x in zip(*files))


@pytest.fixture(scope="module")
def recordings():
    # different t0 per file: timestamps are only comparable inside one file
    return stack(make_file("ambient", 1.478e9, 0, 0, 0),
                 make_file("dos", 1.479e9, 30, 4.5, 3.0, "DoS"),
                 make_file("fuzzy", 1.480e9, 30, 5.2, 3.0, "Fuzzy"),
                 make_file("rpm", 1.481e9, 30, 3.5, 3.0, "RPM"),
                 make_file("gear", 1.482e9, 30, 3.4, 3.0, "Gear"))


# ------------------------------------------------------------------ Technique A
def _labels(n=40000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(CLASSES, size=n, p=[0.83, 0.04, 0.035, 0.05, 0.045])


@pytest.mark.parametrize("stratified", [False, True])
def test_a_exact_count_partition_sorted(stratified):
    lab = _labels()
    for seed in SEEDS:
        tr, te = random_split(lab, seed, stratified=stratified)
        assert len(te) == round(0.2 * len(lab))
        assert np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(len(lab)))
        assert np.all(np.diff(tr) > 0) and np.all(np.diff(te) > 0)


def test_a_stratified_exact_per_class():
    lab = _labels()
    tr, te = random_split(lab, 42, stratified=True)
    for c in CLASSES:
        assert (lab[te] == c).sum() == round(0.2 * (lab == c).sum())


def test_a_plain_shuffle_keeps_every_class_about_20_percent():
    lab = _labels()
    tr, te = random_split(lab, 42)
    for c in CLASSES:
        assert abs((lab[te] == c).sum() / (lab == c).sum() - 0.2) < 0.03


@pytest.mark.parametrize("stratified", [False, True])
def test_a_seed_controls_the_draw(stratified):
    lab = _labels()
    a, b, c = (random_split(lab, s, stratified=stratified)[1] for s in (42, 42, 43))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_a_is_not_chronological(recordings):
    t, lab, f = recordings
    tr, te = random_split(lab, 42)
    # random split mixes time: some test frame is earlier than some train frame inside the same file
    m = f == "dos"
    assert t[te][m[te]].min() < t[tr][m[tr]].max()


# ------------------------------------------------------------------ Technique B
def test_b_train_entirely_before_test_in_every_file(recordings):
    t, lab, f = recordings
    tr, te, info = chronological_split(f, t, lab != "Normal")
    for name in np.unique(f):
        m = f == name
        assert t[tr][m[tr]].max() < t[te][m[te]].min()


def test_b_partition_and_deterministic(recordings):
    t, lab, f = recordings
    a = chronological_split(f, t, lab != "Normal")
    b = chronological_split(f, t, lab != "Normal")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.array_equal(np.sort(np.concatenate([a[0], a[1]])), np.arange(len(t)))


def test_b_never_cuts_a_burst(recordings):
    t, lab, f = recordings
    tr, te, info = chronological_split(f, t, lab != "Normal")
    for name in ["dos", "fuzzy", "rpm", "gear"]:
        m = f == name
        att = (lab != "Normal") & m
        # attack frames: every train attack frame is older than every test attack frame by a quiet gap
        gap = t[te][att[te]].min() - t[tr][att[tr]].max()
        assert gap > 1.0                      # the cut lies in a quiet stretch (burst gap is 1.0 s)


def test_b_cut_is_in_the_middle_of_a_quiet_stretch(recordings):
    t, lab, f = recordings
    tr, te, info = chronological_split(f, t, lab != "Normal")
    for _, r in info.iterrows():
        if r.n_bursts:
            assert r.quiet_at_cut_s == pytest.approx(3.0, abs=0.05)
            m = f == r.src_file
            att_t = t[m & (lab != "Normal")]
            last_before = att_t[att_t <= r.cut_time].max()
            first_after = att_t[att_t > r.cut_time].min()
            assert r.cut_time == pytest.approx((last_before + first_after) / 2, abs=0.02)


def test_b_all_classes_on_both_sides_and_shares_in_range(recordings):
    t, lab, f = recordings
    tr, te, _ = chronological_split(f, t, lab != "Normal")
    tab = class_table(lab, tr, te)
    assert set(tab.label) == set(CLASSES)
    assert ((tab.train > 0) & (tab.test > 0)).all()
    assert tab.test_share.between(0.10, 0.30).all()


def test_b_ambient_file_cut_at_80_percent_of_duration(recordings):
    t, lab, f = recordings
    tr, te, info = chronological_split(f, t, lab != "Normal")
    r = info[info.src_file == "ambient"].iloc[0]
    assert r.n_bursts == 0 and not r.cut_moved_out_of_burst
    assert r.cut_rel_s == pytest.approx(0.8 * (r.t_end - r.t0))


def test_b_files_are_independent_of_absolute_time(recordings):
    t, lab, f = recordings
    a = chronological_split(f, t, lab != "Normal")[2].cut_rel_s.to_numpy()
    b = chronological_split(f, t + 5.0e8, lab != "Normal")[2].cut_rel_s.to_numpy()
    assert a == pytest.approx(b, abs=1e-3)


def test_b_80_percent_point_inside_a_burst_is_moved_to_the_next_quiet_stretch():
    # 9 bursts [3+7k, 7+7k], k=0..8; file ends at 66 -> 80% point 52.8 lies in burst k=7 = [52, 56].
    t, lab, f = make_file("x", 0.0, 9, 4.0, 3.0, "DoS")
    _, _, info = chronological_split(f, t, lab != "Normal")
    r = info.iloc[0]
    assert r.cut_moved_out_of_burst
    assert r.cut_time == pytest.approx(57.5, abs=0.02)          # midpoint of the quiet stretch (56, 59)


def test_b_80_percent_point_in_a_quiet_stretch_uses_that_stretch():
    # bursts [3+5k, 5+5k], k=0..9; file ends at 53 -> 80% point 42.4 lies in quiet stretch (40, 43)
    t, lab, f = make_file("x", 0.0, 10, 2.0, 3.0, "DoS")
    _, _, info = chronological_split(f, t, lab != "Normal")
    r = info.iloc[0]
    assert not r.cut_moved_out_of_burst
    assert r.cut_time == pytest.approx(41.5, abs=0.02)


def test_b_warns_when_the_table_looks_unsanitised():
    t, lab, f = make_file("x", 0.0, 10, 4.0, 3.0, "DoS")
    tail = np.arange(1000.0, 1100.0, 0.01)
    t2, lab2 = np.concatenate([t, tail]), np.concatenate([lab, np.array(["Normal"] * len(tail))])
    with pytest.warns(UserWarning, match="unsanitised"):
        chronological_split(np.array(["x"] * len(t2)), t2, lab2 != "Normal")


def test_b_no_warning_on_a_sanitised_table(recordings):
    import warnings
    t, lab, f = recordings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chronological_split(f, t, lab != "Normal")


@pytest.mark.filterwarnings("ignore:.*unsanitised")
def test_b_without_sanitisation_test_side_has_no_attacks():
    # attack-free tail (copy of ambient) after a long gap: 80% point falls in the tail
    t, lab, f = make_file("x", 0.0, 10, 4.0, 3.0, "DoS")
    tail = np.arange(1000.0, 1100.0, 0.01)
    t2, lab2 = np.concatenate([t, tail]), np.concatenate([lab, np.array(["Normal"] * len(tail))])
    f2 = np.array(["x"] * len(t2))
    tr, te, _ = chronological_split(f2, t2, lab2 != "Normal")
    assert (lab2[te] != "Normal").sum() == 0                    # this is why sanitisation comes first


# ------------------------------------------------------------------ memorisation view (A vs B)
def test_seen_share_single_signature_vs_unique_signature(recordings):
    t, lab, f = recordings
    sig = np.where(lab == "Normal", np.arange(len(t)) % 300,                 # repeated, diverse
                   np.where(lab == "Fuzzy", 10_000 + np.arange(len(t)),      # all different
                            {"DoS": 5000, "RPM": 5001, "Gear": 5002}.get(lab[0], 5000)))
    sig = np.where(lab == "DoS", 5000, np.where(lab == "RPM", 5001, np.where(lab == "Gear", 5002, sig)))
    for tr, te in (random_split(lab, 42), chronological_split(f, t, lab != "Normal")[:2]):
        s = seen_share_by_class(lab, sig, tr, te)
        assert s["DoS"] == s["RPM"] == s["Gear"] == 1.0
        assert s["Fuzzy"] == 0.0
        assert s["Normal"] > 0.9


# ------------------------------------------------------------------ old group split (diagnostic)
def test_group_split_diagnostic_still_clean():
    rng = np.random.default_rng(0)
    sig = rng.integers(0, 500, size=20000)
    tr, te = signature_group_split(sig, 42)
    assert overlap_share(sig, tr, te) == 0.0
    assert abs(share_in_test(tr, te) - 0.2) < 0.03


def test_bad_inputs():
    with pytest.raises(ValueError):
        random_split(np.array(["a", "b"] * 10), 42, test_size=1.5)
    with pytest.raises(ValueError):
        chronological_split(np.array(["f"] * 5), np.arange(5.0), np.zeros(5, bool), train_fraction=0)
