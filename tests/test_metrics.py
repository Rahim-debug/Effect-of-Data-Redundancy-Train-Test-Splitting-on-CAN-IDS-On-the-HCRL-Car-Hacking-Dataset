"""M1 tests: our metrics must match scikit-learn on toy data."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix as sk_cm, f1_score,
                             precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight

from src.metrics import (accuracy, all_normal_baseline, class_weights, confusion_matrix,
                         evaluate, macro_f1, per_class, trivial_baseline_macro_f1)

LABELS = [0, 1, 2, 3, 4]


def _toy(seed, n=2000):
    rng = np.random.default_rng(seed)
    y = rng.choice(5, size=n, p=[0.6, 0.15, 0.1, 0.1, 0.05])
    pred = np.where(rng.random(n) < 0.7, y, rng.integers(0, 5, size=n))
    return y, pred


def test_confusion_matrix_matches_sklearn():
    for seed in range(5):
        y, p = _toy(seed)
        assert (confusion_matrix(y, p) == sk_cm(y, p, labels=LABELS)).all()


def test_per_class_and_macro_match_sklearn():
    for seed in range(5):
        y, p = _toy(seed)
        cm = confusion_matrix(y, p)
        pc = per_class(cm)
        assert np.allclose(pc["precision"], precision_score(y, p, labels=LABELS, average=None))
        assert np.allclose(pc["recall"], recall_score(y, p, labels=LABELS, average=None))
        assert np.allclose(pc["f1"], f1_score(y, p, labels=LABELS, average=None))
        assert np.isclose(macro_f1(cm), f1_score(y, p, labels=LABELS, average="macro"))
        assert np.isclose(accuracy(cm), accuracy_score(y, p))


def test_perfect_prediction():
    y, _ = _toy(0)
    r = evaluate(y, y)
    assert r["accuracy"] == 1.0 and r["macro_f1"] == 1.0
    assert all(r[f"recall_{c}"] == 1.0 for c in ["Normal", "DoS", "Fuzzy", "RPM", "Gear"])


def test_all_normal_baseline_matches_proposal_number():
    # Proposal Step 8: pi_N = 0.9944, C = 5  ->  0.1994
    assert round(trivial_baseline_macro_f1(0.9944, 5), 4) == 0.1994


def test_all_normal_baseline_formula_equals_empirical():
    rng = np.random.default_rng(1)
    y = rng.choice(5, size=100000, p=[0.9944, 0.0014, 0.0014, 0.0014, 0.0014])
    emp = all_normal_baseline(y)
    assert np.isclose(emp["macro_f1"], trivial_baseline_macro_f1(emp["pi_normal"], 5))
    assert np.isclose(emp["accuracy"], emp["pi_normal"])


def test_undefined_cases_are_explicit():
    # class 4 never occurs and is never predicted; class 3 occurs but is never predicted
    y = np.array([0, 0, 1, 2, 3, 3])
    p = np.array([0, 1, 1, 2, 0, 0])
    pc = per_class(confusion_matrix(y, p))
    assert np.isnan(pc["recall"][4]) and np.isnan(pc["precision"][4])   # absent class
    assert np.isnan(pc["precision"][3]) and pc["recall"][3] == 0.0       # never predicted
    assert pc["f1"][4] == 0.0                                            # zero_division=0
    assert np.isclose(macro_f1(confusion_matrix(y, p)),
                      f1_score(y, p, labels=LABELS, average="macro", zero_division=0))


def test_class_weights_match_sklearn_balanced():
    y, _ = _toy(3)
    ours = class_weights(y)
    sk = compute_class_weight("balanced", classes=np.array(LABELS), y=y)
    assert np.allclose(ours, sk)


def test_bad_labels_rejected():
    try:
        confusion_matrix([0, 5], [0, 1])
    except ValueError:
        return
    raise AssertionError("label 5 should be rejected")
