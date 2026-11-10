import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.common import load_config
from src.models import build_model
from src.synthetic import make_synthetic_arrays

CFG = load_config("config.yaml")


def test_builders_and_seeds():
    m = build_model("LR", 43, CFG)
    assert isinstance(m.steps[-1][1], LogisticRegression)
    assert m.steps[-1][1].random_state == 43 and m.steps[-1][1].class_weight == "balanced"
    d = build_model("DT", 44, CFG)
    assert isinstance(d, DecisionTreeClassifier) and d.random_state == 44 and d.class_weight == "balanced"
    r = build_model("RF", 45, CFG)
    assert isinstance(r, RandomForestClassifier) and r.random_state == 45 and r.class_weight == "balanced"


def test_scaler_fitted_on_training_side_only():
    X, y = make_synthetic_arrays(6000, seed=1)
    Xtr, Xte, ytr = X[:4800], X[4800:], y[:4800]
    m = build_model("LR", 42, CFG).fit(Xtr, ytr)
    scaler = m.steps[0][1]
    assert np.allclose(scaler.mean_, Xtr.mean(axis=0), rtol=1e-4)
    assert not np.allclose(scaler.mean_, X.mean(axis=0), rtol=1e-6)  # test rows were not used


def test_same_seed_same_predictions():
    X, y = make_synthetic_arrays(5000, seed=2)
    cfg = {**CFG, "models": {**CFG["models"], "RF": {**CFG["models"]["RF"], "n_estimators": 5}}}
    for name in ["DT", "RF"]:
        a = build_model(name, 42, cfg).fit(X[:4000], y[:4000]).predict(X[4000:])
        b = build_model(name, 42, cfg).fit(X[:4000], y[:4000]).predict(X[4000:])
        assert (a == b).all()


def test_unknown_model_rejected():
    try:
        build_model("SVM", 42, CFG)
    except ValueError:
        return
    raise AssertionError("SVM is out of scope")
