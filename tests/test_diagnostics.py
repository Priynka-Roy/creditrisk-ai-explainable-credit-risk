"""Holdout transformations, subgroup denominators and distribution edge cases."""

import numpy as np
import pandas as pd
import pytest

from creditrisk.data import FEATURES, TARGET
from creditrisk.fairness import subgroup_diagnostics
from creditrisk.features import engineer_features
from creditrisk.modeling import make_model, select_model
from creditrisk.stability import population_stability_index


def sample(n=40):
    df = pd.DataFrame({c: np.tile([1.0, 2.0], n // 2) for c in FEATURES})
    df["LIMIT_BAL"] = np.linspace(10000, 100000, n)
    df["AGE"] = np.arange(n) + 21
    df["ID"] = np.arange(n)
    df[TARGET] = np.tile([0, 1], n // 2)
    return df


def test_preprocessor_fits_training_only_and_handles_unseen_categories():
    train = sample()
    x = engineer_features(train)
    model = make_model(x, "l2").fit(x, train[TARGET])
    processor = model.named_steps["preprocessing"]
    before = processor.named_transformers_["numeric"].mean_.copy()
    heldout = train.copy()
    heldout["LIMIT_BAL"] = 1e12
    heldout["EDUCATION"] = 999
    p = model.predict_proba(engineer_features(heldout))[:, 1]
    np.testing.assert_array_equal(before, processor.named_transformers_["numeric"].mean_)
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    assert not any("ID" == c or TARGET == c for c in processor.get_feature_names_out())


def test_model_selection_uses_joint_tolerance_and_simplicity():
    base = {"log_loss": 0.40, "brier": 0.12, "roc_auc": 0.8}
    results = {"baseline": dict(base), "l2": dict(base), "l1": dict(base), "boosting": dict(base)}
    results["boosting"]["log_loss"] = 0.399
    assert select_model(results) == "baseline"
    results["boosting"]["log_loss"] = 0.38
    assert select_model(results) == "boosting"
    results["baseline"]["log_loss"] = 0.381
    results["baseline"]["brier"] = 0.15
    assert select_model(results) == "boosting"


def test_subgroups_counts_and_absent_events():
    df = sample()
    df.loc[0, "SEX"] = 999
    y = np.zeros(len(df), dtype=int)
    p = np.full(len(df), 0.1)
    rows = subgroup_diagnostics(df, y, p)
    for dimension in {r["dimension"] for r in rows}:
        assert sum(r["n"] for r in rows if r["dimension"] == dimension) == len(df)
    for r in rows:
        assert r["events"] == 0 and r["small_group"]
        assert r["fnr"] is None
        assert r["ci_lower"] <= r["observed"] <= r["ci_upper"]
        assert r["calibration_gap"] == pytest.approx(0.1)
        assert r["fp"] + r["tn"] == r["n"]


def test_psi_equal_distribution_zero_and_shift_visible():
    reference = np.arange(100, dtype=float)
    same = population_stability_index(reference, reference)
    assert same["psi"] == pytest.approx(0)
    assert population_stability_index(reference, reference + 100)["psi"] > 0
    constant = population_stability_index(np.ones(100), np.ones(100) * 2)
    assert constant["psi"] is None or constant["psi"] > 0
