"""Deterministic mathematical and leakage edge cases, without UCI downloads."""

import numpy as np
import pandas as pd
import pytest

from creditrisk.calibration import ProbabilityCalibrator, calibration_parameters
from creditrisk.data import FEATURES, TARGET, make_splits, split_hash, validate_data
from creditrisk.evaluation import bootstrap_metrics, curves, metrics, reliability, threshold_metrics
from creditrisk.features import engineer_features, signed_log
from creditrisk.loss import expected_loss, stress_probabilities


@pytest.fixture
def customers():
    data = {c: np.arange(1, 21, dtype=float) for c in FEATURES}
    data.update(
        ID=np.arange(1, 21),
        LIMIT_BAL=np.repeat(10000.0, 20),
        SEX=np.tile([1, 2], 10),
        EDUCATION=np.tile([0, 1, 2, 3, 5], 4),
        MARRIAGE=np.tile([0, 1, 2, 3], 5),
    )
    for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        data[c] = np.tile([-2, -1, 0, 1, 2], 4)
    data[TARGET] = np.tile([0, 1], 10)
    return pd.DataFrame(data)


def test_schema_audit_preserves_unusual_codes(customers):
    before = customers.copy(deep=True)
    audit = validate_data(customers, 20)
    assert audit["missing_cells"] == audit["duplicate_ids"] == 0
    assert audit["events"] == 10
    assert "0" in audit["category_counts"]["EDUCATION"]
    pd.testing.assert_frame_equal(before, customers)


@pytest.mark.parametrize(
    "problem", ["missing_column", "duplicate_id", "missing", "target", "limit"]
)
def test_schema_rejects_invalid_data(customers, problem):
    if problem == "missing_column":
        customers = customers.drop(columns="AGE")
    elif problem == "duplicate_id":
        customers.loc[0, "ID"] = customers.loc[1, "ID"]
    elif problem == "missing":
        customers.loc[0, "AGE"] = np.nan
    elif problem == "target":
        customers.loc[0, TARGET] = 2
    else:
        customers.loc[0, "LIMIT_BAL"] = 0
    with pytest.raises(ValueError):
        validate_data(customers)


def test_split_isolation_reproducibility_and_target_balance():
    y = np.tile([0, 0, 0, 0, 1], 1000)
    first, second = make_splits(y), make_splits(y)
    expected = {"training": 2500, "validation": 750, "calibration": 750, "test": 1000}
    for part in first:
        np.testing.assert_array_equal(first[part], second[part])
        assert len(first[part]) == expected[part]
        assert y[first[part]].mean() == pytest.approx(0.2)
        assert split_hash(first[part]) == split_hash(first[part][::-1])
    groups = list(first.values())
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            assert not set(a) & set(b)
    assert len(set(np.concatenate(groups))) == len(y)


def test_ratio_zero_negative_and_feature_target_exclusion(customers):
    customers.loc[0, "BILL_AMT1"] = 0
    customers.loc[1, "BILL_AMT1"] = -100
    customers.loc[2, "BILL_AMT1"] = 10
    customers.loc[2, "PAY_AMT1"] = 20
    before = customers.copy(deep=True)
    x = engineer_features(customers)
    assert "ID" not in x and TARGET not in x
    assert x.loc[0, "PAYMENT_BILL_RATIO"] == x.loc[1, "PAYMENT_BILL_RATIO"] == 0
    assert x.loc[0, "RATIO_UNDEFINED"] == x.loc[1, "RATIO_UNDEFINED"] == 1
    assert x.loc[2, "PAYMENT_BILL_RATIO"] == 2
    assert np.isfinite(x.to_numpy()).all()
    other = customers.copy()
    other[TARGET] = 1 - other[TARGET]
    pd.testing.assert_frame_equal(x, engineer_features(other))
    pd.testing.assert_frame_equal(before, customers)
    np.testing.assert_allclose(signed_log(np.array([-9, 0, 9])), [-np.log(10), 0, np.log(10)])


def test_known_discrimination_and_probability_metrics():
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    result = metrics(y, p)
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["pr_auc"] == pytest.approx(5 / 6)
    assert result["ks"] == pytest.approx(0.5)
    assert result["brier"] == pytest.approx((0.01 + 0.16 + 0.4225 + 0.04) / 4)
    assert result["log_loss"] == pytest.approx(-np.log([0.9, 0.6, 0.35, 0.8]).mean())
    r = threshold_metrics(y, p, 0.4)
    assert (r["tn"], r["fp"], r["fn"], r["tp"]) == (1, 1, 1, 1)
    assert r["precision"] == r["recall"] == r["specificity"] == 0.5


def test_equal_predictions_and_single_class_are_explicit():
    r = metrics([0, 1, 0, 1], [0.5] * 4)
    assert r["roc_auc"] == 0.5 and r["pr_auc"] == 0.5 and r["ks"] == 0
    assert r["slope"] is None and r["intercept"] is None
    r = metrics([0, 0], [0.2, 0.3])
    assert r["roc_auc"] is None and r["pr_auc"] is None and r["ks"] is None
    assert threshold_metrics([0, 0], [0.1, 0.2])["recall"] is None
    assert curves([1, 1], [0.1, 0.2])["roc"]["fpr"] == []
    assert bootstrap_metrics([1, 1], [0.1, 0.2]) == []


def test_reliability_empty_bins_and_closed_probability_endpoint():
    rows = reliability([0, 1], [0.0, 1.0], bins=10)
    assert sum(r["n"] for r in rows) == 2
    assert rows[0]["n"] == rows[-1]["n"] == 1
    assert rows[4]["predicted"] is None and rows[4]["ci_lower"] is None
    assert 0 <= rows[0]["ci_lower"] <= rows[0]["ci_upper"] <= 1


def test_calibration_parameters_on_exact_empirical_probabilities():
    y, p = [], []
    for rate in [0.2, 0.4, 0.6, 0.8]:
        y.extend([1] * int(rate * 100) + [0] * (100 - int(rate * 100)))
        p.extend([rate] * 100)
    r = calibration_parameters(y, p)
    assert r["intercept"] == pytest.approx(0, abs=0.02)
    assert r["slope"] == pytest.approx(1, abs=0.02)


@pytest.mark.parametrize("method", ["none", "sigmoid", "isotonic"])
def test_independent_calibration_bounds(method):
    fit_p = np.linspace(0.05, 0.95, 100)
    fit_y = np.tile([0, 0, 0, 1, 1], 20)
    model = ProbabilityCalibrator(method).fit(fit_p, fit_y)
    heldout = np.array([0.0, 0.03, 0.5, 0.98, 1.0])
    predicted = model.predict(heldout)
    assert len(predicted) == 5 and np.isfinite(predicted).all()
    assert ((predicted >= 0) & (predicted <= 1)).all()


def test_stratified_bootstrap_is_deterministic_and_contains_point_estimate():
    y = np.tile([0, 1], 30)
    p = np.linspace(0.1, 0.9, 60)
    a = bootstrap_metrics(y, p, repetitions=30, seed=17)
    assert a == bootstrap_metrics(y, p, repetitions=30, seed=17)
    for row in a:
        assert row["lower"] <= row["estimate"] <= row["upper"]


def test_loss_units_zero_assumptions_and_stress_order():
    assert expected_loss([0.1, 0.3], 0.5, 1000, 100) == pytest.approx(10000)
    assert expected_loss([0.5], 0, 1000, 100) == 0
    assert expected_loss([0.5], 0.5, 0, 100) == 0
    p = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    np.testing.assert_allclose(stress_probabilities(p, 1), p)
    stress = stress_probabilities(p, 2)
    assert stress[2] == pytest.approx(2 / 3)
    assert stress[0] == 0 and stress[-1] == 1
    assert (np.diff(stress) >= 0).all()


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_stress_multiplier(value):
    with pytest.raises(ValueError):
        stress_probabilities([0.2], value)


@pytest.mark.parametrize("p", [[-0.1], [1.1], [np.nan]])
def test_invalid_probability_inputs(p):
    with pytest.raises(ValueError):
        expected_loss(p, 0.5, 1000)
    with pytest.raises(ValueError):
        metrics([0], p)


@pytest.mark.parametrize("lgd,ead,n", [(-0.1, 100, 1), (1.1, 100, 1), (0.5, -1, 1), (0.5, 100, -1)])
def test_invalid_loss_assumptions(lgd, ead, n):
    with pytest.raises(ValueError):
        expected_loss([0.1], lgd, ead, n)
