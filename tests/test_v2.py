"""Independent paired-inference, simulation and governance numerical checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from creditrisk.governance import diagnostic_status, evaluate_governance, monitoring_values
from creditrisk.simulation import portfolio_from_histogram, simulate_portfolio, summarize_losses
from creditrisk.v2_inference import (
    calibration_uncertainty,
    paired_bootstrap,
    subgroup_uncertainty,
)


def test_paired_bootstrap_identical_predictions_cancel_exactly():
    y = np.tile([0, 1], 20)
    p = np.linspace(0.1, 0.9, len(y))
    rows = paired_bootstrap(y, {"boosting": p, "l2": p.copy()}, comparators=("l2",), repetitions=30)
    assert len(rows) == 5
    for row in rows:
        assert row["difference"] == row["lower"] == row["upper"] == 0
        assert row["includes_zero"]


def test_paired_signs_known_perfect_and_reversed_ranking():
    y = np.tile([0, 1], 20)
    good = np.where(y == 1, 0.9, 0.1)
    bad = 1 - good
    rows = paired_bootstrap(y, {"boosting": good, "l2": bad}, comparators=("l2",), repetitions=30)
    rows = {r["metric"]: r for r in rows}
    assert rows["roc_auc"]["difference"] == rows["roc_auc"]["lower"] == 1
    assert rows["roc_auc"]["favorable_direction"] == "higher"
    assert rows["brier"]["difference"] == pytest.approx(-0.8)
    assert rows["brier"]["upper"] < 0
    assert rows["log_loss"]["upper"] < 0
    assert rows["brier"]["favorable_direction"] == "lower"


def test_paired_reproducibility_and_absent_class():
    y = np.tile([0, 1], 20)
    p = np.linspace(0.1, 0.9, 40)
    args = dict(
        predictions={"boosting": p, "l2": p[::-1]}, comparators=("l2",), repetitions=30, seed=22
    )
    assert paired_bootstrap(y, **args) == paired_bootstrap(y, **args)
    missing = paired_bootstrap(np.zeros(40), **args)
    assert all(
        r["difference"] is None for r in missing if r["metric"] in {"roc_auc", "pr_auc", "ks"}
    )
    assert all(
        np.isfinite(r["difference"]) for r in missing if r["metric"] in {"brier", "log_loss"}
    )


def test_calibration_constant_scores_keep_identifiable_oe_uncertainty():
    y = np.r_[np.ones(20), np.zeros(80)]
    data = calibration_uncertainty(y, np.repeat(0.2, 100), repetitions=30)
    rows = {r["metric"]: r for r in data["metrics"]}
    assert rows["slope"]["estimate"] is None
    assert rows["intercept"]["estimate"] is None
    oe = rows["observed_expected_ratio"]
    assert oe["estimate"] == pytest.approx(1)
    assert oe["lower"] < 1 < oe["upper"]


def test_zero_expected_events_have_explicit_undefined_oe():
    result = calibration_uncertainty(np.tile([0, 1], 20), np.zeros(40), repetitions=30)
    oe = next(r for r in result["metrics"] if r["metric"] == "observed_expected_ratio")
    assert oe["estimate"] is None and oe["lower"] is None and oe["upper"] is None


def test_calibration_bootstrap_is_deterministic_and_finite():
    y = np.tile([0, 0, 1, 1], 30)
    p = np.linspace(0.1, 0.9, len(y))
    a = calibration_uncertainty(y, p, repetitions=30, seed=5)
    assert a == calibration_uncertainty(y, p, repetitions=30, seed=5)
    for row in a["metrics"]:
        assert np.isfinite([row["lower"], row["upper"]]).all()
        assert row["lower"] <= row["upper"]


def test_sparse_group_uncertainty_preserves_denominators_and_caution():
    y = np.r_[np.zeros(2), np.tile([0, 1], 50)]
    p = np.linspace(0.1, 0.9, len(y))
    rows = subgroup_uncertainty(y, p, {"example": ["tiny"] * 2 + ["large"] * 100}, repetitions=30)
    assert sum(r["n"] for r in rows) == len(y)
    assert sum(r["events"] for r in rows) == y.sum()
    tiny = next(r for r in rows if r["group"] == "tiny")
    assert tiny["small_group"] and tiny["fnr"] is None
    assert tiny["fnr_lower"] is None
    assert tiny["gap_lower"] is None and tiny["gap_upper"] is None
    assert tiny["fpr_lower"] <= tiny["fpr"] <= tiny["fpr_upper"]


def test_empirical_var_and_fractional_tail_es_definitions():
    # Inverse CDF at 75% is 2. Average exactly top25% mass is 100, not mean(L>=2).
    r = summarize_losses([0, 1, 2, 100], alpha=0.75)
    assert r["var"] == 2 and r["es"] == 100
    # Top 37.5% of four observations: all100 +half2, divided by1.5.
    r = summarize_losses([0, 1, 2, 100], alpha=0.625)
    assert r["var"] == 2 and r["es"] == pytest.approx(101 / 1.5)
    r = summarize_losses([0, 10, 10, 10], alpha=0.75)
    assert r["var"] == r["es"] == 10
    assert summarize_losses([50])["std"] == 0


@pytest.mark.parametrize("rho", [0, 0.25, 1])
def test_simulation_endpoints_and_heterogeneous_exposure_arithmetic(rho):
    out = simulate_portfolio(
        [0, 1, 1], lgd=[0.5, 0.5, 1], ead=[100, 100, 200], n_simulations=100, rho=rho
    )
    np.testing.assert_array_equal(out["marginal_probabilities"], [0, 1, 1])
    np.testing.assert_array_equal(out["losses"], np.repeat(250.0, 100))
    assert out["analytic_expected_loss"] == 250


@pytest.mark.parametrize("lgd,ead", [(0, 1000), (0.5, 0)])
def test_zero_lgd_or_ead_produces_zero_loss(lgd, ead):
    result = simulate_portfolio([0.2, 0.4], lgd=lgd, ead=ead, n_simulations=100, rho=0.2)
    assert np.count_nonzero(result["losses"]) == 0
    assert result["analytic_expected_loss"] == 0


def test_mc_reproducibility_and_chunk_invariance():
    args = dict(probabilities=[0.1, 0.2, 0.6, 0.9], n_simulations=503, seed=12, rho=0.25)
    a = simulate_portfolio(**args, batch_size=7)
    b = simulate_portfolio(**args, batch_size=250)
    np.testing.assert_array_equal(a["losses"], b["losses"])
    np.testing.assert_array_equal(a["marginal_probabilities"], b["marginal_probabilities"])


@pytest.mark.parametrize("rho", [0, 0.3, 1])
def test_marginals_and_mc_mean_agree_with_analytic_expectation(rho):
    p = np.array([0.1, 0.3, 0.6])
    out = simulate_portfolio(p, ead=100, lgd=0.5, n_simulations=20000, rho=rho, seed=8)
    marginal_se = np.sqrt(p * (1 - p) / 20000)
    assert (np.abs(out["marginal_probabilities"] - p) < 6 * marginal_se).all()
    summary = summarize_losses(out["losses"])
    assert abs(summary["mean"] - out["analytic_expected_loss"]) < 6 * summary["mean_se"]
    assert out["analytic_expected_loss"] == pytest.approx(50)


def test_assumed_dependence_changes_tail_not_theoretical_mean():
    p = np.repeat(0.2, 100)
    independent = simulate_portfolio(p, ead=100, lgd=0.5, n_simulations=5000, rho=0, seed=98)
    dependent = simulate_portfolio(p, ead=100, lgd=0.5, n_simulations=5000, rho=0.3, seed=98)
    assert independent["analytic_expected_loss"] == dependent["analytic_expected_loss"]
    a, b = summarize_losses(independent["losses"]), summarize_losses(dependent["losses"])
    assert b["std"] > a["std"]
    assert b["es"] > a["es"]


def test_quantile_portfolio_retains_histogram_weights_deterministically():
    hist = [{"p": 0.8, "n": 1}, {"p": 0.1, "n": 3}]
    p = portfolio_from_histogram(hist, size=4)
    np.testing.assert_array_equal(p, [0.1, 0.1, 0.1, 0.8])
    assert p.mean() == pytest.approx(0.275)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rho": -0.1},
        {"rho": 1.1},
        {"rho": np.nan},
        {"ead": -1},
        {"lgd": 1.1},
        {"n_simulations": 0},
        {"n_simulations": 20001},
        {"batch_size": 0},
    ],
)
def test_simulation_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        simulate_portfolio([0.2], **kwargs)


@pytest.mark.parametrize(
    "losses,alpha", [([], 0.95), ([-1], 0.95), ([np.nan], 0.95), ([0], 1), ([0], 0)]
)
def test_tail_metrics_reject_invalid_inputs(losses, alpha):
    with pytest.raises(ValueError):
        summarize_losses(losses, alpha)


def test_governance_boundaries_and_unavailable_values():
    rule = {"minimum": 0.8, "maximum": 1.2}
    assert diagnostic_status(0.8, rule) == "Within illustrative tolerance"
    assert diagnostic_status(1.2, rule) == "Within illustrative tolerance"
    assert diagnostic_status(0.79, rule) == "Review suggested"
    assert diagnostic_status(None, rule) == diagnostic_status(np.nan, rule) == "Unavailable"
    with pytest.raises(ValueError):
        diagnostic_status(1, {"minimum": 2, "maximum": 1})


def test_governance_is_educational_and_can_flag_actual_retrospective_value():
    root = Path(__file__).resolve().parents[1]
    v1 = json.loads((root / "artifacts/dashboard.json").read_text())
    config = json.loads((root / "configs/v2_governance.json").read_text())
    values = monitoring_values(v1, config)
    rows = evaluate_governance(values, config)
    assert len(rows) == 10
    assert all(r["follow_up"] and r["provenance"] for r in rows)
    config["rules"][0]["minimum"] = 0.8
    changed = evaluate_governance(values, config)
    assert changed[0]["status"] == "Review suggested"
    assert v1["selected_model"] == "boosting"
