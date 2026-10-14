"""Audit published aggregate contracts independently of raw-customer downloads."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from creditrisk.loss import expected_loss, stress_probabilities

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dashboard():
    return json.loads((ROOT / "artifacts/dashboard.json").read_text())


def test_artifact_hashes_and_frozen_protocol(dashboard):
    manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert (
        hashlib.sha256((ROOT / "configs/protocol.json").read_bytes()).hexdigest()
        == dashboard["selection"]["protocol_sha256"]
    )
    assert dashboard["selected_model"] == dashboard["selection"]["selected_model"]
    assert (
        dashboard["calibration"]["selected_method"]
        == dashboard["selection"]["selected_calibration"]
    )
    assert dashboard["selection"]["test_access"] == "Not accessed before writing this decision file"


def test_source_population_and_test_denominators(dashboard):
    data = dashboard["dataset"]
    assert (data["rows"], data["features"], data["events"]) == (30000, 23, 6636)
    assert sum(s["n"] for s in data["splits"].values()) == data["rows"]
    assert sum(s["events"] for s in data["splits"].values()) == data["events"]
    n, events = data["splits"]["test"]["n"], data["splits"]["test"]["events"]
    bins = dashboard["calibration"]["reliability"]
    assert sum(b["n"] for b in bins) == n
    assert sum(b["events"] for b in bins) == events
    for row in dashboard["selected_thresholds"]:
        assert row["tp"] + row["fn"] == events
        assert row["tn"] + row["fp"] == n - events
    for dimension in {r["dimension"] for r in dashboard["fairness"]}:
        rows = [r for r in dashboard["fairness"] if r["dimension"] == dimension]
        assert sum(r["n"] for r in rows) == n
        assert sum(r["events"] for r in rows) == events


def test_loss_approximation_is_bounded_by_histogram_rounding(dashboard):
    histogram = dashboard["loss"]["probability_histogram"]
    p = np.array([r["p"] for r in histogram])
    counts = np.array([r["n"] for r in histogram])
    assert counts.sum() == dashboard["dataset"]["splits"]["test"]["n"]
    assert ((p >= 0) & (p <= 1)).all()
    for scenario in dashboard["loss"]["scenarios"]:
        stressed = stress_probabilities(p, scenario["odds_multiplier"])
        mean = np.average(stressed, weights=counts)
        estimated = expected_loss(
            [mean], scenario["lgd"], scenario["ead"], scenario["portfolio_size"]
        )
        # Odds transformation derivative <= max(m,1/m); rounding each p <=.0005.
        m = scenario["odds_multiplier"]
        bound = (
            0.0005 * max(m, 1 / m) * scenario["lgd"] * scenario["ead"] * scenario["portfolio_size"]
        )
        assert abs(estimated - scenario["expected_loss"]) <= bound


def test_every_selected_metric_and_uncertainty_is_finite(dashboard):
    values = dashboard["selected_metrics"]
    for name in ["roc_auc", "pr_auc", "ks", "brier"]:
        assert 0 <= values[name] <= 1
    for row in dashboard["uncertainty"]:
        assert row["replicates"] == 300
        assert row["lower"] <= row["estimate"] <= row["upper"]
    assert np.isfinite(list(values.values())).all()
    assert len(dashboard["cases"]) == 4
    assert dashboard["metadata"]["license"] == "CC BY 4.0"
    assert dashboard["metadata"]["currency"] == "NTD"
