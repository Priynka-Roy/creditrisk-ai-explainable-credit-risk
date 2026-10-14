"""Versioned-result contracts and actual frozen baseline preservation."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def test_v1_bytes_preserved_and_v2_results_are_separate():
    manifest = load("artifacts/v2/manifest.json")
    assert len(manifest["v1_preserved_sha256"]) == 7
    for name, digest in manifest["v1_preserved_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert load("artifacts/dashboard.json")["schema_version"] == 1
    assert load("artifacts/v2/dashboard.json")["schema_version"] == 2


def test_paired_differences_match_frozen_v1_point_estimates():
    v1, v2 = load("artifacts/dashboard.json"), load("artifacts/v2/dashboard.json")
    scores = {m["name"]: m["metrics"] for m in v1["models"]}
    assert len(v2["paired_comparisons"]) == 15
    for r in v2["paired_comparisons"]:
        expected = scores[r["model_a"]][r["metric"]] - scores[r["model_b"]][r["metric"]]
        assert np.isclose(r["difference"], expected, atol=1e-10)
        assert r["lower"] <= r["upper"]
        assert r["includes_zero"] == (r["lower"] <= 0 <= r["upper"])
        assert r["replicates"] == 1000
        assert r["favorable_direction"] == (
            "lower" if r["metric"] in {"brier", "log_loss"} else "higher"
        )


def test_retrospective_calibration_and_group_denominators():
    v1, v2 = load("artifacts/dashboard.json"), load("artifacts/v2/dashboard.json")
    assert v2["metadata"]["test_usage"].startswith("Retrospective")
    assert v2["metadata"]["calibration_method"] == "V1 none unchanged"
    for row in v2["calibration_uncertainty"]["metrics"]:
        assert np.isclose(row["estimate"], v1["selected_metrics"][row["metric"]])
        assert row["valid_replicates"] == 1000
        assert row["lower"] <= row["upper"]
    for dimension in {r["dimension"] for r in v2["subgroup_uncertainty"]}:
        rows = [r for r in v2["subgroup_uncertainty"] if r["dimension"] == dimension]
        assert sum(r["n"] for r in rows) == 6000
        assert sum(r["events"] for r in rows) == 1327
        for row in rows:
            if row["small_group"]:
                assert row["gap_lower"] is None and row["gap_upper"] is None


def test_coefficient_stability_fixed_names_and_development_only():
    v1, v2 = load("artifacts/dashboard.json"), load("artifacts/v2/dashboard.json")
    original_names = {r["feature"] for r in v1["explainability"]["coefficients"]}
    resampled = v2["stability"]["coefficient_bootstrap"]
    assert {r["feature"] for r in resampled} == original_names
    assert len(resampled) == len(original_names)
    assert v2["stability"]["replicates"] == 100
    for row in resampled:
        assert row["lower"] <= row["upper"]
        assert 0 <= row["positive_fraction"] <= 1


def test_simulated_marginal_mean_and_tail_contracts():
    v2 = load("artifacts/v2/dashboard.json")
    a, b = v2["simulations"][:2]
    assert a["rho"] == 0 and b["rho"] == 0.15
    assert a["analytic_expected_loss"] == b["analytic_expected_loss"]
    for r in v2["simulations"]:
        assert r["mean_pd"] >= 0 and r["mean_pd"] <= 1
        assert r["mean"] >= 0 and r["std"] >= 0
        assert r["es"] >= r["var"]
        assert r["n_simulations"] == sum(bin["n"] for bin in r["histogram"])
        assert abs(r["mean_error"]) <= 6 * r["mean_se"]
        assert r["currency"] == "NTD"
    assert b["std"] > a["std"] and b["es"] > a["es"]
