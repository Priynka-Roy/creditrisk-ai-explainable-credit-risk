"""Saved-artifact application flows; no network or model fitting in CI."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.main import SECTIONS, load_dashboard, number, percent, stressed_mean

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "streamlit_app.py"


@pytest.mark.parametrize("section", SECTIONS)
def test_every_application_section_runs(section):
    app = AppTest.from_file(str(ENTRY)).run(timeout=30)
    assert not app.exception
    app.radio[0].set_value(section).run(timeout=30)
    assert not app.exception
    assert not app.error
    assert app.title


def test_explorer_empty_selection_has_friendly_message():
    app = AppTest.from_file(str(ENTRY)).run(timeout=30)
    app.radio[0].set_value("Borrower Data Explorer").run()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("at least one group" in item.value for item in app.info)


@pytest.mark.parametrize("threshold", [0.1, 0.7])
def test_validation_threshold_controls(threshold):
    app = AppTest.from_file(str(ENTRY)).run(timeout=30)
    app.radio[0].set_value("Model Validation").run()
    app.select_slider[0].set_value(threshold).run()
    assert not app.exception
    assert app.select_slider[0].value == threshold
    assert len(app.metric) >= 7


@pytest.mark.parametrize("field", ["Hypothetical LGD", "Hypothetical EAD per account (NT$)"])
def test_zero_loss_inputs_produce_zero_loss(field):
    app = AppTest.from_file(str(ENTRY)).run(timeout=30)
    app.radio[0].set_value("Credit Loss & Stress Scenarios").run()
    widget = next(item for item in [*app.slider, *app.number_input] if item.label == field)
    widget.set_value(0).run()
    assert not app.exception
    loss_metric = next(item for item in app.metric if item.label == "Illustrative expected loss")
    assert loss_metric.value == "NT$ 0"


def test_governance_small_group_and_threshold_diagnostics():
    app = AppTest.from_file(str(ENTRY)).run(timeout=30)
    app.radio[0].set_value("Model Governance & About").run()
    app.selectbox[0].set_value("Education").run()
    assert not app.exception
    assert any("fewer than" in item.value for item in app.warning)
    app.select_slider[0].set_value(0.7).run()
    assert not app.exception


def test_probability_stress_uses_weighted_aggregate_counts():
    histogram = [{"p": 0, "n": 2}, {"p": 0.25, "n": 4}, {"p": 1, "n": 2}]
    assert stressed_mean(histogram, 1) == pytest.approx(0.375)
    assert stressed_mean(histogram, 3) == pytest.approx(0.5)
    assert 0 <= stressed_mean(histogram, 0.25) <= 1


@pytest.mark.parametrize("multiplier", [0, -1, np.inf, np.nan])
def test_invalid_stress_inputs_raise(multiplier):
    with pytest.raises(ValueError):
        stressed_mean([{"p": 0.2, "n": 1}], multiplier)


def test_missing_or_corrupt_artifact_is_rejected(tmp_path):
    artifact = tmp_path / "dashboard.json"
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        load_dashboard(str(artifact), 1)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"sha256": {"artifacts/dashboard.json": "invalid"}})
    )
    with pytest.raises(ValueError, match="integrity"):
        load_dashboard(str(artifact), 2)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"sha256": {"artifacts/dashboard.json": digest}})
    )
    with pytest.raises(ValueError, match="schema"):
        load_dashboard(str(artifact), 3)


def test_undefined_metrics_are_not_fabricated():
    assert number(None) == "Undefined"
    assert percent(np.nan) == "Undefined"
    assert percent(0) == "0.0%"
