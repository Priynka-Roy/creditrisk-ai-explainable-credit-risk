"""V2 dashboard provenance, version compatibility and bounded scenario controls."""

import hashlib
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app.v2 as v2_ui
from app.v2 import load_v2, run_custom_simulation

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "streamlit_app.py"


@pytest.mark.parametrize(
    "section,case",
    [
        ("Model Validation", "E"),
        ("Probability Calibration", "F"),
        ("Credit Loss & Stress Scenarios", "G"),
        ("Model Governance & About", "H"),
    ],
)
def test_v2_panels_and_computed_guided_cases(section, case):
    app = AppTest.from_file(str(ENTRY)).run(timeout=30)
    app.radio[0].set_value(section).run(timeout=30)
    assert not app.exception
    assert any("Retrospective" in caption.value for caption in app.caption)
    assert any(item.label.startswith(f"Guided case {case} ·") for item in app.expander)


def test_paired_metric_selection_uses_saved_differences():
    app = AppTest.from_file(str(ENTRY)).run(timeout=30)
    app.radio[0].set_value("Model Validation").run()
    app.selectbox(key="v2_paired_metric").set_value("log_loss").run()
    assert not app.exception
    frames = [frame.value for frame in app.dataframe]
    compared = next(frame for frame in frames if "difference" in frame and len(frame) == 3)
    assert (compared.difference < 0).all()
    assert any("negative difference favors Model A" in caption.value for caption in app.caption)


def test_sparse_v2_calibration_groups_stay_visible_and_cautious():
    app = AppTest.from_file(str(ENTRY)).run(timeout=30)
    app.radio[0].set_value("Probability Calibration").run()
    app.selectbox(key="v2_calibration_group").set_value("Education").run()
    assert not app.exception
    assert any("Sparse groups" in warning.value for warning in app.warning)


def test_governance_tolerance_changes_flag_but_not_frozen_model():
    frozen_before = hashlib.sha256((ROOT / "artifacts/dashboard.json").read_bytes()).hexdigest()
    app = AppTest.from_file(str(ENTRY)).run(timeout=30)
    app.radio[0].set_value("Model Governance & About").run()
    app.slider(key="v2_governance_auc").set_value(0.80).run()
    assert not app.exception
    assert any("Review suggested" in frame.value.to_string() for frame in app.dataframe)
    assert (
        hashlib.sha256((ROOT / "artifacts/dashboard.json").read_bytes()).hexdigest()
        == frozen_before
    )


@pytest.mark.parametrize("field", ["v2_mc_lgd", "v2_mc_ead"])
def test_requested_zero_loss_simulation_is_safe(field):
    app = AppTest.from_file(str(ENTRY)).run(timeout=30)
    app.radio[0].set_value("Credit Loss & Stress Scenarios").run()
    widgets = app.slider if field == "v2_mc_lgd" else app.number_input
    widgets(key=field).set_value(0).run()
    app.button(key="v2_run_simulation").click().run(timeout=30)
    assert not app.exception
    result = app.session_state["v2_simulation_result"]["rows"]
    assert all(row["mean"] == row["var_95"] == row["es_95"] == 0 for row in result)
    assert all(row["histogram"][0]["lower"] == 0 for row in result)
    app.slider(key="v2_mc_odds").set_value(1.5).run()
    assert any("Controls have changed" in item.value for item in app.info)


def test_simulation_is_deterministic_and_expected_loss_is_common():
    histogram = json.dumps([{"p": 0.1, "n": 2}, {"p": 0.4, "n": 3}])
    first = run_custom_simulation(histogram, 20, 500, 0.2, 0.45, 1000, 1)
    second = run_custom_simulation(histogram, 20, 500, 0.2, 0.45, 1000, 1)
    assert first == second
    assert first[0]["analytic_expected_loss"] == first[1]["analytic_expected_loss"]
    assert first[0]["var_95"] <= first[0]["es_95"]


@pytest.mark.parametrize(
    "size,repetitions,rho", [(1001, 500, 0.2), (100, 5001, 0.2), (100, 500, 0.65), (9, 500, 0.2)]
)
def test_simulation_workload_limits(size, repetitions, rho):
    with pytest.raises(ValueError, match="bounds"):
        run_custom_simulation("[]", size, repetitions, rho, 0.45, 1000, 1)


def test_v2_missing_falls_back_without_inventing_statistics(monkeypatch, tmp_path):
    monkeypatch.setattr(v2_ui, "V2_ARTIFACT", tmp_path / "missing.json")
    app = AppTest.from_file(str(ENTRY)).run(timeout=30)
    assert not app.exception
    assert any("no Version 2 statistics" in item.value for item in app.info)
    app.radio[0].set_value("Model Validation").run()
    assert not any(item.label.startswith("Guided case E") for item in app.expander)


def test_v2_schema_compatibility_and_integrity(tmp_path):
    path = tmp_path / "dashboard.json"
    path.write_text(json.dumps({"schema_version": 1}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sha256": {"artifacts/v2/dashboard.json": digest}}))
    with pytest.raises(ValueError, match="schema"):
        load_v2(str(path), 1, 1)
    path.write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(ValueError, match="integrity"):
        load_v2(str(path), 2, 1)
