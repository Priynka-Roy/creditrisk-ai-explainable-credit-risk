"""Illustrative review criteria; these statuses never approve a lending model."""

from __future__ import annotations

import math
from typing import Any


def diagnostic_status(value: float | None, rule: dict[str, Any]) -> str:
    """Inclusive bounds and explicit unavailable values; no regulatory pass/fail."""
    minimum, maximum = rule.get("minimum"), rule.get("maximum")
    for bound in (minimum, maximum):
        if bound is not None and not math.isfinite(bound):
            raise ValueError("Governance bounds must be finite")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("Governance minimum cannot exceed maximum")
    if minimum is None and maximum is None:
        raise ValueError("At least one illustrative bound is required")
    if value is None or not math.isfinite(value):
        return "Unavailable"
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        return "Review suggested"
    return "Within illustrative tolerance"


def monitoring_values(v1: dict[str, Any], config: dict[str, Any]) -> dict[str, float | None]:
    """Use preserved retrospective aggregates without changing model decisions."""
    values = dict(v1["selected_metrics"])
    psi_rows = v1["stability"]["psi"]
    feature_psi = [
        r["psi"]
        for r in psi_rows
        if r["feature"] != "Estimated probability" and r["psi"] is not None
    ]
    values["max_feature_psi"] = max(feature_psi) if feature_psi else None
    values["prediction_psi"] = next(
        (r["psi"] for r in psi_rows if r["feature"] == "Estimated probability"), None
    )
    data = v1["dataset"]
    values["missing_rate"] = (
        data["missing_cells"] / (data["rows"] * (data["features"] + 2)) if data["rows"] else None
    )
    groups = [
        abs(r["calibration_gap"])
        for r in v1["fairness"]
        if r["n"] >= config["group_min_n"] and r["events"] >= config["group_min_events"]
    ]
    values["max_group_calibration_gap"] = max(groups) if groups else None
    return values


def evaluate_governance(
    values: dict[str, float | None], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Describe observed values, assumed tolerances and analytical follow-up."""
    rows = []
    for rule in config["rules"]:
        value = values.get(rule["metric"])
        status = diagnostic_status(value, rule)
        bounds = f"[{rule.get('minimum', '-∞')}, {rule.get('maximum', '∞')}]"
        rows.append(
            {
                "metric": rule["metric"],
                "label": rule["label"],
                "observed": value,
                "reference": rule["reference"],
                "minimum": rule.get("minimum"),
                "maximum": rule.get("maximum"),
                "illustrative_tolerance": bounds,
                "status": status,
                "interpretation": "Point estimate lies inside an assumed range; sampling uncertainty and external validity still require review."
                if status == "Within illustrative tolerance"
                else "Point estimate warrants analytical follow-up under the selected educational assumption."
                if status == "Review suggested"
                else "Insufficient evidence; no numeric diagnostic is substituted.",
                "follow_up": rule["follow_up"],
                "provenance": config["provenance"],
            }
        )
    return rows
