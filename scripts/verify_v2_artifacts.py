"""Verify V2 deployment integrity and frozen V1 bytes without source downloads."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / "artifacts/v2/manifest.json").read_text())
    for path, digest in {**manifest["sha256"], **manifest["v1_preserved_sha256"]}.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Artifact fingerprint mismatch: {path}")
    v1 = json.loads((ROOT / "artifacts/dashboard.json").read_text())
    v2 = json.loads((ROOT / "artifacts/v2/dashboard.json").read_text())
    assert v2["schema_version"] == 2
    assert v2["metadata"]["test_usage"].startswith("Retrospective")
    assert v2["metadata"]["calibration_method"] == "V1 none unchanged"
    for model, result in v2["metadata"]["model_reproduction"].items():
        assert result["maximum_absolute_metric_difference"] <= 1e-7, model
    original = {m["name"]: m["metrics"] for m in v1["models"]}
    assert len(v2["paired_comparisons"]) == 15
    for row in v2["paired_comparisons"]:
        expected = original[row["model_a"]][row["metric"]] - original[row["model_b"]][row["metric"]]
        assert np.isclose(row["difference"], expected, atol=1e-10)
        assert row["lower"] <= row["upper"]
        assert row["replicates"] == 1000
    assert v2["stability"]["replicates"] == 100
    for dimension in {r["dimension"] for r in v2["subgroup_uncertainty"]}:
        assert (
            sum(r["n"] for r in v2["subgroup_uncertainty"] if r["dimension"] == dimension) == 6000
        )
    simulations = v2["simulations"]
    assert np.isclose(
        simulations[0]["analytic_expected_loss"], simulations[1]["analytic_expected_loss"]
    )
    for row in simulations:
        assert row["es"] >= row["var"] >= 0
        assert sum(r["n"] for r in row["histogram"]) == row["n_simulations"]
        assert row["currency"] == "NTD"
    assert not any(k in v2 for k in ["customer_records", "individual_predictions", "ids"])
    print(
        f"Verified V2 {len(manifest['sha256'])} hashes; frozen V1 {len(manifest['v1_preserved_sha256'])} hashes and diagnostic contracts."
    )
    return v2


if __name__ == "__main__":
    verify()
