"""Validate committed aggregate artifacts without downloading customer data."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Artifact hash mismatch: {name}")
    d = json.loads((ROOT / "artifacts/dashboard.json").read_text())
    assert d["schema_version"] == 1
    assert d["dataset"]["rows"] == 30000
    assert sum(s["n"] for s in d["dataset"]["splits"].values()) == 30000
    assert (
        sum(r["n"] for r in d["calibration"]["reliability"]) == d["dataset"]["splits"]["test"]["n"]
    )
    assert (
        sum(r["n"] for r in d["loss"]["probability_histogram"])
        == d["dataset"]["splits"]["test"]["n"]
    )
    assert d["selected_model"] == d["selection"]["selected_model"]
    assert d["calibration"]["selected_method"] == d["selection"]["selected_calibration"]
    assert (
        d["selection"]["protocol_sha256"]
        == hashlib.sha256((ROOT / "configs/protocol.json").read_bytes()).hexdigest()
    )
    assert not any(k in d for k in ["customers", "raw_rows", "ids", "individual_records"])
    for dimension in {r["dimension"] for r in d["fairness"]}:
        assert (
            sum(r["n"] for r in d["fairness"] if r["dimension"] == dimension)
            == d["dataset"]["splits"]["test"]["n"]
        )
    for model in d["models"]:
        assert 0 <= model["metrics"]["roc_auc"] <= 1
        assert 0 <= model["metrics"]["brier"] <= 1
    print(f"Verified {len(manifest['sha256'])} file hashes and aggregate dashboard contracts.")
    return d


if __name__ == "__main__":
    verify()
