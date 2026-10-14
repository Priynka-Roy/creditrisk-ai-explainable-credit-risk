"""Export independently reviewable interactive diagnostic figures from saved results."""

import hashlib
import json
from pathlib import Path

import plotly.graph_objects as go

from creditrisk.reporting import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = json.loads((ROOT / "artifacts/dashboard.json").read_text())
    folder = ROOT / "reports/figures"
    folder.mkdir(exist_ok=True)
    roc = go.Figure()
    for m in d["models"]:
        roc.add_scatter(x=m["roc"]["fpr"], y=m["roc"]["tpr"], name=m["name"], mode="lines")
    roc.update_layout(
        title="Final-test ROC audit (predeclared alternatives)",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
    )
    roc.write_html(folder / "roc.html", include_plotlyjs="cdn")
    cal = go.Figure()
    for m in d["calibration"]["comparisons"]:
        rows = [r for r in m["reliability"] if r["n"]]
        cal.add_scatter(
            x=[r["predicted"] for r in rows],
            y=[r["observed"] for r in rows],
            name=m["method"],
            mode="lines+markers",
        )
    cal.add_scatter(x=[0, 1], y=[0, 1], name="Ideal", mode="lines", line={"dash": "dash"})
    cal.update_layout(
        title="Final-test calibration audit",
        xaxis_title="Mean estimated next-month probability",
        yaxis_title="Observed next-month default fraction",
    )
    cal.write_html(folder / "calibration.html", include_plotlyjs="cdn")
    files = [
        ROOT / "artifacts/dashboard.json",
        ROOT / "reports/results.json",
        ROOT / "reports/frozen_decision.json",
        ROOT / "reports/data_audit.json",
        ROOT / "reports/source_manifest.json",
        ROOT / "configs/protocol.json",
    ]
    write_json(
        ROOT / "artifacts/manifest.json",
        {
            "schema_version": 1,
            "sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
            },
        },
    )


if __name__ == "__main__":
    main()
