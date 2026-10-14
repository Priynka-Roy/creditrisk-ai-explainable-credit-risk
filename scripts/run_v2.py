"""Frozen-model V2 retrospective inference, development stability and assumed losses."""

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
import sklearn
from sklearn.base import clone

from creditrisk.data import TARGET, load_data, make_splits
from creditrisk.evaluation import metrics
from creditrisk.fairness import group_columns
from creditrisk.features import engineer_features
from creditrisk.governance import evaluate_governance, monitoring_values
from creditrisk.loss import stress_probabilities
from creditrisk.modeling import make_model
from creditrisk.reporting import write_json
from creditrisk.simulation import portfolio_from_histogram, simulate_portfolio, summarize_losses
from creditrisk.stability import population_stability_index
from creditrisk.v2_inference import calibration_uncertainty, paired_bootstrap, subgroup_uncertainty

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    config = json.loads((ROOT / "configs/v2.json").read_text())
    v1 = json.loads((ROOT / "artifacts/dashboard.json").read_text())
    original_manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
    preserved = {
        **original_manifest["sha256"],
        "artifacts/manifest.json": hashlib.sha256(
            (ROOT / "artifacts/manifest.json").read_bytes()
        ).hexdigest(),
    }
    for path, digest in preserved.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"V1 changed: {path}")
        baseline = subprocess.check_output(["git", "show", f"v1.0.0:{path}"], cwd=ROOT)
        if hashlib.sha256(baseline).hexdigest() != digest:
            raise ValueError(f"V1 differs from preserved tag: {path}")
    source = v1["metadata"]["source_manifest"]
    for name, entry in source["files"].items():
        if hashlib.sha256((ROOT / "data/raw" / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Source hash mismatch")
    df = load_data(ROOT / "data/raw/default of credit card clients.xls")
    y = df[TARGET].to_numpy()
    splits = make_splits(y)
    X = engineer_features(df, v1["selection"]["include_engineered"])
    tr = splits["training"]
    va = splits["validation"]
    te = splits["test"]
    models = {}
    predictions = {}
    reproduction = {}
    for kind in ["baseline", "l2", "l1", "boosting"]:
        model = make_model(X, kind)
        params = v1["selection"]["hyperparameters"][kind].get("best_params", {})
        model.set_params(**params).fit(X.iloc[tr], y[tr])
        models[kind] = model
        predictions[kind] = model.predict_proba(X.iloc[te])[:, 1]
        result = metrics(y[te], predictions[kind])
        original = next(m["metrics"] for m in v1["models"] if m["name"] == kind)
        differences = {
            k: abs(result[k] - original[k])
            for k in result
            if result[k] is not None and original[k] is not None
        }
        if max(differences.values()) > 1e-7:
            raise ValueError(f"Frozen {kind} model metrics do not reproduce V1")
        reproduction[kind] = {
            "maximum_absolute_metric_difference": max(differences.values()),
            "parameters": params,
        }
    np.savez_compressed(
        ROOT / "data/processed/v2/private_predictions.npz", outcomes=y[te], **predictions
    )
    p = predictions["boosting"]
    print("V1 reproduced; paired bootstrap", flush=True)
    paired = paired_bootstrap(
        y[te],
        predictions,
        repetitions=config["paired_bootstrap"]["replicates"],
        seed=config["paired_bootstrap"]["seed"],
    )
    print("Calibration and subgroup uncertainty", flush=True)
    calibration = calibration_uncertainty(
        y[te],
        p,
        repetitions=config["calibration_bootstrap"]["replicates"],
        seed=config["calibration_bootstrap"]["seed"],
    )
    groups = group_columns(df.iloc[te])
    credit_edges = np.unique(np.quantile(df.LIMIT_BAL.iloc[tr], [0, 0.25, 0.5, 0.75, 1]))
    credit_edges[0] = -np.inf
    credit_edges[-1] = np.inf
    groups["Credit limit band (training quartiles)"] = pd.cut(
        df.LIMIT_BAL.iloc[te], credit_edges
    ).astype(str)
    groups["September repayment status"] = df.PAY_0.iloc[te].astype(str)
    subgroups = subgroup_uncertainty(
        y[te],
        p,
        groups,
        repetitions=config["subgroup_bootstrap"]["replicates"],
        seed=config["subgroup_bootstrap"]["seed"],
    )
    print("Development-only coefficient and validation bootstrap stability", flush=True)
    original = models["l2"]
    names = original.named_steps["preprocessing"].get_feature_names_out()
    categories = original.named_steps["preprocessing"].named_transformers_["category"].categories_
    rng = np.random.default_rng(config["stability_bootstrap"]["seed"])
    coef = []
    performance = []
    for _ in range(config["stability_bootstrap"]["replicates"]):
        idx = rng.choice(tr, len(tr), replace=True)
        model = clone(original)
        model.set_params(preprocessing__category__categories=categories)
        model.fit(X.iloc[idx], y[idx])
        if not np.array_equal(names, model.named_steps["preprocessing"].get_feature_names_out()):
            raise ValueError("Coefficient category alignment failure")
        coef.append(model.named_steps["model"].coef_[0])
        performance.append(metrics(y[va], model.predict_proba(X.iloc[va])[:, 1]))
    coef = np.asarray(coef)
    coefs = [
        {
            "feature": str(name),
            "mean": float(coef[:, i].mean()),
            "lower": float(np.quantile(coef[:, i], 0.025)),
            "upper": float(np.quantile(coef[:, i], 0.975)),
            "positive_fraction": float((coef[:, i] > 0).mean()),
            "replicates": len(coef),
        }
        for i, name in enumerate(names)
    ]
    variability = [
        {
            "metric": key,
            "mean": float(np.mean([r[key] for r in performance])),
            "lower": float(np.quantile([r[key] for r in performance], 0.025)),
            "upper": float(np.quantile([r[key] for r in performance], 0.975)),
            "replicates": len(performance),
        }
        for key in [
            "roc_auc",
            "pr_auc",
            "brier",
            "log_loss",
            "intercept",
            "slope",
            "observed_expected_ratio",
        ]
    ]
    valp = models["boosting"].predict_proba(X.iloc[va])[:, 1]
    stability = {
        "coefficient_bootstrap": coefs,
        "validation_performance": variability,
        "seed": config["stability_bootstrap"]["seed"],
        "replicates": len(coef),
        "reference_categories": "Fixed V1 training dictionary; original reference levels preserved",
        "prediction_psi": population_stability_index(
            models["boosting"].predict_proba(X.iloc[tr])[:, 1], valp
        ),
        "feature_psi": [
            {"feature": c, **population_stability_index(df[c].iloc[tr], df[c].iloc[va])}
            for c in ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]
        ],
        "interpretation": "Development-only logistic refits evaluated on the same original validation partition; descriptive training sensitivity, not new validation or temporal monitoring",
    }
    print("Hypothetical Monte Carlo portfolios", flush=True)
    simconfig = config["simulation"]
    portfolio = portfolio_from_histogram(
        v1["loss"]["probability_histogram"], simconfig["portfolio_size"]
    )
    simulations = []
    definitions = [
        ("Baseline · independent", 1, 0.45, 100000, 0),
        ("Baseline · common factor", 1, 0.45, 100000, 0.15),
        ("Mild adverse", 1.5, 0.45, 100000, 0.15),
        ("Severe adverse", 2, 0.60, 100000, 0.15),
        ("LGD increase only", 1, 0.60, 100000, 0.15),
        ("EAD increase only", 1, 0.45, 125000, 0.15),
    ]
    for label, odds, lgd, ead, rho in definitions:
        stressed = stress_probabilities(portfolio, odds)
        simulation = simulate_portfolio(
            stressed,
            lgd,
            ead,
            n_simulations=simconfig["portfolios"],
            rho=rho,
            seed=simconfig["seed"],
            batch_size=simconfig["batch_size"],
        )
        summary = summarize_losses(simulation["losses"], simconfig["alpha"])
        counts, edges = np.histogram(simulation["losses"], bins=30)
        simulations.append(
            {
                "label": label,
                "odds_multiplier": odds,
                "lgd": lgd,
                "ead": ead,
                "rho": rho,
                "portfolio_size": len(portfolio),
                "currency": "NTD",
                "seed": simconfig["seed"],
                "mean_pd": float(stressed.mean()),
                "analytic_expected_loss": simulation["analytic_expected_loss"],
                **summary,
                "mean_error": summary["mean"] - simulation["analytic_expected_loss"],
                "mean_error_standard_errors": (
                    summary["mean"] - simulation["analytic_expected_loss"]
                )
                / summary["mean_se"]
                if summary["mean_se"]
                else None,
                "marginal_check": {
                    "max_absolute_error": float(
                        np.max(np.abs(simulation["marginal_probabilities"] - stressed))
                    ),
                    "mean_probability_difference": float(
                        (simulation["marginal_probabilities"] - stressed).mean()
                    ),
                    "conditional_expected_loss_mean": simulation["conditional_expected_loss_mean"],
                },
                "histogram": [
                    {"lower": float(edges[i]), "upper": float(edges[i + 1]), "n": int(n)}
                    for i, n in enumerate(counts)
                ],
            }
        )
    gconfig = json.loads((ROOT / "configs/v2_governance.json").read_text())
    governance = {
        "criteria": gconfig,
        "rows": evaluate_governance(monitoring_values(v1, gconfig), gconfig),
    }
    primary = next(r for r in paired if r["model_b"] == "l2" and r["metric"] == "log_loss")
    calrow = next(r for r in calibration["metrics"] if r["metric"] == "slope")
    ind, dependent = simulations[:2]
    review_count = sum(r["status"] == "Review suggested" for r in governance["rows"])
    cases = [
        {
            "title": "E · Paired model comparison",
            "question": "Are fitted-model differences distinguishable on the inspected V1 test?",
            "method": "1,000 paired stratified customer bootstrap replicates; Model A minus Model B.",
            "result": f"Boosting−L2 log loss difference {primary['difference']:.5f}; percentile 95% interval [{primary['lower']:.5f}, {primary['upper']:.5f}].",
            "interpretation": "Negative log-loss differences favor boosting on this historical sample; intervals do not establish practical banking superiority.",
            "limitation": "Retrospective, conditional on fitted models and observed class counts; no selection uncertainty.",
        },
        {
            "title": "F · Calibration uncertainty",
            "question": "How uncertain is calibration around the V1 point estimate?",
            "method": "Ordinary customer bootstrap retaining event-count variability.",
            "result": f"Calibration slope {calrow['estimate']:.4f}; approximate 95% interval [{calrow['lower']:.4f}, {calrow['upper']:.4f}].",
            "interpretation": "An overall estimate near one does not guarantee local or subgroup calibration.",
            "limitation": "Fixed fitted predictions, unknown customer dependence, historical population only.",
        },
        {
            "title": "G · Dependence and portfolio tails",
            "question": "How does assumed dependence alter a hypothetical loss tail?",
            "method": "5,000 Gaussian latent-factor simulations; identical PD/LGD/EAD, rho 0 versus assumed rho 0.15.",
            "result": f"Independent ES95 {ind['es']:,.0f} NTD; common-factor ES95 {dependent['es']:,.0f} NTD; common theoretical expected loss {ind['analytic_expected_loss']:,.0f} NTD.",
            "interpretation": "Common shocks change loss variability and tail risk while preserving marginal expected losses.",
            "limitation": "Dependence, LGD and EAD are assumed; tail risk is neither observed bank loss nor regulatory capital.",
        },
        {
            "title": "H · Illustrative monitoring",
            "question": "How can assumed monitoring tolerances direct analytical follow-up?",
            "method": "Compare frozen diagnostic values with labeled educational criteria.",
            "result": f"Default criteria suggest review for {review_count} diagnostics; this is a demonstration rather than model approval.",
            "interpretation": "A diagnostic status supports investigation; changing assumed tolerances changes the status.",
            "limitation": "No external bank validation, regulatory certification or model reselection.",
        },
    ]
    elapsed = time.perf_counter() - started
    registry = {
        "experiment_id": config["experiment_id"],
        "version": 2,
        "v1_tag": "v1.0.0",
        "v1_sha": subprocess.check_output(
            ["git", "rev-parse", "v1.0.0^{commit}"], cwd=ROOT, text=True
        ).strip(),
        "source": source,
        "v1_file_hashes": preserved,
        "configuration": config,
        "calibration_method": "V1 none unchanged",
        "model_reproduction": reproduction,
        "split_configuration": v1["dataset"]["splits"],
        "runtime_seconds": elapsed,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "test_usage": "Retrospective V2 diagnostic analysis of previously inspected V1 test; no reselection",
        "governance_tolerances": gconfig,
        "simulation_probability_source": "Deterministic midpoint quantiles of V1 anonymous probability histogram rounded to .001; not an actual customer portfolio",
        "credit_limit_band_edges": [float(v) if np.isfinite(v) else None for v in credit_edges],
    }
    dashboard = {
        "schema_version": 2,
        "metadata": registry,
        "paired_comparisons": paired,
        "calibration_uncertainty": calibration,
        "subgroup_uncertainty": subgroups,
        "stability": stability,
        "simulations": simulations,
        "scenarios": simulations,
        "governance": governance,
        "cases": cases,
    }
    write_json(ROOT / "artifacts/v2/dashboard.json", dashboard)
    for name, data in [
        ("experiment_registry", registry),
        ("paired_comparisons", paired),
        ("calibration_uncertainty", calibration),
        ("subgroups", subgroups),
        ("stability", stability),
        ("simulations", simulations),
    ]:
        write_json(ROOT / f"reports/v2/{name}.json", data)
    pd.DataFrame(paired).to_csv(ROOT / "reports/v2/paired_comparisons.csv", index=False)
    export_figures(dashboard)
    files = (
        [
            ROOT / "artifacts/v2/dashboard.json",
            ROOT / "configs/v2.json",
            ROOT / "configs/v2_governance.json",
        ]
        + sorted((ROOT / "reports/v2").glob("*.json"))
        + [ROOT / "reports/v2/paired_comparisons.csv"]
        + sorted((ROOT / "reports/v2/figures").glob("*.html"))
    )
    write_json(
        ROOT / "artifacts/v2/manifest.json",
        {
            "schema_version": 2,
            "sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
            },
            "v1_preserved_sha256": preserved,
        },
    )
    print(
        json.dumps(
            {
                "runtime_seconds": elapsed,
                "primary_log_loss": primary,
                "calibration": calibration["metrics"],
                "independent": ind["es"],
                "dependent": dependent["es"],
            },
            indent=2,
        ),
        flush=True,
    )


def export_figures(dashboard):
    folder = ROOT / "reports/v2/figures"
    folder.mkdir(exist_ok=True)
    rows = [r for r in dashboard["paired_comparisons"] if r["model_b"] == "l2"]
    comparison = go.Figure(
        go.Scatter(
            x=[r["difference"] for r in rows],
            y=[r["metric"] for r in rows],
            mode="markers",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [r["upper"] - r["difference"] for r in rows],
                "arrayminus": [r["difference"] - r["lower"] for r in rows],
            },
        )
    )
    comparison.add_vline(x=0, line_dash="dash")
    comparison.update_layout(
        title="Retrospective boosting minus L2: paired 95% intervals",
        xaxis_title="Metric difference (units differ by metric)",
    )
    comparison.write_html(folder / "paired-comparison.html", include_plotlyjs="cdn")
    loss = go.Figure()
    for scenario in dashboard["simulations"][:2]:
        rows = scenario["histogram"]
        loss.add_scatter(
            x=[(r["lower"] + r["upper"]) / 2 for r in rows],
            y=[r["n"] / scenario["n_simulations"] for r in rows],
            mode="lines+markers",
            name=scenario["label"],
        )
    loss.update_layout(
        title="Hypothetical next-month loss: independent versus common-factor defaults",
        xaxis_title="Simulated loss (NTD)",
        yaxis_title="Histogram probability mass",
    )
    loss.write_html(folder / "dependence-loss.html", include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
