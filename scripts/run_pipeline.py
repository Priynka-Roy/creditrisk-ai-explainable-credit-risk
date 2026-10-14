"""Reproduce frozen protocol, fit, freeze decisions, then audit final holdout."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from creditrisk.calibration import ProbabilityCalibrator
from creditrisk.data import FEATURES, TARGET, load_data, make_splits, split_hash, validate_data
from creditrisk.evaluation import (
    bootstrap_metrics,
    curves,
    metrics,
    reliability,
    threshold_metrics,
    wilson,
)
from creditrisk.explainability import logistic_coefficients, permutation_rows
from creditrisk.fairness import LABELS, group_columns, subgroup_diagnostics
from creditrisk.features import engineer_features
from creditrisk.loss import expected_loss, stress_probabilities
from creditrisk.modeling import fit_candidates, make_model, select_model
from creditrisk.reporting import write_json
from creditrisk.stability import population_stability_index

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol = json.loads((ROOT / "configs/protocol.json").read_text())
    source_manifest = json.loads((ROOT / "reports/source_manifest.json").read_text())
    for name, metadata in source_manifest["files"].items():
        if (
            hashlib.sha256((ROOT / "data/raw" / name).read_bytes()).hexdigest()
            != metadata["sha256"]
        ):
            raise ValueError(f"Raw source integrity mismatch: {name}")
    df = load_data(ROOT / "data/raw/default of credit card clients.xls")
    y = df[TARGET].to_numpy()
    splits = make_splits(y)
    audit = validate_data(df)
    audit["splits"] = {
        k: {"n": len(v), "events": int(y[v].sum()), "sha256_ids": split_hash(df.ID.to_numpy()[v])}
        for k, v in splits.items()
    }
    write_json(ROOT / "reports/data_audit.json", audit)
    # IDs remain local-only for exact audit; never copied into public artifacts.
    np.savez_compressed(
        ROOT / "data/processed/splits.npz", **{k: df.ID.to_numpy()[v] for k, v in splits.items()}
    )
    tr, va, ca, te = [splits[k] for k in ["training", "validation", "calibration", "test"]]
    ablation = {}
    for engineered in [False, True]:
        X = engineer_features(df, engineered)
        model = make_model(X, "l2")
        model.set_params(model__C=1).fit(X.iloc[tr], y[tr])
        ablation["engineered" if engineered else "raw"] = metrics(
            y[va], model.predict_proba(X.iloc[va])[:, 1]
        )
    use_engineered = ablation["raw"]["log_loss"] - ablation["engineered"]["log_loss"] >= 0.002
    X = engineer_features(df, use_engineered)
    print("Fitting candidates", flush=True)
    models, params = fit_candidates(X.iloc[tr], y[tr])
    validation = {k: metrics(y[va], m.predict_proba(X.iloc[va])[:, 1]) for k, m in models.items()}
    selected = select_model(validation)
    calfit, calselect = train_test_split(ca, test_size=0.5, stratify=y[ca], random_state=42)
    pfit = models[selected].predict_proba(X.iloc[calfit])[:, 1]
    pselect = models[selected].predict_proba(X.iloc[calselect])[:, 1]
    calibrators = {
        m: ProbabilityCalibrator(m).fit(pfit, y[calfit]) for m in ["none", "sigmoid", "isotonic"]
    }
    calmetrics = {k: metrics(y[calselect], m.predict(pselect)) for k, m in calibrators.items()}
    best = min(calmetrics, key=lambda k: calmetrics[k]["log_loss"])
    chosen = next(
        k
        for k in ["none", "sigmoid", "isotonic"]
        if calmetrics[k]["log_loss"] <= calmetrics[best]["log_loss"] + 0.002
    )
    decision = {
        "selected_model": selected,
        "selected_calibration": chosen,
        "validation_metrics": validation,
        "hyperparameters": params,
        "feature_ablation": ablation,
        "include_engineered": use_engineered,
        "calibration_selection_metrics": calmetrics,
        "protocol_sha256": hashlib.sha256(
            (ROOT / "configs/protocol.json").read_bytes()
        ).hexdigest(),
        "test_access": "Not accessed before writing this decision file",
    }
    write_json(ROOT / "reports/frozen_decision.json", decision)
    # Final-test audit begins only here. No selection or tuning below this line.
    print("Frozen decisions:", selected, chosen, "Final audit", flush=True)
    pc = models[selected].predict_proba(X.iloc[ca])[:, 1]
    calibrators = {k: ProbabilityCalibrator(k).fit(pc, y[ca]) for k in calibrators}
    rawtest = models[selected].predict_proba(X.iloc[te])[:, 1]
    p = calibrators[chosen].predict(rawtest)
    published = []
    for k, m in models.items():
        prediction = m.predict_proba(X.iloc[te])[:, 1]
        distribution = reliability(y[te], prediction)
        published.append(
            {
                "name": k,
                "metrics": metrics(y[te], prediction),
                "validation_metrics": validation[k],
                **curves(y[te], prediction),
                "thresholds": [
                    threshold_metrics(y[te], prediction, t) for t in protocol["thresholds"]
                ],
                "risk_distribution": distribution,
            }
        )
    calcomparisons = [
        {
            "method": k,
            "selection_metrics": calmetrics[k],
            "test_metrics": metrics(y[te], m.predict(rawtest)),
            "reliability": reliability(y[te], m.predict(rawtest)),
        }
        for k, m in calibrators.items()
    ]
    finalmetrics = metrics(y[te], p)
    uncertainty = bootstrap_metrics(y[te], p, protocol["bootstrap_replicates"])
    fairness = subgroup_diagnostics(df.iloc[te], y[te], p)
    coefficients = logistic_coefficients(models["l2"])
    permutation = permutation_rows(models["boosting"], X.iloc[va], y[va])
    # Training resampling for coefficient stability; never uses test outcomes.
    rng = np.random.default_rng(42)
    bs = []
    names = models["l2"].named_steps["preprocessing"].get_feature_names_out()
    for _ in range(protocol["coefficient_bootstrap_replicates"]):
        idx = rng.choice(tr, len(tr), replace=True)
        # Preserve the training-only category dictionary/reference levels across resamples.
        fitted = clone(models["l2"])
        fitted.set_params(
            preprocessing__category__categories=models["l2"]
            .named_steps["preprocessing"]
            .named_transformers_["category"]
            .categories_
        )
        fitted.fit(X.iloc[idx], y[idx])
        lookup = dict(
            zip(
                fitted.named_steps["preprocessing"].get_feature_names_out(),
                fitted.named_steps["model"].coef_[0],
            )
        )
        bs.append([lookup.get(n, np.nan) for n in names])
    names = models["l2"].named_steps["preprocessing"].get_feature_names_out()
    bs = np.asarray(bs)
    coefficient_stability = [
        {
            "feature": str(n),
            "lower": float(np.nanquantile(bs[:, i], 0.025)),
            "upper": float(np.nanquantile(bs[:, i], 0.975)),
            "positive_fraction": float((bs[np.isfinite(bs[:, i]), i] > 0).mean()),
            "available_replicates": int(np.isfinite(bs[:, i]).sum()),
        }
        for i, n in enumerate(names)
    ]
    psi = [
        {"feature": c, **population_stability_index(df[c].iloc[tr], df[c].iloc[te])}
        for c in ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]
    ]
    trainp = calibrators[chosen].predict(models[selected].predict_proba(X.iloc[tr])[:, 1])
    psi.append({"feature": "Estimated probability", **population_stability_index(trainp, p)})
    # Aggregate anonymous probability histogram, rounded to .001; not customer rows.
    vals, counts = np.unique(np.round(p, 3), return_counts=True)
    histogram = [{"p": float(v), "n": int(n)} for v, n in zip(vals, counts)]
    scenarios = []
    for label, mult, lgd in [
        ("Baseline", 1, 0.45),
        ("Mild adverse", 1.5, 0.45),
        ("Severe adverse", 2, 0.60),
    ]:
        stressed = stress_probabilities(p, mult)
        scenarios.append(
            {
                "label": label,
                "odds_multiplier": mult,
                "lgd": lgd,
                "ead": 100000,
                "portfolio_size": 1000,
                "mean_probability": float(stressed.mean()),
                "expected_loss": expected_loss(stressed, lgd, 100000, 1000),
            }
        )
    eda = make_eda(df)
    cases = [
        {
            "title": "A · Historical repayment patterns",
            "question": "How do repayment-status categories relate to next-month default?",
            "method": "Full-data descriptive rates with Wilson intervals, no causal inference.",
            "result": f"Observed prevalence {y.mean():.2%}; {y.sum():,} events among {len(y):,} released records.",
            "interpretation": "Repayment status is a predictive association to investigate, not a causal lever.",
            "limitation": "Historical Taiwan 2005 sample; undocumented codes retained.",
        },
        {
            "title": "B · Logistic versus boosting",
            "question": "Does nonlinear modeling improve probability quality?",
            "method": "Three-fold training CV; separate validation; untouched final-test descriptive audit.",
            "result": f"Selected {selected} by validation log loss. Final raw L2 log loss {next(r for r in published if r['name'] == 'l2')['metrics']['log_loss']:.4f}; boosting {next(r for r in published if r['name'] == 'boosting')['metrics']['log_loss']:.4f}.",
            "interpretation": "Discrimination and probability quality both matter; final comparisons do not select the model.",
            "limitation": "One frozen random split; no out-of-time evidence.",
        },
        {
            "title": "C · Discrimination and calibration",
            "question": "Does independent calibration improve estimated probabilities?",
            "method": "Calibration half-fit/half-selection; chosen method refit full calibration only.",
            "result": f"Chosen {chosen}. Final log loss {finalmetrics['log_loss']:.4f}, Brier {finalmetrics['brier']:.4f}, calibration slope {finalmetrics['slope']:.3f}.",
            "interpretation": "Calibration changes probability scale; improved AUC alone does not imply accurate probabilities.",
            "limitation": "Selection partition uncertainty and calibration estimates do not certify transportability.",
        },
        {
            "title": "D · Illustrative loss sensitivity",
            "question": "How do assumed shocks affect modeled loss?",
            "method": "Odds stress times hypothetical LGD times hypothetical EAD times 1,000 accounts.",
            "result": f"Baseline modeled loss {scenarios[0]['expected_loss']:,.0f} NTD; severe assumed scenario {scenarios[2]['expected_loss']:,.0f} NTD.",
            "interpretation": "Loss assumptions materially affect a scenario arithmetic result.",
            "limitation": "No observed LGD/EAD, no IFRS9 or regulatory loss calculation.",
        },
    ]
    dashboard = {
        "schema_version": 1,
        "metadata": {
            "project": "CreditRisk AI",
            "outcome": "Estimated next-month payment default probability",
            "source": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
            "doi": "https://doi.org/10.24432/C55S3H",
            "license": "CC BY 4.0",
            "currency": "NTD",
            "seed": 42,
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "protocol": protocol,
            "source_manifest": json.loads((ROOT / "reports/source_manifest.json").read_text()),
        },
        "dataset": audit,
        "eda": eda,
        "models": published,
        "selected_model": selected,
        "selection": decision,
        "selected_metrics": finalmetrics,
        "selected_curves": curves(y[te], p),
        "selected_thresholds": [threshold_metrics(y[te], p, t) for t in protocol["thresholds"]],
        "calibration": {
            "selected_method": chosen,
            "comparisons": calcomparisons,
            "reliability": reliability(y[te], p),
            "intercept": finalmetrics["intercept"],
            "slope": finalmetrics["slope"],
        },
        "uncertainty": uncertainty,
        "explainability": {
            "coefficients": coefficients,
            "coefficient_model": "l2",
            "permutation": permutation,
            "permutation_model": "boosting",
            "permutation_partition": "validation",
        },
        "fairness": fairness,
        "stability": {
            "psi": psi,
            "bootstrap_coefficients": coefficient_stability,
            "replicates": len(bs),
        },
        "loss": {
            "probability_histogram": histogram,
            "histogram_rounding": 0.001,
            "scenarios": scenarios,
        },
        "cases": cases,
    }
    write_json(ROOT / "artifacts/dashboard.json", dashboard)
    write_json(
        ROOT / "reports/results.json",
        {k: v for k, v in dashboard.items() if k not in ["eda", "metadata"]},
    )
    print(
        json.dumps({"selected": selected, "calibration": chosen, "test": finalmetrics}, indent=2),
        flush=True,
    )


def make_eda(df):
    y = df[TARGET].to_numpy()
    categories = []
    groups = group_columns(df)
    groups["September repayment status"] = df.PAY_0.astype(str)
    for dimension, values in groups.items():
        for group in sorted(values.dropna().astype(str).unique()):
            mask = values.astype(str).to_numpy() == group
            n = int(mask.sum())
            events = int(y[mask].sum())
            lo, hi = wilson(events, n)
            categories.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "n": n,
                    "events": events,
                    "rate": events / n,
                    "ci_lower": lo,
                    "ci_upper": hi,
                }
            )
    distributions = []
    for feature in ["LIMIT_BAL", "AGE"]:
        edges = np.linspace(df[feature].min(), df[feature].max(), 21)
        for outcome in [0, 1]:
            counts, _ = np.histogram(df.loc[df[TARGET] == outcome, feature], edges)
            distributions.extend(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "lower": float(edges[i]),
                    "upper": float(edges[i + 1]),
                    "n": int(n),
                }
                for i, n in enumerate(counts)
            )
    monthly = []
    for i, month in enumerate(["September", "August", "July", "June", "May", "April"], 1):
        monthly.append(
            {
                "month": month,
                "bill_mean": float(df[f"BILL_AMT{i}"].mean()),
                "bill_median": float(df[f"BILL_AMT{i}"].median()),
                "payment_mean": float(df[f"PAY_AMT{i}"].mean()),
                "payment_median": float(df[f"PAY_AMT{i}"].median()),
            }
        )
    corrcols = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "BILL_AMT1",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT6",
        TARGET,
    ]
    dictionary = [
        {
            "variable": c,
            "definition": "Credit limit in NTD (not measured EAD)"
            if c == "LIMIT_BAL"
            else "Age in years"
            if c == "AGE"
            else "Historical bill statement in NTD; negative values retained"
            if c.startswith("BILL")
            else "Historical payment amount in NTD"
            if c.startswith("PAY_AMT")
            else "Repayment-status categorical code: -1 paid duly, positive month delay; -2 and 0 not defined by UCI narrative and retained separately"
            if c.startswith("PAY_")
            else json.dumps(LABELS.get(c, {})),
        }
        for c in FEATURES
    ]
    return {
        "categories": categories,
        "distributions": distributions,
        "monthly": monthly,
        "correlations": {"variables": corrcols, "matrix": df[corrcols].corr().to_numpy().tolist()},
        "dictionary": dictionary,
        "summary": df[["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]].describe().round(2).to_dict(),
    }


if __name__ == "__main__":
    main()
