"""Version 2 extensions inside the preserved eight-section dashboard.

Default views use computed aggregates. Optional portfolio simulation runs only
after an explicit button press, uses synthetic accounts and bounded workloads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.main import (
    BLUE,
    COLORS,
    MODEL_LABELS,
    NAVY,
    TEAL,
    chart,
    context,
    feature_label,
    metric_row,
    table,
)

ROOT = Path(__file__).resolve().parents[1]
V2_ARTIFACT = ROOT / "artifacts" / "v2" / "dashboard.json"
METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "Average precision",
    "ks": "KS",
    "brier": "Brier score",
    "log_loss": "Log loss",
    "intercept": "Calibration intercept",
    "slope": "Calibration slope",
    "observed_expected_ratio": "Observed / expected defaults",
}


@st.cache_data(show_spinner=False)
def load_v2(path: str, modified_ns: int, manifest_modified_ns: int) -> dict:
    """Validate version and content hash before displaying computed V2 results."""
    del modified_ns, manifest_modified_ns
    artifact = Path(path)
    raw = artifact.read_bytes()
    manifest = json.loads(artifact.with_name("manifest.json").read_text(encoding="utf-8"))
    expected = manifest.get("sha256", {}).get("artifacts/v2/dashboard.json")
    if expected != hashlib.sha256(raw).hexdigest():
        raise ValueError("Version 2 artifact does not match its integrity manifest.")
    result = json.loads(raw)
    if result.get("schema_version") != 2:
        raise ValueError("Unsupported Version 2 artifact schema.")
    required = {
        "paired_comparisons",
        "calibration_uncertainty",
        "subgroup_uncertainty",
        "stability",
        "simulations",
    }
    if not required.issubset(result):
        raise ValueError("Version 2 artifact is incomplete.")
    return result


def attach_v2(data: dict) -> dict:
    """A missing V2 supplement preserves the honest, frozen V1 dashboard."""
    if not V2_ARTIFACT.exists():
        st.info(
            "Version 1 results are available. Version 2 supplementary artifacts have not been generated; no Version 2 statistics are displayed."
        )
        return {**data, "_v2": None}
    try:
        manifest = V2_ARTIFACT.with_name("manifest.json")
        v2 = load_v2(str(V2_ARTIFACT), V2_ARTIFACT.stat().st_mtime_ns, manifest.stat().st_mtime_ns)
        expected_v1 = (
            v2.get("metadata", {}).get("v1_file_hashes", {}).get("artifacts/dashboard.json")
        )
        actual_v1 = hashlib.sha256((ROOT / "artifacts" / "dashboard.json").read_bytes()).hexdigest()
        if expected_v1 != actual_v1:
            raise ValueError(
                "Version 2 supplement does not match the preserved Version 1 dashboard."
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.warning(
            "Version 2 supplement is unavailable or inconsistent. Showing preserved Version 1 results only."
        )
        st.caption(str(exc))
        return {**data, "_v2": None}
    st.caption(
        "VERSION 2 EXTENSIONS · Frozen V1 models and choices · Retrospective analysis of the previously inspected test set · Development-only resampling · Hypothetical simulations"
    )
    return {**data, "_v2": v2}


def case_v2(v2: dict, letter: str) -> None:
    cases = v2.get("cases", [])
    item = next(
        (
            item
            for item in cases
            if item.get("case", item.get("id", "")) == letter
            or item.get("title", "").startswith(f"{letter} ·")
        ),
        None,
    )
    if item is None:
        position = {"E": 0, "F": 1, "G": 2, "H": 3}[letter]
        item = cases[position] if position < len(cases) else None
    if item is None:
        return
    title = item.get("title", "Computed analytical case").removeprefix(f"{letter} · ")
    with st.expander(f"Guided case {letter} · {title}", expanded=True):
        for key, label in [
            ("question", "Question"),
            ("method", "Method"),
            ("result", "Computed result"),
            ("interpretation", "Interpretation"),
        ]:
            if item.get(key):
                st.markdown(f"**{label}**  \n{item[key]}")
        if item.get("limitation"):
            st.caption(f"Limitation: {item['limitation']}")


def interval_plot(
    frame: pd.DataFrame,
    *,
    estimate: str,
    label: str,
    title: str,
    axis: str,
    reference: float = 0,
    height: int = 350,
) -> None:
    """Draw CI segments separately so biased intervals cannot create negative error bars."""
    fig = go.Figure()
    for row in frame.to_dict("records"):
        if row.get("lower") is not None and row.get("upper") is not None:
            fig.add_trace(
                go.Scatter(
                    x=[row["lower"], row["upper"]],
                    y=[row[label], row[label]],
                    mode="lines",
                    line=dict(color=BLUE, width=4),
                    showlegend=False,
                    hovertemplate="95% interval: %{x:.4f}<extra></extra>",
                )
            )
    fig.add_trace(
        go.Scatter(
            x=frame[estimate],
            y=frame[label],
            mode="markers",
            marker=dict(color=TEAL, size=10),
            showlegend=False,
            hovertemplate="%{y}<br>Estimate: %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=reference, line_dash="dash", line_color=NAVY)
    fig.update_xaxes(title=axis)
    chart(fig, title, height=height)


def validation_panel(v2: dict | None) -> None:
    if not v2:
        return
    st.divider()
    st.subheader("V2 · Paired model differences")
    context(
        "Retrospective Version 2 analysis: these frozen models are compared on the original, already inspected 6,000-customer test partition. No new model or calibration choice is made from these results."
    )
    rows = v2.get("paired_comparisons", [])
    if rows:
        frame = pd.DataFrame(rows)
        metric = st.selectbox(
            "Paired comparison metric",
            list(frame.metric.drop_duplicates()),
            format_func=lambda key: METRIC_LABELS.get(key, key),
            key="v2_paired_metric",
        )
        sample = frame[frame.metric == metric].copy()
        sample["Pair"] = (
            sample.model_a.map(lambda key: MODEL_LABELS.get(key, key))
            + " − "
            + sample.model_b.map(lambda key: MODEL_LABELS.get(key, key))
        )
        favorable = "positive" if metric in {"roc_auc", "pr_auc", "ks"} else "negative"
        st.caption(
            f"Sign convention: Model A − Model B. For {METRIC_LABELS.get(metric, metric)}, a {favorable} difference favors Model A. Intervals concern fitted predictions, not training or hyperparameter-selection uncertainty."
        )
        interval_plot(
            sample,
            estimate="difference",
            label="Pair",
            title=f"Paired 95% intervals · {METRIC_LABELS.get(metric, metric)}",
            axis="Metric difference (Model A − Model B)",
            height=320,
        )
        table(
            sample[["Pair", "metric", "difference", "lower", "upper", "includes_zero"]],
            formats={key: "{:.5f}" for key in ["difference", "lower", "upper"]},
        )
        with st.expander("All predeclared comparisons and methods"):
            table(rows)
            st.write(
                v2.get("metadata", {}).get(
                    "paired_method",
                    "Stratified customer-level paired bootstrap, identical resampled indices for both models within each replicate; 95% percentile intervals. Conditional on the frozen fitted models and the original split.",
                )
            )
    st.caption(
        "A confidence interval excluding zero does not establish business usefulness, regulatory suitability or practical banking superiority. Secondary metrics and model pairs are exploratory comparisons, with no multiplicity correction."
    )
    case_v2(v2, "E")


def calibration_panel(v2: dict | None) -> None:
    if not v2:
        return
    st.divider()
    st.subheader("V2 · How uncertain is calibration?")
    context(
        "Retrospective diagnostics on the previously inspected test partition. The frozen Version 1 choice—no additional calibration—is preserved. These intervals are not used to choose a different calibrator."
    )
    uncertainty = v2.get("calibration_uncertainty", {})
    rows = uncertainty.get("metrics", [])
    if rows:
        frame = pd.DataFrame(rows)
        frame["Parameter"] = frame.metric.map(lambda key: METRIC_LABELS.get(key, key))
        table(frame, formats={key: "{:.4f}" for key in ["estimate", "lower", "upper"]})
        st.caption(
            "Reference values: intercept 0, slope 1 and observed/expected ratio 1. Customer bootstrap uncertainty is approximate under unknown customer dependence; undefined fits are reported explicitly."
        )
    with st.expander("Probability-band calibration uncertainty", expanded=True):
        bands = uncertainty.get("bands", [])
        table(
            bands,
            formats={
                key: "{:.1%}"
                for key in [
                    "lower",
                    "upper",
                    "predicted",
                    "observed",
                    "ci_lower",
                    "ci_upper",
                    "gap",
                    "gap_lower",
                    "gap_upper",
                ]
            },
        )
        st.caption(
            "Counts and defaults define each band's denominator. Sparse or empty bands do not support strong claims; empty estimates stay undefined rather than being displayed as zero."
        )
    st.subheader("Overall agreement can hide local differences")
    groups = v2.get("subgroup_uncertainty", [])
    if groups:
        frame = pd.DataFrame(groups)
        dimension = st.selectbox(
            "V2 calibration grouping",
            list(frame.dimension.drop_duplicates()),
            key="v2_calibration_group",
        )
        sample = frame[frame.dimension == dimension].copy()
        shown = sample.rename(columns={"gap_lower": "lower", "gap_upper": "upper"})
        interval_plot(
            shown,
            estimate="gap",
            label="group",
            title="Mean predicted − observed probability gap",
            axis="Probability gap (estimate − observed)",
            height=max(330, len(sample) * 35 + 150),
        )
        table(
            sample.drop(columns=["calibration_metrics"], errors="ignore"),
            formats={
                key: "{:.1%}"
                for key in [
                    "observed",
                    "predicted",
                    "gap",
                    "gap_lower",
                    "gap_upper",
                    "fpr",
                    "fnr",
                    "fpr_lower",
                    "fpr_upper",
                    "fnr_lower",
                    "fnr_upper",
                ]
            },
        )
        if "small_group" in sample and sample.small_group.any():
            st.warning(
                "Sparse groups are present. Uncertainty can be wide or undefined; source categories and denominators must remain visible."
            )
        st.caption(
            "Positive gaps indicate overestimation in this historical subgroup; negative gaps indicate underestimation. Neither is a causal explanation or a legal fairness conclusion."
        )
    case_v2(v2, "F")


def simulation_table(rows: list[dict]) -> None:
    """Put financial outcomes first; retain assumptions in a separate detail view."""
    if not rows:
        return
    frame = pd.DataFrame(rows)
    labels = {
        "label": "Scenario",
        "analytic_expected_loss": "Expected loss",
        "mean": "MC mean",
        "std": "Loss SD",
        "var_95": "VaR 95%",
        "es_95": "ES 95%",
    }
    primary = frame[[key for key in labels if key in frame]].rename(columns=labels)
    st.caption(
        "All loss amounts are NTD. Expected loss is analytical; MC mean and loss SD summarize simulated outcomes. VaR and ES use the documented 95% tail definitions."
    )
    st.dataframe(
        primary.style.format(
            {key: "{:,.0f}" for key in primary if key != "Scenario"}, na_rep="Undefined"
        ),
        hide_index=True,
        width="stretch",
        column_config={
            key: st.column_config.Column(width="medium" if key == "Scenario" else "small")
            for key in primary
        },
    )
    with st.expander("Simulation assumptions and marginal checks"):
        assumptions = []
        checks = []
        for row in rows:
            assumptions.append(
                {
                    "Scenario": row["label"],
                    "Synthetic accounts": row["portfolio_size"],
                    "Simulations": row.get("n_simulations", row.get("repetitions")),
                    "Mean probability": row.get("mean_pd", row.get("mean_probability")),
                    "Assumed factor correlation": row["rho"],
                    "Odds multiplier": row.get("odds_multiplier"),
                    "Assumed LGD": row.get("lgd"),
                    "Assumed EAD (NTD)": row.get("ead"),
                }
            )
            if row.get("marginal_check"):
                checks.append({"Scenario": row["label"], **row["marginal_check"]})
        table(
            assumptions,
            formats={
                "Mean probability": "{:.1%}",
                "Assumed LGD": "{:.0%}",
                "Assumed EAD (NTD)": "{:,.0f}",
            },
        )
        if checks:
            st.caption(
                "Marginal checks describe Monte Carlo sampling error under the assumed construction; they do not validate real-world dependence."
            )
            table(checks)


def simulation_plot(rows: list[dict], *, title: str) -> None:
    fig = go.Figure()
    for index, row in enumerate(rows):
        histogram = row.get("histogram", [])
        if isinstance(histogram, dict):
            highs = histogram.get("upper", histogram.get("edges", [])[1:])
            counts = histogram.get("counts", [])
        else:
            highs = [item["upper"] for item in histogram]
            counts = [item.get("n", item.get("count", 0)) for item in histogram]
        if not counts:
            continue
        counts = np.asarray(counts, dtype=float)
        fig.add_trace(
            go.Scatter(
                x=np.asarray(highs),
                y=1 - np.cumsum(counts) / counts.sum(),
                mode="lines+markers" if len(counts) == 1 else "lines",
                name=row.get("label", f"Assumed scenario {index + 1}"),
                line=dict(color=COLORS[index % len(COLORS)], width=3),
                hovertemplate="Loss-band upper boundary: NTD %{x:,.0f}<br>Simulations exceeding boundary: %{y:.1%}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_xaxes(title="Simulated hypothetical next-month portfolio loss (NTD)")
    fig.update_yaxes(
        title="Fraction of simulations exceeding loss boundary", tickformat=".0%", range=[0, 1]
    )
    chart(fig, title, height=400)


@st.cache_data(show_spinner=False, max_entries=8)
def run_custom_simulation(
    histogram_json: str,
    size: int,
    repetitions: int,
    rho: float,
    lgd: float,
    ead: float,
    odds_multiplier: float,
) -> list[dict]:
    """Bound deterministic synthetic simulation work before calling the numerical API."""
    if not 10 <= size <= 1000 or not 500 <= repetitions <= 5000 or not 0 <= rho <= 0.6:
        raise ValueError("Simulation exceeds the dashboard workload bounds.")
    if not np.isfinite(odds_multiplier) or not 0.25 <= odds_multiplier <= 5:
        raise ValueError("The dashboard odds multiplier must be finite and between 0.25 and 5.")
    # Imported only for a requested calculation, never for model fitting.
    from creditrisk.simulation import portfolio_from_histogram, simulate_portfolio, summarize_losses

    histogram = json.loads(histogram_json)
    p = portfolio_from_histogram(histogram, size)
    p = odds_multiplier * p / (1 - p + odds_multiplier * p)
    rows = []
    for label, correlation in [("Independent defaults", 0.0), ("Assumed common factor", rho)]:
        simulation = simulate_portfolio(
            p, lgd=lgd, ead=ead, n_simulations=repetitions, rho=correlation, seed=2026
        )
        losses = simulation["losses"]
        summary = summarize_losses(losses)
        if np.ptp(losses) == 0:
            histogram_rows = [
                {"lower": float(losses[0]), "upper": float(losses[0]), "n": len(losses)}
            ]
        else:
            counts, edges = np.histogram(losses, bins=35)
            histogram_rows = [
                {"lower": float(edges[i]), "upper": float(edges[i + 1]), "n": int(count)}
                for i, count in enumerate(counts)
            ]
        rows.append(
            {
                "label": label,
                "rho": correlation,
                "portfolio_size": size,
                "repetitions": repetitions,
                "mean_probability": float(p.mean()),
                "lgd": lgd,
                "ead": ead,
                "odds_multiplier": odds_multiplier,
                "analytic_expected_loss": float(p.sum() * lgd * ead),
                **summary,
                "histogram": histogram_rows,
            }
        )
    return rows


def simulation_panel(data: dict, v2: dict | None) -> None:
    if not v2:
        return
    st.divider()
    st.subheader("V2 · Simulated portfolio loss, with dependence")
    context(
        "Synthetic portfolio simulation, not observed customer losses. Probabilities come from the frozen model's anonymous rounded distribution. EAD, LGD, systematic dependence and stress parameters are hypothetical assumptions; currency is NTD."
    )
    rows = v2.get("simulations", [])
    if rows:
        baseline = rows[0]
        dependent = rows[1] if len(rows) > 1 else baseline
        st.caption(
            f"Saved baseline: {baseline['portfolio_size']:,} synthetic accounts · {baseline['n_simulations']:,} simulated portfolios · assumed LGD {baseline['lgd']:.0%} · EAD NTD {baseline['ead']:,.0f} per account · factor correlation {baseline['rho']:.2f} versus {dependent['rho']:.2f}."
        )
        increased_lgd = [row for row in rows if row.get("lgd", 0) > baseline["lgd"]]
        if increased_lgd:
            st.caption(
                "Additional stress scenarios use higher hypothetical LGD: "
                + "; ".join(f"{row['label']} ({row['lgd']:.0%})" for row in increased_lgd)
                + ". Other changed assumptions are recorded below."
            )
    simulation_table(rows)
    simulation_plot(rows[:2], title="Independent versus assumed dependent defaults")
    st.write(
        "Expected loss describes the probability-weighted mean. Monte Carlo standard deviation describes scenario outcome variability; it is different from uncertainty in fitted model performance. Dependence changes joint outcomes and tails, while intended marginal default probabilities and analytical expected loss remain fixed when the other assumptions are held constant."
    )
    with st.expander("Run a bounded, synthetic scenario comparison"):
        st.caption(
            "Default page views use saved results. The calculation below runs only when requested; it cannot accept or retain real borrower records."
        )
        a, b, c = st.columns(3)
        size = a.number_input(
            "Synthetic portfolio accounts",
            min_value=10,
            max_value=1000,
            value=250,
            step=10,
            key="v2_mc_size",
        )
        repetitions = b.selectbox(
            "Simulated portfolios", [500, 1000, 2500, 5000], index=1, key="v2_mc_repetitions"
        )
        rho = c.slider(
            "Assumed Gaussian factor correlation",
            0.0,
            0.6,
            0.25,
            0.05,
            key="v2_mc_rho",
            help="A latent-variable assumption, not an empirically estimated asset correlation or a Basel capital parameter.",
        )
        a, b, c = st.columns(3)
        lgd = a.slider("Simulation hypothetical LGD (%)", 0, 100, 45, 5, key="v2_mc_lgd") / 100
        ead = b.number_input(
            "Simulation hypothetical EAD per account (NTD)",
            min_value=0.0,
            max_value=1000000.0,
            value=100000.0,
            step=10000.0,
            key="v2_mc_ead",
        )
        odds = c.slider(
            "Simulation default-odds multiplier", 0.25, 5.0, 1.0, 0.25, key="v2_mc_odds"
        )
        parameters = (size, repetitions, rho, lgd, ead, odds)
        if st.button("Run synthetic comparison", key="v2_run_simulation", type="primary"):
            with st.spinner("Simulating bounded synthetic portfolios…"):
                try:
                    results = run_custom_simulation(
                        json.dumps(data["loss"]["probability_histogram"], sort_keys=True),
                        *parameters,
                    )
                    st.session_state["v2_simulation_result"] = {
                        "parameters": parameters,
                        "rows": results,
                    }
                except (ValueError, TypeError, KeyError) as exc:
                    st.error(f"The scenario could not be calculated: {exc}")
        saved = st.session_state.get("v2_simulation_result")
        if saved and tuple(saved["parameters"]) == parameters:
            simulation_table(saved["rows"])
            simulation_plot(saved["rows"], title="Requested synthetic simulation · fixed seed 2026")
            st.caption(
                "Fixed synthetic account sample and random seed. Mean losses can differ between dependence scenarios by Monte Carlo sampling error; this does not mean the theoretical expected loss changed."
            )
        elif saved:
            st.info(
                "Controls have changed. Run the comparison to calculate these assumptions; previous simulation results are not presented as current."
            )
    with st.expander("Tail-risk definitions and uncertainty"):
        st.write(
            "95% Value at Risk is the empirical loss-distribution 95th percentile using the documented quantile convention. Expected Shortfall is the average of the worst 5% probability mass, with fractional mass at the boundary when necessary. It is not a regulatory capital calculation."
        )
        st.caption(
            "Simulation variation, bootstrap model uncertainty and economic scenario uncertainty are different quantities. The systematic-factor parameter and stresses are assumptions, not identified economic relationships."
        )
    case_v2(v2, "G")


def governance_panel(data: dict, v2: dict | None) -> None:
    if not v2:
        return
    st.divider()
    st.subheader("V2 · Illustrative model monitoring")
    context(
        "Educational diagnostic tolerances, not bank policy, regulatory certification or a production approval gate. Original test-period values illustrate the framework and are not used to reselect the model."
    )
    governance = v2.get("governance", {})
    rows = (
        governance.get("monitoring", governance.get("rows", []))
        if isinstance(governance, dict)
        else governance
    )
    criteria = governance.get("criteria") if isinstance(governance, dict) else None
    if criteria:
        from creditrisk.governance import evaluate_governance, monitoring_values

        with st.expander("Explore illustrative monitoring assumptions"):
            a, b = st.columns(2)
            auc_floor = a.slider(
                "Illustrative ROC-AUC minimum", 0.50, 0.95, 0.75, 0.01, key="v2_governance_auc"
            )
            brier_ceiling = b.slider(
                "Illustrative Brier maximum", 0.10, 0.25, 0.15, 0.01, key="v2_governance_brier"
            )
            st.caption(
                "Changing a tolerance updates educational review flags only. It neither refits nor reselects the frozen model."
            )
        custom = json.loads(json.dumps(criteria))
        for rule in custom["rules"]:
            if rule["metric"] == "roc_auc":
                rule["minimum"] = auc_floor
            elif rule["metric"] == "brier":
                rule["maximum"] = brier_ceiling
        rows = evaluate_governance(monitoring_values(data, custom), custom)
    if rows:
        metric_row(
            [
                (
                    "Within assumed tolerance",
                    str(sum(row["status"] == "Within illustrative tolerance" for row in rows)),
                    "No production suitability or regulatory certification follows.",
                ),
                (
                    "Review suggested",
                    str(sum(row["status"] == "Review suggested" for row in rows)),
                    "Flags are consequences of explicitly assumed educational tolerances.",
                ),
                (
                    "Unavailable",
                    str(sum(row["status"] == "Unavailable" for row in rows)),
                    "Insufficient or undefined evidence is not substituted by fabricated values.",
                ),
            ]
        )
    table(
        [
            {
                key: row.get(key)
                for key in ["label", "observed", "reference", "illustrative_tolerance", "status"]
            }
            for row in rows
        ]
    )
    with st.expander("Interpretation and analytical follow-up"):
        for row in rows:
            st.markdown(f"**{row['label']}** · {row['status']}")
            st.write(row.get("interpretation", ""))
            st.caption(row.get("follow_up", ""))
    if isinstance(governance, dict):
        summary = governance.get("model_risk_summary", governance.get("summary"))
        if summary:
            st.write(summary)
    st.subheader("Development-only resampling stability")
    stable = v2.get("stability", {})
    performance = stable.get("validation_performance", [])
    if performance:
        table(performance)
        st.caption(
            "These results come from development-only refitting and fixed validation observations. They are not a fresh independent test, temporal drift evidence or a new model selection exercise."
        )
    coefficients = stable.get("coefficient_bootstrap", [])
    if coefficients:
        with st.expander("Aligned logistic coefficient stability"):
            frame = pd.DataFrame(coefficients)
            if "feature" in frame:
                frame["Predictor"] = frame.feature.map(feature_label)
            table(frame)
            st.caption(
                "Category definitions and reference levels remain aligned across bootstrap fits, including rare source categories. Coefficient variation concerns the logistic comparator rather than the frozen selected boosting model."
            )
    st.caption(
        "Known limits: historical Taiwan data; inspected test set; no verified external or out-of-time outcomes; unmeasured LGD/EAD; assumed dependence; approximate conditional uncertainty; no legal fairness certification."
    )
    case_v2(v2, "H")
