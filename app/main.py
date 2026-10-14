"""Aggregate-only, eight-section model validation application.

No fitting, source download, personal-input collection or eligibility decision
is performed here. Numerical results are loaded from the offline pipeline.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.styles import CSS

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "dashboard.json"
SECTIONS = (
    "Credit Risk Overview",
    "Borrower Data Explorer",
    "Default Probability Models",
    "Model Validation",
    "Probability Calibration",
    "Explainable AI",
    "Credit Loss & Stress Scenarios",
    "Model Governance & About",
)
NAVY, TEAL, BLUE, GOLD = "#132C45", "#137F89", "#7395B5", "#B9894D"
COLORS = [TEAL, BLUE, NAVY, GOLD, "#8D81A1"]
GITHUB = "https://github.com/Priynka-Roy/creditrisk-ai-explainable-credit-risk"
SOURCE = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
MODEL_LABELS = {
    "baseline": "Logistic regression",
    "l1": "L1 regularized logistic",
    "l2": "L2 regularized logistic",
    "boosting": "Histogram gradient boosting",
}
SHORT_MODEL_LABELS = {**MODEL_LABELS, "boosting": "Hist. boosting"}


@st.cache_data(show_spinner=False)
def load_dashboard(path: str, modified_ns: int, manifest_modified_ns: int = 0) -> dict:
    """Cache by modification time so regenerated artifacts cannot stay stale."""
    del modified_ns, manifest_modified_ns
    path_obj = Path(path)
    raw = path_obj.read_bytes()
    manifest_path = path_obj.with_name("manifest.json")
    if not manifest_path.exists():
        raise ValueError("Artifact integrity manifest is missing; regenerate artifacts.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("sha256", {}).get("artifacts/dashboard.json")
    if expected_hash != hashlib.sha256(raw).hexdigest():
        raise ValueError(
            "The artifact does not match its integrity manifest; regenerate artifacts."
        )
    data = json.loads(raw)
    required = {
        "metadata",
        "dataset",
        "eda",
        "models",
        "selected_model",
        "calibration",
        "selected_metrics",
        "selected_curves",
        "selected_thresholds",
    }
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError("The dashboard schema is incomplete; regenerate deployment artifacts.")
    if data.get("schema_version") != 1:
        raise ValueError("The dashboard schema version is unsupported; regenerate artifacts.")
    if not data["models"] or not any(
        model.get("name") == data["selected_model"] for model in data["models"]
    ):
        raise ValueError("The selected model is absent from the saved model results.")
    return data


def number(value, digits: int = 3) -> str:
    """Explicit undefined display rather than a fabricated zero."""
    if value is None:
        return "Undefined"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:,.{digits}f}" if np.isfinite(numeric) else "Undefined"


def percent(value, digits: int = 1) -> str:
    if value is None:
        return "Undefined"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "Undefined"
    return f"{numeric:.{digits}%}" if np.isfinite(numeric) else "Undefined"


def context(text: str) -> None:
    st.markdown(f'<div class="context-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def header(index: int, title: str, intro: str) -> None:
    st.markdown(f'<div class="eyebrow">CreditRisk AI / {index:02d}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="hero-intro">{html.escape(intro)}</p>', unsafe_allow_html=True)


def chart(fig: go.Figure, title: str, *, height: int = 365) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=NAVY), x=0.03),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color=NAVY),
        colorway=COLORS,
        margin=dict(l=55, r=25, t=70, b=55),
        height=height,
        legend=dict(orientation="h", y=-0.22, x=0),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(gridcolor="#EBF0F5", zeroline=False)
    fig.update_yaxes(gridcolor="#EBF0F5", zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def table(rows: list[dict] | pd.DataFrame, *, formats: dict | None = None) -> None:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty:
        st.info("No observations are available for this selection.")
        return
    if formats:
        st.dataframe(
            frame.style.format(formats, na_rep="Undefined"), hide_index=True, width="stretch"
        )
    else:
        st.dataframe(frame, hide_index=True, width="stretch")


def selected(data: dict) -> dict:
    return {
        "name": f"{data['selected_model']} · selected probability estimates",
        "metrics": data["selected_metrics"],
        **data["selected_curves"],
        "thresholds": data["selected_thresholds"],
        "risk_distribution": data["calibration"]["reliability"],
    }


def model_views(data: dict) -> list[dict]:
    return [selected(data), *data["models"]]


def model_label(value: str) -> str:
    if " · " in value:
        base, suffix = value.split(" · ", 1)
        return f"{MODEL_LABELS.get(base, base)} · {suffix}"
    return f"{MODEL_LABELS.get(value, value)} · uncalibrated"


def selection_text(data: dict) -> str:
    decision = data.get("selection", {})
    validation = decision.get("validation_metrics", {})
    model_name = data["selected_model"]
    score = validation.get(model_name, {}).get("log_loss")
    return f"{MODEL_LABELS.get(model_name, model_name)} was selected using validation log loss ({number(score, 4)}) with a predefined preference for simpler models within the documented tolerance. Calibration method: {data['calibration']['selected_method']}. The selection record was saved before final-test audit."


def case(data: dict, index: int) -> None:
    cases = data.get("cases", [])
    if index >= len(cases):
        return
    item = cases[index]
    title = item["title"].removeprefix(f"{chr(65 + index)} · ")
    with st.expander(f"Guided case {chr(65 + index)} · {title}", expanded=True):
        st.markdown(f"**Question**  \n{item['question']}")
        st.caption("Dataset: UCI historical credit-card repayment records, Taiwan, 2005.")
        st.markdown(f"**Method**  \n{item['method']}")
        st.markdown(f"**Computed result**  \n{item['result']}")
        st.markdown(f"**Banking interpretation**  \n{item['interpretation']}")
        st.caption(f"Limitation: {item['limitation']}")


def metric_row(items: list[tuple[str, str, str]]) -> None:
    for column, (label, value, help_text) in zip(st.columns(len(items)), items, strict=True):
        column.metric(label, value, help=help_text)


def overview(data: dict) -> None:
    header(
        1,
        "Credit risk, explained.",
        "A transparent view of historical repayment behavior, estimated next-month payment default probabilities, and the evidence behind a model.",
    )
    dataset = data["dataset"]
    model = selected(data)
    metric_row(
        [
            (
                "Historical customers",
                number(dataset.get("n", dataset.get("rows")), 0),
                "One released record per customer; no customer-level observation timestamp.",
            ),
            (
                "Observed default prevalence",
                percent(dataset.get("prevalence")),
                "Positive class: next-month payment default, as recorded by the original source.",
            ),
            (
                "Selected model",
                SHORT_MODEL_LABELS.get(data["selected_model"], data["selected_model"]),
                "Selected using development evidence before opening the final test set.",
            ),
            (
                "Final-test ROC-AUC",
                number(model["metrics"].get("roc_auc")),
                "Discrimination: ability to rank default events above non-events. It does not establish calibration.",
            ),
        ]
    )
    context(
        "Educational historical analysis. These estimates describe next-month payment default; they are not regulatory one-year PDs, lending decisions or a validated European banking model."
    )
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("The analytical question")
        st.write(
            "Can repayment histories support useful probability estimates, and how well do those probabilities withstand independent validation? This project treats discrimination, probability calibration and interpretation as separate questions."
        )
        st.subheader("What the data can tell us")
        st.write(
            "UCI’s publicly released, anonymized credit-card dataset contains credit limits, demographic categories and six months of repayment, bill and payment history. Amounts are reported in New Taiwan dollars (NT$)."
        )
        st.caption(
            "The observations relate to Taiwan in 2005. Geographic, economic and population transfer is unvalidated."
        )
    with right:
        counts = [
            dataset.get("n", dataset.get("rows", 0)) - dataset.get("events", 0),
            dataset.get("events", 0),
        ]
        fig = go.Figure(
            go.Bar(
                x=["No recorded default", "Next-month default"],
                y=counts,
                marker_color=[BLUE, TEAL],
                text=[f"{n:,}" for n in counts],
                textposition="outside",
                hovertemplate="%{x}<br>Customers: %{y:,}<extra></extra>",
            )
        )
        fig.update_yaxes(title="Customers")
        chart(fig, "Observed outcome balance", height=315)
    st.subheader("A probability-first validation workflow")
    st.write(
        "Fit on training customers → compare on validation customers → fit probability calibration on a separate calibration partition → evaluate the frozen choice once on untouched test customers."
    )
    with st.expander("Selected model and key evidence", expanded=True):
        st.write(selection_text(data))
        metric_row(
            [
                (
                    "Final-test Brier score",
                    number(model["metrics"].get("brier")),
                    "Mean squared error of probabilities; lower is better.",
                ),
                (
                    "Final-test log loss",
                    number(model["metrics"].get("log_loss")),
                    "Penalizes confident incorrect estimates; lower is better.",
                ),
                (
                    "Calibration method",
                    data["calibration"].get("selected_method", "Recorded in artifact"),
                    "Method chosen on development data, not on final-test outcomes.",
                ),
            ]
        )
    case(data, 0)


def explorer(data: dict) -> None:
    header(
        2,
        "Historical borrower patterns",
        "Explore aggregate credit limits, repayment behavior and category-level event rates. Associations describe the released sample; they do not establish causes.",
    )
    eda = data["eda"]
    categories = eda.get("categories", [])
    if isinstance(categories, dict):
        categories = [{"dimension": dim, **row} for dim, rows in categories.items() for row in rows]
    st.subheader("Default prevalence by category")
    if categories:
        frame = pd.DataFrame(categories)
        dimensions = list(frame["dimension"].drop_duplicates())
        dimension = st.selectbox("Descriptive grouping", dimensions)
        groups = frame.loc[frame.dimension == dimension, "group"].astype(str).tolist()
        chosen = st.multiselect(
            "Displayed groups",
            groups,
            default=groups,
            help="Filters the saved aggregate groups only. No individual customer records are shown or collected.",
        )
        filtered = frame[(frame.dimension == dimension) & frame.group.astype(str).isin(chosen)]
        if filtered.empty:
            st.info("Choose at least one group to display its aggregate observations.")
        else:
            rate = "observed" if "observed" in filtered else "rate"
            upper = filtered.ci_upper - filtered[rate] if "ci_upper" in filtered else None
            lower = filtered[rate] - filtered.ci_lower if "ci_lower" in filtered else None
            fig = go.Figure(
                go.Bar(
                    x=filtered.group.astype(str),
                    y=filtered[rate],
                    marker_color=TEAL,
                    error_y=dict(type="data", array=upper, arrayminus=lower),
                    customdata=filtered[["n", "events"]],
                    hovertemplate="%{x}<br>Observed default: %{y:.1%}<br>Customers: %{customdata[0]:,}<br>Events: %{customdata[1]:,}<extra></extra>",
                )
            )
            fig.update_yaxes(
                title="Observed next-month default rate", tickformat=".0%", rangemode="tozero"
            )
            chart(fig, f"Observed event rates · {dimension}")
            table(filtered, formats={rate: "{:.1%}", "ci_lower": "{:.1%}", "ci_upper": "{:.1%}"})
            st.caption(
                "Intervals are 95% Wilson intervals. Small groups have wider uncertainty; zero events do not establish zero underlying risk."
            )
    distributions = eda.get("distributions", {})
    if isinstance(distributions, list):
        distributions = {
            feature: [row for row in distributions if row["feature"] == feature]
            for feature in dict.fromkeys(row["feature"] for row in distributions)
        }
    st.subheader("Credit and demographic distributions")
    if distributions:
        feature = st.selectbox("Distribution", list(distributions))
        rows = distributions[feature]
        frame = pd.DataFrame(rows)
        fig = go.Figure()
        outcomes = frame["outcome"].drop_duplicates() if "outcome" in frame else [None]
        for i, outcome in enumerate(outcomes):
            sample = frame[frame.outcome == outcome] if outcome is not None else frame
            mid = (sample.lower + sample.upper) / 2
            label = (
                {0: "No recorded default", 1: "Next-month default"}.get(outcome, str(outcome))
                if outcome is not None
                else "All customers"
            )
            fig.add_trace(
                go.Bar(
                    x=mid,
                    y=sample.n,
                    name=label,
                    marker_color=COLORS[i % len(COLORS)],
                    opacity=0.8,
                    customdata=sample[["lower", "upper"]],
                    hovertemplate="Band: %{customdata[0]:,.0f}–%{customdata[1]:,.0f}<br>Customers: %{y:,}<extra>%{fullData.name}</extra>",
                )
            )
        fig.update_layout(barmode="group")
        fig.update_xaxes(title="Age in years" if feature == "AGE" else "Credit limit (NT$)")
        fig.update_yaxes(title="Customers")
        chart(fig, f"Historical distribution · {feature}")
        st.caption(
            "Counts reflect unequal outcome prevalence; these are descriptive distributions, not adjusted comparisons."
        )
    months = eda.get("monthly", [])
    if months:
        frame = pd.DataFrame(months)
        fig = go.Figure()
        for field, label in [
            ("bill_mean", "Mean bill statement"),
            ("payment_mean", "Mean payment"),
        ]:
            if field in frame:
                fig.add_trace(
                    go.Scatter(
                        x=frame.month,
                        y=frame[field],
                        mode="lines+markers",
                        name=label,
                        hovertemplate="%{x}<br>%{y:,.0f} NT$<extra>%{fullData.name}</extra>",
                    )
                )
        fig.update_yaxes(title="NT$ per customer")
        chart(fig, "Six-month billing and payment history")
        st.caption(
            "Source months refer to the historical repayment fields, not a longitudinal customer-level validation split."
        )
    correlation = eda.get("correlations", {})
    if correlation:
        with st.expander("Correlation structure"):
            names = correlation.get("variables", [])
            fig = go.Figure(
                go.Heatmap(
                    z=correlation.get("matrix", []),
                    x=names,
                    y=names,
                    colorscale="RdBu",
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    hovertemplate="%{x} / %{y}<br>Correlation: %{z:.2f}<extra></extra>",
                )
            )
            chart(fig, "Pairwise correlations · predictors and outcome", height=490)
            st.caption(
                "Correlation measures association; it does not identify causal effects or remove confounding."
            )
    dictionary(data)


def comparison_rows(data: dict) -> list[dict]:
    return [
        {
            "Model": model_label(item["name"]),
            "ROC-AUC": item["metrics"].get("roc_auc"),
            "PR-AUC (AP)": item["metrics"].get("pr_auc"),
            "KS": item["metrics"].get("ks"),
            "Brier": item["metrics"].get("brier"),
            "Log loss": item["metrics"].get("log_loss"),
        }
        for item in model_views(data)
    ]


def models(data: dict) -> None:
    header(
        3,
        "Models, compared transparently",
        "Interpretable logistic models and a nonlinear competitor are assessed using distinct evidence for ranking, probability accuracy and practical complexity.",
    )
    context(
        "The final-test table reports frozen candidates for comparison. It did not determine the selected model or calibration method; those decisions used development data."
    )
    st.subheader("Final-test model comparison")
    table(
        comparison_rows(data),
        formats={
            field: "{:.3f}" for field in ["ROC-AUC", "PR-AUC (AP)", "KS", "Brier", "Log loss"]
        },
    )
    st.caption(
        "PR-AUC uses average precision. Lower Brier score and log loss indicate better probability accuracy; higher ROC-AUC, average precision and KS indicate stronger separation."
    )
    st.subheader("Selected model")
    st.success(MODEL_LABELS.get(data["selected_model"], data["selected_model"]))
    st.write(selection_text(data))
    with st.expander("Development evidence used for selection"):
        table(
            [
                {"Model": MODEL_LABELS.get(key, key), **value}
                for key, value in data.get("selection", {}).get("validation_metrics", {}).items()
            ]
        )
        st.caption(
            "These metrics refer to the separate validation partition. The final-test table above is reporting only."
        )
    views = model_views(data)
    model_name = st.selectbox(
        "Saved model distribution", [model["name"] for model in views], format_func=model_label
    )
    model = next(item for item in views if item["name"] == model_name)
    rows = model.get("risk_distribution", [])
    if rows:
        frame = pd.DataFrame(rows)
        fig = go.Figure(
            go.Bar(
                x=(frame.lower + frame.upper) / 2,
                y=frame.n,
                width=(frame.upper - frame.lower) * 0.9,
                marker_color=TEAL,
                customdata=frame[["lower", "upper"]],
                hovertemplate="Estimated probability: %{customdata[0]:.0%}–%{customdata[1]:.0%}<br>Test customers: %{y:,}<extra></extra>",
            )
        )
        fig.update_xaxes(
            title="Estimated next-month payment default probability", tickformat=".0%", range=[0, 1]
        )
        fig.update_yaxes(title="Customers")
        chart(fig, "Final-test probability distribution")
    with st.expander("Model families and terminology"):
        st.markdown(
            "**Logistic regression** estimates log odds using encoded predictors. **L1 regularization** can reduce coefficients to zero; **L2 regularization** shrinks coefficients. **Histogram gradient boosting** models nonlinear predictive associations through ensembles of trees."
        )
        st.write(
            "A probability of 0.20 means a model estimate of 20% next-month payment default under its historical learned relationships. It is neither a guaranteed individual outcome nor a lending eligibility recommendation."
        )
    case(data, 1)


def validation(data: dict) -> None:
    header(
        4,
        "Evidence on untouched customers",
        "Inspect final-test discrimination, threshold sensitivity and statistical uncertainty. Positive class: recorded next-month payment default.",
    )
    views = model_views(data)
    model_name = st.selectbox(
        "Validation model", [model["name"] for model in views], format_func=model_label
    )
    model = next(item for item in views if item["name"] == model_name)
    metrics = model["metrics"]
    metric_row(
        [
            (label, number(metrics.get(key)), help_text)
            for label, key, help_text in [
                ("ROC-AUC", "roc_auc", "Ranking across all thresholds; higher is better."),
                (
                    "Average precision",
                    "pr_auc",
                    "Precision-recall summary; baseline is test event prevalence.",
                ),
                (
                    "KS separation",
                    "ks",
                    "Maximum difference between true and false positive rates.",
                ),
                ("Brier score", "brier", "Mean squared probability error; lower is better."),
            ]
        ]
    )
    left, right = st.columns(2)
    with left:
        curve = model.get("roc", {})
        fig = go.Figure(
            go.Scatter(
                x=curve.get("fpr", []),
                y=curve.get("tpr", []),
                mode="lines",
                name=model_name,
                line=dict(color=TEAL, width=3),
                hovertemplate="False positive rate: %{x:.1%}<br>True positive rate: %{y:.1%}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance ranking",
                line=dict(color=BLUE, dash="dash"),
            )
        )
        fig.update_xaxes(title="False positive rate", range=[0, 1], tickformat=".0%")
        fig.update_yaxes(title="True positive rate", range=[0, 1], tickformat=".0%")
        chart(fig, "ROC curve")
    with right:
        curve = model.get("pr", {})
        fig = go.Figure(
            go.Scatter(
                x=curve.get("recall", []),
                y=curve.get("precision", []),
                mode="lines",
                name=model_name,
                line=dict(color=TEAL, width=3),
                hovertemplate="Recall: %{x:.1%}<br>Precision: %{y:.1%}<extra></extra>",
            )
        )
        prevalence = metrics.get("observed_rate", data["dataset"].get("prevalence"))
        if prevalence is not None:
            fig.add_hline(
                y=prevalence, line_dash="dash", line_color=BLUE, annotation_text="Event prevalence"
            )
        fig.update_xaxes(title="Recall", range=[0, 1], tickformat=".0%")
        fig.update_yaxes(title="Precision", range=[0, 1], tickformat=".0%")
        chart(fig, "Precision–recall curve")
    st.subheader("Threshold behavior")
    thresholds = model.get("thresholds", [])
    if thresholds:
        available = [float(row["threshold"]) for row in thresholds]
        threshold = st.select_slider(
            "Illustrative analytical threshold",
            options=available,
            value=min(available, key=lambda value: abs(value - 0.5)),
            format_func=lambda value: f"{value:.0%}",
            help="Labels probabilities for diagnostic purposes only. No approval or rejection recommendation is produced.",
        )
        row = thresholds[available.index(threshold)]
        metric_row(
            [
                (
                    "Precision",
                    percent(row.get("precision")),
                    "True defaults / all threshold-positive customers.",
                ),
                ("Recall", percent(row.get("recall")), "True defaults / all observed defaults."),
                (
                    "Specificity",
                    percent(row.get("specificity")),
                    "True non-defaults / all observed non-defaults.",
                ),
            ]
        )
        left, right = st.columns([0.8, 1.2])
        with left:
            matrix = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
            fig = go.Figure(
                go.Heatmap(
                    z=matrix,
                    x=["Threshold-negative", "Threshold-positive"],
                    y=["Observed no default", "Observed default"],
                    text=matrix,
                    texttemplate="%{text:,}",
                    colorscale=[[0, "#EDF4F6"], [1, TEAL]],
                    showscale=False,
                    hovertemplate="%{y}<br>%{x}<br>Customers: %{z:,}<extra></extra>",
                )
            )
            chart(fig, "Confusion matrix", height=305)
        with right:
            frame = pd.DataFrame(thresholds)
            fig = go.Figure()
            for key, label in [
                ("recall", "Recall / true positive rate"),
                ("precision", "Precision"),
                ("specificity", "Specificity"),
            ]:
                fig.add_trace(go.Scatter(x=frame.threshold, y=frame[key], mode="lines", name=label))
            fig.add_vline(x=threshold, line_dash="dot", line_color=NAVY)
            fig.update_xaxes(title="Analytical probability threshold", tickformat=".0%")
            fig.update_yaxes(title="Rate", tickformat=".0%", range=[0, 1])
            chart(fig, "Threshold sensitivity", height=305)
        st.caption(
            "Thresholds shown here were not optimized on final-test outcomes. Rates with an absent denominator are displayed as undefined."
        )
    st.subheader("Bootstrap uncertainty")
    st.caption(
        f"Intervals below always refer to the selected {MODEL_LABELS.get(data['selected_model'], data['selected_model'])} probability estimates with {data['calibration']['selected_method']} calibration, regardless of the curve selector above."
    )
    uncertainty = data.get("uncertainty", [])
    if isinstance(uncertainty, dict):
        uncertainty = uncertainty.get("intervals", [])
    table(uncertainty)
    context(
        "Customer-level stratified bootstrap intervals measure sampling variability within this historical holdout. They do not capture future economic change, geographic transfer or all dependence between customers."
    )


def calibration(data: dict) -> None:
    header(
        5,
        "When probabilities match outcomes",
        "A model can rank customers well and still estimate event probabilities poorly. Calibration compares expected default frequency with observed frequency.",
    )
    cal = data["calibration"]
    metric_row(
        [
            (
                "Selected calibration",
                cal.get("selected_method", "Not available"),
                "Chosen using development outcomes before final-test evaluation.",
            ),
            (
                "Calibration intercept",
                number(cal.get("intercept")),
                "Ideal value 0. Estimated on final test for diagnostic evaluation, not probability fitting.",
            ),
            (
                "Calibration slope",
                number(cal.get("slope")),
                "Ideal value 1. A diagnostic association between observed outcome and predicted log odds.",
            ),
        ]
    )
    rows = cal.get("reliability", [])
    if rows:
        frame = pd.DataFrame(rows)
        displayed = frame[frame.n > 0]
        fig = go.Figure(
            go.Scatter(
                x=displayed.predicted,
                y=displayed.observed,
                mode="lines+markers",
                name="Saved selected probability estimates",
                marker=dict(size=9, color=TEAL),
                error_y=dict(
                    type="data",
                    array=displayed.ci_upper - displayed.observed,
                    arrayminus=displayed.observed - displayed.ci_lower,
                ),
                customdata=displayed[["n", "events"]],
                hovertemplate="Mean estimated probability: %{x:.1%}<br>Observed event rate: %{y:.1%}<br>Customers: %{customdata[0]:,}<br>Defaults: %{customdata[1]:,}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect agreement",
                line=dict(color=BLUE, dash="dash"),
            )
        )
        fig.update_xaxes(
            title="Mean estimated next-month default probability", range=[0, 1], tickformat=".0%"
        )
        fig.update_yaxes(
            title="Observed next-month default frequency", range=[0, 1], tickformat=".0%"
        )
        chart(fig, "Final-test reliability diagram", height=420)
        st.caption(
            "Fixed probability bands; empty bands are omitted from the curve. Vertical intervals are 95% Wilson intervals for observed rates, not individual prediction intervals."
        )
        with st.expander("Probability bands, events and uncertainty", expanded=True):
            table(
                frame,
                formats={
                    field: "{:.1%}"
                    for field in ["lower", "upper", "predicted", "observed", "ci_lower", "ci_upper"]
                },
            )
    comparisons = cal.get("comparisons", [])
    if comparisons:
        st.subheader("Calibration methods compared")
        output = []
        for item in comparisons:
            selection_metrics, test_metrics = (
                item.get("selection_metrics", {}),
                item.get("test_metrics", {}),
            )
            output.append(
                {
                    "Method": item["method"],
                    "Development Brier": selection_metrics.get("brier"),
                    "Development log loss": selection_metrics.get("log_loss"),
                    "Final-test Brier": test_metrics.get("brier"),
                    "Final-test log loss": test_metrics.get("log_loss"),
                }
            )
        table(
            output,
            formats={
                field: "{:.4f}"
                for field in [
                    "Development Brier",
                    "Development log loss",
                    "Final-test Brier",
                    "Final-test log loss",
                ]
            },
        )
        st.caption(
            "Final-test comparisons are descriptive reports after selection. A method’s worse result is retained rather than hidden."
        )
    st.subheader("Observed versus expected defaults")
    metrics = data["selected_metrics"]
    test_n = data["dataset"].get("splits", {}).get("test", {}).get("n")
    observed = metrics.get("observed_rate")
    predicted = metrics.get("mean_probability")
    metric_row(
        [
            (
                "Observed test defaults",
                number(
                    observed * test_n if observed is not None and test_n is not None else None, 0
                ),
                "Actual defaults among final-test customers; source positive class is next-month payment default.",
            ),
            (
                "Expected test defaults",
                number(
                    predicted * test_n if predicted is not None and test_n is not None else None, 1
                ),
                "Sum of selected-model estimated probabilities. This is an expected count, not a realized count.",
            ),
            (
                "Observed / expected",
                number(metrics.get("observed_expected_ratio")),
                "A ratio near 1 describes aggregate agreement but can conceal disagreement within probability bands or groups.",
            ),
        ]
    )
    st.subheader("How to read calibration")
    st.write(
        "Points above the diagonal indicate underestimation in that band; points below indicate overestimation. Sparse bands need caution. Calibration intercept and slope summarize different aspects of agreement and can themselves be uncertain."
    )
    case(data, 2)


def explanations(data: dict) -> None:
    header(
        6,
        "Predictive associations, made visible",
        "Examine interpretable logistic coefficients and held-out permutation importance. These explain model behavior; they are not causal effects or credit eligibility advice.",
    )
    exp = data.get("explainability", {})
    coefficients = exp.get("coefficients", [])
    st.subheader("Logistic model coefficients")
    st.caption(
        f"Comparator: {MODEL_LABELS.get(exp.get('coefficient_model', 'l2'), 'L2 logistic regression')}. These coefficients explain that comparator, not the selected gradient-boosting model."
    )
    if coefficients:
        frame = pd.DataFrame(coefficients)
        limit = st.slider(
            "Coefficients displayed",
            min_value=5,
            max_value=min(30, max(5, len(frame))),
            value=min(15, max(5, len(frame))),
            help="Largest absolute fitted coefficients. Scaling and reference categories affect comparisons.",
        )
        shown = (
            frame.assign(magnitude=frame.coefficient.abs())
            .nlargest(limit, "magnitude")
            .sort_values("coefficient")
        )
        shown["display_name"] = shown.feature.map(feature_label)
        fig = go.Figure(
            go.Bar(
                x=shown.coefficient,
                y=shown.display_name,
                orientation="h",
                marker_color=[TEAL if value >= 0 else BLUE for value in shown.coefficient],
                customdata=shown.odds_ratio,
                hovertemplate="%{y}<br>Log-odds coefficient: %{x:.3f}<br>Odds ratio: %{customdata:.3f}<extra></extra>",
            )
        )
        fig.update_xaxes(title="Fitted log-odds coefficient")
        chart(
            fig, "Direction and size of fitted associations", height=max(380, 23 * len(shown) + 140)
        )
        st.caption(
            "Numeric coefficients are per standardized model input unit. Category coefficients compare to a declared reference. Odds ratios multiply odds; they are not percentage-point changes in probability."
        )
        with st.expander("Coefficient and odds-ratio table"):
            table(frame, formats={"coefficient": "{:.3f}", "odds_ratio": "{:.3f}"})
    permutation = exp.get("permutation", [])
    st.subheader("Nonlinear model permutation importance")
    if permutation:
        frame = pd.DataFrame(permutation).sort_values("importance").tail(15)
        fig = go.Figure(
            go.Bar(
                x=frame.importance,
                y=frame.feature.map(feature_label),
                orientation="h",
                marker_color=TEAL,
                error_x=dict(type="data", array=frame["std"]),
                hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
            )
        )
        fig.update_xaxes(title="Increase in validation log loss after permutation")
        chart(fig, "Held-out permutation importance", height=max(380, len(frame) * 23 + 140))
        st.caption(
            "Bars show repeated-permutation variability. Correlated variables can share or mask importance; negative values are retained. No causal interpretation follows."
        )
    st.subheader("Plain-language definitions")
    table(
        [
            {
                "Concept": "Credit limit",
                "Meaning": "Historical account credit limit in NT$; not a measured exposure at default.",
            },
            {
                "Concept": "Repayment delay",
                "Meaning": "Source-coded repayment status, including special non-delay codes; positive values indicate months of delay.",
            },
            {
                "Concept": "Bill statement",
                "Meaning": "Historical statement amount in NT$; zero and negative values can occur.",
            },
            {
                "Concept": "Historical payment",
                "Meaning": "Recorded payment amount in NT$ for a source month.",
            },
            {
                "Concept": "Estimated probability",
                "Meaning": "Model estimate of next-month payment default in this historical population.",
            },
        ]
    )
    context(
        "No real-person prediction form is provided. The dashboard neither accepts personal customer data nor produces lending approval or rejection decisions."
    )


def stressed_mean(histogram: list[dict], multiplier: float) -> float:
    """Odds-scale shift on an anonymous weighted probability histogram."""
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("The odds multiplier must be positive and finite.")
    if not histogram:
        raise ValueError("The saved probability histogram is unavailable.")
    probabilities = np.asarray([row["p"] for row in histogram], dtype=float)
    counts = np.asarray([row["n"] for row in histogram], dtype=float)
    if (
        np.any(~np.isfinite(probabilities))
        or np.any((probabilities < 0) | (probabilities > 1))
        or np.any(~np.isfinite(counts))
        or np.any(counts < 0)
        or counts.sum() <= 0
    ):
        raise ValueError("The aggregate probability histogram is invalid.")
    shifted = multiplier * probabilities / (1 - probabilities + multiplier * probabilities)
    return float(np.average(shifted, weights=counts))


def loss(data: dict) -> None:
    header(
        7,
        "Credit loss under stated assumptions",
        "Translate estimated next-month payment default probabilities into illustrative expected losses, while keeping hypothetical exposure, LGD and stress assumptions explicit.",
    )
    context(
        "Expected loss = estimated next-month payment default probability × hypothetical LGD × hypothetical EAD. This is not an IFRS 9 or regulatory impairment calculation, and no realized bank loss is measured."
    )
    loss_data = data.get("loss", {})
    histogram = loss_data.get("probability_histogram", [])
    st.subheader("Construct an illustrative portfolio")
    a, b, c, d = st.columns(4)
    lgd = (
        a.slider(
            "Hypothetical LGD",
            0,
            100,
            45,
            5,
            format="%d%%",
            help="Assumed fraction of exposure lost conditional on default. Not estimated from this dataset.",
        )
        / 100
    )
    ead = b.number_input(
        "Hypothetical EAD per account (NT$)",
        min_value=0.0,
        max_value=10000000.0,
        value=100000.0,
        step=10000.0,
        help="Uniform assumed exposure at default; credit limits are not substituted for measured EAD.",
    )
    size = c.number_input(
        "Illustrative accounts",
        min_value=1,
        max_value=1000000,
        value=1000,
        step=100,
        help="Hypothetical portfolio size. This does not multiply or reveal actual customer records.",
    )
    multiplier = d.slider(
        "Default-odds stress multiplier",
        0.25,
        5.0,
        1.0,
        0.25,
        help="Assumed shift to default odds, not a fitted macroeconomic shock or a direct probability multiplier.",
    )
    if histogram:
        mean = stressed_mean(histogram, multiplier)
        baseline = stressed_mean(histogram, 1)
        expected = mean * lgd * ead * size
        metric_row(
            [
                (
                    "Mean stressed probability",
                    percent(mean),
                    "Weighted mean after an odds-scale transformation of the saved rounded probability distribution.",
                ),
                (
                    "Modeled expected defaults",
                    number(mean * size, 1),
                    "Expected count, not a realized count or a prediction interval.",
                ),
                (
                    "Illustrative expected loss",
                    f"NT$ {expected:,.0f}",
                    "Uniform EAD × hypothetical LGD × assumed accounts × mean stressed next-month default probability.",
                ),
            ]
        )
        st.caption(
            f"Baseline mean: {percent(baseline)}. Transformation: p′ = k·p / (1 − p + k·p). k = {multiplier:.2f}; probabilities remain within [0, 1] and ranking is preserved. Distribution rounding introduces a small approximation."
        )
        scenario_rows = []
        for label, k in [
            ("Baseline", 1.0),
            ("Mild adverse", 1.5),
            ("Severe adverse", 2.5),
            ("Current scenario", multiplier),
        ]:
            pd_mean = stressed_mean(histogram, k)
            scenario_rows.append(
                {
                    "Scenario": label,
                    "Odds multiplier": k,
                    "Mean probability": pd_mean,
                    "LGD": lgd,
                    "EAD (NT$)": ead,
                    "Accounts": size,
                    "Expected loss (NT$)": pd_mean * lgd * ead * size,
                }
            )
        left, right = st.columns([1, 1.1])
        with left:
            frame = pd.DataFrame(scenario_rows)
            fig = go.Figure(
                go.Bar(
                    x=frame.Scenario,
                    y=frame["Expected loss (NT$)"],
                    marker_color=[BLUE, TEAL, NAVY, GOLD],
                    hovertemplate="%{x}<br>Illustrative expected loss: NT$ %{y:,.0f}<extra></extra>",
                )
            )
            fig.update_yaxes(title="Hypothetical next-month expected loss (NT$)")
            chart(fig, "Scenario comparison")
        with right:
            multipliers = np.linspace(0.25, 5, 39)
            fig = go.Figure()
            for assumption in [0.25, 0.45, 0.65]:
                fig.add_trace(
                    go.Scatter(
                        x=multipliers,
                        y=[
                            stressed_mean(histogram, k) * assumption * ead * size
                            for k in multipliers
                        ],
                        mode="lines",
                        name=f"LGD {assumption:.0%}",
                    )
                )
            fig.update_xaxes(title="Hypothetical default-odds multiplier")
            fig.update_yaxes(title="Illustrative expected loss (NT$)")
            chart(fig, "Sensitivity to probability stress and LGD")
        table(
            scenario_rows,
            formats={
                "Mean probability": "{:.1%}",
                "LGD": "{:.0%}",
                "EAD (NT$)": "{:,.0f}",
                "Expected loss (NT$)": "{:,.0f}",
            },
        )
    else:
        st.warning(
            "The saved probability histogram is unavailable. Regenerate artifacts to enable scenario calculations."
        )
    st.caption(
        "Assumed scenarios are not estimated recession effects. Uniform exposures and losses deliberately simplify the example; correlation, recoveries, maturity and realized LGD/EAD are not modeled."
    )
    if loss_data.get("scenarios"):
        with st.expander("Reproducible baseline scenarios saved by the pipeline"):
            table(loss_data["scenarios"])
    case(data, 3)


def dictionary(data: dict) -> None:
    with st.expander("Data dictionary and source coding"):
        entries = data.get("eda", {}).get("dictionary", [])
        if entries:
            table(entries)
        else:
            st.markdown(
                "**LIMIT_BAL:** credit limit (NT$). **SEX, EDUCATION, MARRIAGE:** source-coded demographic categories. **AGE:** years. **PAY_0, PAY_2–PAY_6:** repayment status, September–April 2005. **BILL_AMT1–6:** monthly bill statement amounts. **PAY_AMT1–6:** monthly prior payments. **default.payment.next.month:** binary next-month payment default target, 1 = event."
            )
        st.caption(
            "Unusual and undocumented codes are preserved or explicitly encoded by the offline pipeline, not silently merged or treated as missing. See the data card for exact code definitions and cleaning decisions."
        )


def feature_label(name: str) -> str:
    """Readable financial vocabulary while preserving source coding."""
    labels = {
        "LIMIT_BAL": "Credit limit",
        "AGE": "Age",
        "SEX": "Sex category",
        "EDUCATION": "Education category",
        "MARRIAGE": "Marital status",
        "MAX_DELAY": "Maximum repayment delay",
        "DELAY_MONTHS": "Months with repayment delay",
        "MEAN_STATUS": "Mean repayment status",
    }
    for prefix, label in [("BILL_AMT", "Bill statement"), ("PAY_AMT", "Historical payment")]:
        if name.startswith(prefix):
            month = name.removeprefix(prefix)
            return f"{label} · month {month}"
    if name.startswith("PAY_"):
        parts = name.split("_")
        return f"Repayment status {parts[1]}" + (
            f" · category {parts[2]}" if len(parts) > 2 else ""
        )
    for code, label in labels.items():
        if name == code:
            return label
        if name.startswith(f"{code}_"):
            return f"{label} · {name.removeprefix(code + '_')}"
    return name.replace("_", " ").capitalize()


def governance(data: dict) -> None:
    header(
        8,
        "Governance, evidence and boundaries",
        "Understand the experimental protocol, subgroup diagnostics, reproducibility and the limits of this historical research demonstration.",
    )
    protocol, subgroups, stability, about = st.tabs(
        ["Protocol & provenance", "Subgroup diagnostics", "Stability", "Responsible use & about"]
    )
    with protocol:
        st.subheader("Frozen experimental design")
        metadata = data["metadata"]
        protocol_data = metadata.get("protocol", {})
        st.write(
            "Customer-level stratified training, validation, calibration and final-test partitions; fixed random seed 42. The source does not support a genuine out-of-time split."
        )
        if data["dataset"].get("splits"):
            splits = data["dataset"]["splits"]
            table(
                splits
                if isinstance(splits, list)
                else [
                    {
                        "Partition": key,
                        **(value if isinstance(value, dict) else {"Customers": value}),
                    }
                    for key, value in splits.items()
                ]
            )
        st.write(
            protocol_data.get(
                "selection",
                "Selection balances probability accuracy, discrimination and simplicity using development evidence.",
            )
        )
        st.write(
            protocol_data.get(
                "calibration",
                "Calibration is fitted on a separate partition; final-test outcomes are not used for tuning.",
            )
        )
        st.subheader("Official source")
        st.markdown(
            f"[UCI Default of Credit Card Clients]({SOURCE}) · [Dataset DOI](https://doi.org/10.24432/C55S3H)"
        )
        st.write(
            "Attribution: I-Cheng Yeh, Default of Credit Card Clients (UCI Machine Learning Repository). The released data are historical and anonymized; the application displays only aggregate summaries."
        )
        st.caption(
            f"License: {metadata.get('license', 'See the data card and official UCI license record.')}"
        )
        with st.expander("Artifact provenance"):
            st.json(metadata, expanded=False)
        dictionary(data)
    with subgroups:
        st.subheader("Historical demographic group diagnostics")
        fairness = data.get("fairness", [])
        if isinstance(fairness, dict):
            fairness = fairness.get("groups", [])
        if fairness:
            frame = pd.DataFrame(fairness)
            dimension = st.selectbox("Diagnostic grouping", list(frame.dimension.drop_duplicates()))
            filtered = frame[frame.dimension == dimension].copy()
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=filtered.group.astype(str),
                    y=filtered.observed,
                    name="Observed event rate",
                    marker_color=TEAL,
                    error_y=dict(
                        type="data",
                        array=filtered.ci_upper - filtered.observed,
                        arrayminus=filtered.observed - filtered.ci_lower,
                    ),
                    customdata=filtered[["n", "events"]],
                    hovertemplate="%{x}<br>Observed: %{y:.1%}<br>Customers: %{customdata[0]:,}<br>Events: %{customdata[1]:,}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=filtered.group.astype(str),
                    y=filtered.predicted,
                    name="Mean estimated probability",
                    marker_color=BLUE,
                    hovertemplate="%{x}<br>Mean estimate: %{y:.1%}<extra></extra>",
                )
            )
            fig.update_layout(barmode="group")
            fig.update_yaxes(title="Next-month default rate / probability", tickformat=".0%")
            chart(fig, "Observed and estimated event frequency by group")
            table(
                filtered.drop(columns=["thresholds"], errors="ignore"),
                formats={
                    field: "{:.1%}"
                    for field in [
                        "observed",
                        "predicted",
                        "ci_lower",
                        "ci_upper",
                        "fpr",
                        "fnr",
                        "calibration_gap",
                    ]
                },
            )
            st.caption(
                "FPR denominator: observed non-defaults. FNR denominator: observed defaults. See threshold field for the fixed diagnostic cutoff. Undefined denominators are not replaced with zeros."
            )
            if "small_group" in filtered and filtered.small_group.any():
                st.warning(
                    "Some groups have fewer than 100 test observations or 20 events. Interpret these estimates cautiously; small or absent denominators do not support strong conclusions."
                )
            if "thresholds" in filtered:
                with st.expander("Group threshold sensitivity"):
                    threshold = st.select_slider(
                        "Subgroup diagnostic threshold",
                        options=[0.1, 0.2, 0.3, 0.5, 0.7],
                        value=0.5,
                        format_func=lambda value: f"{value:.0%}",
                    )
                    group_rows = []
                    for row in filtered.to_dict("records"):
                        chosen = next(
                            item for item in row["thresholds"] if item["threshold"] == threshold
                        )
                        group_rows.append(
                            {
                                "Group": row["group"],
                                "Customers": row["n"],
                                "Events": row["events"],
                                **chosen,
                            }
                        )
                    table(
                        group_rows,
                        formats={
                            field: "{:.1%}"
                            for field in [
                                "threshold",
                                "precision",
                                "recall",
                                "specificity",
                                "fpr",
                                "fnr",
                            ]
                        },
                    )
        sensitivity = data.get("fairness_thresholds", [])
        if sensitivity:
            with st.expander("Group threshold sensitivity"):
                table(sensitivity)
        context(
            "Unexplained group differences do not by themselves establish unlawful discrimination, and statistical parity does not establish fairness compliance. These historical demographic categories require context, sufficient denominators and uncertainty."
        )
    with stability:
        st.subheader("Random-holdout distribution comparisons")
        stable = data.get("stability", {})
        rows = stable.get("psi", [])
        if rows:
            frame = pd.DataFrame(rows).sort_values("psi", ascending=True)
            fig = go.Figure(
                go.Bar(
                    x=frame.psi,
                    y=frame.feature,
                    orientation="h",
                    marker_color=TEAL,
                    hovertemplate="%{y}<br>PSI: %{x:.4f}<extra></extra>",
                )
            )
            fig.update_xaxes(title="Population Stability Index")
            chart(
                fig, "Training versus random test partition", height=max(360, len(frame) * 24 + 140)
            )
            table(rows)
        st.write(
            stable.get(
                "method",
                "PSI uses training-defined bins and explicit empty-bin smoothing. Cutoff conventions are heuristics, not universal stability guarantees.",
            )
        )
        if stable.get("bootstrap_coefficients"):
            with st.expander("Resampling coefficient stability"):
                table(stable["bootstrap_coefficients"])
        context(
            "A random training-versus-test comparison is not temporal drift monitoring. The source has no trustworthy observation timestamp for year-to-year validation, and this analysis does not establish future or European-population stability."
        )
    with about:
        st.subheader("Intended purpose")
        st.write(
            "An educational portfolio project demonstrating statistical credit modeling, probability calibration, model validation and reproducible software engineering. Developed by Priynka Roy; no bank is a collaborator or endorser."
        )
        st.subheader("Unsupported uses")
        st.markdown(
            "- Real customer credit eligibility or underwriting decisions.\n- Regulatory IRB one-year PD or capital calculations.\n- IFRS 9 impairment measurement.\n- Claims of EBA, EU AI Act, GDPR or Finnish regulatory compliance.\n- Causal conclusions from predictive associations."
        )
        st.subheader("Reproduce and inspect")
        st.markdown(
            f"[GitHub repository]({GITHUB}) · [Model card]({GITHUB}/blob/main/docs/MODEL_CARD.md) · [Validation report]({GITHUB}/blob/main/docs/MODEL_VALIDATION.md) · [Data card]({GITHUB}/blob/main/docs/DATA_CARD.md)"
        )
        st.write(
            "Models and aggregate artifacts are computed offline. Opening the application neither downloads raw data nor retrains models. There is no personal-input collection or individual lending eligibility service."
        )
        st.caption(
            "The technical review is internal project QA, not independent validation by a bank or regulatory certification."
        )


def main() -> None:
    st.set_page_config(
        page_title="CreditRisk AI · Model Validation",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">CreditRisk <span>AI</span></div>', unsafe_allow_html=True
        )
        st.caption("EXPLAINABLE CREDIT RISK\n& MODEL VALIDATION")
        st.markdown('<div class="sidebar-label">Analysis workspace</div>', unsafe_allow_html=True)
        section = st.radio("Analysis section", SECTIONS, label_visibility="collapsed")
        st.divider()
        st.caption(
            "HISTORICAL DATA · TAIWAN · 2005\n\nNext-month payment default\n\nAggregate analysis only"
        )
        st.markdown(f"[Project source ↗]({GITHUB})")
    if not ARTIFACT.exists():
        st.error("Deployment artifacts are not available yet.")
        st.write(
            "Generate the saved analysis artifacts using the reproduction command in the README, then restart this application. No model is trained or source data downloaded by the dashboard."
        )
        st.stop()
    try:
        manifest_path = ARTIFACT.with_name("manifest.json")
        manifest_mtime = manifest_path.stat().st_mtime_ns if manifest_path.exists() else 0
        data = load_dashboard(str(ARTIFACT), ARTIFACT.stat().st_mtime_ns, manifest_mtime)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error("The saved dashboard artifact could not be loaded safely.")
        st.caption(str(exc))
        st.write(
            "Regenerate artifacts with the documented offline pipeline. A partial or stale analysis is not displayed as a valid result."
        )
        st.stop()
    renderers = (
        overview,
        explorer,
        models,
        validation,
        calibration,
        explanations,
        loss,
        governance,
    )
    renderers[SECTIONS.index(section)](data)
    st.markdown(
        '<div class="app-footer">CreditRisk AI · Priynka Roy · Historical research demonstration<br>Estimated next-month payment default probabilities. No real customer eligibility decisions. No personal data collection.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
