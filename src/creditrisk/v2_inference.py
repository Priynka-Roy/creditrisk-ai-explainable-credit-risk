"""Retrospective paired inference and customer-bootstrap calibration uncertainty."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from .calibration import calibration_parameters
from .evaluation import metrics_simple, reliability, validate_predictions, wilson


def _replicates(repetitions):
    if not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("At least two integer bootstrap replicates required")


def _scores(y, p):
    if len(np.unique(y)) == 2:
        return metrics_simple(y, p)
    return {
        "roc_auc": None,
        "pr_auc": None,
        "ks": None,
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }


def paired_bootstrap(
    y,
    predictions,
    model_a="boosting",
    comparators=("l2", "l1", "baseline"),
    repetitions=1000,
    seed=142,
):
    y = np.asarray(y)
    for p in predictions.values():
        validate_predictions(y, p)
    _replicates(repetitions)
    if model_a not in predictions or any(b not in predictions for b in comparators):
        raise ValueError("Requested comparison model is absent")
    classes = [np.flatnonzero(y == c) for c in (0, 1)]
    rng = np.random.default_rng(seed)
    points = {k: _scores(y, np.asarray(p)) for k, p in predictions.items()}
    samples = {(b, m): [] for b in comparators for m in points[model_a]}
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(c, len(c), replace=True) for c in classes])
        scores = {k: _scores(y[indices], np.asarray(p)[indices]) for k, p in predictions.items()}
        for (b, m), values in samples.items():
            if scores[model_a][m] is not None:
                values.append(scores[model_a][m] - scores[b][m])
    rows = []
    for (b, m), values in samples.items():
        lower, upper = np.quantile(values, [0.025, 0.975]) if values else (None, None)
        rows.append(
            {
                "model_a": model_a,
                "model_b": b,
                "metric": m,
                "difference": float(points[model_a][m] - points[b][m])
                if points[model_a][m] is not None
                else None,
                "lower": float(lower) if lower is not None else None,
                "upper": float(upper) if upper is not None else None,
                "includes_zero": bool(lower <= 0 <= upper) if lower is not None else None,
                "favorable_direction": "higher" if m in {"roc_auc", "pr_auc", "ks"} else "lower",
                "replicates": repetitions,
                "seed": seed,
                "n": len(y),
                "sign_convention": "Model A minus Model B",
                "resampling": "Customer-level stratified paired percentile bootstrap; conditional on fitted models and observed class counts",
            }
        )
    return rows


def calibration_uncertainty(y, p, repetitions=1000, seed=242, min_events=20):
    y, p = validate_predictions(y, p)
    _replicates(repetitions)
    if not isinstance(min_events, int) or min_events < 1:
        raise ValueError("Minimum events must be a positive integer")
    events = int(y.sum())
    nonevents = len(y) - events
    ratio = float(y.sum() / p.sum()) if p.sum() > 0 else None
    point = {**calibration_parameters(y, p), "observed_expected_ratio": ratio}
    sparse = events < min_events or nonevents < min_events or np.std(p) < 1e-12
    if sparse:
        lo, hi = wilson(events, len(y))
        sparse_rows = [
            {"metric": k, "estimate": v, "lower": None, "upper": None, "valid_replicates": 0}
            for k, v in point.items()
        ]
        if p.mean() > 0:
            sparse_rows[-1].update(
                lower=float(lo / p.mean()),
                upper=float(hi / p.mean()),
                method="Wilson event-rate interval divided by fixed mean prediction; conditional approximation",
            )
        return {
            "metrics": sparse_rows,
            "reason": "Insufficient events/non-events or prediction variation for calibration regression uncertainty",
            "n": len(y),
            "events": events,
            "replicates": repetitions,
            "bands": reliability(y, p),
        }
    rng = np.random.default_rng(seed)
    values = {k: [] for k in point}
    for _ in range(repetitions):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        pp = p[idx]
        params = calibration_parameters(yy, pp)
        params["observed_expected_ratio"] = float(yy.sum() / pp.sum()) if pp.sum() > 0 else None
        for key, value in params.items():
            if value is not None and np.isfinite(value):
                values[key].append(value)
    rows = []
    for key, v in values.items():
        bounds = (
            np.quantile(v, [0.025, 0.975]) if len(v) >= max(20, repetitions * 0.8) else [None, None]
        )
        rows.append(
            {
                "metric": key,
                "estimate": point[key],
                "lower": float(bounds[0]) if bounds[0] is not None else None,
                "upper": float(bounds[1]) if bounds[1] is not None else None,
                "valid_replicates": len(v),
            }
        )
    return {
        "metrics": rows,
        "n": len(y),
        "events": events,
        "replicates": repetitions,
        "seed": seed,
        "resampling": "Ordinary customer bootstrap; prevalence varies; percentile 95% intervals conditional on fitted predictions",
        "bands": reliability(y, p),
    }


def subgroup_uncertainty(y, p, groups, repetitions=1000, seed=342, min_events=20):
    y, p = validate_predictions(y, p)
    _replicates(repetitions)
    rng = np.random.default_rng(seed)
    rows = []
    for dimension, labels in groups.items():
        labels = np.asarray(labels).astype(str)
        if labels.ndim != 1 or len(labels) != len(y):
            raise ValueError("Group labels must align with prediction rows")
        for group in sorted(np.unique(labels)):
            mask = labels == group
            yy = y[mask]
            pp = p[mask]
            n = len(yy)
            events = int(yy.sum())
            pred = pp >= 0.5
            neg = yy == 0
            pos = yy == 1
            fp = int((pred & neg).sum())
            fn = int((~pred & pos).sum())
            fpr = fp / int(neg.sum()) if neg.sum() else None
            fnr = fn / events if events else None
            fpr_ci = wilson(fp, int(neg.sum()))
            fnr_ci = wilson(fn, events)
            gap_samples = []
            for _ in range(repetitions):
                idx = rng.integers(0, n, n)
                gap_samples.append(float((pp[idx] - yy[idx]).mean()))
            small_group = n < 100 or events < min_events or n - events < min_events
            gap_lo, gap_hi = (
                np.quantile(gap_samples, [0.025, 0.975]) if not small_group else (None, None)
            )
            observed_ci = wilson(events, n)
            rows.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "n": n,
                    "events": events,
                    "observed": float(yy.mean()),
                    "predicted": float(pp.mean()),
                    "gap": float((pp - yy).mean()),
                    "gap_lower": float(gap_lo) if gap_lo is not None else None,
                    "gap_upper": float(gap_hi) if gap_hi is not None else None,
                    "fpr": fpr,
                    "fnr": fnr,
                    "fpr_lower": fpr_ci[0],
                    "fpr_upper": fpr_ci[1],
                    "fnr_lower": fnr_ci[0],
                    "fnr_upper": fnr_ci[1],
                    "small_group": small_group,
                    "observed_lower": observed_ci[0],
                    "observed_upper": observed_ci[1],
                    "gap_interval_method": "Ordinary customer bootstrap; unavailable for sparse groups",
                    "calibration_metrics": calibration_uncertainty(
                        yy,
                        pp,
                        repetitions=min(repetitions, 300),
                        seed=seed + len(rows),
                        min_events=min_events,
                    ),
                }
            )
    return rows
