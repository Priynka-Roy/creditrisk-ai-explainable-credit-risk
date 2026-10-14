"""Explicit binary metrics, reliability and conditional bootstrap intervals."""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .calibration import calibration_parameters


def wilson(events, n):
    if n == 0:
        return None, None
    z = norm.ppf(0.975)
    rate = events / n
    center = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(center - half), float(center + half)


def validate_predictions(y, p):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError(
            "Nonempty equal-length one-dimensional outcomes and probabilities required"
        )
    if not np.isfinite(y).all() or not set(np.unique(y)) <= {0, 1}:
        raise ValueError("Outcomes must be binary 0/1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities outside [0,1]")
    return y, p


def metrics(y, p):
    y, p = validate_predictions(y, p)
    both = len(np.unique(y)) == 2
    fpr, tpr, _ = roc_curve(y, p) if both else (None, None, None)
    return {
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "ks": float(np.max(tpr - fpr)) if both else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_probability": float(p.mean()),
        "observed_rate": float(y.mean()),
        "observed_expected_ratio": float(y.sum() / p.sum()) if p.sum() > 0 else None,
        **calibration_parameters(y, p),
    }


def threshold_metrics(y, p, threshold=0.5):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0,1]")
    y, p = validate_predictions(y, p)
    tn, fp, fn, tp = confusion_matrix(y, np.asarray(p) >= threshold, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "fnr": float(fn / (tp + fn)) if tp + fn else None,
    }


def reliability(y, p, bins=10):
    rows = []
    y, p = validate_predictions(y, p)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("Positive integer bins required")
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        n = int(mask.sum())
        events = int(y[mask].sum())
        lo, hi = wilson(events, n)
        rows.append(
            {
                "lower": float(edges[i]),
                "upper": float(edges[i + 1]),
                "n": n,
                "events": events,
                "predicted": float(p[mask].mean()) if n else None,
                "observed": float(y[mask].mean()) if n else None,
                "ci_lower": lo,
                "ci_upper": hi,
            }
        )
    return rows


def curves(y, p):
    y, p = validate_predictions(y, p)
    if len(np.unique(y)) < 2:
        return {"roc": {"fpr": [], "tpr": []}, "pr": {"recall": [], "precision": []}}
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)

    # Thin display only; reported metrics computed on complete curves.
    def thin(a, b):
        idx = np.unique(np.linspace(0, len(a) - 1, min(len(a), 250)).astype(int))
        return a[idx].tolist(), b[idx].tolist()

    fpr, tpr = thin(fpr, tpr)
    recall, precision = thin(recall, precision)
    return {"roc": {"fpr": fpr, "tpr": tpr}, "pr": {"recall": recall, "precision": precision}}


def bootstrap_metrics(y, p, repetitions=300, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    groups = [np.flatnonzero(y == c) for c in (0, 1)]
    if any(len(g) == 0 for g in groups):
        return []
    values = {key: [] for key in ["roc_auc", "pr_auc", "ks", "brier", "log_loss"]}
    estimate = metrics(y, p)
    for _ in range(repetitions):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        # Avoid fitting a calibration regression each bootstrap.
        sample = metrics_simple(y[idx], p[idx])
        for key in values:
            values[key].append(sample[key])
    return [
        {
            "metric": k,
            "estimate": estimate[k],
            "lower": float(np.quantile(v, 0.025)),
            "upper": float(np.quantile(v, 0.975)),
            "replicates": repetitions,
        }
        for k, v in values.items()
    ]


def metrics_simple(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "ks": np.max(tpr - fpr),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }
