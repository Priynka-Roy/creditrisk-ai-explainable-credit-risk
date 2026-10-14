"""Random-holdout distribution checks; not temporal monitoring."""

import numpy as np


def population_stability_index(reference, current, bins=10):
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        if np.all(cur == ref[0]):
            return {"psi": 0.0, "edges": edges.tolist()}
        return {
            "psi": None,
            "edges": edges.tolist(),
            "reason": "Constant reference has no quantile bins; shifted current distribution cannot be summarized by this PSI.",
        }
    edges[0] = -np.inf
    edges[-1] = np.inf
    a = np.histogram(ref, edges)[0].astype(float)
    b = np.histogram(cur, edges)[0].astype(float)
    # Jeffreys half-count smoothing prevents infinite empty-cell comparisons.
    a = (a + 0.5) / (a.sum() + 0.5 * len(a))
    b = (b + 0.5) / (b.sum() + 0.5 * len(b))
    return {
        "psi": float(np.sum((a - b) * np.log(a / b))),
        "edges": [None if not np.isfinite(v) else float(v) for v in edges],
    }
