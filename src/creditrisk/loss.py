"""Hypothetical next-month loss arithmetic; no realized EAD or LGD."""

import numpy as np
from scipy.special import expit, logit


def stress_probabilities(p, odds_multiplier=1.0):
    p = np.asarray(p, dtype=float)
    if (
        not np.isfinite(odds_multiplier)
        or odds_multiplier <= 0
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Finite positive odds multiplier and probabilities in [0,1] required")
    result = expit(logit(np.clip(p, 1e-12, 1 - 1e-12)) + np.log(odds_multiplier))
    return np.where(p == 0, 0, np.where(p == 1, 1, result))


def expected_loss(pd, lgd, ead, portfolio_size=1):
    if (
        not np.isfinite([lgd, ead, portfolio_size]).all()
        or not 0 <= lgd <= 1
        or ead < 0
        or portfolio_size < 0
    ):
        raise ValueError("LGD in [0,1], EAD and portfolio size nonnegative required")
    p = np.asarray(pd, dtype=float)
    if p.size == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probability")
    return float(p.mean() * lgd * ead * portfolio_size)
