"""Hypothetical next-month credit losses under assumed Gaussian dependence."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri


def portfolio_from_histogram(histogram: list[dict], size: int = 1000) -> np.ndarray:
    """Deterministic midpoint-quantile portfolio from anonymous weighted PD bins."""
    if not isinstance(size, int) or not 1 <= size <= 10000:
        raise ValueError("Portfolio size must be an integer from 1 to 10000")
    probabilities = np.asarray([r["p"] for r in histogram], dtype=float)
    counts = np.asarray([r["n"] for r in histogram], dtype=float)
    if len(counts) == 0 or not np.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("Histogram requires finite nonnegative counts with positive sum")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Histogram probabilities must lie in [0,1]")
    order = np.argsort(probabilities)
    cumulative = np.cumsum(counts[order]) / counts.sum()
    indices = np.searchsorted(cumulative, (np.arange(size) + 0.5) / size, side="left")
    return probabilities[order][indices]


def summarize_losses(losses, alpha: float = 0.95) -> dict:
    """Empirical inverse-CDF VaR and exact upper-(1-alpha) mass ES with ties."""
    x = np.asarray(losses, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Losses must be a nonempty finite nonnegative vector")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Tail confidence alpha must lie strictly between zero and one")
    ordered = np.sort(x)
    var = ordered[max(0, int(np.ceil(alpha * len(x))) - 1)]
    tail_mass = (1 - alpha) * len(x)
    whole = int(np.floor(tail_mass))
    fraction = tail_mass - whole
    descending = ordered[::-1]
    numerator = descending[:whole].sum()
    if fraction > 0:
        numerator += fraction * descending[whole]
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": sd,
        "mean_se": sd / np.sqrt(len(x)),
        "alpha": alpha,
        "var": float(var),
        "es": float(numerator / tail_mass),
        "var_95": float(var) if alpha == 0.95 else None,
        "es_95": float(numerator / tail_mass) if alpha == 0.95 else None,
        "minimum": float(x.min()),
        "maximum": float(x.max()),
        "n_simulations": len(x),
    }


def simulate_portfolio(
    probabilities, lgd=0.45, ead=100000.0, n_simulations=5000, rho=0.0, seed=442, batch_size=250
) -> dict:
    """D_i = 1[sqrt(rho) Z + sqrt(1-rho) epsilon_i < Phi^-1(p_i)].

    rho is an assumed latent asset correlation, never estimated here. Marginal
    default probabilities remain p_i. Endpoints are exact; no probability clipping.
    """
    p = np.asarray(probabilities, dtype=float)
    if (
        p.ndim != 1
        or not 1 <= len(p) <= 10000
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Provide 1–10000 finite probabilities in [0,1]")
    if not isinstance(n_simulations, int) or not 1 <= n_simulations <= 20000:
        raise ValueError("Simulation count must be integer 1–20000")
    if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 1000:
        raise ValueError("Batch size must be integer 1–1000")
    if not np.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("Assumed latent correlation rho must lie in [0,1]")
    lgd_values = np.broadcast_to(np.asarray(lgd, dtype=float), p.shape)
    ead_values = np.broadcast_to(np.asarray(ead, dtype=float), p.shape)
    if not np.isfinite(lgd_values).all() or ((lgd_values < 0) | (lgd_values > 1)).any():
        raise ValueError("LGD must be finite and lie in [0,1]")
    if not np.isfinite(ead_values).all() or (ead_values < 0).any():
        raise ValueError("EAD must be finite nonnegative")
    exposures = lgd_values * ead_values
    rng = np.random.default_rng(seed)
    # Generate systematic factors separately to make results batch-size invariant.
    systematic = rng.standard_normal(n_simulations)
    thresholds = ndtri(p)
    losses = np.empty(n_simulations)
    defaults = np.zeros(len(p), dtype=np.int64)
    conditional_expected_losses = np.empty(n_simulations)
    for begin in range(0, n_simulations, batch_size):
        end = min(begin + batch_size, n_simulations)
        idiosyncratic = rng.standard_normal((end - begin, len(p)))
        latent = np.sqrt(rho) * systematic[begin:end, None] + np.sqrt(1 - rho) * idiosyncratic
        event = latent < thresholds
        losses[begin:end] = event @ exposures
        defaults += event.sum(axis=0)
        if rho == 1:
            conditional = systematic[begin:end, None] < thresholds
        else:
            conditional = ndtr(
                (thresholds[None, :] - np.sqrt(rho) * systematic[begin:end, None])
                / np.sqrt(1 - rho)
            )
        conditional_expected_losses[begin:end] = conditional @ exposures
    return {
        "losses": losses,
        "marginal_probabilities": defaults / n_simulations,
        "analytic_expected_loss": float(np.dot(p, exposures)),
        "conditional_expected_loss_mean": float(conditional_expected_losses.mean()),
        "rho": float(rho),
        "seed": int(seed),
        "portfolio_size": len(p),
        "mean_pd": float(p.mean()),
    }
