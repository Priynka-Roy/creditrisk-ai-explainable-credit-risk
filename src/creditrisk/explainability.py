"""Coefficients and held-out permutation importance are predictive associations."""

import numpy as np
from sklearn.inspection import permutation_importance


def logistic_coefficients(model):
    names = model.named_steps["preprocessing"].get_feature_names_out()
    coeff = model.named_steps["model"].coef_[0]
    return sorted(
        [
            {
                "feature": str(name),
                "coefficient": float(c),
                "odds_ratio": float(np.exp(np.clip(c, -30, 30))),
            }
            for name, c in zip(names, coeff)
        ],
        key=lambda r: abs(r["coefficient"]),
        reverse=True,
    )


def permutation_rows(model, X, y):
    result = permutation_importance(
        model, X, y, scoring="neg_log_loss", n_repeats=5, random_state=42, n_jobs=1
    )
    return sorted(
        [
            {"feature": str(c), "importance": float(m), "std": float(s)}
            for c, m, s in zip(X.columns, result.importances_mean, result.importances_std)
        ],
        key=lambda r: r["importance"],
        reverse=True,
    )
