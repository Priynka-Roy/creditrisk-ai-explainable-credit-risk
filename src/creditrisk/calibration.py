"""Independent probability calibrators and descriptive calibration diagnostics."""

import numpy as np
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def __init__(self, method="none"):
        if method not in {"none", "sigmoid", "isotonic"}:
            raise ValueError("Unknown calibration method")
        self.method = method

    def fit(self, p, y):
        if self.method == "sigmoid":
            self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(
                logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1), y
            )
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return self

    def predict(self, p):
        if self.method == "none":
            return np.asarray(p)
        if self.method == "sigmoid":
            return self.model.predict_proba(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]
        return self.model.predict(p)


def calibration_parameters(y, p):
    if len(np.unique(y)) < 2 or np.std(p) < 1e-12:
        return {"intercept": None, "slope": None}
    model = LogisticRegression(C=1e8, solver="lbfgs", max_iter=1000).fit(
        logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1), y
    )
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}
