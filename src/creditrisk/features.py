"""Conservative feature summaries; no outcome-dependent transformations."""

import numpy as np

from .data import CATEGORICAL, NUMERIC

ENGINEERED = [
    "MAX_POSITIVE_DELAY",
    "DELAY_MONTHS",
    "RECENT_PAYMENT_CHANGE_SIGNED_LOG",
    "PAYMENT_VARIABILITY_LOG",
    "BILL_CHANGE_SIGNED_LOG",
    "UTILIZATION_PROXY",
    "PAYMENT_BILL_RATIO",
]


def signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


def engineer_features(df, include_engineered=True):
    out = df[NUMERIC + CATEGORICAL].copy()
    for c in NUMERIC:
        if c != "AGE":
            out[c] = signed_log(out[c].to_numpy(dtype=float))
    if include_engineered:
        delays = df[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].to_numpy()
        payments = df[[f"PAY_AMT{i}" for i in range(1, 7)]].to_numpy(dtype=float)
        bills = df[[f"BILL_AMT{i}" for i in range(1, 7)]].to_numpy(dtype=float)
        out["MAX_POSITIVE_DELAY"] = np.maximum(delays, 0).max(axis=1)
        out["DELAY_MONTHS"] = (delays > 0).sum(axis=1)
        out["RECENT_PAYMENT_CHANGE_SIGNED_LOG"] = signed_log(payments[:, 0] - payments[:, 1])
        out["PAYMENT_VARIABILITY_LOG"] = np.log1p(payments.std(axis=1))
        out["BILL_CHANGE_SIGNED_LOG"] = signed_log(bills[:, 0] - bills[:, 5])
        out["UTILIZATION_PROXY"] = bills[:, 0] / df.LIMIT_BAL.to_numpy()
        # Explicit nonpositive denominator -> zero; indicator captures unavailable ratio.
        out["PAYMENT_BILL_RATIO"] = np.divide(
            payments[:, 0], bills[:, 0], out=np.zeros(len(df)), where=bills[:, 0] > 0
        ).clip(0, 10)
        out["RATIO_UNDEFINED"] = (bills[:, 0] <= 0).astype(int)
    return out.astype(float)
