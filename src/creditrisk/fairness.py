"""Historical subgroup descriptive diagnostics, not compliance conclusions."""

import numpy as np
import pandas as pd

from .evaluation import threshold_metrics, wilson

LABELS = {
    "SEX": {1: "1 · male", 2: "2 · female"},
    "EDUCATION": {
        1: "1 · graduate school",
        2: "2 · university",
        3: "3 · high school",
        4: "4 · others",
        0: "0 · undocumented",
        5: "5 · undocumented",
        6: "6 · undocumented",
    },
    "MARRIAGE": {1: "1 · married", 2: "2 · single", 3: "3 · others", 0: "0 · undocumented"},
}


def group_columns(df):
    return {
        "Sex (historical source coding)": df.SEX.map(LABELS["SEX"]).fillna(
            df.SEX.astype(str) + " · unseen"
        ),
        "Education": df.EDUCATION.map(LABELS["EDUCATION"]).fillna(
            df.EDUCATION.astype(str) + " · unseen"
        ),
        "Marital status": df.MARRIAGE.map(LABELS["MARRIAGE"]).fillna(
            df.MARRIAGE.astype(str) + " · unseen"
        ),
        "Age band": pd.cut(
            df.AGE, [0, 29, 39, 49, 59, 100], labels=["20–29", "30–39", "40–49", "50–59", "60+"]
        ),
    }


def subgroup_diagnostics(df, y, p, min_n=100):
    y = np.asarray(y)
    p = np.asarray(p)
    rows = []
    for dimension, values in group_columns(df).items():
        for group in sorted(values.dropna().astype(str).unique()):
            mask = values.astype(str).to_numpy() == group
            n = int(mask.sum())
            events = int(y[mask].sum())
            lo, hi = wilson(events, n)
            thresholds = [threshold_metrics(y[mask], p[mask], t) for t in [0.1, 0.2, 0.3, 0.5, 0.7]]
            rows.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "n": n,
                    "events": events,
                    "observed": float(y[mask].mean()),
                    "predicted": float(p[mask].mean()),
                    "calibration_gap": float(p[mask].mean() - y[mask].mean()),
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "small_group": n < min_n or events < 20,
                    "thresholds": thresholds,
                    **thresholds[3],
                }
            )
    return rows
