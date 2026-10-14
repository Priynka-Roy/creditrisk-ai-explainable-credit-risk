"""Official XLS loading, schema validation and reproducible customer partitions."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "default.payment.next.month"
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
NUMERIC = (
    ["LIMIT_BAL", "AGE"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
FEATURES = NUMERIC + CATEGORICAL


def validate_data(df, expected_rows=None):
    if set(df.columns) != {"ID", TARGET, *FEATURES}:
        raise ValueError("Dataset columns do not match official schema")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError("Unexpected row count")
    if df.ID.duplicated().any():
        raise ValueError("Customer identifiers must be unique")
    if df.isna().any().any():
        raise ValueError("Unexpected missing observations")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError("Outcome must be binary 0/1")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite source data")
    if (df.LIMIT_BAL <= 0).any():
        raise ValueError("Credit limits must be positive")
    return {
        "rows": len(df),
        "features": len(FEATURES),
        "events": int(df[TARGET].sum()),
        "prevalence": float(df[TARGET].mean()),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_ids": int(df.ID.duplicated().sum()),
        "duplicate_predictor_rows": int(df[FEATURES].duplicated().sum()),
        "category_counts": {
            c: {str(k): int(v) for k, v in df[c].value_counts().sort_index().items()}
            for c in CATEGORICAL
        },
        "negative_bill_cells": int((df[[f"BILL_AMT{i}" for i in range(1, 7)]] < 0).sum().sum()),
    }


def load_data(path):
    df = pd.read_excel(Path(path), header=1, engine="xlrd")
    df = df.rename(columns={"default payment next month": TARGET})
    validate_data(df, 30000)
    return df


def make_splits(y, seed=42):
    idx = np.arange(len(y))
    development, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    train, remainder = train_test_split(
        development, test_size=0.375, stratify=np.asarray(y)[development], random_state=seed
    )
    validation, calibration = train_test_split(
        remainder, test_size=0.5, stratify=np.asarray(y)[remainder], random_state=seed
    )
    result = {"training": train, "validation": validation, "calibration": calibration, "test": test}
    assert len(set(np.concatenate(list(result.values())))) == len(y)
    return result


def split_hash(ids):
    return hashlib.sha256(np.asarray(sorted(ids), dtype="<i8").tobytes()).hexdigest()
