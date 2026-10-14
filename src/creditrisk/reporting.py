"""Aggregate publication artifacts, with no customer-level records."""

import json
from pathlib import Path

import numpy as np


def write_json(path, data):
    def convert(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o).__name__)

    Path(path).write_text(json.dumps(data, indent=2, default=convert, allow_nan=False))
