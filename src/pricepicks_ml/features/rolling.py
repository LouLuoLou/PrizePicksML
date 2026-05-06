"""Rolling aggregates with optional exponential recency weights."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _exp_weights(n: int, lam: float) -> np.ndarray:
    if n <= 0:
        return np.array([])
    ages = np.arange(n - 1, -1, -1, dtype=float)
    w = np.exp(-lam * ages)
    return w / w.sum()


def rolling_weighted_mean(values: np.ndarray, lam: float = 0.15) -> float:
    n = values.size
    if n == 0:
        return float("nan")
    w = _exp_weights(n, lam)
    return float(np.dot(values, w))


def rolling_stats(values: np.ndarray, *, lam: float = 0.15) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "wmean": float("nan"),
        }
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v, ddof=0)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "wmean": rolling_weighted_mean(v, lam=lam),
    }
