"""Baselines: naive rolling-mean vs line and sparse logistic regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def naive_prob_over(
    rolling_mean: np.ndarray,
    line: np.ndarray,
    *,
    scale: float = 0.35,
) -> np.ndarray:
    """Heuristic probability from ``mean - line`` without training."""
    return sigmoid(scale * (rolling_mean.astype(float) - line.astype(float)))


def naive_side(prob: np.ndarray, threshold: float = 0.55) -> np.ndarray:
    out = np.full(prob.shape, -1, dtype=int)
    out[prob >= threshold] = 1
    out[prob <= 1.0 - threshold] = 0
    return out


def fit_logistic_baseline(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    """L2 logistic regression on standardized features (small, interpretable)."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X.to_numpy(dtype=float), y.astype(int))
    return pipe


def logistic_predict_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(X.to_numpy(dtype=float))[:, 1]
