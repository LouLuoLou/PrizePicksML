"""Metrics: Brier, log loss, ROC-AUC, reliability bins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y = y_true.astype(float)
    pp = np.clip(p.astype(float), 0.0, 1.0)
    return float(np.mean((pp - y) ** 2))


def safe_log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    pp = np.clip(p.astype(float), eps, 1 - eps)
    return float(log_loss(y_true.astype(int), pp))


def safe_roc_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    y = y_true.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p.astype(float)))


def sklearn_brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y_true.astype(int), np.clip(p.astype(float), 1e-6, 1 - 1e-6)))


@dataclass
class EvalReport:
    brier: float
    logloss: float
    roc_auc: float


def eval_report(y_true: np.ndarray, p: np.ndarray) -> EvalReport:
    return EvalReport(
        brier=brier_score(y_true, p),
        logloss=safe_log_loss(y_true, p),
        roc_auc=safe_roc_auc(y_true, p),
    )


def reliability_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs empirical frequency per probability bin."""
    df = pd.DataFrame({"y": y_true.astype(int), "p": np.clip(p.astype(float), 0, 1)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    g = df.groupby("bin", observed=True)
    out = g.agg(mean_p=("p", "mean"), emp_rate=("y", "mean"), n=("y", "size")).reset_index()
    return out
