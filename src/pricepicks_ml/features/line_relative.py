"""Line-relative transforms with safe denominators."""

from __future__ import annotations

import numpy as np


def mean_minus_line(rolling_mean: float, line: float) -> float:
    return float(rolling_mean - line)


def z_gap(rolling_mean: float, line: float, rolling_std: float, eps_std: float = 0.5) -> float:
    denom = max(rolling_std, eps_std)
    return float((rolling_mean - line) / denom)


def vectorized_mean_minus_line(rolling_mean: np.ndarray, line: np.ndarray) -> np.ndarray:
    return (rolling_mean.astype(float) - line.astype(float)).astype(float)


def vectorized_z_gap(rolling_mean: np.ndarray, line: np.ndarray, rolling_std: np.ndarray, eps_std: float = 0.5) -> np.ndarray:
    denom = np.maximum(rolling_std.astype(float), eps_std)
    return (rolling_mean.astype(float) - line.astype(float)) / denom
