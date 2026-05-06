"""Label computation: Over / Under vs line; push exclusion and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class LabelReport:
    n_total: int
    n_over: int
    n_under: int
    n_push_excluded: int
    push_rate: float


def compute_y_over(realized: float | np.ndarray, line: float | np.ndarray) -> np.ndarray:
    """Return 1 (Over), 0 (Under), np.nan for exact push."""
    r = np.asarray(realized, dtype=float)
    l = np.asarray(line, dtype=float)
    out = np.full(r.shape, np.nan)
    out[r > l] = 1.0
    out[r < l] = 0.0
    return out


def label_report(y: Iterable[float | int]) -> LabelReport:
    arr = np.asarray(list(y), dtype=float)
    n = arr.size
    n_push = int(np.isnan(arr).sum())
    valid = arr[~np.isnan(arr)]
    n_over = int((valid == 1).sum())
    n_under = int((valid == 0).sum())
    push_rate = (n_push / n) if n else 0.0
    return LabelReport(
        n_total=n,
        n_over=n_over,
        n_under=n_under,
        n_push_excluded=n_push,
        push_rate=push_rate,
    )


def attach_labels(
    df: pd.DataFrame,
    *,
    col_realized: str = "realized_stat",
    col_line: str = "line_value",
    col_y: str = "y_over",
) -> pd.DataFrame:
    out = df.copy()
    out[col_y] = compute_y_over(out[col_realized].to_numpy(), out[col_line].to_numpy())
    return out


def drop_pushes(df: pd.DataFrame, col_y: str = "y_over") -> pd.DataFrame:
    return df[df[col_y].notna()].copy()
