"""Walk-forward date splits with optional embargo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterator, List, Optional

import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def monthly_walk_forward_indices(
    dates: pd.Series,
    *,
    embargo_days: int = 0,
) -> Iterator[Fold]:
    """Yield train/val folds: expanding train, next calendar month as val.

    dates: series of official_start_ts (timezone-aware or naive, but consistent).
    """
    s = pd.to_datetime(pd.Series(dates)).sort_values()
    if s.empty:
        return
    months = s.dt.to_period("M").unique()
    if len(months) < 2:
        return
    embargo = timedelta(days=embargo_days)
    for i in range(1, len(months)):
        val_month = months[i]
        train_months = months[:i]
        train_start = s.min()
        train_end = s[s.dt.to_period("M").isin(train_months)].max()
        val_mask = s.dt.to_period("M") == val_month
        val_start = s[val_mask].min()
        val_end = s[val_mask].max()
        if embargo_days:
            train_end = train_end - embargo
        yield Fold(
            fold_id=i,
            train_start=pd.Timestamp(train_start),
            train_end=pd.Timestamp(train_end),
            val_start=pd.Timestamp(val_start),
            val_end=pd.Timestamp(val_end),
        )


def split_frame(
    df: pd.DataFrame,
    ts_col: str,
    fold: Fold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(df[ts_col])
    train = df[(ts >= fold.train_start) & (ts <= fold.train_end)]
    val = df[(ts >= fold.val_start) & (ts <= fold.val_end)]
    return train, val


def all_folds(df: pd.DataFrame, ts_col: str, *, embargo_days: int = 0) -> List[Fold]:
    return list(monthly_walk_forward_indices(df[ts_col], embargo_days=embargo_days))
