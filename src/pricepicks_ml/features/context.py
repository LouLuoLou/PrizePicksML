"""Rest, home/away, congestion proxies."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def rest_days(team_games_ts: list[pd.Timestamp], current_ts: pd.Timestamp) -> float:
    past = [t for t in team_games_ts if t < current_ts]
    if not past:
        return float("nan")
    last = max(past)
    return float((current_ts - last).total_seconds() / 86400.0)


def is_back_to_back(rest: float) -> float:
    if np.isnan(rest):
        return float("nan")
    return 1.0 if rest <= 1.5 else 0.0


def matches_in_last_days(team_games_ts: list[pd.Timestamp], current_ts: pd.Timestamp, days: int = 7) -> int:
    cutoff = current_ts - pd.Timedelta(days=days)
    return int(sum(1 for t in team_games_ts if cutoff <= t < current_ts))
