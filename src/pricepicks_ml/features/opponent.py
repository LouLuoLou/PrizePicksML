"""Opponent rolling allowed stats (shifted: only prior games of opponent)."""

from __future__ import annotations

import pandas as pd

from pricepicks_ml.features.rolling import rolling_stats


def opponent_allowed_prior(
    opp_games: pd.DataFrame,
    stat_col: str,
    _position_bucket: str,
    *,
    windows: tuple[int, ...] = (5, 10, 20),
    lam: float = 0.15,
) -> dict[str, float]:
    """opp_games: rows are prior games of opponent vs same position_bucket, oldest first."""
    out: dict[str, float] = {}
    vals = opp_games[stat_col].to_numpy(dtype=float)
    for n in windows:
        tail = vals[-n:] if len(vals) else vals
        rs = rolling_stats(tail, lam=lam)
        for k, v in rs.items():
            out[f"opp_allowed_{n}_{k}"] = v
    return out


def gather_opponent_prior_games(
    history_df: pd.DataFrame,
    opponent_team_id: str,
    position_bucket: str,
    before_ts: pd.Timestamp,
    stat_col: str,
) -> pd.DataFrame:
    """Rows where the defensive team is ``opponent_team_id`` (opponent of scorer).

    history_df columns include ``opponent_team_id``, ``position_bucket`` (offensive player bucket),
    ``official_start_ts``, and ``{stat_col}`` (stat accumulated vs that defense).
    """
    h = history_df.copy()
    h["official_start_ts"] = pd.to_datetime(h["official_start_ts"])
    mask = (
        (h["opponent_team_id"] == opponent_team_id)
        & (h["position_bucket"] == position_bucket)
        & (h["official_start_ts"] < before_ts)
    )
    return h.loc[mask].sort_values("official_start_ts")
