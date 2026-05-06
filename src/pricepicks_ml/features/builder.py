"""Assemble leakage-safe feature rows from chronological player games."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pricepicks_ml.features.context import is_back_to_back, matches_in_last_days, rest_days
from pricepicks_ml.features.line_relative import vectorized_mean_minus_line, vectorized_z_gap
from pricepicks_ml.features.opponent import gather_opponent_prior_games, opponent_allowed_prior
from pricepicks_ml.features.rolling import rolling_stats
from pricepicks_ml.features.time_adjusted import per_36, per_90


def build_feature_matrix(
    games: pd.DataFrame,
    *,
    stat_col: str,
    minutes_col: str = "minutes",
    sport: str = "basketball",
    windows: tuple[int, ...] = (5, 10, 20),
    lam: float = 0.15,
    team_games_lookup: dict[str, list[pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Return one row per input game with prior-only features.

    Required columns on ``games``:
    ``player_id, team_id, opponent_team_id, position_bucket, official_start_ts, is_home``,
    plus ``stat_col`` and ``minutes_col``.

    Rows with insufficient history still emitted; rolling stats may be NaN.
    """
    g = games.copy()
    g["official_start_ts"] = pd.to_datetime(g["official_start_ts"])
    g = g.sort_values(["player_id", "official_start_ts"]).reset_index(drop=True)

    feats: list[dict] = []
    history_by_player: dict[str, list[dict]] = {}

    for idx, row in g.iterrows():
        pid = row["player_id"]
        ts = row["official_start_ts"]
        prior_same = [h for h in history_by_player.get(pid, []) if h["ts"] < ts]
        stat_hist = np.array([h["stat"] for h in prior_same], dtype=float)
        min_hist = np.array([h["minutes"] for h in prior_same], dtype=float)

        # Current-game boxscore is kept only as label source (`realized_stat`), not as a feature.
        out: dict[str, object] = {
            "player_id": pid,
            "match_id": row.get("match_id", f"row_{idx}"),
            "official_start_ts": ts,
            "team_id": row["team_id"],
            "opponent_team_id": row["opponent_team_id"],
            "position_bucket": row["position_bucket"],
            "is_home": float(row["is_home"]),
            "realized_stat": float(row[stat_col]),
        }

        for n in windows:
            tail_s = stat_hist[-n:] if len(stat_hist) else stat_hist
            tail_m = min_hist[-n:] if len(min_hist) else min_hist
            rs = rolling_stats(tail_s, lam=lam)
            for k, v in rs.items():
                out[f"roll_{n}_{k}"] = v
            if sport == "basketball":
                per36 = np.array([per_36(s, m) for s, m in zip(tail_s, tail_m)])
                out[f"roll_{n}_per36_mean"] = float(np.nanmean(per36)) if per36.size else float("nan")
            elif sport == "soccer":
                per90 = np.array([per_90(s, m) for s, m in zip(tail_s, tail_m)])
                out[f"roll_{n}_per90_mean"] = float(np.nanmean(per90)) if per90.size else float("nan")

        opp_g = gather_opponent_prior_games(
            g,
            row["opponent_team_id"],
            row["position_bucket"],
            ts,
            stat_col,
        )
        opp_feats = opponent_allowed_prior(opp_g, stat_col, str(row["position_bucket"]), windows=windows, lam=lam)
        out.update(opp_feats)

        if team_games_lookup is not None:
            tss = team_games_lookup.get(str(row["team_id"]), [])
            rd = rest_days(tss, ts)
            out["rest_days"] = rd
            out["is_b2b"] = is_back_to_back(rd)
            out["matches_last7"] = float(matches_in_last_days(tss, ts, days=7))
        else:
            out["rest_days"] = float("nan")
            out["is_b2b"] = float("nan")
            out["matches_last7"] = float("nan")

        feats.append(out)
        history_by_player.setdefault(pid, []).append(
            {
                "ts": ts,
                "stat": float(row[stat_col]),
                "minutes": float(row[minutes_col]),
            }
        )

    return pd.DataFrame(feats)


def attach_line_features(feats: pd.DataFrame, lines: pd.DataFrame, *, stat_col: str, line_col: str = "line_value") -> pd.DataFrame:
    """Join lines to feature rows on (player_id, match_id) and add line-relative columns."""
    out = feats.merge(
        lines[["player_id", "match_id", line_col]],
        on=["player_id", "match_id"],
        how="left",
    )
    rolling_mean = out["roll_10_mean"].to_numpy(dtype=float)
    rolling_std = out["roll_10_std"].to_numpy(dtype=float)
    lv = out[line_col].to_numpy(dtype=float)
    out["mean_minus_line"] = vectorized_mean_minus_line(rolling_mean, lv)
    out["z_gap"] = vectorized_z_gap(rolling_mean, lv, rolling_std)
    return out
