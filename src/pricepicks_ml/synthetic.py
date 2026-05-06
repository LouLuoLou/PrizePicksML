"""Synthetic multi-sport schedules for pipeline demos (no external API)."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_basketball_demo(
    *,
    n_players: int = 16,
    n_matches: int = 90,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[pd.Timestamp]]]:
    """Return ``(player_games, lines, team_games_lookup)`` for integration tests / demos."""
    rng = _rng(seed)
    teams = [f"T{i}" for i in range(4)]
    players = [f"P{i}" for i in range(n_players)]
    buckets = rng.choice(["G", "F", "C"], size=n_players)
    # Naive timestamps avoid Period warnings; span >=3 calendar months for walk-forward folds.
    base = pd.Timestamp("2026-01-05")
    rows: list[dict] = []
    team_ts: dict[str, list] = {t: [] for t in teams}

    for m in range(n_matches):
        ts = base + pd.Timedelta(days=m, hours=19)
        home, away = rng.choice(teams, size=2, replace=False)
        team_ts[home].append(ts)
        team_ts[away].append(ts)
        match_id = f"M{m:04d}"
        for t in (home, away):
            opp = away if t == home else home
            idx = rng.choice(n_players, size=5, replace=False)
            for i in idx:
                ii = int(i)
                pid = players[ii]
                minutes = float(rng.uniform(18, 38))
                talent = hashlib.md5(pid.encode()).digest()[0] / 255.0 * 10.0
                noise = rng.normal(0, 4.0)
                pts = max(0.0, talent + noise + (5.0 if t == home else 0.0))
                rows.append(
                    {
                        "match_id": match_id,
                        "player_id": pid,
                        "team_id": t,
                        "opponent_team_id": opp,
                        "position_bucket": buckets[ii],
                        "official_start_ts": ts,
                        "is_home": 1.0 if t == home else 0.0,
                        "minutes": minutes,
                        "stat_pts": float(pts),
                    }
                )

    games = pd.DataFrame(rows).sort_values(["official_start_ts", "match_id", "player_id"]).reset_index(drop=True)

    line_rows: list[dict] = []
    for _, grp in games.groupby("player_id", sort=False):
        grp = grp.sort_values("official_start_ts")
        prev_mean = grp["stat_pts"].shift(1).rolling(5, min_periods=1).mean()
        for i in range(len(grp)):
            row = grp.iloc[i]
            pm = prev_mean.iloc[i]
            base_line = float(pm) if not np.isnan(pm) else float(row["stat_pts"])
            line_val = float(np.clip(base_line + rng.normal(0, 1.5), 5, 35))
            line_rows.append(
                {
                    "player_id": row["player_id"],
                    "match_id": row["match_id"],
                    "line_value": line_val,
                }
            )

    lines = pd.DataFrame(line_rows)
    return games, lines, {k: sorted(v) for k, v in team_ts.items()}
