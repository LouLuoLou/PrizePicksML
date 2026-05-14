"""NBA API data loading helpers for the demo pipeline.

The NBA API provides real player-game box scores. Prop lines are estimated from
prior player performance so the project can run without paid historical odds.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playergamelogs


DEFAULT_SEASONS = ("2023-24",)
DEFAULT_MAX_PLAYERS = 60


def fetch_nba_player_game_logs(
    *,
    seasons: Iterable[str] = DEFAULT_SEASONS,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """Fetch NBA player game logs for one or more seasons."""
    frames: list[pd.DataFrame] = []
    for season in seasons:
        logs = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable=season_type,
            timeout=60,
        ).get_data_frames()[0]
        logs["SEASON"] = season
        frames.append(logs)

    if not frames:
        raise ValueError("At least one NBA season is required.")

    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        raise ValueError("NBA API returned no player game logs.")
    return out


def _parse_minutes(value: object) -> float:
    """Handle either numeric minutes or NBA-style MM:SS strings."""
    if pd.isna(value):
        return float("nan")
    if isinstance(value, str) and ":" in value:
        minutes, seconds = value.split(":", maxsplit=1)
        return float(minutes) + (float(seconds) / 60.0)
    return float(value)


def normalize_nba_player_games(logs: pd.DataFrame, *, source_stat_col: str = "PTS") -> pd.DataFrame:
    """Map NBA API game logs into the columns expected by ``build_feature_matrix``."""
    required = {
        "GAME_ID",
        "GAME_DATE",
        "MATCHUP",
        "PLAYER_ID",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "MIN",
        source_stat_col,
    }
    missing = sorted(required - set(logs.columns))
    if missing:
        raise ValueError(f"NBA game logs missing required columns: {missing}")

    team_id_by_abbr = (
        logs[["TEAM_ABBREVIATION", "TEAM_ID"]]
        .dropna()
        .drop_duplicates("TEAM_ABBREVIATION")
        .set_index("TEAM_ABBREVIATION")["TEAM_ID"]
        .astype(str)
        .to_dict()
    )
    opponent_abbr = logs["MATCHUP"].astype(str).str.extract(r"(?:vs\.|@)\s+([A-Z]{2,3})")[0]

    games = pd.DataFrame(
        {
            "match_id": logs["GAME_ID"].astype(str),
            "player_id": logs["PLAYER_ID"].astype(str),
            "team_id": logs["TEAM_ID"].astype(str),
            "opponent_team_id": opponent_abbr.map(team_id_by_abbr).fillna(opponent_abbr).astype(str),
            "position_bucket": "NBA",
            "official_start_ts": pd.to_datetime(logs["GAME_DATE"]),
            "is_home": logs["MATCHUP"].astype(str).str.contains(" vs. ", regex=False).astype(float),
            "minutes": logs["MIN"].map(_parse_minutes).astype(float),
            "stat_pts": logs[source_stat_col].astype(float),
        }
    )
    return games.sort_values(["official_start_ts", "match_id", "player_id"]).reset_index(drop=True)


def estimate_prop_lines(
    games: pd.DataFrame,
    *,
    stat_col: str = "stat_pts",
    window: int = 5,
) -> pd.DataFrame:
    """Estimate half-point prop lines from prior player games only."""
    ordered = games.sort_values(["player_id", "official_start_ts", "match_id"]).copy()
    prior_mean = ordered.groupby("player_id", sort=False)[stat_col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    lines = ordered[["player_id", "match_id"]].copy()
    # Basketball points are integer-valued, so half-point lines avoid pushes.
    lines["line_value"] = np.floor(prior_mean.to_numpy(dtype=float)) + 0.5
    return lines.dropna(subset=["line_value"]).reset_index(drop=True)


def limit_to_top_players(games: pd.DataFrame, *, max_players: int | None = DEFAULT_MAX_PLAYERS) -> pd.DataFrame:
    """Keep the players with the most game logs so the demo stays quick."""
    if max_players is None:
        return games
    if max_players <= 0:
        raise ValueError("max_players must be positive or None.")

    keepers = games["player_id"].value_counts().head(max_players).index
    limited = games[games["player_id"].isin(keepers)].copy()
    return limited.sort_values(["official_start_ts", "match_id", "player_id"]).reset_index(drop=True)


def build_team_games_lookup(games: pd.DataFrame) -> dict[str, list[pd.Timestamp]]:
    """Create team schedule lookup used by rest-days and back-to-back features."""
    team_games = games[["team_id", "match_id", "official_start_ts"]].drop_duplicates()
    return {
        str(team_id): sorted(pd.to_datetime(group["official_start_ts"]).tolist())
        for team_id, group in team_games.groupby("team_id", sort=False)
    }


def load_nba_demo_data(
    *,
    seasons: Iterable[str] = DEFAULT_SEASONS,
    season_type: str = "Regular Season",
    max_players: int | None = DEFAULT_MAX_PLAYERS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[pd.Timestamp]]]:
    """Return ``(games, estimated_lines, team_games_lookup)`` for the demo."""
    logs = fetch_nba_player_game_logs(seasons=seasons, season_type=season_type)
    games = normalize_nba_player_games(logs, source_stat_col="PTS")
    games = limit_to_top_players(games, max_players=max_players)
    lines = estimate_prop_lines(games, stat_col="stat_pts")
    team_lookup = build_team_games_lookup(games)
    return games, lines, team_lookup
