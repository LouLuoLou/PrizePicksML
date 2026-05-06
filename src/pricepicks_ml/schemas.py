"""Silver-layer schema documentation and light dataclass helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Match:
    match_id: str
    league_id: str
    home_team_id: str
    away_team_id: str
    official_start_ts: datetime


@dataclass
class PlayerGameStat:
    player_id: str
    match_id: str
    team_id: str
    opponent_team_id: str
    official_start_ts: datetime
    minutes: float
    is_start: bool
    position_bucket: str
    # Generic stat column names are sport-specific in parquet; code passes stat_col.
    stats: dict[str, float]


@dataclass
class PropLine:
    player_id: str
    match_id: str
    market_code: str
    line_value: float
    line_snapshot_ts: datetime


@dataclass
class TrainingRow:
    player_id: str
    match_id: str
    market_code: str
    line_value: float
    line_snapshot_ts: datetime
    official_start_ts: datetime
    feature_as_of_ts: datetime
    realized_stat: float
    y_over: Optional[int]  # None = push excluded
