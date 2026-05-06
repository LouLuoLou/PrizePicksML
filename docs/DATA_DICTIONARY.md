# Data dictionary

Entities, keys, timestamps, and **as-of** rules for the silver layer and training rows.

## 1. Global conventions

| Field | Type | Description |
|-------|------|-------------|
| `sport` | string | `basketball`, `baseball`, `soccer` |
| `league_id` | string | e.g. `nba`, `mlb`, `epl` |
| `official_start_ts` | datetime UTC | Match scheduled start (or actual if only that exists—document source) |
| `feature_as_of_ts` | datetime UTC | Latest timestamp allowed when building a prediction row |

**As-of rule (non-negotiable):**  
`feature_as_of_ts = min(official_start_ts of predicted match - 1 microsecond, line_snapshot_ts if policy uses line, last_completed_game_end_ts used in rolls)`  
Implementation detail: use strict `< official_start_ts` filters when joining historical games.

## 2. `dim_players`

| Column | Key | Description |
|--------|-----|-------------|
| `player_id` | PK | Stable internal id |
| `display_name` | | Human-readable |
| `primary_position` | | Sport-specific code (e.g. PG, SP, CM) |
| `team_id` | FK | Current roster mapping optional; historical via `fact_player_game_stats` |

## 3. `dim_teams`

| Column | Key | Description |
|--------|-----|-------------|
| `team_id` | PK | Internal id |
| `league_id` | | League |
| `name` | | Display |

## 4. `fact_matches`

| Column | Key | Description |
|--------|-----|-------------|
| `match_id` | PK | Internal id |
| `league_id` | | |
| `home_team_id` | FK | |
| `away_team_id` | FK | |
| `official_start_ts` | | Used for ordering and walk-forward |
| `venue_id` | optional | For MLB park factors if used |

## 5. `fact_player_game_stats` (boxscore grain)

One row per `(player_id, match_id)` appearance.

| Column | Description |
|--------|-------------|
| `player_id`, `match_id` | Composite logical key |
| `team_id` | Player's team that game |
| `opponent_team_id` | Derived |
| `minutes` | BB: seconds or decimal minutes; soccer: minutes played; MLB: custom (innings pitched, PA, etc.) |
| `is_start` | bool if known |
| `stat_*` | Sport-specific columns (e.g. `stat_pts`, `stat_hits`, `stat_shots`) |
| `boxscore_final_ts` | When stat row is considered final for labeling |

**DNP / zero minutes:** Keep row with `minutes=0` if rostered; label may be missing or excluded for props that require participation (document per market).

## 6. `fact_lines` (optional for MVP)

| Column | Description |
|--------|-------------|
| `line_id` | PK |
| `player_id`, `match_id`, `market_code` | e.g. `PTS`, `HITS`, `SHOTS` |
| `line_value` | float |
| `line_snapshot_ts` | when line observed |
| `source` | sportsbook / PP / synthetic |

## 7. Training row (`training_snapshot` logical entity)

| Column | Description |
|--------|-------------|
| `snapshot_id` | PK |
| `player_id`, `match_id`, `market_code`, `line_value`, `line_snapshot_ts` | |
| `realized_stat` | from post-game boxscore |
| `y_over` | 0/1/NaN if push excluded |
| `feature_as_of_ts` | audit |

## 8. Bronze layer

Raw JSON/CSV blobs keyed by `ingest_batch_id`, `source_uri_hash`, `ingested_at`. Immutable append-only.
