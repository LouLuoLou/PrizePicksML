import pandas as pd

from pricepicks_ml.nba_data import (
    build_team_games_lookup,
    estimate_prop_lines,
    limit_to_top_players,
    normalize_nba_player_games,
)


def test_normalize_nba_logs_and_estimate_lines():
    logs = pd.DataFrame(
        [
            {
                "GAME_ID": "001",
                "GAME_DATE": "2024-01-01",
                "MATCHUP": "LAL vs. BOS",
                "PLAYER_ID": 23,
                "TEAM_ID": 1610612747,
                "TEAM_ABBREVIATION": "LAL",
                "MIN": "30:30",
                "PTS": 20,
            },
            {
                "GAME_ID": "001",
                "GAME_DATE": "2024-01-01",
                "MATCHUP": "BOS @ LAL",
                "PLAYER_ID": 0,
                "TEAM_ID": 1610612738,
                "TEAM_ABBREVIATION": "BOS",
                "MIN": 31,
                "PTS": 18,
            },
            {
                "GAME_ID": "002",
                "GAME_DATE": "2024-01-03",
                "MATCHUP": "LAL @ BOS",
                "PLAYER_ID": 23,
                "TEAM_ID": 1610612747,
                "TEAM_ABBREVIATION": "LAL",
                "MIN": 29,
                "PTS": 24,
            },
        ]
    )

    games = normalize_nba_player_games(logs)

    assert set(
        [
            "match_id",
            "player_id",
            "team_id",
            "opponent_team_id",
            "position_bucket",
            "official_start_ts",
            "is_home",
            "minutes",
            "stat_pts",
        ]
    ).issubset(games.columns)
    assert games.loc[games["player_id"] == "23", "minutes"].iloc[0] == 30.5
    home_by_player = games.loc[games["match_id"] == "001"].set_index("player_id")["is_home"].to_dict()
    assert home_by_player == {"0": 0.0, "23": 1.0}
    assert games.loc[games["player_id"] == "23", "opponent_team_id"].iloc[0] == "1610612738"

    lines = estimate_prop_lines(games)

    assert len(lines) == 1
    assert lines.iloc[0]["player_id"] == "23"
    assert lines.iloc[0]["line_value"] == 20.5

    team_lookup = build_team_games_lookup(games)

    assert len(team_lookup["1610612747"]) == 2
    assert len(team_lookup["1610612738"]) == 1


def test_limit_to_top_players_keeps_most_common_players():
    games = pd.DataFrame(
        {
            "player_id": ["a", "a", "a", "b", "b", "c"],
            "match_id": ["1", "2", "3", "1", "2", "1"],
            "official_start_ts": pd.date_range("2024-01-01", periods=6),
        }
    )

    limited = limit_to_top_players(games, max_players=2)

    assert set(limited["player_id"]) == {"a", "b"}
