from pricepicks_ml.features.builder import attach_line_features, build_feature_matrix
from pricepicks_ml.labels import attach_labels, drop_pushes
from pricepicks_ml.synthetic import generate_basketball_demo


def test_feature_label_pipeline_smoke():
    games, lines, team = generate_basketball_demo(n_matches=40, n_players=10, seed=1)
    feats = build_feature_matrix(
        games,
        stat_col="stat_pts",
        sport="basketball",
        team_games_lookup=team,
    )
    df = attach_line_features(feats, lines, stat_col="stat_pts", line_col="line_value")
    df = attach_labels(df, col_realized="realized_stat", col_line="line_value")
    df = drop_pushes(df)
    assert "roll_10_mean" in df.columns
    assert "mean_minus_line" in df.columns
    assert df["y_over"].notna().all()
    assert "realized_stat" in df.columns
