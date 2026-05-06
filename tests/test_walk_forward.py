import pandas as pd

from pricepicks_ml.walk_forward import all_folds, monthly_walk_forward_indices


def test_monthly_folds_generate():
    ts = pd.to_datetime(
        ["2026-01-05", "2026-01-20", "2026-02-05", "2026-02-20", "2026-03-10", "2026-03-25"]
    )
    folds = list(monthly_walk_forward_indices(ts))
    assert len(folds) >= 2


def test_all_folds_dataframe():
    df = pd.DataFrame(
        {
            "official_start_ts": pd.to_datetime(
                ["2026-01-05", "2026-01-20", "2026-02-05", "2026-02-20", "2026-03-10", "2026-03-25"]
            ),
            "x": range(6),
        }
    )
    folds = all_folds(df, "official_start_ts")
    assert len(folds) >= 2
