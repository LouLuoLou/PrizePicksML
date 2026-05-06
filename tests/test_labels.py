import numpy as np

from pricepicks_ml.labels import compute_y_over, drop_pushes, label_report


def test_push_nan():
    y = compute_y_over(np.array([10.0, 10.0, 9.0]), np.array([10.0, 9.0, 10.0]))
    assert np.isnan(y[0])
    assert y[1] == 1.0
    assert y[2] == 0.0


def test_label_report():
    r = label_report([1, 0, np.nan])
    assert r.n_push_excluded == 1
    assert r.n_over == 1
    assert r.n_under == 1


def test_drop_pushes():
    import pandas as pd

    df = pd.DataFrame({"y_over": [1.0, 0.0, np.nan]})
    out = drop_pushes(df)
    assert len(out) == 2
