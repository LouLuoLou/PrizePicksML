"""Primary GBDT model (HistGradientBoosting) + hyperparameter protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


META_COLS = {
    "player_id",
    "match_id",
    "official_start_ts",
    "team_id",
    "opponent_team_id",
    "position_bucket",
    "realized_stat",
    "y_over",
    "line_snapshot_ts",
}


def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if df[c].dtype == object:
            continue
        cols.append(c)
    return cols


@dataclass
class TrainResult:
    model: HistGradientBoostingClassifier
    feature_columns: List[str]


def fit_hgb(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    feature_columns: Iterable[str] | None = None,
    learning_rate: float = 0.06,
    max_depth: int | None = 5,
    max_iter: int = 200,
    l2_regularization: float = 1.0,
    random_state: int = 0,
) -> TrainResult:
    feats = list(feature_columns) if feature_columns is not None else infer_feature_columns(X)
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        l2_regularization=l2_regularization,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
    )
    model.fit(X[feats].to_numpy(dtype=float), y.astype(int))
    return TrainResult(model=model, feature_columns=feats)


def predict_proba(model: HistGradientBoostingClassifier, X: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
    return model.predict_proba(X[feature_columns].to_numpy(dtype=float))[:, 1]
