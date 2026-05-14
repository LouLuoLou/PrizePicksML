#!/usr/bin/env python3
"""End-to-end demo: NBA data, features, walk-forward, HGB + calibration, metrics."""

from __future__ import annotations

import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    import pricepicks_ml  # noqa: F401
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))

from pricepicks_ml.baselines import fit_logistic_baseline, logistic_predict_proba, naive_prob_over  # noqa: E402
from pricepicks_ml.calibration_util import choose_calibrator_kind, fit_calibrator  # noqa: E402
from pricepicks_ml.evaluate import eval_report, reliability_table  # noqa: E402
from pricepicks_ml.features.builder import attach_line_features, build_feature_matrix  # noqa: E402
from pricepicks_ml.labels import LabelReport, attach_labels, drop_pushes, label_report  # noqa: E402
from pricepicks_ml.nba_data import DEFAULT_MAX_PLAYERS, DEFAULT_SEASONS, load_nba_demo_data  # noqa: E402
from pricepicks_ml.train import fit_hgb, predict_proba  # noqa: E402
from pricepicks_ml.walk_forward import all_folds, split_frame  # noqa: E402


def median_impute(train: pd.DataFrame, val: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = train[cols].median()
    return train[cols].fillna(med), val[cols].fillna(med)


def _section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _print_label_block(rep: LabelReport) -> None:
    _section("Dataset / labels")
    rows_kv = [
        ("Rows used (pushes dropped)", str(rep.n_total)),
        ("Over label (1)", str(rep.n_over)),
        ("Under label (0)", str(rep.n_under)),
        ("Pushes excluded from training row count", str(rep.n_push_excluded)),
        ("Push rate (before drop)", f"{rep.push_rate:.4f}"),
    ]
    key_w = max(len(k) for k, _ in rows_kv)
    for key, val in rows_kv:
        print(f"  {key:<{key_w}}  {val}")
    print()
    blurb = (
        "Each row is a player-game pick vs the line: Over=1, Under=0. "
        "The walk-forward section scores models only on the validation time slice for that fold "
        "(the model never trains on those games). "
        "For Brier score and log loss, lower is better; for ROC-AUC, higher is better (1.0 is perfect separation)."
    )
    for line in textwrap.wrap(blurb, width=78):
        print(line)


def _fmt_float(x: float, width: int, decimals: int) -> str:
    if isinstance(x, (float, np.floating)) and math.isnan(float(x)):
        return f"{'nan':>{width}}"
    return f"{float(x):>{width}.{decimals}f}"


def _print_validation_metrics_table(naive_d: dict, logreg_d: dict, hgb_d: dict) -> None:
    _section("Validation metrics")
    w_model = 18
    w_num = 10
    header = f"{'model':<{w_model}}  {'Brier':>{w_num}}  {'log loss':>{w_num}}  {'ROC-AUC':>{w_num}}"
    print(header)
    print("-" * len(header))
    specs = [
        ("naive", naive_d),
        ("logreg", logreg_d),
        ("hgb (calibrated)", hgb_d),
    ]
    for name, d in specs:
        b = _fmt_float(d["brier"], w_num, 4)
        ll = _fmt_float(d["logloss"], w_num, 4)
        auc = _fmt_float(d["roc_auc"], w_num, 4)
        print(f"{name:<{w_model}}  {b}  {ll}  {auc}")


def _print_reliability(rel: pd.DataFrame) -> None:
    _section("Reliability (calibrated HGB)")
    explain = (
        "Bins sort validation rows by predicted P(Over): mean_p is the average prediction in the bin, "
        "emp_rate is the actual fraction that went Over, n is how many rows fell in that bin."
    )
    for line in textwrap.wrap(explain, width=78):
        print(line)
    print()
    rel = rel.copy()
    rel["bin"] = rel["bin"].astype(str)
    w_bin = min(max(int(rel["bin"].str.len().max()), 10), 44)
    w_mp = 10
    w_er = 10
    w_n = 6
    hdr = f"{'bin':<{w_bin}}  {'mean_p':>{w_mp}}  {'emp_rate':>{w_er}}  {'n':>{w_n}}"
    print(hdr)
    print("-" * len(hdr))
    for _, row in rel.iterrows():
        b = str(row["bin"])[:w_bin]
        print(
            f"{b:<{w_bin}}  {_fmt_float(row['mean_p'], w_mp, 6)}  "
            f"{_fmt_float(row['emp_rate'], w_er, 6)}  {int(row['n']):{w_n}d}"
        )


def _write_input_csvs(games: pd.DataFrame, lines: pd.DataFrame, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    games.to_csv(artifact_dir / "nba_player_games.csv", index=False)
    lines.to_csv(artifact_dir / "nba_estimated_lines.csv", index=False)
    print("Wrote data/artifacts/nba_player_games.csv")
    print("Wrote data/artifacts/nba_estimated_lines.csv")


def _write_metrics_csv(rows: list[dict], artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.json_normalize(rows).to_csv(artifact_dir / "walkforward_metrics.csv", index=False)
    print("Wrote data/artifacts/walkforward_metrics.csv")


def main() -> None:
    artifact_dir = Path("data/artifacts")
    print(
        f"Loading NBA player game logs for seasons: {', '.join(DEFAULT_SEASONS)} "
        f"(top {DEFAULT_MAX_PLAYERS} players by games played)"
    )
    games, lines, team_lookup = load_nba_demo_data(seasons=DEFAULT_SEASONS, max_players=DEFAULT_MAX_PLAYERS)
    print(
        f"Loaded {len(games):,} real player-game rows and "
        f"{len(lines):,} estimated prior-average lines."
    )
    _write_input_csvs(games, lines, artifact_dir)
    raw_feats = build_feature_matrix(
        games,
        stat_col="stat_pts",
        sport="basketball",
        team_games_lookup=team_lookup,
    )
    df = attach_line_features(raw_feats, lines, stat_col="stat_pts", line_col="line_value")
    df = attach_labels(df, col_realized="realized_stat", col_line="line_value")
    df = drop_pushes(df)
    df.to_csv(artifact_dir / "nba_training_rows.csv", index=False)
    print("Wrote data/artifacts/nba_training_rows.csv")
    rep = label_report(df["y_over"].tolist())
    _print_label_block(rep)

    folds = all_folds(df, "official_start_ts", embargo_days=0)
    if len(folds) < 2:
        print("Not enough months for walk-forward; add more NBA seasons.")
        return

    feature_cols = [c for c in df.columns if c not in {"player_id", "match_id", "official_start_ts", "team_id", "opponent_team_id", "position_bucket", "realized_stat", "y_over"}]

    rows = []
    for fold in folds:
        tr, va = split_frame(df, "official_start_ts", fold)
        if len(tr) < 50 or len(va) < 10:
            continue
        tr_x, va_x = median_impute(tr, va, feature_cols)
        y_tr = tr["y_over"].to_numpy()
        y_va = va["y_over"].to_numpy()

        naive_p_va = naive_prob_over(va["roll_10_mean"].to_numpy(), va["line_value"].to_numpy())

        log_pipe = fit_logistic_baseline(tr_x[feature_cols], y_tr)
        log_p_va = logistic_predict_proba(log_pipe, va_x[feature_cols])

        res = fit_hgb(tr_x, y_tr, feature_columns=feature_cols)
        raw_va = predict_proba(res.model, va_x, res.feature_columns)
        raw_tr = predict_proba(res.model, tr_x, res.feature_columns)
        kind = choose_calibrator_kind(raw_va, y_va, raw_tr, y_tr)
        cal = fit_calibrator(raw_tr, y_tr, kind=kind)
        p_va = cal.predict(raw_va)

        naive_val = eval_report(y_va, naive_p_va).__dict__
        logreg_val = eval_report(y_va, log_p_va).__dict__
        hgb_cal_val = eval_report(y_va, p_va).__dict__
        rows.append(
            {
                "fold": fold.fold_id,
                "n_train": len(tr),
                "n_val": len(va),
                "calibrator": kind,
                "naive_val": naive_val,
                "logreg_val": logreg_val,
                "hgb_cal_val": hgb_cal_val,
            }
        )

        _section(f"Walk-forward fold {fold.fold_id}")
        print(f"  Calibrator: {kind}  |  train rows: {len(tr)}  |  validation rows: {len(va)}")
        _print_validation_metrics_table(naive_val, logreg_val, hgb_cal_val)
        rel = reliability_table(y_va, p_va, n_bins=8)
        _print_reliability(rel)

    _write_metrics_csv(rows, artifact_dir)
    with open(artifact_dir / "demo_walkforward.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print()
    print("Wrote data/artifacts/demo_walkforward.json")


if __name__ == "__main__":
    main()
