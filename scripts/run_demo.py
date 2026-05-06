#!/usr/bin/env python3
"""End-to-end demo: synthetic BB data, features, walk-forward, HGB + calibration, metrics."""

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

from pricepicks_ml.baselines import fit_logistic_baseline, logistic_predict_proba, naive_prob_over
from pricepicks_ml.calibration_util import choose_calibrator_kind, fit_calibrator
from pricepicks_ml.evaluate import eval_report, reliability_table
from pricepicks_ml.features.builder import attach_line_features, build_feature_matrix
from pricepicks_ml.labels import LabelReport, attach_labels, drop_pushes, label_report
from pricepicks_ml.synthetic import generate_basketball_demo
from pricepicks_ml.train import fit_hgb, predict_proba
from pricepicks_ml.walk_forward import all_folds, split_frame


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


def main() -> None:
    games, lines, team_lookup = generate_basketball_demo()
    raw_feats = build_feature_matrix(
        games,
        stat_col="stat_pts",
        sport="basketball",
        team_games_lookup=team_lookup,
    )
    df = attach_line_features(raw_feats, lines, stat_col="stat_pts", line_col="line_value")
    df = attach_labels(df, col_realized="realized_stat", col_line="line_value")
    df = drop_pushes(df)
    rep = label_report(df["y_over"].tolist())
    _print_label_block(rep)

    folds = all_folds(df, "official_start_ts", embargo_days=0)
    if len(folds) < 2:
        print("Not enough months for walk-forward; extend synthetic span.")
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

        naive_p = naive_prob_over(tr["roll_10_mean"].to_numpy(), tr["line_value"].to_numpy())
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

    Path("data/artifacts").mkdir(parents=True, exist_ok=True)
    with open("data/artifacts/demo_walkforward.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print()
    print("Wrote data/artifacts/demo_walkforward.json")


if __name__ == "__main__":
    main()
