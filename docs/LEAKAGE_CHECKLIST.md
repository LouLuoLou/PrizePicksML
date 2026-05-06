# Leakage audit checklist

Use before every submission / model freeze. Sign: name + date.

## A. Temporal integrity

- [ ] Every feature column is computable from data with timestamp **strictly before** `official_start_ts` of the predicted match.
- [ ] Rolling / cumulative stats use only prior games: `prior_game_end_ts < official_start_ts`.
- [ ] Opponent “allowed” aggregates are **shifted**: computed from games **before** the opponent’s game vs our team (not including same match).
- [ ] No post-game boxscore fields from the predicted match appear in `X`.
- [ ] Train/val/test splits are **by date** (walk-forward), not random row shuffle for headline metrics.

## B. Line and market integrity

- [ ] `line_value` used for training matches the **declared** `line_snapshot_ts` policy (e.g. opening vs T-60m); no future line snapshots.
- [ ] If using line movement features, each row uses only snapshots **≤ feature_as_of_ts**.

## C. Target leakage

- [ ] `y_over` derived only from finalized `realized_stat` and `line_value` used in that row (same market).
- [ ] Injuries / scratches after `feature_as_of_ts` are **not** encoded as known facts (only unknown flags allowed).

## D. Join duplication

- [ ] One training row per `(player_id, match_id, market_code, line_snapshot_ts)` or explicit dedupe rule documented.
- [ ] No duplicate counting from multiple books unless modeling explicitly requires it.

## E. Adversarial self-review (“pretend I cheat”)

1. List the **top 10** features by importance; for each, state the **latest data** it could legally see.
2. Artificially set a future stat in a held-out row; confirm pipeline **cannot** access it (unit test).
3. Re-run with **shuffle labels** on same `X`: performance should collapse toward chance on val (sanity).

## F. Semester note

[Feb: we caught same-match opponent defensive rating leaking from the current game—fixed with shift-by-one.]
