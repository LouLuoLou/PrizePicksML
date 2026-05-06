# Inference checklist and output template

## Inputs required before `official_start_ts`

- [ ] `sport`, `league_id`
- [ ] `player_id`, `primary_position` / `position_bucket`
- [ ] `match_id`, `official_start_ts`, `home_team_id`, `away_team_id`, player `team_id`
- [ ] `market_code` (e.g. `PTS`, `SHOTS`, `HITS`)
- [ ] `line_value` and `line_snapshot_ts` (per policy)
- [ ] Prior boxscore rows for player (last ≥20 games preferred)
- [ ] Schedule context: team games timestamps for rest/B2B/congestion
- [ ] Injury/status: structured fields or explicit `unknown` flag

## Model call

- Build `feature_as_of_ts` per [DATA_DICTIONARY.md](DATA_DICTIONARY.md).
- Run frozen `feature_columns` in model order; apply median imputers from training if used.

## Output template (per prediction)

1. **`P(Over)`** (calibrated)
2. **Discrete side** (`Over` / `Under` / `Abstain`) using `max(p,1-p) >= 0.55` unless overridden
3. **Top drivers:** SHAP for tree models (add `shap` optional dependency) or **coefficients** for logistic baseline
4. **Uncertainty bullets:** minutes volatility, cold start, missing injury, opponent bucket coarse, line policy mismatch

## Semester note

[Apr 14: added SHAP plan—optional dependency to keep base env light]
