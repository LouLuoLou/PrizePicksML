# Robustness slices and ablations

## Robustness slices (reporting plan)

Report **Brier + log loss + reliability** on each slice of the **concatenated forward validation** predictions.

| Slice | Definition |
|-------|------------|
| Low minutes | Prior `roll_10_mean` of minutes `< p25` league distribution |
| Home vs away | Stratify on `is_home` |
| Early season | First 15 team games in season |
| Soccer sparse | Markets like `GOALS` with expected count `< 0.35` in rolling window |

## Ablations (feature groups)

Train/eval the same protocol with feature groups removed:

1. **No opponent:** drop columns matching `opp_allowed_*`.
2. **No line-relative:** drop `mean_minus_line`, `z_gap`, and optionally `line_value` (document if removed).
3. **No context:** drop `rest_days`, `is_b2b`, `matches_last7`.

Expectation: performance should **not** magically improve when removing informative groups; large gains may indicate leakage or overfitting.

## Semester note

[Apr: ablations finally automated in eval script backlog—manual drop list acceptable for report]
