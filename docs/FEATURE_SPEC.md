# Feature specification (rolling, time-adjusted, opponent, context, line-relative)

## 1. Rolling history (all sports)

**Windows:** last `N` games where `N ∈ {5, 10, 20}` plus optional season-to-date (same logic with larger N).

**Per window compute:**

- Mean, median, std (population)
- Robust spread: IQR = `p75 - p25` (use `numpy.percentile`)
- **Recency weights:** exponential weights `w_i = exp(-λ * age_games)` with tunable `λ` (default `0.15` on game index age).

**Eligibility:** only games with `official_start_ts < target_official_start_ts`. Sort ascending before rolling.

## 2. Minutes-adjusted rates

| Sport | Adjustment | Example |
|-------|------------|---------|
| Basketball | per-36 minutes | `pts_per36 = pts * 36 / max(minutes, ε)` |
| Soccer | per-90 | `shots_per90 = shots * 90 / max(minutes, ε)` |
| Baseball (hitting) | per PA if PA tracked; else per game | document if PA missing |
| Baseball (pitching) | per inning or per BF | market-dependent |

Use `ε = 1e-3` to avoid division blowups; **cap** rates at sport-specific max if needed for stability.

## 3. Opponent defense (shifted rolling “allowed”)

For opponent team `O` when predicting player `P` in match `M`:

1. Take all **prior** games of `O` (before `M.official_start_ts`).
2. Among opponent games, aggregate **allowed** stat to **position_bucket** of `P` (coarse buckets acceptable: e.g. `G/F/C`, `IF/OF/C`, `DEF/MID/FWD`).
3. Apply same window sizes (5/10/20) on allowed series.

**Shift-by-one-game rule:** the game immediately before `M` for team `O` is included only if it ended before `M` starts (natural ordering); never use stats from the concurrent match.

## 4. Context: rest, travel, congestion, home/away

| Feature | BB | MLB | Soccer |
|---------|----|----|--------|
| `is_home` | 1 if player team is home | same | same |
| `rest_days` | days since last team game | days since last game (team) | days since last match |
| `is_b2b` | 1 if `rest_days == 1` | rare; optional | congestion: matches in last 7 days |

**Missingness:** if schedule gap unknown, set `rest_days = median_league` and add `rest_days_imputed = 1` indicator.

## 5. Line-relative features

- `mean_minus_line = rolling_mean_stat - line_value`
- `z_gap = (rolling_mean_stat - line_value) / max(rolling_std_stat, ε_std)` with `ε_std = 0.5` league units for counts.

**Optional deferred:** `line_delta` between two snapshots requires clean `lines` history—see PRD non-goals.

## 6. Role stability

- `minutes_vol_10` = rolling std of minutes over last 10
- `start_rate_10` = fraction started last 10

## Semester notes

[Feb: dropped season-only averages] [Mar: added opponent allowed + per-36] [Apr: standardized z_gap epsilon]
