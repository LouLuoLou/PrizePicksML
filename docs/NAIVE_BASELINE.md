# Naive rolling-mean vs line baseline

## Definition

For each row, use prior rolling mean \(\hat{\mu}\) of the stat (default window **10 games**, column `roll_10_mean`) and posted line \(L\).

**Heuristic probability (no training):**

\[
P(\text{Over}) = \sigma\big(\alpha \cdot (\hat{\mu} - L)\big)
\]

with \(\sigma\) the logistic sigmoid and **default** \(\alpha = 0.35\) (tunable in [src/pricepicks_ml/baselines.py](../src/pricepicks_ml/baselines.py) as `naive_prob_over(..., scale=0.35)`).

## Discrete decision rule

- `Over` if `P(Over) >= 0.55`
- `Under` if `P(Over) <= 0.45`
- Else **abstain** / no call (`naive_side` returns `-1` for abstain in code)

## Limitations

Ignores opponent, minutes volatility, and calibration to market; included only as a **sanity floor**.

## Semester note

[Jan: started with raw mean > line rule; upgraded to sigmoid for probabilistic metrics]
