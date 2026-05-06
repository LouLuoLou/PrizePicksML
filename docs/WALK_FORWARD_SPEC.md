# Walk-forward validation specification

## Objective

Estimate out-of-sample calibration and discrimination **without peeking** at future seasons/months when choosing hyperparameters for reporting.

## Default protocol (monthly validation blocks)

1. Sort all training rows by `official_start_ts`.
2. **Expanding train:** first train window starts at season start (or min date).
3. **Validation:** next calendar month (or fixed day span, e.g. 21 days) immediately following train max date.
4. Slide forward: train max expands to end of previous val; repeat until data ends.
5. **Embargo:** optional `embargo_days` gap between train end and val start to mimic operational latency (default `0`; set to `1` if simulating late boxscore ingestion).

## Hyperparameter tuning rule

- Tune only **inside each train fold** (inner time split or last chunk of train as inner-val).
- **Never** tune on the forward test month that is reserved for reporting.

## Alternatives (document if used)

- **Rolling train window:** last `K` days only (non-stationary leagues).
- **Seasonal folds:** train on season `S`, validate on `S+1` opener months.

## Reporting

- Aggregate metrics: mean ± std across folds (optional), plus **concatenated** val predictions for calibration plots.
- Store per-fold: train date range, val date range, row counts, push exclusion counts.

## Semester note

[Mar: switched from random 80/20 to monthly walk-forward after inflated AUC.]
