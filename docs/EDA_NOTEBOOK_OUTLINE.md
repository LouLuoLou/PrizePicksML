# EDA + baseline notebook outline (`notebooks/00_eda_baseline.ipynb`)

## Cells 1–2: Imports and load silver tables

- `pandas` read `data/silver/*.parquet` (or CSV)  
- Print row counts, date ranges, missingness heatmap (optional)

## Cells 3–4: Label distribution

- Histogram of `realized_stat` by `market_code`  
- Push rate table  
- Class balance for `y_over`

## Cells 5–6: Rolling sanity

- Example player: rolling mean vs realized for `PTS`  
- Correlation matrix of engineered features (subsample)

## Cells 7–8: Baselines

- Naive `P(Over)` vs logistic vs HGB on **single time split** (not for headline—teaching only)  
- Compare to walk-forward script for real metrics

## Cell 9: Leakage spot checks

- Verify `official_start_ts` monotonic joins  
- “Max future timestamp in features” assert per row

## Semester note

[Jan–Feb: EDA caught duplicate match rows before silver MERGE]
