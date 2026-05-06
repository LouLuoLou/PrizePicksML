# Bronze → silver ETL layout

## Directory layout (local default)

```
data/
  bronze/
    {source}/{YYYY}/{MM}/{DD}/
      raw_{batch_id}.jsonl   # immutable append
  silver/
    players.parquet
    teams.parquet
    matches.parquet
    player_game_stats.parquet
    lines.parquet            # optional
  artifacts/
    models/
    calibrators/
```

## Idempotency

- Each ingest batch writes to a new `batch_id` folder; dedupe on `(source, logical_key, content_hash)`.
- Silver builds are **partitioned** by `league_id` and `season` where applicable; rebuild job uses `MERGE` semantics on primary keys.

## Orchestration

- Phase 1: Makefile / `python -m pricepicks_ml.etl` targets: `ingest`, `silver_refresh`, `validate`.
- Validation gates: non-null `match_id`, monotonic `official_start_ts` within season, no duplicate `(player_id, match_id)`.

## Cloud TBD

Swap `data/` roots for `s3://` or `azblob://` URIs via config; keep the same logical tables.
