# April 15 submission freeze procedure

## Versioning

1. Create git tag: `submission-2026-04-15` (or course-required tag name).
2. Record `git rev-parse HEAD` and `git status` (must be clean or document dirty paths).
3. Export `pip freeze > artifacts/requirements-freeze.txt` from the venv used for final numbers.

## Config hash

- Serialize training config (hyperparameters, feature list, walk-forward months) to `artifacts/train_config.json`.
- `sha256sum artifacts/train_config.json` (or PowerShell `Get-FileHash`) stored in report appendix.

## Model artifact

- Save `model.joblib` + `calibrator.joblib` + `feature_columns.json` + `impute_medians.json` under `data/artifacts/models/apr15/`.

## Reproducible command

Document exact command line, e.g.:

```bash
python scripts/run_demo.py
```

For full training: `python -m pricepicks_ml.train_run --config artifacts/train_config.json` (add script if extended).

## Semester note

[Apr 15: frozen demo config after final reliability pass]
