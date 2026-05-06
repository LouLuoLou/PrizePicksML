# PricePicksML

Educational **multi-sport player prop Over/Under** modeling pipeline: leakage-safe rolling features, walk-forward validation, baselines, **HistGradientBoosting** classifier, and **Platt / isotonic** calibration. **Not wagering advice.**

## Quickstart

```bash
cd PricePicksML
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python scripts/run_demo.py
```

The demo prints a short **report** to the terminal: a **Dataset / labels** summary, then for each walk-forward month a **Validation metrics** table (naive vs logistic vs calibrated HGB: Brier, log loss, ROC-AUC) and a **Reliability (calibrated HGB)** bin table (mean predicted vs empirical Over rate). The same numbers are still saved as JSON for notebooks or tooling.

Artifacts: `data/artifacts/demo_walkforward.json`

## Layout

- [docs/PRD.md](docs/PRD.md) — scope, ethics, label policy  
- [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) — entities and as-of rules  
- [docs/FEATURE_SPEC.md](docs/FEATURE_SPEC.md) — rolling, opponent, context, line-relative  
- [docs/WALK_FORWARD_SPEC.md](docs/WALK_FORWARD_SPEC.md) — time splits  
- [docs/LEAKAGE_CHECKLIST.md](docs/LEAKAGE_CHECKLIST.md) — audit before submission  
- `src/pricepicks_ml/` — implementation  
- [notebooks/01_eval_metrics.ipynb](notebooks/01_eval_metrics.ipynb) — metrics walkthrough  

## Semester framing

[Jan: repo + PRD] [Feb: labels + leakage checklist] [Mar: features + calibration] [Apr 15: freeze doc + demo run]
