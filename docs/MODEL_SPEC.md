# Primary model specification (GBDT + calibration)

## Primary learner

**Implementation:** `sklearn.ensemble.HistGradientBoostingClassifier` in [src/pricepicks_ml/train.py](../src/pricepicks_ml/train.py).

**Rationale:** Native handling of numeric features, strong tabular performance, fewer extra native deps than some GBDT libraries while remaining portable for coursework. **Optional upgrade:** swap estimator for **LightGBM** or **XGBoost** if your environment supports them; keep the same walk-forward outer loop.

## Hyperparameter search protocol (forward data only)

1. Within each **outer walk-forward train window**, reserve the **last chronological 20%** of rows as **inner validation** (or last N days).
2. Run small grid on inner val only, e.g.  
   - `learning_rate ∈ {0.03, 0.06, 0.1}`  
   - `max_depth ∈ {4, 6, None}`  
   - `l2_regularization ∈ {0.1, 1.0, 10.0}`  
3. Choose by **lowest inner Brier score** (or log loss).
4. Refit on **full outer train** with chosen hyperparameters; evaluate on **outer val month** (never used for selection).

`HistGradientBoostingClassifier` also uses `early_stopping=True` with `validation_fraction` (random sub-split of train) for iteration control—acceptable for demos; for final report prefer inner time-based val only and set `early_stopping=False` after tuning.

## Calibration

Post-hoc on **train predictions** vs `y_train` (demo path): [src/pricepicks_ml/calibration_util.py](../src/pricepicks_ml/calibration_util.py).

**Selection rule:** `choose_calibrator_kind` compares **isotonic** vs **Platt** on the **outer validation fold** using Brier score, fit on train probabilities (see demo script). **Production recommendation:** fit calibrator on **out-of-fold** train predictions or a dedicated calibration slice to reduce optimism.

## Artifacts

- Serialized model: `joblib` or `skops` (add if needed) under `data/artifacts/models/`.
- Calibrator bundle + `feature_columns.json` alongside.

## Semester note

[Mar: picked HGB over deep nets for auditability + tabular focus]
