# Logistic regression baseline (interpretable)

## Purpose

A **sparse, standardized logistic regression** provides an interpretable ceiling/floor and sanity check for the GBDT model.

## Specification

- **Pipeline:** `StandardScaler` → `LogisticRegression` (L2, `lbfgs`, `class_weight='balanced'`, `max_iter=200`).
- **Features:** same numeric matrix as the GBDT (see `scripts/run_demo.py` column filter), after median imputation from train fold.
- **Output:** `predict_proba[:, 1]` as `P(Over)` (uncalibrated unless you wrap with calibration).

## Interpretation deliverable

For coursework, export **coefficients × feature name** sorted by magnitude for one fold.

## Semester note

[Feb: added logreg after naive sigmoid baseline to test linear separability of engineered features]
