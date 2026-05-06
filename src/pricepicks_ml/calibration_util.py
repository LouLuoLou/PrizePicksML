"""Probability calibration (isotonic vs Platt) with simple diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


Kind = Literal["isotonic", "platt"]


@dataclass
class CalibratorBundle:
    kind: Kind
    iso: IsotonicRegression | None = None
    platt: LogisticRegression | None = None

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
        if self.kind == "isotonic" and self.iso is not None:
            return np.clip(self.iso.predict(p), 1e-6, 1 - 1e-6)
        if self.kind == "platt" and self.platt is not None:
            logits = np.log(p / (1 - p)).reshape(-1, 1)
            return np.clip(self.platt.predict_proba(logits)[:, 1], 1e-6, 1 - 1e-6)
        return p


def fit_calibrator(p_train: np.ndarray, y_train: np.ndarray, *, kind: Kind) -> CalibratorBundle:
    y = y_train.astype(int)
    p = np.clip(p_train.astype(float), 1e-6, 1 - 1e-6)
    if kind == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        return CalibratorBundle(kind="isotonic", iso=iso)
    lr = LogisticRegression(max_iter=200, solver="lbfgs")
    lr.fit(np.log(p / (1 - p)).reshape(-1, 1), y)
    return CalibratorBundle(kind="platt", platt=lr)


def choose_calibrator_kind(p_val: np.ndarray, y_val: np.ndarray, p_train: np.ndarray, y_train: np.ndarray) -> Kind:
    """Pick lower Brier on validation between isotonic and Platt."""
    from pricepicks_ml.evaluate import brier_score

    iso = fit_calibrator(p_train, y_train, kind="isotonic")
    platt = fit_calibrator(p_train, y_train, kind="platt")
    bi = brier_score(y_val, iso.predict(p_val))
    bp = brier_score(y_val, platt.predict(p_val))
    return "isotonic" if bi <= bp else "platt"


def sklearn_calibrated_classifier(base_estimator, X, y, *, method: str = "isotonic") -> CalibratedClassifierCV:
    """Sklearn wrapper for CV-calibrated classifier (alternative path)."""
    return CalibratedClassifierCV(base_estimator, method=method, cv=3).fit(X, y)
