"""Minutes / playing-time adjusted rates."""

from __future__ import annotations

import numpy as np


def per_36(stat: float, minutes: float, eps: float = 1e-3) -> float:
    return float(stat * 36.0 / max(minutes, eps))


def per_90(stat: float, minutes: float, eps: float = 1e-3) -> float:
    return float(stat * 90.0 / max(minutes, eps))


def per_pa(stat: float, pa: float, eps: float = 1e-3) -> float:
    return float(stat / max(pa, eps))
