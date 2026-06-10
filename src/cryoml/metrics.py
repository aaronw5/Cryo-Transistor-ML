"""Paper-exact RRMS metrics from the companion RMS_functions.ipynb."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CurveMetric:
    rrms: float
    rmse_uA: float
    mean_abs_meas: float


def per_curve_rrms(sim: np.ndarray, meas: np.ndarray) -> CurveMetric:
    """Equation 5.1, including the companion notebook's zero convention."""
    sim = np.asarray(sim, dtype=np.float64)
    meas = np.asarray(meas, dtype=np.float64)
    n = min(len(sim), len(meas))
    if n == 0:
        return CurveMetric(np.nan, np.nan, 0.0)
    sim = sim[:n]
    meas = meas[:n]
    rmse = float(np.sqrt(np.mean((sim - meas) ** 2)))
    mean_abs = float(np.mean(np.abs(meas)))
    rrms = 0.0 if mean_abs == 0.0 else rmse / mean_abs
    return CurveMetric(rrms=rrms, rmse_uA=rmse * 1e6, mean_abs_meas=mean_abs)


def device_rrms(
    sim_curves: list[np.ndarray],
    meas_curves: list[np.ndarray],
) -> dict[str, float | int]:
    """Equations 5.2-5.3: mean and standard deviation over every curve."""
    metrics = [per_curve_rrms(s, m) for s, m in zip(sim_curves, meas_curves)]
    finite = [m for m in metrics if np.isfinite(m.rrms)]
    if not finite:
        return {
            "rrms": float("nan"),
            "sigma": float("nan"),
            "rmse_uA": float("nan"),
            "n_curves": 0,
        }
    return {
        "rrms": float(np.mean([m.rrms for m in finite])),
        "sigma": float(np.std([m.rrms for m in finite])),
        "rmse_uA": float(np.mean([m.rmse_uA for m in finite])),
        "n_curves": len(finite),
    }
