"""Paper RRMS for pdk2 targets.

Per curve, exactly the ``cryoml.metrics.curve_trim`` recipe: clean the
target (``clean_current``), drop leading points up to the first
``|I| > TRIM_FLOOR`` (one floor for every device and curve: 1e-10 A, above
ngspice's GMIN x VDS off-state floor of ~3e-12 A), denominator = mean |I| of
the trimmed target, curve RRMS = RMSE / denominator; device RRMS = mean over
the included curves.

Inclusion is frozen per target: a curve is scored when its trimmed mean |I|
exceeds the rrmsCalc nMOS thresholds (3e-7 A Id-Vd, 2e-7 A Id-Vg).  It
depends only on the target, never on a candidate, so no fit can change which
curves count.  There is no separate dynamic-inclusion score.
"""

from __future__ import annotations

import math

import numpy as np

from ..metrics import clean_current
from .gf180 import curve_tag

TRIM_FLOOR = 1e-10
INCLUDE_MIN = {"idvd": 3e-7, "idvg": 2e-7}
PENALTY = 10.0          # curve RRMS of a failed simulation (as ngspice_fit)


def curve_trim(kind: str, target: np.ndarray) -> tuple[int, np.ndarray, float]:
    cleaned = clean_current(np.asarray(target, dtype=np.float64))
    above = np.abs(cleaned) > TRIM_FLOOR
    start = int(np.argmax(above)) if above.any() else 0
    den = float(np.mean(np.abs(cleaned[start:]))) if len(cleaned) > start else 0.0
    return start, cleaned, den


def curve_layout(curves) -> list[dict]:
    """Included curves: ``index``, ``start``, ``tag``, ``target`` (trimmed),
    ``denominator`` (the ``DeviceProblem.layout`` row format)."""
    layout = []
    for index, curve in enumerate(curves):
        start, cleaned, den = curve_trim(curve.kind, curve.Id)
        if den > INCLUDE_MIN[curve.kind] and np.isfinite(den):
            layout.append({"index": index, "start": start,
                           "tag": curve_tag(curve),
                           "target": np.asarray(cleaned[start:], np.float64),
                           "denominator": den})
    return layout


def include_tags(curves) -> list[str]:
    return [row["tag"] for row in curve_layout(curves)]


def curve_rrms(sims, layout: list[dict]) -> np.ndarray:
    values = []
    for row in layout:
        sim = np.asarray(sims[row["index"]], dtype=np.float64)[row["start"]:]
        if not np.all(np.isfinite(sim)):
            values.append(PENALTY)
            continue
        rmse = math.sqrt(float(np.mean((sim - row["target"]) ** 2)))
        values.append(rmse / row["denominator"])
    return np.asarray(values)


def device_rrms(sims, curves) -> dict:
    layout = curve_layout(curves)
    values = curve_rrms(sims, layout)
    return {"rrms": float(np.mean(values)) if len(values) else float("nan"),
            "sigma": float(np.std(values)) if len(values) else float("nan"),
            "n_curves": len(values),
            "per_curve": {row["tag"]: float(v) for row, v in zip(layout, values)}}
