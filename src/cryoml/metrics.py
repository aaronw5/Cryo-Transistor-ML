"""RRMS metrics.

Two metric families live here:

* the legacy paper-companion-notebook all-curve RRMS (``per_curve_rrms`` /
  ``device_rrms``), kept for continuity, and
* the confirmed-setup metric ported from the new repository's
  ``rrmsCalc.py`` (``device_rrms_new`` and helpers): the same per-curve
  RRMS = RMSE / mean|I_meas|, but computed on 11 fixed curves per device
  with measured-glitch cleaning, per-device start-trim floors, and
  low-current curve-exclusion rules.
"""

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


# ---------------------------------------------------------------------------
# Confirmed-setup metric (port of CryoPDK_Skywater130nm_ML/rrmsCalc.py)
# ---------------------------------------------------------------------------

# The 11 scored curves per device, keyed by (kind, |fixed bias|).
VD_METRIC_BIASES: tuple[float, ...] = (0.37, 0.74, 1.11, 1.48, 1.85)
VG_METRIC_BIASES: tuple[float, ...] = (0.01, 0.37, 0.74, 1.11, 1.48, 1.85)

# Per-device (vg_floor, vd_floor) start-trim current thresholds for pMOS,
# keyed by (L_um, W_um) — copied from rrmsCalc.py's pmos_geometries table.
PMOS_FLOORS: dict[tuple[float, float], tuple[float, float]] = {
    (0.35, 0.55): (1e-10, 0.0),
    (0.35, 1.6): (6e-10, 0.0),
    (0.35, 5.0): (6e-6, 1e-7),
    (0.5, 0.42): (1e-7, 5e-8),
    (0.5, 0.64): (0.0, 0.0),
    (2.0, 5.0): (6e-10, 0.0),
    (4.0, 7.0): (1.2e-9, 0.75),
    (8.0, 0.84): (1.8e-9, 0.0),
    (8.0, 1.6): (1e-9, 0.0),
    (8.0, 5.0): (4.5e-9, 0.0),
}

# nMOS curve-inclusion thresholds on mean|I_meas| of the trimmed curve
# (rrmsCalc.py: generic values, with L100/W100 special-cased).
_NMOS_INCLUDE_VD = 3e-7
_NMOS_INCLUDE_VG = 2e-7
_NMOS_100_INCLUDE_VD = 3e-5
_NMOS_100_INCLUDE_VG = 9e-10

# pMOS curve-inclusion thresholds on |I_sim| at the last trimmed point.
_PMOS_SIM_LAST_VD = 1e-10
_PMOS_SIM_LAST_VG = 1e-11


def clean_current(i_meas: np.ndarray) -> np.ndarray:
    """rrmsCalc.py cleanCurrent: zero every point before the *last* exact
    zero of the measured curve (kills SMU glitch spikes on off-curves).
    No zeros -> unchanged. Operates on a copy."""
    out = np.array(i_meas, dtype=np.float64, copy=True)
    zero_idx = np.flatnonzero(out == 0.0)
    if zero_idx.size:
        out[: int(zero_idx[-1])] = 0.0
    return out


def _trim_pair(sim: np.ndarray, meas: np.ndarray,
               floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Clean the measured curve, drop leading points up to the first sample
    with |I_meas| > floor, and truncate both arrays to equal length."""
    meas = clean_current(meas)
    above = np.abs(meas) > floor
    start = int(np.argmax(above)) if above.any() else 0
    sim_t, meas_t = np.asarray(sim, dtype=np.float64)[start:], meas[start:]
    n = min(len(sim_t), len(meas_t))
    return sim_t[:n], meas_t[:n]


def device_floors(dev_type: str, L_um: float, W_um: float) -> tuple[float, float]:
    """(vg_floor, vd_floor) start-trim thresholds for a device."""
    if dev_type != "pmos":
        return 0.0, 0.0
    key = min(PMOS_FLOORS, key=lambda k: abs(k[0] - L_um) + abs(k[1] - W_um))
    return PMOS_FLOORS[key]


def curve_trim(dev_type: str, L_um: float, W_um: float, kind: str,
               meas: np.ndarray) -> tuple[int, np.ndarray, float]:
    """(start index, cleaned measured curve, trimmed denominator) for one
    metric curve — the shared preprocessing used by both the scorer and the
    optimization residuals."""
    vg_floor, vd_floor = device_floors(dev_type, L_um, W_um)
    floor = vd_floor if kind == "idvd" else vg_floor
    cleaned = clean_current(np.asarray(meas, dtype=np.float64))
    above = np.abs(cleaned) > floor
    start = int(np.argmax(above)) if above.any() else 0
    den = float(np.mean(np.abs(cleaned[start:]))) if len(cleaned) > start else 0.0
    return start, cleaned, den


def curve_rrms_new(sim: np.ndarray, meas: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((np.asarray(sim, dtype=np.float64)
                                  - np.asarray(meas, dtype=np.float64)) ** 2)))
    mean_abs = float(np.mean(np.abs(meas)))
    return rmse / mean_abs if mean_abs > 0 else float("nan")


def device_rrms_new(
    dev_type: str,
    L_um: float,
    W_um: float,
    tagged_sim: dict[tuple[str, float], np.ndarray],
    tagged_meas: dict[tuple[str, float], np.ndarray],
    include_tags: set[str] | None = None,
) -> dict:
    """Confirmed-setup device score.

    ``tagged_sim`` / ``tagged_meas`` map ``(kind, |fixed bias|)`` — kind in
    {"idvd", "idvg"}, bias one of the metric biases — to same-grid current
    arrays (signed or magnitude convention, as long as both match).
    Returns rrms (mean over included curves), sigma (population std over
    included curves), per-curve values and the inclusion mask.

    ``include_tags`` freezes the curve-inclusion set (tags like
    ``"idvd@0.37"``): thresholds are bypassed and exactly those curves are
    scored. Used during optimization/selection so a candidate cannot game
    the sim-dependent exclusion rules; the official dynamic-inclusion score
    is this function with ``include_tags=None``.
    """
    vg_floor, vd_floor = device_floors(dev_type, L_um, W_um)
    is_n100 = dev_type == "nmos" and L_um >= 99 and W_um >= 99

    per_curve: dict[str, dict] = {}
    included: list[float] = []
    rmses: list[float] = []
    for kind, biases, floor in (("idvd", VD_METRIC_BIASES, vd_floor),
                                ("idvg", VG_METRIC_BIASES, vg_floor)):
        for b in biases:
            tag = f"{kind}@{b:g}"
            if (kind, b) not in tagged_sim or (kind, b) not in tagged_meas:
                per_curve[tag] = {"rrms": None, "included": False,
                                  "reason": "missing"}
                continue
            sim_t, meas_t = _trim_pair(tagged_sim[(kind, b)],
                                       tagged_meas[(kind, b)], floor)
            if len(meas_t) == 0:
                per_curve[tag] = {"rrms": None, "included": False,
                                  "reason": "empty"}
                continue
            mean_abs = float(np.mean(np.abs(meas_t)))
            if include_tags is not None:
                keep = tag in include_tags and mean_abs > 0
            elif dev_type == "pmos":
                sim_last = float(abs(sim_t[-1])) if len(sim_t) else 0.0
                thresh = _PMOS_SIM_LAST_VD if kind == "idvd" else _PMOS_SIM_LAST_VG
                keep = mean_abs > 0 and sim_last > thresh
            else:
                if kind == "idvd":
                    lim = _NMOS_100_INCLUDE_VD if is_n100 else _NMOS_INCLUDE_VD
                else:
                    lim = _NMOS_100_INCLUDE_VG if is_n100 else _NMOS_INCLUDE_VG
                keep = mean_abs > lim
            rrms = curve_rrms_new(sim_t, meas_t) if keep else None
            per_curve[tag] = {"rrms": rrms, "included": bool(keep),
                              "mean_abs_meas": mean_abs}
            if keep and rrms is not None and np.isfinite(rrms):
                included.append(rrms)
                rmses.append(rrms * mean_abs)
    if included:
        arr = np.asarray(included)
        rrms_bar = float(np.mean(arr))
        sigma = float(np.sqrt(np.mean((arr - rrms_bar) ** 2)))
        rmse_uA = float(np.mean(rmses) * 1e6)
    else:
        rrms_bar, sigma, rmse_uA = float("nan"), float("nan"), float("nan")
    return {
        "rrms": rrms_bar,
        "sigma": sigma,
        "rmse_uA": rmse_uA,
        "n_curves": len(included),
        "per_curve": per_curve,
    }


def tag_metric_curves(curves) -> dict[tuple[str, float], object]:
    """Select the 11 scored curves from a device's curve list by
    (kind, |fixed bias|); extra curves are ignored."""
    out: dict[tuple[str, float], object] = {}
    for kind, biases in (("idvd", VD_METRIC_BIASES), ("idvg", VG_METRIC_BIASES)):
        for b in biases:
            for c in curves:
                if c.kind == kind and abs(abs(float(c.fixed)) - b) < 5e-3:
                    out[(kind, b)] = c
                    break
    return out


def score_device_new(dev_type: str, L_um: float, W_um: float,
                     curves, sims, include_tags: set[str] | None = None) -> dict:
    """Confirmed-setup score for parallel lists of measured Curves and
    simulated current arrays (same order/grids)."""
    tagged = tag_metric_curves(curves)
    index = {id(c): i for i, c in enumerate(curves)}
    tsim = {k: np.asarray(sims[index[id(c)]], dtype=np.float64)
            for k, c in tagged.items()}
    tmeas = {k: np.asarray(c.Id, dtype=np.float64) for k, c in tagged.items()}
    return device_rrms_new(dev_type, L_um, W_um, tsim, tmeas,
                           include_tags=include_tags)


def family_totals(device_results: dict[str, dict]) -> dict:
    """rrmsCalc.py totals: per-family mean of device scores, combined =
    average of the two family means (same for sigma)."""
    fams: dict[str, list[dict]] = {"nmos": [], "pmos": []}
    for tag, res in device_results.items():
        fam = "nmos" if tag.startswith("nmos") else "pmos"
        fams[fam].append(res)
    out: dict = {}
    for fam, rows in fams.items():
        vals = [r["rrms"] for r in rows if np.isfinite(r["rrms"])]
        sigs = [r["sigma"] for r in rows if np.isfinite(r["sigma"])]
        out[f"{fam}_rrms"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{fam}_sigma"] = float(np.mean(sigs)) if sigs else float("nan")
        out[f"{fam}_n_devices"] = len(vals)
    out["combined_rrms"] = float(
        (out["nmos_rrms"] + out["pmos_rrms"]) / 2.0)
    out["combined_sigma"] = float(
        (out["nmos_sigma"] + out["pmos_sigma"]) / 2.0)
    return out
