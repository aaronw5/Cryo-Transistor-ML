"""I25-1: sky130 77 K pipeline rerun from the stock SkyWater tt card.

Pure helpers (no NGSpice): round-one boxes centered on the stock bin values,
parameter-set overlap, the card-independent secondary curve inclusion
("M-rule"), secondary scoring of a record, GMR labels and the edge fraction
of a fitted box.  Pre-registration: out/research/explore/I25_1/prereg.md.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..cold_start import half_width, spec_from_centers
from ..metrics import (_NMOS_100_INCLUDE_VD, _NMOS_100_INCLUDE_VG, _NMOS_INCLUDE_VD,
                       _NMOS_INCLUDE_VG, _PMOS_SIM_LAST_VD, _PMOS_SIM_LAST_VG, _trim_pair,
                       device_floors, device_rrms_new)

EDGE = 0.9


def stock_round_one_spec(names: Sequence[str], centers: dict[str, dict],
                         rule: str = "stock SkyWater tt bin value (showmod readback), I25-1") -> dict:
    """Round-one box per device: center = the template card's value (stock
    bin, or the multiplier-1 card under amendment A3), the unchanged
    cold-start half-widths (``cold_start.half_width``)."""
    names = tuple(names)
    rows = {tag: {n: float(vals[n]) for n in names} for tag, vals in centers.items()}
    widths = {tag: {n: half_width(n) for n in names} for tag in rows}
    return spec_from_centers(names, rows, widths, extra={
        "round": 1, "center_rule": rule})


def overlap(old: Sequence[str], new: Sequence[str]) -> dict:
    a, b = set(old), set(new)
    return {"n_old": len(a), "n_new": len(b), "n_both": len(a & b),
            "both": sorted(a & b), "only_old": sorted(a - b), "only_new": sorted(b - a),
            "jaccard": len(a & b) / max(len(a | b), 1)}


def m_rule_keep(dev_type: str, L_um: float, W_um: float, kind: str, meas) -> bool:
    """Card-independent inclusion of one metric curve from the measured
    current only: nMOS the upstream measured-mean rule; pMOS the upstream
    thresholds applied to the measured |I| at the last trimmed point."""
    vg_floor, vd_floor = device_floors(dev_type, L_um, W_um)
    floor = vd_floor if kind == "idvd" else vg_floor
    m = np.asarray(meas, dtype=np.float64)
    _, mt = _trim_pair(m, m, floor)
    if len(mt) == 0:
        return False
    mean_abs = float(np.mean(np.abs(mt)))
    if dev_type == "pmos":
        thresh = _PMOS_SIM_LAST_VD if kind == "idvd" else _PMOS_SIM_LAST_VG
        return bool(mean_abs > 0 and abs(float(mt[-1])) > thresh)
    is_n100 = L_um >= 99 and W_um >= 99
    if kind == "idvd":
        lim = _NMOS_100_INCLUDE_VD if is_n100 else _NMOS_INCLUDE_VD
    else:
        lim = _NMOS_100_INCLUDE_VG if is_n100 else _NMOS_INCLUDE_VG
    return bool(mean_abs > lim)


def m_rule_tags(rec) -> set[str]:
    dev_type = rec.meta.get("dev_type", rec.family)
    return {c.key for c in rec.curves
            if m_rule_keep(dev_type, rec.L_um, rec.W_um, c.kind, c.target)}


def score_rule(rec, sims, include: set[str]) -> dict:
    """Paper RRMS of ``sims`` (one per ``rec.curves``) with the curve set
    ``include``."""
    tsim, tmeas = {}, {}
    for c, s in zip(rec.curves, sims):
        kind, bias = c.key.split("@")
        tsim[(kind, float(bias))] = np.asarray(s, dtype=np.float64)
        tmeas[(kind, float(bias))] = np.asarray(c.target, dtype=np.float64)
    r = device_rrms_new(rec.meta.get("dev_type", rec.family), rec.L_um, rec.W_um,
                        tsim, tmeas, include_tags=set(include))
    return {"rrms": float(r["rrms"]), "n_curves": int(r["n_curves"]),
            "per_curve": {k: v["rrms"] for k, v in r["per_curve"].items() if v.get("included")}}


def frozen_tags(rec) -> set[str]:
    return {c.key for c in rec.curves if c.included}


def gmr_label(g: dict) -> str:
    lo, hi = g.get("ci_low"), g.get("ci_high")
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if hi < 1.0:
        return "lower"
    if lo > 1.0:
        return "higher"
    return "inconclusive"


def aggregate(rows: dict[str, float]) -> dict:
    """all-device mean, family means, combined (mean of the two family
    means), median; non-finite rows counted."""
    vals = {t: float(v) for t, v in rows.items()}
    fin = {t: v for t, v in vals.items() if np.isfinite(v)}
    n = [v for t, v in fin.items() if t.startswith("nmos")]
    p = [v for t, v in fin.items() if t.startswith("pmos")]
    nm = float(np.mean(n)) if n else float("nan")
    pm = float(np.mean(p)) if p else float("nan")
    return {"all_device": float(np.mean(list(fin.values()))) if fin else float("nan"),
            "nmos": nm, "pmos": pm, "combined": (nm + pm) / 2.0,
            "median": float(np.median(list(fin.values()))) if fin else float("nan"),
            "n": len(vals), "n_failed": len(vals) - len(fin)}


def edge_fraction(spec: dict, local_by_tag: dict[str, Sequence[float]], edge: float = EDGE) -> dict:
    """Fraction of active (log-scaled, non-degenerate) coordinates with
    |u| >= edge, per device and pooled."""
    names = spec["selected_parameters"]
    per, hits, total = {}, 0, 0
    for tag, local in local_by_tag.items():
        flags = spec["log_scale_by_device"][tag]
        bounds = spec["bounds_by_device"][tag]
        act = [i for i, n in enumerate(names) if flags[n] and bounds[n][1] > bounds[n][0]]
        h = sum(1 for i in act if abs(float(local[i])) >= edge)
        per[tag] = h / max(len(act), 1)
        hits += h
        total += len(act)
    return {"per_device": per, "pooled": hits / max(total, 1), "n_active": total}
