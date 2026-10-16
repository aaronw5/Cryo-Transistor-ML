"""Re-simulate fixed per-device cards in NGSpice and score them.

Every score uses the frozen published-baseline curve inclusion (primary) and
also records the upstream dynamic-inclusion value.  Aggregates report both the
all-device mean and the upstream combined (family-mean) convention.
"""

from __future__ import annotations

import json
from multiprocessing import get_context

import numpy as np

from .cold_start import BASELINE_JSON, device_bins
from .data_io import load_device_curves
from .devices import PAPER_DEVICES
from .metrics import family_totals, score_device_new
from .spice_pdk import simulate_pdk
from .utils import device_tag

_INCLUDE = None


def include_tags(tag: str) -> set[str]:
    global _INCLUDE
    if _INCLUDE is None:
        per_curve = json.loads(BASELINE_JSON.read_text())["per_curve"]
        _INCLUDE = {key: {t for t, v in rows.items() if v.get("included")}
                    for key, rows in per_curve.items()}
    return _INCLUDE[tag]


def device_by_tag(tag: str):
    for device in PAPER_DEVICES:
        if device_tag(device.dev_type, device.L_um, device.W_um) == tag:
            return device
    raise KeyError(tag)


def score_card(tag: str, params: dict[str, float] | None,
               curves=None, bin_index: int | None = None,
               return_sims: bool = False) -> dict:
    device = device_by_tag(tag)
    curves = load_device_curves(device) if curves is None else curves
    if bin_index is None:
        bin_index = device_bins()[tag]
    sims = simulate_pdk(device.dev_type, device.L_um, device.W_um, curves,
                        params=params or None, bin_index=bin_index)
    fixed = score_device_new(device.dev_type, device.L_um, device.W_um,
                             curves, sims, include_tags=include_tags(tag))
    dynamic = score_device_new(device.dev_type, device.L_um, device.W_um,
                               curves, sims)
    out = {"rrms": float(fixed["rrms"]), "sigma": float(fixed["sigma"]),
           "n_curves": int(fixed["n_curves"]),
           "per_curve": {k: v["rrms"] for k, v in fixed["per_curve"].items()
                         if v.get("included")},
           "official_dynamic_rrms": float(dynamic["rrms"])}
    if not np.isfinite(out["rrms"]):
        out["rrms"] = float("inf")
    if return_sims:
        out["sims"] = sims
    return out


def _score_job(job):
    tag, params = job
    return tag, score_card(tag, params)


def score_cards(cards: dict[str, dict[str, float] | None],
                workers: int = 6) -> dict:
    """``cards`` maps device tag -> parameter overrides (None = released)."""
    jobs = list(cards.items())
    if workers <= 1:
        results = dict(map(_score_job, jobs))
    else:
        with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            results = dict(pool.map(_score_job, jobs))
    return {"devices": results, **aggregate(results)}


def aggregate(results: dict[str, dict]) -> dict:
    values = [row["rrms"] for row in results.values()]
    totals = family_totals({tag: {"sigma": float("nan"), **row}
                            for tag, row in results.items()})
    return {
        "all_device_mean_rrms": float(np.mean(values)),
        "nmos_rrms": totals["nmos_rrms"], "pmos_rrms": totals["pmos_rrms"],
        "combined_rrms": totals["combined_rrms"],
        "official_dynamic_all_device_mean": float(np.mean(
            [row.get("official_dynamic_rrms", np.nan)
             for row in results.values()])),
        "n_devices": len(results),
    }
