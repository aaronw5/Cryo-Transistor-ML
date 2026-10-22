"""I17-1: re-registration of the I16-1 Part B test of the I15-2 selected
correction (``full|sym_e1_l1``, lambda 1 V) with the Q1 V(d,s) = 0 check
evaluated at tight tolerance only (threshold 1e-12 A kept).  The check was
changed after the observed I16-1 failure; the I16-1 outcome stays 'not
deployable'.

Pre-registration: ``out/research/explore/I17_1/prereg.md`` / ``prereg.json``.

Pure helpers: the V(d,s) = 0 values per tolerance from the forward Q1 grid
arrays, the Part B decision under the I17-1 rule (every other check is
``i12_1_symm.decide_deploy`` unchanged) and under the unchanged I16-1 rule,
and the outcome table.
"""

from __future__ import annotations

import numpy as np

from . import i12_1_symm as v2

VDS0_TOL = "tight"           # I17-1: the V(d,s) = 0 check uses this tolerance only
VDS0_THR = 1e-12             # A, unchanged from I16-1 (v2.THR["q1_vds0_abs"])
TOLS = ("default", "r4", "tight")


def vds0_column(grid_vds) -> int:
    """Index of the V(d,s) = 0 column of the Q1 grid (must be exactly 0)."""
    g = np.asarray(grid_vds, float)
    j = int(np.argmin(np.abs(g)))
    if g[j] != 0.0:
        raise ValueError(f"grid has no V(d,s) = 0 column (closest {g[j]})")
    return j


def vds0_values(arrays: dict, j0: int, grid_vgs) -> dict:
    """Per tolerance: max |I| of the element run and of the card + replica
    run at the V(d,s) = 0 column of the forward grid, the |V(g,s)| at the
    element maximum and the number of points above ``VDS0_THR``."""
    out = {}
    for tol in TOLS:
        el = np.abs(np.asarray(arrays[f"grid_fwd_{tol}_elem"], float)[:, j0])
        cr = np.abs(np.asarray(arrays[f"grid_fwd_{tol}_card_rep"], float)[:, j0])
        k = int(np.nanargmax(el))
        out[tol] = {"elem_max": float(np.nanmax(el)), "vgs_abs_at_max": float(np.asarray(grid_vgs)[k]),
                    "card_rep_max": float(np.nanmax(cr)), "n_above": int(np.sum(el > VDS0_THR)),
                    "n_points": int(el.size), "n_nonfinite": int(np.sum(~np.isfinite(el)))}
    return out


def summarize_vds0(per_device: dict) -> dict:
    """Max over devices per tolerance, the device / V(g,s) of the maximum."""
    out = {}
    for tol in TOLS:
        best = max(per_device, key=lambda t: per_device[t][tol]["elem_max"])
        out[tol] = {"elem_max": per_device[best][tol]["elem_max"], "at": [best, per_device[best][tol]["vgs_abs_at_max"]],
                    "card_rep_max": max(v[tol]["card_rep_max"] for v in per_device.values()),
                    "n_above": sum(v[tol]["n_above"] for v in per_device.values()),
                    "n_points": sum(v[tol]["n_points"] for v in per_device.values()),
                    "n_nonfinite": sum(v[tol]["n_nonfinite"] for v in per_device.values())}
    return out


def values_i16_rule(values: dict, vds0_summary: dict) -> dict:
    """Part B values with q1_vds0_max = max over default and tight (I16-1)."""
    return {**values, "q1_vds0_max": max(vds0_summary["default"]["elem_max"], vds0_summary["tight"]["elem_max"])}


def values_i17_rule(values: dict, vds0_summary: dict) -> dict:
    """Part B values with q1_vds0_max = tight tolerance only (I17-1); a
    non-finite tight value fails the check."""
    t = vds0_summary[VDS0_TOL]
    v = t["elem_max"] if t["n_nonfinite"] == 0 else float("inf")
    return {**values, "q1_vds0_max": v}


def decide_deploy_i17(values: dict, vds0_summary: dict) -> dict:
    """``v2.decide_deploy`` with the I17-1 V(d,s) = 0 value; also returns
    the label under the unchanged I16-1 rule (reported, no decision)."""
    assert v2.THR["q1_vds0_abs"] == VDS0_THR
    dec = v2.decide_deploy(values_i17_rule(values, vds0_summary))
    old = v2.decide_deploy(values_i16_rule(values, vds0_summary))
    return {**dec, "under_I16_1_rule": old}


def decide_outcome(validity: dict, dep: dict) -> dict:
    """I17-1 outcome table: invalid / deployable / not deployable."""
    if not validity["valid"]:
        return {"outcome": "invalid", "action": "stop: fix and re-register"}
    if dep["deploy"] == "not deployable":
        return {"outcome": "not deployable", "action": "stop; diagnose the failed core checks"}
    return {"outcome": "deployable", "label": dep["deploy"], "restrictions": list(dep["restrictions"]),
            "action": ("the I15-2 selected variant is the deployable add-on candidate (the V(d,s) = 0 check "
                       "was re-registered after an observed failure); report restrictions")}
