"""I15-2: which g v2 training change costs accuracy (symmetric inputs, smooth
abs EPS, gm/gds penalty lambda), and a training-data-only nested selection of
lambda.  Pure helpers (no simulation, no training).

Pre-registration: ``out/research/explore/I15_2/prereg.md`` / ``prereg.json``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

NI_MARGIN = 1.05
GRID_LAMBDA = (0.0, 1.0, 2.0, 5.0, 10.0)
EPS_ON = 0.01
N_INNER = 3
TIE_TOL = 1e-12

# name -> (inputs, eps, lambda); "stored" variants are read from I13-1 outputs
VARIANTS = {
    "v1": ("abs_v1", None, 0.0),
    "sym_e0_l0": ("sym_v2", 0.0, 0.0),
    "sym_e1_l0": ("sym_v2", EPS_ON, 0.0),
    "sym_e0_l10": ("sym_v2", 0.0, 10.0),
    "sym_e1_l10": ("sym_v2", EPS_ON, 10.0),
    "sym_e1_l1": ("sym_v2", EPS_ON, 1.0),
    "sym_e1_l2": ("sym_v2", EPS_ON, 2.0),
    "sym_e1_l5": ("sym_v2", EPS_ON, 5.0),
}
STORED = {"v1": ("all18", "M_bias"), "sym_e1_l10": ("lodo", "M_bias_v2")}
NEW = tuple(v for v in VARIANTS if v not in STORED)

CONTRASTS = {
    "inputs_abs_to_sym": [("sym_e0_l0", "v1")],
    "eps_0_to_0.01": [("sym_e1_l0", "sym_e0_l0"), ("sym_e1_l10", "sym_e0_l10")],
    "penalty_0_to_10": [("sym_e0_l10", "sym_e0_l0"), ("sym_e1_l10", "sym_e1_l0")],
}
LAMBDA_CURVE = ([(f"sym_e1_l{k}", "sym_e1_l0") for k in (1, 2, 5, 10)]
                + [(f"sym_e1_l{k}", "v1") for k in (0, 1, 2, 5, 10)])


def variant_for_lambda(lam: float) -> str:
    """Grid lambda -> the symmetric EPS 0.01 variant name."""
    lam = float(lam)
    if lam not in GRID_LAMBDA:
        raise ValueError(f"lambda {lam} not in the grid {GRID_LAMBDA}")
    return f"sym_e1_l{int(lam)}"


def inner_folds(train_tags: Sequence[str], is_pmos: dict, k: int = N_INNER) -> list[list[str]]:
    """Training devices sorted by (polarity, nMOS first; tag); the i-th device
    goes to inner fold i mod k.  Returns k lists (each in that sorted order)."""
    tags = list(train_tags)
    if len(set(tags)) != len(tags):
        raise ValueError("duplicate training tags")
    order = sorted(tags, key=lambda t: (bool(is_pmos[t]), t))
    folds = [[] for _ in range(k)]
    for i, t in enumerate(order):
        folds[i % k].append(t)
    return folds


def inner_train(train_tags_load_order: Sequence[str], held: Sequence[str]) -> list[str]:
    """Inner training set in record load order (all outer training devices
    outside the inner fold ``held``)."""
    held = set(held)
    return [t for t in train_tags_load_order if t not in held]


def inner_score(rrms: Sequence[float]) -> float:
    """Mean of ln RRMS over the inner held-out devices; a non-finite or
    non-positive RRMS counts as ln(1.0) = 0."""
    vals = [math.log(r) if (r is not None and np.isfinite(r) and r > 0) else 0.0 for r in rrms]
    if not vals:
        raise ValueError("no inner held-out devices")
    return float(np.mean(vals))


def select_lambda(scores: dict) -> float:
    """argmin over lambda of the inner score; ties within TIE_TOL go to the
    larger lambda."""
    if set(map(float, scores)) != set(GRID_LAMBDA):
        raise ValueError("scores must cover the lambda grid")
    best = min(float(v) for v in scores.values())
    tied = [float(k) for k, v in scores.items() if float(v) <= best + TIE_TOL]
    return max(tied)


def contrast_label(gmr: dict, margin: float = NI_MARGIN) -> str:
    """'cost' if CI lower > 1; 'no cost beyond the margin' if CI upper <=
    margin; otherwise 'inconclusive'."""
    lo, hi = gmr.get("ci_low"), gmr.get("ci_high")
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if lo > 1.0:
        return "cost"
    if hi <= margin:
        return "no cost beyond the margin"
    return "inconclusive"


def decide_outcome(validity_ok: bool, primary: dict, margin: float = NI_MARGIN) -> dict:
    """Pre-registered primary outcome from GMR(selected, v1)."""
    if not validity_ok:
        return {"outcome": "invalid", "action": "stop: fix and re-register"}
    hi = primary.get("ci_high")
    ok = bool(hi is not None and np.isfinite(hi) and hi <= margin)
    return {"outcome": ("selected variant non-inferior to M_bias v1" if ok
                        else "selected variant not shown non-inferior to M_bias v1"),
            "ci_upper_below_1": bool(hi is not None and np.isfinite(hi) and hi < 1.0)}
