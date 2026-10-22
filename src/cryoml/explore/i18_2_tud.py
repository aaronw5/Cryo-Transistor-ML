"""I18-2: the selected g v2 recipe (symmetric inputs, EPS 0.01 V, nested
selection of the gm/gds penalty lambda) on measured TU Delft 40-nm MOSFETs
(I6-1 data, BSIM4 CMOSedu card), 4.13 K and 295.0 K, 45 DEV devices,
leave-one-geometry-group-out; BSIM4 clamp state C.

Pre-registration: ``out/research/explore/I18_2/prereg.md`` / ``prereg.json``.

Pure helpers: the b4check.c A1/A2 clamp of the card parameters (state C),
penalty-grid axes and curve specs, inner folds over geometry groups, the
pre-registered validity, primary / secondary labels.  No simulation, no
training.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .i15_2_ablate import GRID_LAMBDA, inner_score, select_lambda  # noqa: F401
from .i17_2_ihp import primary_label, secondary_label, sign_check  # noqa: F401

TEMPS = (4, 295)
VDD = 1.1                      # V, I6-1 normalization and the measured range
EPS = 0.01                     # V
PEN_STEP = 0.025               # V (sky130 step)
PEN_N = 45                     # 0 .. 1.1 V
N_INNER = 3
NI_MARGIN = 1.05
THR_V1 = 1e-9
CARD_A1, CARD_A2 = 0.0, 1.0    # N1 / P1 card values (no l/w/p A1/A2 terms)
A2_LOW, A2_HIGH = 0.01, 1.0
V1C_TAGS = ("t40_nmos_W1_L1_d130", "t40_pmos_W1_L1_d330")
SMOKE_GROUPS = ("t40_nmos_W0p36_L0p04", "t40_nmos_W1_L1", "t40_pmos_W1_L0p04", "t40_pmos_W1_L1")


def clamp_params(params: Mapping[str, float] | None) -> tuple[dict, dict]:
    """State C: write the ngspice-41 b4check.c clamp into the parameters
    (a2 > 1 -> a2 = 1 and a1 = 0 if nonzero; a2 < 0.01 -> a2 = 0.01).  The
    card values (a1 = 0, a2 = 1) apply when a key is absent.  Returns
    (params, changes); unclamped parameters are returned unchanged."""
    p = dict(params or {})
    for k in ("la1", "wa1", "pa1", "la2", "wa2", "pa2"):
        if k in p:
            raise NotImplementedError(f"{k} present: effective A1/A2 is size dependent")
    a2 = float(p.get("a2", CARD_A2))
    a1 = float(p.get("a1", CARD_A1))
    changes = {}
    if a2 > A2_HIGH:
        changes["a2"] = (a2, A2_HIGH)
        p["a2"] = A2_HIGH
        if a1 != 0.0:
            changes["a1"] = (a1, 0.0)
            p["a1"] = 0.0
    elif a2 < A2_LOW:
        changes["a2"] = (a2, A2_LOW)
        p["a2"] = A2_LOW
    return p, changes


def pen_axis(n: int = PEN_N, step: float = PEN_STEP) -> np.ndarray:
    return np.round(np.arange(int(n)) * float(step), 6)


def grid_curve_specs(is_pmos: bool) -> list[dict]:
    """One Id-Vd sweep per u_g: V(g,s) = s u_g, V(d,s) = s u_d (0 .. 1.1 V), V(b,s) = 0."""
    s = -1.0 if is_pmos else 1.0
    ax = pen_axis()
    return [{"fixed": float(s * ug), "sweep": s * ax, "vb": 0.0} for ug in ax]


def is_pmos_group(group: str) -> bool:
    return "_pmos_" in group


def inner_group_folds(groups: Sequence[str], k: int = N_INNER) -> list[list[str]]:
    """Groups sorted by (polarity, nMOS first; name); group i -> inner fold i mod k."""
    gs = list(groups)
    if len(set(gs)) != len(gs):
        raise ValueError("duplicate groups")
    order = sorted(gs, key=lambda g: (is_pmos_group(g), g))
    folds = [[] for _ in range(k)]
    for i, g in enumerate(order):
        folds[i % k].append(g)
    return folds


def variant(lam: float) -> str:
    lam = float(lam)
    if lam not in GRID_LAMBDA:
        raise ValueError(f"lambda {lam} not in {GRID_LAMBDA}")
    return f"sym_e1_l{int(lam)}"


# ---------------------------------------------------------------------------
# decision (pure, pre-registered)
# ---------------------------------------------------------------------------
def decide_validity(v: dict) -> dict:
    """V1 (a) state-L re-simulation reproduces the I6-1 records, (b) state C
    = state L on unclamped cards, (c) state-L refits reproduce the I6-1
    rows; V2 unit tests; V3 45 rows per arm and temperature, inner and full
    selection complete; V4 penalty grids."""
    fails = []
    if not (v["v1a_max"] <= THR_V1 and v["v1a_n"] == 90):
        fails.append("V1a state-L re-simulation reproduces I6-1 records")
    if not (v["v1b_max"] == 0.0 and v["v1b_n"] > 0):
        fails.append("V1b state C = state L on unclamped cards")
    if not (v["v1c_max"] <= THR_V1 and v["v1c_n"] == 4):
        fails.append("V1c state-L refits reproduce I6-1 rows")
    if not v["v2_unit_tests_pass"]:
        fails.append("V2 unit tests")
    if not (all(n == 45 for n in v["v3_rows"].values()) and v["v3_inner_complete"] and v["v3_full_complete"]):
        fails.append("V3 rows / inner / full selection")
    if not v["v4_grids_ok"]:
        fails.append("V4 penalty grids")
    return {"valid": not fails, "fails": fails}


def decide_outcome(validity: dict, primary: dict, n_failed: int, secondary: dict) -> dict:
    if not validity["valid"]:
        return {"outcome": "invalid", "action": "stop: fix and re-register"}
    return {"outcome": primary_label(primary, n_failed), "secondary": secondary_label(secondary, NI_MARGIN)}
