"""I17-2: second-PDK replication of the selected g v2 recipe (symmetric
inputs, EPS 0.01 V, gm/gds penalty, nested selection of lambda) on measured
IHP SG13G2 MOSFETs (PSP103, I5-1 data), 300 K, 18 DEV DUTs, leave-one-DUT-out
within class.  Accuracy only; Part B not run (PSP103 has no switch for its
intrinsic charge, see the pre-registration).

Pre-registration: ``out/research/explore/I17_2/prereg.md`` / ``prereg.json``.

Pure helpers: penalty-grid axes and grid curves, fold training sets, inner
folds, geometry clusters, the pre-registered validity and decision tables.
No simulation and no training here.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .i15_2_ablate import GRID_LAMBDA, inner_folds, inner_score, inner_train, select_lambda  # noqa: F401

T_K = 300
PEN_N = 75                   # grid values per axis (sky130 count)
EPS = 0.01                   # V
NI_MARGIN = 1.05
THR_V1 = 1e-9                # V1: |rerun - I5-1 row|
SMOKE_FOLDS = ("lv_nmos_W0p15_L0p18", "hv_nmos_W0p3_L0p5")


def pen_axis(vdd: float, n: int = PEN_N) -> np.ndarray:
    """Forward-quadrant axis 0 .. vdd with n values (step vdd / (n - 1))."""
    return np.linspace(0.0, float(vdd), int(n))


def pen_step(vdd: float, n: int = PEN_N) -> float:
    return float(vdd) / (int(n) - 1)


def grid_curve_specs(vdd: float, is_pmos: bool, n: int = PEN_N) -> list[dict]:
    """One Id-Vd sweep per u_g value: fixed V(g,s) = s u_g, swept V(d,s) =
    s u_d (u_d = 0 .. vdd), V(b,s) = 0; s = -1 for pMOS."""
    s = -1.0 if is_pmos else 1.0
    ax = pen_axis(vdd, n)
    return [{"fixed": float(s * ug), "sweep": s * ax, "vb": 0.0} for ug in ax]


def class_of(tag: str) -> str:
    return tag.split("_", 1)[0]


def fold_train(load_order: Sequence[str], held: str) -> list[str]:
    """Training DUTs of the held-out DUT's class (other DUTs), load order."""
    c = class_of(held)
    return [t for t in load_order if t != held and class_of(t) == c]


def is_pmos_map(tags: Sequence[str]) -> dict:
    return {t: "_pmos_" in t for t in tags}


def clusters(tags: Sequence[str]) -> dict:
    """Geometry cluster: class and W/L (LV nMOS and pMOS of one W/L, and HV
    nMOS / pMOS of one W/L, share a cluster)."""
    out = {}
    for t in tags:
        cls, _, geom = t.split("_", 2)
        out[t] = f"{cls}_{geom}"
    return out


def variant(lam: float) -> str:
    lam = float(lam)
    if lam not in GRID_LAMBDA:
        raise ValueError(f"lambda {lam} not in {GRID_LAMBDA}")
    return f"sym_e1_l{int(lam)}"


# ---------------------------------------------------------------------------
# decision (pure, pre-registered)
# ---------------------------------------------------------------------------
def decide_validity(v: dict) -> dict:
    """V1 C0_refit / M_bias v1 reruns reproduce the I5-1 rows (<= 1e-9, 18
    each); V2 unit tests; V3 18 rows per arm and inner complete; V4 18 finite
    75 x 75 grids with positive forward current at the corner."""
    fails = []
    if not (v["v1_c0_max"] <= THR_V1 and v["v1_mb_max"] <= THR_V1 and v["v1_n"] == 18):
        fails.append("V1 C0_refit / M_bias v1 rerun reproduces I5-1 rows")
    if not v["v2_unit_tests_pass"]:
        fails.append("V2 unit tests")
    if not (all(n == 18 for n in v["v3_rows"].values()) and v["v3_inner_complete"]):
        fails.append("V3 18 rows per arm / inner selection complete")
    if not v["v4_grids_ok"]:
        fails.append("V4 penalty grids")
    return {"valid": not fails, "fails": fails}


def primary_label(g: dict, n_failed: int) -> str:
    lo, hi = g.get("ci_low"), g.get("ci_high")
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if hi < 1.0 and n_failed == 0:
        return "gain over C0_refit replicated"
    if lo > 1.0:
        return "worse than C0_refit"
    return "inconclusive"


def secondary_label(g: dict, margin: float = NI_MARGIN) -> str:
    hi = g.get("ci_high")
    ok = hi is not None and np.isfinite(hi) and hi <= margin
    return "non-inferior to M_bias v1" if ok else "not shown non-inferior to M_bias v1"


def decide_outcome(validity: dict, primary: dict, n_failed: int, secondary: dict) -> dict:
    if not validity["valid"]:
        return {"outcome": "invalid", "action": "stop: fix and re-register", "part_B": "not run"}
    return {"outcome": primary_label(primary, n_failed), "secondary": secondary_label(secondary),
            "part_B": "not run"}


def sign_check(ratio: float, predicted: str) -> bool:
    """Predicted sign of ln GMR ('> 0' or '< 0') vs the point estimate."""
    lr = float(np.log(ratio))
    return lr > 0 if predicted == "> 0" else lr < 0
