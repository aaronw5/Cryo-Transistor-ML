"""I23-1: call-matched IHP control for B1 (Part A) and g v2 with lambda fixed
at 1 V (Part B).

Pre-registration: ``out/research/explore/I23_1/prereg.md`` (``prereg.json``).

This module holds the dataset-independent pieces: ``polish_capped`` (the
``ngspice_fit.polish`` refit with a hard limit on NGSpice calls), the Part A
and Part B rules, and small helpers.  The drivers are in
``scripts/research/explore/i23_1_budget.py`` and ``i23_1_report.py``.
"""

from __future__ import annotations

import math
import time
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from .i22_1_baselines import label_vs_control, label_vs_reference, non_inferior, pooled_gmr  # noqa: F401

IHP = ("ihp233", "ihp300", "ihp343", "ihp398")
DATASETS = ("sky130", "nist4", "nistrt", "ihp233", "ihp300", "ihp343", "ihp398", "tud4", "tud295")
GAIN_SET = ("sky130", "nist4", "nistrt")
FIXED_LAMBDA = 1.0
NI_MARGIN = 1.05
REPRO_TOL = 1e-9
VA1_TOL = 1e-12
MAX_FAIL_FRACTION = 0.10
REUSED = {"sky130": "I15_2/lodo_sym_e1_l1", "ihp300": "I17_2/lodo_sym_e1_l1", "tud4": "I18_2/lodo_4K_sym_e1_l1",
          "tud295": "I18_2/lodo_295K_sym_e1_l1", "nist4": "I20_1/lodo_nist4 (lambda* = 1 V devices)"}
VB1_SAMPLES = {"sky130": ("nmos_L1_W1p6", "pmos_L2_W5"), "ihp300": ("lv_nmos_W0p15_L0p18", "hv_pmos_W0p6_L0p5"),
               "tud4": ("t40_nmos_W0p36_L0p04",), "tud295": ("t40_nmos_W0p36_L0p04",)}


class BudgetExhausted(Exception):
    """Raised inside a residual call when the NGSpice-call limit is reached."""


def polish_capped(problem, local0: np.ndarray, *, max_calls: int | None = None, max_nfev: int = 200,
                  diff_step: float = 0.01, irls_rounds: int = 3, active_only: bool = True) -> dict:
    """``ngspice_fit.polish`` with at most ``max_calls`` calls of
    ``problem.simulate`` (None: no limit; then identical to ``polish``).

    At the limit the running TRF round stops, the remaining IRLS rounds are
    skipped and the best card seen so far is returned.  ``calls`` in the
    result counts the simulations made here; ``capped`` says whether the limit
    was reached."""
    t0 = time.time()
    count = [0]

    def sim(local):
        if max_calls is not None and count[0] >= max_calls:
            raise BudgetExhausted
        count[0] += 1
        return problem.simulate(local)

    active = np.flatnonzero(problem.box.active) if active_only else np.arange(problem.box.n_parameters)
    local = np.asarray(local0, dtype=np.float64).copy()
    best = {"local": local.copy(), "objective": float("inf")}
    history: list[float] = []
    capped = False
    try:
        start_sims = sim(local)
    except BudgetExhausted:
        return {"local": local, "objective": float("nan"), "history": history, "n_sim": problem.n_sim,
                "calls": count[0], "capped": True, "runtime_s": round(time.time() - t0, 1)}
    best["objective"] = float(np.mean(problem.curve_rrms(start_sims)))
    history.append(best["objective"])
    weights = np.ones(problem.n_curves)
    for _ in range(max(1, irls_rounds)):
        def residual(values, weights=weights, local=local):
            trial = local.copy()
            trial[active] = values
            sims = sim(trial)
            value = float(np.mean(problem.curve_rrms(sims)))
            if value < best["objective"]:
                best.update(local=trial.copy(), objective=value)
            return problem.residual_vector(sims, weights)

        def jacobian(values, residual=residual):
            base = residual(values)
            columns = np.empty((base.size, values.size))
            for index in range(values.size):
                shifted = values.copy()
                step = diff_step if shifted[index] + diff_step <= 1.0 else -diff_step
                shifted[index] += step
                columns[:, index] = (residual(shifted) - base) / step
            return columns

        budget = max(2, max_nfev // max(1, irls_rounds) // (len(active) + 2))
        x0 = np.clip(local[active], -1 + 1e-9, 1 - 1e-9)
        try:
            least_squares(residual, x0, method="trf", jac=jacobian,
                          bounds=(-np.ones(len(active)), np.ones(len(active))),
                          xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=budget)
        except ValueError:
            pass
        except BudgetExhausted:
            capped = True
        local = best["local"].copy()
        if capped:
            break
        try:
            curves = problem.curve_rrms(sim(local))
        except BudgetExhausted:
            capped = True
            break
        history.append(float(np.mean(curves)))
        weights = 1.0 / np.sqrt(np.maximum(curves, 1e-3))
        weights *= math.sqrt(problem.n_curves / float(np.sum(weights ** 2)))
    return {"local": best["local"], "objective": best["objective"], "history": history, "n_sim": problem.n_sim,
            "calls": count[0], "capped": capped, "runtime_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------
def rule_partA(b1_vs_c0call: Mapping[str, Mapping], valid: Mapping[str, bool],
               temps: Sequence[str] = IHP) -> dict:
    """Pre-registered Part A rule over GMR(B1, C0_call) per temperature."""
    bad = [t for t in temps if not valid.get(t, False)]
    hi = {t: float(b1_vs_c0call[t].get("ci_high", float("nan"))) for t in temps if t in b1_vs_c0call}
    gain = [t for t in temps if t in hi and np.isfinite(hi[t]) and hi[t] < 1.0]
    if bad:
        return {"label": "not evaluated", "invalid": bad, "gain_temperatures": gain}
    if len(gain) == len(temps):
        label = "B1 gain survives the call match"
    elif not gain:
        label = "B1 gain not shown at equal calls"
    else:
        label = "partial"
    return {"label": label, "invalid": [], "gain_temperatures": gain, "ci_high": hi}


def rule_partB(fixed_vs_sel: Mapping[str, Mapping], valid: Mapping[str, bool],
               gain_set: Sequence[str] = GAIN_SET, margin: float = NI_MARGIN) -> dict:
    """Pre-registered Part B rule: non-inferior (CI upper < margin) on every
    dataset of the gain set."""
    bad = [d for d in gain_set if not valid.get(d, False)]
    ni = {d: non_inferior(fixed_vs_sel[d], margin) for d in gain_set if d in fixed_vs_sel}
    failing = [d for d in gain_set if not ni.get(d, False)]
    if bad:
        return {"label": "not evaluated", "invalid": bad, "non_inferior": ni, "failing": failing}
    label = ("fixed lambda 1 V non-inferior to the selected g v2" if not failing
             else "fixed lambda 1 V not shown non-inferior")
    return {"label": label, "invalid": [], "non_inferior": ni, "failing": failing}


def call_limit_check(n_sim: Mapping[str, int | None], limit: Mapping[str, int]) -> dict:
    """VA3: every recorded n_sim is at most its limit."""
    over = {t: (n_sim.get(t), limit[t]) for t in limit
            if n_sim.get(t) is None or int(n_sim[t]) > int(limit[t])}
    used = [int(n_sim[t]) / int(limit[t]) for t in limit if n_sim.get(t) is not None and limit[t]]
    return {"ok": not over, "over": over, "sum_calls": int(sum(int(v) for v in n_sim.values() if v is not None)),
            "sum_limit": int(sum(int(v) for v in limit.values())),
            "mean_fraction_used": float(np.mean(used)) if used else float("nan")}


def max_abs_diff(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    if not keys or set(a) != set(b):
        return float("inf")
    d = [abs(float(a[k]) - float(b[k])) for k in keys]
    return float(max(d))
