"""Direct NGSpice least-squares fitting of one device in a ParameterBox.

Residuals come from fresh NGSpice simulations compared with the measured
curves, never from a surrogate.  Each included curve contributes its trimmed
points scaled by the paper denominator; an optional iteratively reweighted
outer loop makes the squared objective track the paper's *mean* per-curve
RRMS instead of the mean squared RRMS (w_c = 1/sqrt(rrms_c)).  The paper
metric itself is unchanged and is always recomputed with ``card_eval``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .card_eval import device_by_tag, include_tags, score_card
from .cold_start import device_bins
from .data_io import load_device_curves
from .metrics import curve_trim
from .parameter_space import ParameterBox
from .spice_pdk import simulate_pdk

PENALTY = 10.0


@dataclass
class DeviceProblem:
    tag: str
    box: ParameterBox
    device: object = field(init=False)
    curves: list = field(init=False)
    layout: list = field(init=False)
    bin_index: int = field(init=False)
    n_sim: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.device = device_by_tag(self.tag)
        self.curves = load_device_curves(self.device)
        self.bin_index = device_bins()[self.tag]
        tags = include_tags(self.tag)
        self.layout = []
        for index, curve in enumerate(self.curves):
            key = f"{curve.kind}@{abs(float(curve.fixed)):g}"
            if key not in tags:
                continue
            start, cleaned, denominator = curve_trim(
                self.device.dev_type, self.device.L_um, self.device.W_um,
                curve.kind, curve.Id)
            if denominator > 0 and np.isfinite(denominator):
                self.layout.append({
                    "index": index, "start": start, "tag": key,
                    "target": np.asarray(cleaned[start:], dtype=np.float64),
                    "denominator": float(denominator)})
        self.n_curves = len(self.layout)

    def simulate(self, local: np.ndarray) -> list[np.ndarray]:
        self.n_sim += 1
        params = self.box.local_to_params(np.asarray(local, dtype=np.float64))
        return simulate_pdk(self.device.dev_type, self.device.L_um,
                            self.device.W_um, self.curves, params=params,
                            bin_index=self.bin_index)

    def curve_rrms(self, sims) -> np.ndarray:
        values = []
        for row in self.layout:
            sim = np.asarray(sims[row["index"]], dtype=np.float64)[row["start"]:]
            if not np.all(np.isfinite(sim)):
                values.append(PENALTY)
                continue
            rmse = math.sqrt(float(np.mean((sim - row["target"]) ** 2)))
            values.append(rmse / row["denominator"])
        return np.asarray(values)

    def residual_vector(self, sims, weights: np.ndarray) -> np.ndarray:
        pieces = []
        scale = 1.0 / math.sqrt(self.n_curves)
        for row, weight in zip(self.layout, weights):
            sim = np.asarray(sims[row["index"]], dtype=np.float64)[row["start"]:]
            n = len(row["target"])
            if not np.all(np.isfinite(sim)):
                pieces.append(np.full(n, PENALTY * weight * scale / math.sqrt(n)))
                continue
            pieces.append((sim - row["target"]) * weight * scale
                          / (row["denominator"] * math.sqrt(n)))
        return np.concatenate(pieces)

    def objective(self, local: np.ndarray) -> float:
        """Mean included-curve RRMS (the fixed-inclusion paper metric)."""
        return float(np.mean(self.curve_rrms(self.simulate(local))))


def polish(problem: DeviceProblem, local0: np.ndarray, *, max_nfev: int = 200,
           diff_step: float = 0.01, irls_rounds: int = 3,
           active_only: bool = True) -> dict:
    """Bounded TRF on NGSpice residuals; returns the best card seen."""
    t0 = time.time()
    active = np.flatnonzero(problem.box.active) if active_only else \
        np.arange(problem.box.n_parameters)
    local = np.asarray(local0, dtype=np.float64).copy()
    start_sims = problem.simulate(local)
    best_curves = problem.curve_rrms(start_sims)
    best = {"local": local.copy(), "objective": float(np.mean(best_curves))}
    history = [best["objective"]]
    weights = np.ones(problem.n_curves)
    for _ in range(max(1, irls_rounds)):
        def residual(values):
            trial = local.copy()
            trial[active] = values
            sims = problem.simulate(trial)
            curves = problem.curve_rrms(sims)
            value = float(np.mean(curves))
            if value < best["objective"]:
                best.update(local=trial.copy(), objective=value)
            return problem.residual_vector(sims, weights)

        def jacobian(values):
            # Absolute steps in local units (as in scripts/real_fd_polish.py):
            # SciPy's relative diff_step and xtol collapse at local = 0, which
            # is exactly where a box-centered start sits, and TRF then stops
            # after one rejected step without moving.
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
        local = best["local"].copy()
        curves = problem.curve_rrms(problem.simulate(local))
        history.append(float(np.mean(curves)))
        weights = 1.0 / np.sqrt(np.maximum(curves, 1e-3))
        weights *= math.sqrt(problem.n_curves / float(np.sum(weights ** 2)))
    return {"local": best["local"], "objective": best["objective"],
            "history": history, "n_sim": problem.n_sim,
            "runtime_s": round(time.time() - t0, 1)}


def final_score(problem: DeviceProblem, local: np.ndarray) -> dict:
    params = problem.box.local_to_params(np.asarray(local, dtype=np.float64))
    scored = score_card(problem.tag, params, curves=problem.curves,
                        bin_index=problem.bin_index)
    scored["params"] = params
    scored["local"] = np.asarray(local, dtype=np.float64).tolist()
    return scored
