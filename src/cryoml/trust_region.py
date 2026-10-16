"""Trust-region model management with a first-order-consistent correction.

Alexandrov, Dennis, Lewis and Torczon (1998): an expensive residual ``r_f``
(here NGSpice) is minimized through a cheap nonlinear model ``r_s`` (here the
frozen neural surrogate), corrected at each trust-region centre ``z_c`` so it
reproduces the expensive value and Jacobian there::

    r_c(z) = r_s(z) + [r_f(z_c) - r_s(z_c)] + [J_f(z_c) - J_s(z_c)] (z - z_c)

Each iteration minimizes ``0.5 ||r_c(z)||^2 + 0.5 ||w z||^2`` over the box
``[-1, 1]`` intersected with the infinity-norm ball ``|z - z_c| <= delta``
(bounded TRF with the analytic model Jacobian; model calls only), then spends
one expensive call at the step.  ``rho`` = actual / predicted reduction of
that same objective drives acceptance and the radius; an accepted centre gets
a fresh expensive Jacobian.  Without a surrogate (``r_s = 0``) ``r_c`` is the
Gauss-Newton linearization of ``r_f`` and the loop is plain trust-region
Gauss-Newton on ``r_f`` (the control).

``TrustRegionRun.advance`` takes an absolute call limit and can be resumed,
so several starts can share one budget (``run_schedule``).

The expensive backend needs ``n_calls`` (every simulation, Jacobian probes
included), ``jacobian_cost``, ``residual(z)`` and ``jacobian(z, r)``; the
surrogate needs ``value(z)`` and ``jacobian(z)``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class TrustRegionSettings:
    delta0: float = 0.25
    delta_max: float = 2.0          # the [-1, 1] box diameter
    delta_min: float = 1e-3         # radius collapse: stop the run
    accept: float = 0.1             # accept the step if rho > accept
    good: float = 0.75              # grow if rho > good and the step hit delta
    poor: float = 0.25              # shrink if rho < poor
    grow: float = 2.0
    shrink: float = 0.25            # delta <- shrink * min(delta, |step|_inf)
    boundary_fraction: float = 0.99
    stationary: float = 1e-10       # predicted reduction / objective
    sub_max_nfev: int = 100
    sub_tol: float = 1e-10


class CorrectedModel:
    """``r_c`` and its Jacobian at one centre, plus the prior rows."""

    def __init__(self, surrogate, z_c: np.ndarray, r_f: np.ndarray,
                 J_f: np.ndarray, prior_weight: float):
        self.surrogate = surrogate
        self.z_c = np.asarray(z_c, dtype=np.float64).copy()
        self.prior_weight = float(prior_weight)
        if surrogate is None:
            self.offset = np.asarray(r_f, dtype=np.float64).copy()
            self.slope = np.asarray(J_f, dtype=np.float64).copy()
        else:
            self.offset = r_f - surrogate.value(self.z_c)
            self.slope = J_f - surrogate.jacobian(self.z_c)

    def data_residual(self, z: np.ndarray) -> np.ndarray:
        linear = self.offset + self.slope @ (z - self.z_c)
        if self.surrogate is None:
            return linear
        return self.surrogate.value(z) + linear

    def data_jacobian(self, z: np.ndarray) -> np.ndarray:
        if self.surrogate is None:
            return self.slope
        return self.surrogate.jacobian(z) + self.slope

    def residual(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        return np.concatenate([self.data_residual(z), self.prior_weight * z])

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        return np.vstack([self.data_jacobian(z),
                          self.prior_weight * np.eye(len(z))])

    def cost(self, z: np.ndarray) -> float:
        r = self.residual(z)
        return 0.5 * float(r @ r)

    def consistency(self, r_f: np.ndarray, J_f: np.ndarray) -> dict:
        """Max |r_c - r_f| and |J_c - J_f| at the centre (should be ~0)."""
        value = np.max(np.abs(self.data_residual(self.z_c) - r_f))
        slope = np.max(np.abs(self.data_jacobian(self.z_c) - J_f))
        return {"max_abs_value_error": float(value),
                "max_abs_jacobian_error": float(slope),
                "max_abs_value": float(np.max(np.abs(r_f))),
                "max_abs_jacobian": float(np.max(np.abs(J_f)))}


class TrustRegionRun:
    """One trust-region loop from ``z0``; ``advance`` resumes it."""

    def __init__(self, z0: np.ndarray, expensive, surrogate,
                 prior_weight: float, settings: TrustRegionSettings,
                 label: str = ""):
        self.expensive = expensive
        self.surrogate = surrogate
        self.prior_weight = float(prior_weight)
        self.settings = settings
        self.label = label
        self.z0 = np.clip(np.asarray(z0, dtype=np.float64), -1.0, 1.0)
        self.z_c = self.z0.copy()
        self.r_c = None                 # expensive residual at the centre
        self.J_c = None                 # expensive Jacobian at the centre
        self.F_c = math.inf
        self.F_start = math.inf
        self.model = None
        self.delta = settings.delta0
        self.status = "new"
        self.calls = 0
        self.n_accepted = self.n_rejected = 0
        self.log: list[dict] = []
        self.seconds = {"subproblem": 0.0, "residual": 0.0, "jacobian": 0.0}

    def objective(self, z: np.ndarray, r: np.ndarray) -> float:
        prior = self.prior_weight * z
        return 0.5 * float(r @ r + prior @ prior)

    def _residual(self, z):
        t0, n0 = time.time(), self.expensive.n_calls
        r = self.expensive.residual(z)
        self.seconds["residual"] += time.time() - t0
        self.calls += self.expensive.n_calls - n0
        return r

    def _jacobian(self, z, r):
        t0, n0 = time.time(), self.expensive.n_calls
        J = self.expensive.jacobian(z, r)
        self.seconds["jacobian"] += time.time() - t0
        self.calls += self.expensive.n_calls - n0
        return J

    def solve_subproblem(self) -> tuple[np.ndarray, float]:
        """Minimize the corrected model in the trust region; returns the
        step end point and the predicted reduction."""
        if self.model is None:
            self.model = CorrectedModel(self.surrogate, self.z_c, self.r_c,
                                        self.J_c, self.prior_weight)
        lo = np.maximum(-1.0, self.z_c - self.delta)
        hi = np.minimum(1.0, self.z_c + self.delta)
        t0 = time.time()
        solution = least_squares(
            self.model.residual, self.z_c, jac=self.model.jacobian,
            bounds=(lo, hi), method="trf", x_scale=1.0,
            ftol=self.settings.sub_tol, xtol=self.settings.sub_tol,
            gtol=self.settings.sub_tol, max_nfev=self.settings.sub_max_nfev)
        z_new = np.clip(solution.x, lo, hi)
        predicted = self.model.cost(self.z_c) - self.model.cost(z_new)
        self.seconds["subproblem"] += time.time() - t0
        return z_new, float(predicted)

    def advance(self, limit: int) -> str:
        """Iterate until stationary, the radius collapses, or the next
        expensive evaluation would take the shared counter past ``limit``."""
        s = self.settings
        e = self.expensive
        if self.status == "new":
            if e.n_calls + 1 > limit:
                return self.status
            self.r_c = self._residual(self.z_c)
            self.F_c = self.F_start = self.objective(self.z_c, self.r_c)
            self.status = "active"
        while self.status == "active":
            if self.J_c is None:
                if e.n_calls + e.jacobian_cost > limit:
                    break
                self.J_c = self._jacobian(self.z_c, self.r_c)
                self.model = None
            if e.n_calls + 1 > limit:
                break
            z_new, predicted = self.solve_subproblem()
            step = float(np.max(np.abs(z_new - self.z_c))) if len(z_new) else 0.0
            if predicted <= s.stationary * max(self.F_c, 1e-300) or step == 0.0:
                self.status = "stationary"
                self.log.append({"calls": e.n_calls, "F": self.F_c,
                                 "delta": self.delta, "step": step,
                                 "predicted": predicted, "stationary": True})
                break
            r_new = self._residual(z_new)
            F_new = self.objective(z_new, r_new)
            actual = self.F_c - F_new
            rho = actual / predicted
            hit = step >= s.boundary_fraction * self.delta
            accepted = rho > s.accept
            self.log.append({"calls": e.n_calls, "F": self.F_c, "F_new": F_new,
                             "delta": self.delta, "step": step,
                             "predicted": predicted, "actual": actual,
                             "rho": rho, "accepted": bool(accepted),
                             "boundary": bool(hit)})
            if rho < s.poor:
                self.delta = s.shrink * min(self.delta, step)
            elif rho > s.good and hit:
                self.delta = min(s.grow * self.delta, s.delta_max)
            if accepted:
                self.n_accepted += 1
                self.z_c, self.r_c, self.F_c = z_new, r_new, F_new
                self.J_c = self.model = None
            else:
                self.n_rejected += 1
            if self.delta < s.delta_min:
                self.status = "radius_collapsed"
        return self.status

    def summary(self) -> dict:
        return {"label": self.label, "status": self.status,
                "F_start": self.F_start, "F_final": self.F_c,
                "delta_final": self.delta, "calls": self.calls,
                "accepted": self.n_accepted, "rejected": self.n_rejected,
                "seconds": {k: round(v, 3) for k, v in self.seconds.items()},
                "log": self.log}


def run_schedule(runs: list[TrustRegionRun], expensive, budget: int,
                 schedule: str = "equal") -> None:
    """Share ``budget`` expensive calls among the runs.

    ``equal``: runs go one after another, each with an equal share of what
    is left (so budget a run leaves unused rolls over to later runs).
    ``halving``: successive halving, log2(n) + 1 rounds with equal round
    budgets; after each round the better half (by centre objective) goes on.
    Whatever is left at the end goes to the best run still active."""
    if schedule == "equal":
        for index, run in enumerate(runs):
            share = (budget - expensive.n_calls) // (len(runs) - index)
            run.advance(expensive.n_calls + share)
    elif schedule == "halving":
        survivors = list(runs)
        n_rounds = int(math.ceil(math.log2(max(1, len(runs))))) + 1
        for round_index in range(n_rounds):
            remaining = budget - expensive.n_calls
            share = remaining // ((n_rounds - round_index) * len(survivors))
            for run in survivors:
                run.advance(expensive.n_calls + share)
            survivors = sorted(survivors, key=lambda r: r.F_c)[
                :max(1, len(survivors) // 2)]
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    while expensive.n_calls < budget:
        active = [r for r in runs if r.status in ("active", "new")]
        if not active:
            break
        before = expensive.n_calls
        min(active, key=lambda r: r.F_c).advance(budget)
        if expensive.n_calls == before:
            break
