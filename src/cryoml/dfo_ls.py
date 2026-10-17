"""Model-based derivative-free least squares: the DFO-LS core, re-implemented.

Algorithm of DFO-LS (Cartis, Fiala, Marteau and Roberts 2019, ACM TOMS
45(3):32; Cartis and Roberts 2019, Math. Prog. Comp. 11:631): a Gauss-Newton
trust region whose residual Jacobian comes from linear interpolation on
``p + 1`` points, one expensive residual call per iteration.

* Objective ``F(z) = 0.5 ||r(z)||^2 + 0.5 w^2 ||z||^2`` (``w`` = prior weight,
  ``sqrt(reg / n_active)`` in the DEV arms, 0 for the Moré-Garbow-Hillstrom
  validation), over the coordinates ``z`` within the backend's box (or
  unbounded).
* Interpolation set ``Y = {y_0..y_p}``; the centre ``z_k`` is the point of ``Y``
  with the lowest ``F``.  Initialization: ``y_0 = z0``, ``y_t = z0 + rho_beg
  e_t`` (sign flipped if that leaves the box), ``p + 1`` calls, ``z0``
  included; ``Delta_0 = rho = rho_beg``.
* Linear model: ``J_k`` solves ``(y_t - z_k)^T J_k^T = (r(y_t) - r(z_k))^T``
  for ``t != k`` (``p x p`` solve; least squares when the matrix is singular
  or its condition number exceeds 1e15).  The condition number is recorded.
* Step: ``s_k = argmin ||r_k + J_k s||^2 + w^2 ||z_k + s||^2`` over
  ``||s||_inf <= Delta`` and ``z_k + s`` in the box (``scipy.optimize.
  lsq_linear``, method ``bvls``, ``trf`` on failure).  See ``STEP_NORM_NOTE``
  for the deviation from DFO-LS's Euclidean ball.
* Safety phase if ``||s||_inf < 0.5 rho`` (or the model predicts no
  decrease): ``Delta <- max(rho, 0.5 Delta)``, then a geometry step if a
  point lies farther than ``2 Delta`` from ``z_k``, else a rho reduction.
* Otherwise one call at ``z_k + s``; ``ratio`` = actual / predicted decrease
  of ``F``; DFO-LS default radius update: ``ratio < 0.1``: ``Delta <-
  min(0.5 Delta, ||s||)``; ``ratio <= 0.7``: ``Delta <- max(0.5 Delta,
  ||s||)``; else ``Delta <- min(max(2 Delta, 4 ||s||), delta_max)``; then
  ``Delta <- max(Delta, rho)`` (norms are infinity norms).
* The new point replaces the ``y_t`` (``t != k``; the centre is never
  replaced) maximizing ``|l_t(z_k + s)| max(1, ||y_t - z_k||_inf^4 /
  Delta^4)``, ``l_t`` the linear Lagrange polynomials of ``Y``; the centre
  moves when ``F`` decreased.
* ``ratio < 0.1``: geometry step if the farthest point is more than ``2
  Delta`` from ``z_k`` (that point is replaced by ``z_k + d``, ``d = Delta
  sign(grad l_t)`` clipped to the box, or its negative, whichever gives the
  larger ``|l_t|``; one call); else a rho reduction if ``||s||_inf <= rho``
  (``rho <- max(0.1 rho, rho_end)``, ``Delta <- max(0.5 rho_old, rho)``).
* Soft restart when a rho reduction is requested at ``rho <= rho_end``
  (1e-8): keep the best point, ``rho = Delta = rho_beg``, replace
  ``restart_points`` (3) points by ``z_k + rho_beg u`` along seeded random
  directions ``u`` (``||u||_inf = 1``; components that leave the box are
  reflected).  Restarts are logged and counted.
* Failed points (the backend's ``last_failed`` flag after a call, or a
  non-finite residual) are never added to ``Y``; a failed trial point counts
  as ``ratio = -inf``.  They are counted (``n_failed_points_skipped``).  A
  failed initialization point is retried with ``-h``, ``h/2``, ``-h/2``.

Budget: every residual call counts.  ``advance(limit)`` returns before any
call that would take the backend's shared counter ``n_calls`` past
``limit`` and can be resumed, so several starts can share one budget through
``trust_region.run_schedule`` (it reads ``advance``, ``status`` and
``F_c``).  Status: ``new`` (no call yet), ``active``, ``budget`` (paused at
a limit; the next ``advance`` resumes it), ``stationary`` (stopped: failed
start or initialization, zero objective, or ``max_restarts`` exceeded).
``run_schedule``'s final loop gives leftover calls only to ``active``/``new``
runs; ``finish_budget`` also gives them to runs paused at ``budget``.

Backend interface: ``n_calls``, ``residual(z)`` and optionally ``bounds``
(``(lo, hi)`` arrays or scalars; ``None`` or absent = unbounded) and
``last_failed``.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import lsq_linear

STEP_NORM_NOTE = (
    "deviation from DFO-LS: trust region and step bound in the infinity norm "
    "(||s||_inf <= Delta intersected with the box, solved exactly by "
    "scipy.optimize.lsq_linear); DFO-LS uses the Euclidean ball "
    "||s||_2 <= Delta")

LOG_FIELDS = ("calls", "kind", "F", "delta", "rho", "step", "ratio")
COND_LIMIT = 1e15


@dataclass
class DfoLsSettings:
    rho_beg: float = 0.1
    rho_end: float = 1e-8
    delta_max: float = 1.0          # the DEV box [-1, 1]; 1e10 when unbounded
    eta1: float = 0.1
    eta2: float = 0.7
    gamma_dec: float = 0.5
    gamma_inc: float = 2.0
    gamma_inc_overline: float = 4.0
    safety: float = 0.5             # safety phase if ||s||_inf < safety * rho
    far: float = 2.0                # geometry step if a point is > far * Delta away
    rho_factor: float = 0.1
    restart_points: int = 3
    max_restarts: int | None = None
    replace_power: float = 4.0
    seed: int = 0


class DfoLsRun:
    """One DFO-LS-style run from ``z0``; ``advance`` resumes it."""

    def __init__(self, z0, backend, prior_weight: float = 0.0,
                 settings: DfoLsSettings | None = None, label: str = "",
                 stream: int = 0):
        self.backend = backend
        self.settings = settings or DfoLsSettings()
        self.prior_weight = float(prior_weight)
        self.label = label
        z0 = np.asarray(z0, dtype=np.float64).copy()
        self.p = p = len(z0)
        bounds = getattr(backend, "bounds", None)
        if bounds is None:
            self.lo, self.hi = np.full(p, -np.inf), np.full(p, np.inf)
        else:
            self.lo = np.broadcast_to(np.asarray(bounds[0], np.float64), (p,)).copy()
            self.hi = np.broadcast_to(np.asarray(bounds[1], np.float64), (p,)).copy()
        if np.any(self.hi <= self.lo):
            raise ValueError("every upper bound must exceed its lower bound")
        self.z0 = np.clip(z0, self.lo, self.hi)
        self.Y = np.tile(self.z0, (p + 1, 1))
        self.R: np.ndarray | None = None
        self.Fy = np.full(p + 1, np.inf)
        self.k = 0
        self.rho = self.delta = float(self.settings.rho_beg)
        self.status = "new"
        self.stop_reason: str | None = None
        self.initialized = False
        self.pending: deque = deque()
        self.calls = 0
        self.F_c = math.inf
        self.F_start = math.inf
        self.log: list[list] = []
        self.rng = np.random.default_rng(
            [int(stream) & 0xFFFFFFFF, int(self.settings.seed)])
        self.n_iterations = self.n_trial = self.n_success = 0
        self.n_centre_moves = self.n_safety_steps = self.n_geometry_steps = 0
        self.n_rho_reductions = self.n_restarts = 0
        self.n_failed_points_skipped = 0
        self.n_failed = {"start": 0, "init": 0, "trial": 0, "geometry": 0,
                         "restart": 0}
        self.n_lstsq_fallback = self.n_subproblem_failures = 0
        self.max_condition = 0.0
        self.seconds = {"subproblem": 0.0, "residual": 0.0, "model": 0.0}

    # ------------------------------------------------------------ basics
    @property
    def z_c(self) -> np.ndarray:
        return self.Y[self.k]

    def objective(self, z: np.ndarray, r: np.ndarray) -> float:
        prior = self.prior_weight * z
        return 0.5 * float(r @ r + prior @ prior)

    def _evaluate(self, x: np.ndarray) -> tuple[np.ndarray, bool]:
        t0, n0 = time.time(), self.backend.n_calls
        r = np.asarray(self.backend.residual(x), dtype=np.float64)
        self.seconds["residual"] += time.time() - t0
        self.calls += self.backend.n_calls - n0
        if self.status == "new":
            self.status = "active"
        failed = (bool(getattr(self.backend, "last_failed", False))
                  or not bool(np.all(np.isfinite(r))))
        return r, failed

    def _record(self, kind: str, step: float | None = None,
                ratio: float | None = None) -> None:
        self.log.append([int(self.backend.n_calls), kind, float(self.F_c),
                         float(self.delta), float(self.rho),
                         None if step is None else float(step),
                         None if ratio is None else float(ratio)])

    def _others(self) -> np.ndarray:
        return np.delete(np.arange(self.p + 1), self.k)

    def _set_point(self, t: int, x: np.ndarray, r: np.ndarray, F: float,
                   count_move: bool = True) -> None:
        self.Y[t], self.R[t], self.Fy[t] = x, r, F
        if F < self.Fy[self.k]:
            self.k = int(t)
            if count_move:
                self.n_centre_moves += 1
        self.F_c = float(self.Fy[self.k])

    # ------------------------------------------------------------ model
    def interpolation_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """``D`` (rows ``y_t - z_k``, ``t != k``) and those indices ``t``."""
        idx = self._others()
        return self.Y[idx] - self.Y[self.k], idx

    def _solve(self, D: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        cond = float(np.linalg.cond(D)) if D.size else 1.0
        if not math.isfinite(cond):
            cond = math.inf
        self.max_condition = max(self.max_condition, cond)
        if cond <= COND_LIMIT:
            try:
                out = np.linalg.solve(D, rhs)
                if np.all(np.isfinite(out)):
                    return out
            except np.linalg.LinAlgError:
                pass
        self.n_lstsq_fallback += 1
        return np.linalg.lstsq(D, rhs, rcond=None)[0]

    def model_jacobian(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``J_k`` from the interpolation conditions, plus ``D`` and its
        indices."""
        t0 = time.time()
        D, idx = self.interpolation_matrix()
        dR = self.R[idx] - self.R[self.k]
        J = self._solve(D, dR).T
        self.seconds["model"] += time.time() - t0
        return J, D, idx

    def lagrange_values(self, x: np.ndarray, D: np.ndarray | None = None,
                        idx: np.ndarray | None = None) -> np.ndarray:
        """``l_t(x)`` for every ``t`` (linear Lagrange polynomials of ``Y``):
        ``x - z_k = sum_j c_j (y_{t_j} - z_k)``, ``l_{t_j} = c_j``,
        ``l_k = 1 - sum_j c_j``."""
        if D is None:
            D, idx = self.interpolation_matrix()
        v = np.asarray(x, dtype=np.float64) - self.Y[self.k]
        try:
            c = np.linalg.solve(D.T, v)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(D.T, v, rcond=None)[0]
        out = np.empty(self.p + 1)
        out[idx] = c
        out[self.k] = 1.0 - float(np.sum(c))
        return out

    def lagrange_gradient(self, t: int, D: np.ndarray,
                          idx: np.ndarray) -> np.ndarray:
        """Gradient of ``l_t`` (``t != k``): ``D^{-1} e_j``, ``t = idx[j]``."""
        j = int(np.flatnonzero(idx == t)[0])
        e = np.zeros(self.p)
        e[j] = 1.0
        try:
            return np.linalg.solve(D, e)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(D, e, rcond=None)[0]

    def step(self, J: np.ndarray) -> tuple[np.ndarray, float]:
        """Bounded Gauss-Newton step in the infinity-norm trust region and
        its predicted decrease of ``F``."""
        z, r, w = self.Y[self.k], self.R[self.k], self.prior_weight
        lb = np.maximum(-self.delta, self.lo - z)
        ub = np.minimum(self.delta, self.hi - z)
        if w > 0.0:
            A = np.vstack([J, w * np.eye(self.p)])
            b = np.concatenate([-r, -w * z])
        else:
            A, b = J, -r
        t0 = time.time()
        s = None
        for method in ("bvls", "trf"):
            try:
                with np.errstate(all="ignore"):
                    s = lsq_linear(A, b, bounds=(lb, ub), method=method).x
                if np.all(np.isfinite(s)):
                    break
                s = None
            except (ValueError, np.linalg.LinAlgError):
                s = None
        if s is None:
            self.n_subproblem_failures += 1
            s = np.zeros(self.p)
        s = np.clip(s, lb, ub)
        self.seconds["subproblem"] += time.time() - t0

        def model(step):
            rr = r + J @ step
            prior = w * (z + step)
            return 0.5 * float(rr @ rr + prior @ prior)

        return s, model(np.zeros(self.p)) - model(s)

    def farthest(self) -> tuple[int, float]:
        dist = np.max(np.abs(self.Y - self.Y[self.k]), axis=1)
        dist[self.k] = -1.0
        t = int(np.argmax(dist))
        return t, float(dist[t])

    def replacement_index(self, x: np.ndarray, D: np.ndarray,
                          idx: np.ndarray) -> int:
        """``argmax_{t != k} |l_t(x)| max(1, (||y_t - z_k||_inf / Delta)^4)``."""
        values = np.abs(self.lagrange_values(x, D, idx))
        dist = np.max(np.abs(self.Y - self.Y[self.k]), axis=1)
        weight = values * np.maximum(
            1.0, (dist / self.delta) ** self.settings.replace_power)
        weight[self.k] = -np.inf
        weight[~np.isfinite(weight)] = -np.inf
        if not np.any(np.isfinite(weight)):
            return int(self.farthest()[0])
        return int(np.argmax(weight))

    # ------------------------------------------------------------ actions
    def _queue_geometry(self, t: int, front: bool = True) -> None:
        D, idx = self.interpolation_matrix()
        g = self.lagrange_gradient(t, D, idx)
        sign = np.sign(g)
        if not np.any(sign):
            sign = self.rng.choice([-1.0, 1.0], size=self.p)
        z = self.Y[self.k]
        plus = np.clip(z + self.delta * sign, self.lo, self.hi)
        minus = np.clip(z - self.delta * sign, self.lo, self.hi)
        if abs(g @ (plus - z)) >= abs(g @ (minus - z)):
            first, second = plus, minus
        else:
            first, second = minus, plus
        action = ("geometry", t, first, second)
        if front:
            self.pending.appendleft(action)
        else:
            self.pending.append(action)

    def _reduce_rho(self) -> None:
        s = self.settings
        if self.rho <= s.rho_end * (1.0 + 1e-9):
            self._soft_restart()
            return
        rho_old = self.rho
        self.rho = max(s.rho_factor * self.rho, s.rho_end)
        self.delta = max(0.5 * rho_old, self.rho)
        self.n_rho_reductions += 1
        self._record("rho")

    def _soft_restart(self) -> None:
        s = self.settings
        self.n_restarts += 1
        if s.max_restarts is not None and self.n_restarts > s.max_restarts:
            self.status, self.stop_reason = "stationary", "max_restarts"
            self._record("stop_max_restarts")
            return
        self.rho = self.delta = float(s.rho_beg)
        z = self.Y[self.k].copy()
        n_move = min(int(s.restart_points), self.p)
        for t in self.rng.choice(self._others(), size=n_move, replace=False):
            u = self.rng.standard_normal(self.p)
            u /= max(float(np.max(np.abs(u))), 1e-300)
            x = z + s.rho_beg * u
            out = (x < self.lo) | (x > self.hi)
            x[out] = z[out] - s.rho_beg * u[out]
            self.pending.append(("restart", int(t), np.clip(x, self.lo, self.hi)))
        self._record("restart")

    def _init_candidates(self, j: int) -> list[float]:
        h = float(self.settings.rho_beg)
        sign = 1.0 if self.z0[j] + h <= self.hi[j] else -1.0
        steps = [sign * h, -sign * h, sign * h / 2, -sign * h / 2]
        return [a for a in steps if self.lo[j] <= self.z0[j] + a <= self.hi[j]]

    def _do(self, action: tuple) -> None:
        kind = action[0]
        if kind == "start":
            r, failed = self._evaluate(self.z0)
            if failed:
                self.n_failed["start"] += 1
                self.n_failed_points_skipped += 1
                self.status, self.stop_reason = "stationary", "start_failed"
                self.pending.clear()
                self._record("start_failed")
                return
            self.R = np.tile(r, (self.p + 1, 1))
            F = self.objective(self.z0, r)
            self.Y[0], self.R[0], self.Fy[0], self.k = self.z0, r, F, 0
            self.F_c = self.F_start = F
            self._record("start")
        elif kind == "init":
            _, j, attempt = action
            candidates = self._init_candidates(j)
            if attempt >= len(candidates):
                self.status, self.stop_reason = "stationary", "init_failed"
                self.pending.clear()
                self._record("init_failed")
                return
            x = self.z0.copy()
            x[j] += candidates[attempt]
            r, failed = self._evaluate(x)
            if failed:
                self.n_failed["init"] += 1
                self.n_failed_points_skipped += 1
                if attempt + 1 < len(candidates):
                    self.pending.appendleft(("init", j, attempt + 1))
                else:
                    self.status, self.stop_reason = "stationary", "init_failed"
                    self.pending.clear()
                self._record("init_failed")
            else:
                self._set_point(j + 1, x, r, self.objective(x, r),
                                count_move=False)
                self._record("init")
            if self.status != "stationary" and not any(
                    a[0] == "init" for a in self.pending):
                self.initialized = True
        elif kind == "geometry":
            _, t, x, alternative = action
            self.n_geometry_steps += 1
            r, failed = self._evaluate(x)
            if failed:
                self.n_failed["geometry"] += 1
                self.n_failed_points_skipped += 1
                if alternative is not None:
                    self.pending.appendleft(("geometry", t, alternative, None))
                else:
                    self.delta = max(self.rho,
                                     self.settings.gamma_dec * self.delta)
                self._record("geometry_failed")
            else:
                self._set_point(t, x, r, self.objective(x, r))
                self._record("geometry")
        elif kind == "restart":
            _, t, x = action
            r, failed = self._evaluate(x)
            if failed:
                self.n_failed["restart"] += 1
                self.n_failed_points_skipped += 1
                self._record("restart_failed")
            else:
                self._set_point(t, x, r, self.objective(x, r))
                self._record("restart_point")
        else:
            raise ValueError(f"unknown action {kind!r}")

    def _iterate(self) -> None:
        """One iteration: model, step, then a trial call or a safety phase."""
        s = self.settings
        self.n_iterations += 1
        J, D, idx = self.model_jacobian()
        step, predicted = self.step(J)
        size = float(np.max(np.abs(step))) if self.p else 0.0
        if size < s.safety * self.rho or not predicted > 0.0:
            self.n_safety_steps += 1
            self.delta = max(self.rho, s.gamma_dec * self.delta)
            self._record("safety", size, None)
            t, dist = self.farthest()
            if dist > s.far * self.delta:
                self._queue_geometry(t)
            else:
                self._reduce_rho()
            return
        z = self.Y[self.k]
        x = np.clip(z + step, self.lo, self.hi)
        F_k = self.F_c
        self.n_trial += 1
        r, failed = self._evaluate(x)
        if failed:
            self.n_failed["trial"] += 1
            self.n_failed_points_skipped += 1
            ratio = -math.inf
        else:
            F_new = self.objective(x, r)
            ratio = (F_k - F_new) / predicted
        if ratio < s.eta1:
            self.delta = min(s.gamma_dec * self.delta, size)
        elif ratio <= s.eta2:
            self.delta = max(s.gamma_dec * self.delta, size)
        else:
            self.delta = min(max(s.gamma_inc * self.delta,
                                 s.gamma_inc_overline * size), s.delta_max)
        self.delta = max(self.delta, self.rho)
        if ratio >= s.eta1:
            self.n_success += 1
        if not failed:
            t = self.replacement_index(x, D, idx)
            self._set_point(t, x, r, F_new)
        self._record("trial_failed" if failed else "trial", size, ratio)
        if ratio < s.eta1:
            t, dist = self.farthest()
            if dist > s.far * self.delta:
                self._queue_geometry(t)
            elif size <= self.rho:
                self._reduce_rho()

    # ------------------------------------------------------------ driver
    def advance(self, limit: int) -> str:
        """Iterate until stopped or the next call would take the backend's
        counter past ``limit``; resumable."""
        if self.status == "stationary":
            return self.status
        if self.status == "budget":
            self.status = "active"
        if not self.initialized and self.R is None and not self.pending:
            self.pending.append(("start",))
            self.pending.extend(("init", j, 0) for j in range(self.p))
            if self.p == 0:
                self.initialized = True
        while self.status != "stationary":
            if self.initialized and self.F_c == 0.0:
                self.status, self.stop_reason = "stationary", "zero_objective"
                break
            if self.backend.n_calls + 1 > limit:
                if self.status == "active":
                    self.status = "budget"
                break
            if self.pending:
                self._do(self.pending.popleft())
            else:
                self._iterate()
        return self.status

    def summary(self) -> dict:
        return {
            "label": self.label, "status": self.status,
            "stop_reason": self.stop_reason,
            "F_start": self.F_start, "F_final": self.F_c,
            "delta_final": self.delta, "rho_final": self.rho,
            "calls": self.calls, "n_iterations": self.n_iterations,
            "n_trial": self.n_trial, "n_success": self.n_success,
            "n_centre_moves": self.n_centre_moves,
            "n_safety_steps": self.n_safety_steps,
            "n_geometry_steps": self.n_geometry_steps,
            "n_rho_reductions": self.n_rho_reductions,
            "n_restarts": self.n_restarts,
            "n_failed_points_skipped": self.n_failed_points_skipped,
            "n_failed": dict(self.n_failed),
            "n_lstsq_fallback": self.n_lstsq_fallback,
            "n_subproblem_failures": self.n_subproblem_failures,
            "max_condition": self.max_condition,
            "seconds": {k: round(v, 3) for k, v in self.seconds.items()},
            "settings": asdict(self.settings),
            "step_norm_note": STEP_NORM_NOTE,
            "log_fields": list(LOG_FIELDS), "log": self.log,
        }


def finish_budget(runs: list[DfoLsRun], backend, budget: int) -> int:
    """Calls left after ``trust_region.run_schedule`` go to the run with the
    lowest ``F_c`` that can still use them (``active``, ``new`` or paused at
    a limit, ``budget``); run_schedule's own final loop skips paused runs.
    Returns the number of calls spent here."""
    start = backend.n_calls
    while backend.n_calls < budget:
        live = [r for r in runs if r.status in ("active", "new", "budget")]
        if not live:
            break
        before = backend.n_calls
        min(live, key=lambda r: r.F_c).advance(budget)
        if backend.n_calls == before:
            break
    return backend.n_calls - start
