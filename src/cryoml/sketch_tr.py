"""Trust-region model management with sketched, transported Jacobians.

``cryoml.trust_region.TrustRegionRun`` pays a full finite-difference
Jacobian of the expensive residual (one call per active parameter, 43 here)
at every accepted centre.  ``SketchTrustRegionRun`` keeps the same objective,
corrected model, bounded TRF subproblem, acceptance test, radius rules and
box, and replaces most of those Jacobians by

* a transported prior at the new centre: with a surrogate,
  ``J0 = J_s(z_new) + (J_c - J_s(z_c))`` (the model at ``z_new`` keeps the
  last measured slope correction); without one (``r_s = 0``), ``J0 = J_c``;
* a free secant (good Broyden) update of ``J0`` along the accepted step,
  ``J0 += (y - J0 s) s^T / (s^T s)`` when ``|s|_inf <= secant_max_step``;
* ``k`` directional probes ``Y = (r(z_c + h v) - r_c) / h`` along an
  orthonormal sketch ``V`` (the linearized Gauss-Newton step, the top right
  singular vectors of ``J0`` and one Gaussian direction, on the coordinates
  away from the box faces), merged by the block update
  ``J = J0 + (Y - J0 V) V^T`` so that ``J V = Y`` exactly;
* a projected Broyden update on rejected steps,
  ``J + (y - J s) s_perp^T / (s_perp^T s)`` with ``s_perp = s - V V^T s``,
  which keeps ``J V = Y`` and gives ``J s = y``.

A full Jacobian is still taken at the first centre, after ``refresh_every``
accepted centres, and whenever the estimate is demonstrably wrong: probe
mismatch ``|Y - J0 V|_F / |J0 V|_F > mismatch_refresh``, ``poor_refresh``
poor steps at one centre, a stationary or zero step (only a fresh exact
Jacobian can certify stationarity), or a radius collapse (once per centre,
radius reset to ``collapse_delta``).  A fresh full Jacobian is never
secant-updated, so ``refresh_every = 0`` reproduces ``TrustRegionRun``
iterate for iterate.  With ``surrogate = None`` the same code is a
Broyden / random-subspace Gauss-Newton trust region in the spirit of DFO-LS
(Cartis, Fiala, Marteau and Roberts 2019; Cartis, Fowkes and Shao 2022): the
control that isolates what the surrogate contributes.

The expensive backend needs ``n_calls``, ``jacobian_cost``, ``residual(z)``,
``jacobian(z, r)`` and ``directional(z, r, V, probe_step) -> (Y, kept)``
(``Y`` m x k forward differences along the columns of ``V``; ``kept`` marks
the columns that are usable); the surrogate needs ``value(z)`` and
``jacobian(z)``.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import lsq_linear

from cryoml.trust_region import TrustRegionRun, TrustRegionSettings

REFRESH_REASONS = ("start", "periodic", "mismatch", "poor", "stationary",
                   "collapse")


@dataclass
class SketchSettings:
    k: int = 8                      # directions per sketch
    probe_step: float = 1e-3        # inf-norm probe length, local units
    secant_max_step: float = 0.15   # secant updates only for |s|_inf <= this
    refresh_every: int = 6          # accepted centres between full Jacobians
    mismatch_refresh: float = 0.5   # |Y - J0 V|_F / |J0 V|_F above: full
    poor_refresh: int = 2           # poor steps at one centre: full
    collapse_delta: float = 0.05    # radius after a collapse refresh
    min_dir_norm: float = 1e-8      # Gram-Schmidt drop tolerance
    seed: int = 0


def block_update(J0: np.ndarray, V: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """``J0 + (Y - J0 V) V^T``: equals ``Y`` on the columns of an
    orthonormal ``V`` and ``J0`` on their orthogonal complement."""
    if V.shape[1] == 0:
        return np.array(J0, dtype=np.float64, copy=True)
    return J0 + (Y - J0 @ V) @ V.T


def projected_broyden(J: np.ndarray, s: np.ndarray, y: np.ndarray,
                      V: np.ndarray | None) -> np.ndarray:
    """Rank-one secant update ``J s = y`` that leaves ``J V`` unchanged.

    Returns ``J`` itself (the same object) when the part of ``s``
    orthogonal to ``V`` is below 1e-3 of ``|s|`` (no new information)."""
    s = np.asarray(s, dtype=np.float64)
    s_perp = s if V is None or V.shape[1] == 0 else s - V @ (V.T @ s)
    norm_s = float(np.linalg.norm(s))
    if norm_s == 0.0 or float(np.linalg.norm(s_perp)) < 1e-3 * norm_s:
        return J
    return J + np.outer(y - J @ s, s_perp) / float(s_perp @ s)


def _orthonormalize(columns: list[np.ndarray], tol: float,
                    limit: int) -> list[np.ndarray]:
    """Gram-Schmidt in order with one re-orthogonalization pass; each
    column is normalized first, so ``tol`` is relative."""
    basis: list[np.ndarray] = []
    for column in columns:
        norm = float(np.linalg.norm(column))
        if not np.isfinite(norm) or norm == 0.0:
            continue
        v = column / norm
        for _ in range(2):
            for b in basis:
                v = v - (b @ v) * b
        norm = float(np.linalg.norm(v))
        if norm < tol:
            continue
        basis.append(v / norm)
        if len(basis) == limit:
            break
    return basis


def choose_directions(J0: np.ndarray, r_c: np.ndarray, z_c: np.ndarray,
                      w: float, delta: float, free: np.ndarray, k: int,
                      rng: np.random.Generator,
                      min_norm: float = 1e-8) -> np.ndarray:
    """Orthonormal sketch ``V`` (n x k', k' <= k), zero off ``free``.

    Order: the bounded Gauss-Newton step of the linearized model
    ``min |r_c + J0 s|^2 + |w (z_c + s)|^2`` over (box ∩ trust region) on the
    free coordinates (``lsq_linear``; no expensive calls), then the top
    ``k - 2`` right singular vectors of ``J0[:, free]``, then one Gaussian
    direction; Gram-Schmidt drops near-dependent columns."""
    z_c = np.asarray(z_c, dtype=np.float64)
    n = len(z_c)
    index = np.flatnonzero(free)
    n_free = len(index)
    if n_free == 0 or k <= 0:
        return np.zeros((n, 0))
    J_free = np.asarray(J0, dtype=np.float64)[:, index]
    columns: list[np.ndarray] = []
    lo = np.maximum(-1.0, z_c - delta)[index] - z_c[index]
    hi = np.minimum(1.0, z_c + delta)[index] - z_c[index]
    if np.all(lo < hi):
        A = np.vstack([J_free, w * np.eye(n_free)])
        b = -np.concatenate([r_c, w * z_c[index]])
        columns.append(lsq_linear(A, b, bounds=(lo, hi)).x)
    n_sv = max(0, min(k - 2, n_free))
    if n_sv:
        # Non-finite prior entries (penalized probes) and LAPACK's rare
        # non-convergence fall back to the eigenvectors of J^T J, then to
        # random directions; finite, convergent cases are unchanged.
        J_svd = np.nan_to_num(J_free, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            _, _, vt = np.linalg.svd(J_svd, full_matrices=False)
        except np.linalg.LinAlgError:
            try:
                _, vecs = np.linalg.eigh(J_svd.T @ J_svd)
                vt = vecs[:, ::-1].T
            except np.linalg.LinAlgError:
                vt = rng.standard_normal((n_sv, n_free))
        columns.extend(vt[:n_sv])
    if k >= 2:
        columns.append(rng.standard_normal(n_free))
    basis = _orthonormalize(columns, min_norm, min(k, n_free))
    V = np.zeros((n, len(basis)))
    for j, v in enumerate(basis):
        V[index, j] = v
    return V


class SketchTrustRegionRun(TrustRegionRun):
    """``TrustRegionRun`` with the sketched-Jacobian policy (module doc).

    ``stream`` offsets the direction generator's seed (the driver passes
    ``crc32(tag) + start index``)."""

    def __init__(self, z0: np.ndarray, expensive, surrogate,
                 prior_weight: float, settings: TrustRegionSettings,
                 sketch: SketchSettings | None = None, label: str = "",
                 stream: int = 0):
        super().__init__(z0, expensive, surrogate, prior_weight, settings,
                         label)
        self.sketch = sketch or SketchSettings()
        self.rng = np.random.default_rng(
            (int(self.sketch.seed) + int(stream)) % 2 ** 63)
        self.J_exact = False            # J_c is a fresh full FD at z_c
        self.V_c = None                 # probed directions at z_c
        self.J0_pending = None          # transported prior for the next centre
        self.accepted_since_full = 0
        self.poor_at_centre = 0
        self.collapse_refreshed = False
        self.pending = None             # full refresh owed (reason)
        self.seconds.update({"probe": 0.0, "directions": 0.0})
        self.n_full = {reason: 0 for reason in REFRESH_REASONS}
        self.n_sketch = self.n_secant_accept = self.n_broyden_reject = 0
        self.n_dropped_directions = 0
        self.probe_calls = self.full_calls = 0
        self.mismatches: list[float] = []
        self.events: list[dict] = []

    # expensive calls -------------------------------------------------
    def _directional(self, z, r, V):
        e = self.expensive
        t0, n0 = time.time(), e.n_calls
        Y, kept = e.directional(z, r, V, self.sketch.probe_step)
        self.seconds["probe"] += time.time() - t0
        self.calls += e.n_calls - n0
        self.probe_calls += e.n_calls - n0
        return np.asarray(Y, dtype=np.float64), np.asarray(kept, dtype=bool)

    def _full_refresh(self, reason: str) -> None:
        n0 = self.expensive.n_calls
        self.J_c = self._jacobian(self.z_c, self.r_c)
        self.full_calls += self.expensive.n_calls - n0
        self.J_exact = True
        self.V_c = None
        self.model = None
        self.accepted_since_full = 0
        self.poor_at_centre = 0
        self.pending = None
        self.n_full[reason] += 1
        if reason == "collapse":
            self.delta = max(self.delta, self.sketch.collapse_delta)
            self.collapse_refreshed = True
        self.events.append({"calls": self.expensive.n_calls, "kind": "full",
                            "reason": reason})

    def _full_affordable(self, limit: int) -> bool:
        e = self.expensive
        return e.n_calls + e.jacobian_cost + 1 <= limit

    def _centre_jacobian(self, limit: int) -> bool:
        """Jacobian policy at a new centre; False if the budget cannot pay
        for it plus one trial."""
        e, k = self.expensive, self.sketch
        J0 = self.J0_pending
        if J0 is None or self.accepted_since_full >= k.refresh_every:
            if not self._full_affordable(limit):
                return False
            self._full_refresh("start" if J0 is None else "periodic")
            return True
        free = np.abs(self.z_c) < 1.0 - 2.0 * k.probe_step
        if e.n_calls + min(k.k, int(free.sum())) + 1 > limit:
            return False
        t0 = time.time()
        V = choose_directions(J0, self.r_c, self.z_c, self.prior_weight,
                              self.delta, free, k.k, self.rng, k.min_dir_norm)
        self.seconds["directions"] += time.time() - t0
        event = {"calls": None, "kind": "sketch", "n_free": int(free.sum()),
                 "k": int(V.shape[1])}
        mismatch = None
        if V.shape[1]:
            Y, kept = self._directional(self.z_c, self.r_c, V)
            self.n_sketch += 1
            self.n_dropped_directions += int((~kept).sum())
            V, Y = V[:, kept], Y[:, kept]
            JV = J0 @ V
            mismatch = (float(np.linalg.norm(Y - JV))
                        / max(float(np.linalg.norm(JV)), 1e-30)
                        if V.shape[1] else math.inf)
            self.mismatches.append(mismatch)
            event.update({"kept": int(V.shape[1]), "mismatch": mismatch})
        event["calls"] = e.n_calls
        self.events.append(event)
        if mismatch is not None and mismatch > k.mismatch_refresh:
            if self._full_affordable(limit):
                self._full_refresh("mismatch")
                return True
            event["refresh_unaffordable"] = True
        self.J_c = J0 if not V.shape[1] else block_update(J0, V, Y)
        self.V_c = V
        self.J_exact = False
        self.model = None
        return True

    # trust-region loop -----------------------------------------------
    def _accept(self, z_new, r_new, F_new) -> None:
        s_vec, y = z_new - self.z_c, r_new - self.r_c
        t0 = time.time()
        J0 = self.model.slope.copy()        # J_c - J_s(z_c), or J_c
        if self.surrogate is not None:
            J0 = self.surrogate.jacobian(z_new) + J0
        ss = float(s_vec @ s_vec)
        if ss > 0.0 and np.max(np.abs(s_vec)) <= self.sketch.secant_max_step:
            J0 = J0 + np.outer(y - J0 @ s_vec, s_vec) / ss
            self.n_secant_accept += 1
        self.seconds["directions"] += time.time() - t0
        self.n_accepted += 1
        self.z_c, self.r_c, self.F_c = z_new, r_new, F_new
        self.J_c = self.model = None
        self.J0_pending = J0
        self.J_exact = False
        self.V_c = None
        self.accepted_since_full += 1
        self.poor_at_centre = 0
        self.collapse_refreshed = False

    def _reject(self, z_new, r_new, rho) -> None:
        self.n_rejected += 1
        s_vec = z_new - self.z_c
        if (not self.J_exact
                and np.max(np.abs(s_vec)) <= self.sketch.secant_max_step):
            J = projected_broyden(self.J_c, s_vec, r_new - self.r_c, self.V_c)
            if J is not self.J_c:
                self.J_c, self.model = J, None
                self.n_broyden_reject += 1
        if rho < self.settings.poor:
            self.poor_at_centre += 1
            if (self.poor_at_centre >= self.sketch.poor_refresh
                    and not self.J_exact):
                self.pending = "poor"

    def advance(self, limit: int) -> str:
        """As ``TrustRegionRun.advance`` (same subproblem, acceptance and
        radius rules) with the sketched-Jacobian policy; stops when the next
        Jacobian plus one trial would take the shared counter past
        ``limit``."""
        s = self.settings
        e = self.expensive
        if self.status == "new":
            if e.n_calls + 1 > limit:
                return self.status
            self.r_c = self._residual(self.z_c)
            self.F_c = self.F_start = self.objective(self.z_c, self.r_c)
            self.status = "active"
        while self.status == "active":
            if self.pending is not None:
                if self._full_affordable(limit):
                    self._full_refresh(self.pending)
                elif self.pending in ("stationary", "collapse"):
                    break
                else:                   # poor: go on with the estimate
                    self.pending = None
            if self.J_c is None and not self._centre_jacobian(limit):
                break
            if e.n_calls + 1 > limit:
                break
            z_new, predicted = self.solve_subproblem()
            step = float(np.max(np.abs(z_new - self.z_c))) if len(z_new) else 0.0
            if predicted <= s.stationary * max(self.F_c, 1e-300) or step == 0.0:
                if not self.J_exact:
                    self.pending = "stationary"
                    continue
                self.status = "stationary"
                self.log.append({"calls": e.n_calls, "F": self.F_c,
                                 "delta": self.delta, "step": step,
                                 "predicted": predicted, "stationary": True})
                break
            model_exact = self.J_exact
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
                self._accept(z_new, r_new, F_new)
            else:
                self._reject(z_new, r_new, rho)
            if self.delta < s.delta_min:
                if not model_exact and not self.collapse_refreshed:
                    self.pending = "collapse"
                else:
                    self.status = "radius_collapsed"
        return self.status

    def next_iteration_cost(self) -> int:
        """Expensive calls the next trial needs, Jacobian work included
        (a new run also pays its first residual)."""
        e, k = self.expensive, self.sketch
        if self.status == "new":
            return 2 + e.jacobian_cost
        if self.status != "active":
            return 0
        if self.pending is not None:
            return 1 + e.jacobian_cost
        if self.J_c is not None:
            return 1
        if self.J0_pending is None or self.accepted_since_full >= k.refresh_every:
            return 1 + e.jacobian_cost
        return 1 + k.k

    def sketch_stats(self) -> dict:
        rho = [row["rho"] for row in self.log if "rho" in row]
        return {"k": self.sketch.k, "sketch_settings": asdict(self.sketch),
                "n_full_refresh": int(sum(self.n_full.values())),
                "full_refresh_reasons": dict(self.n_full),
                "n_sketch": self.n_sketch,
                "n_secant_accept": self.n_secant_accept,
                "n_broyden_reject": self.n_broyden_reject,
                "n_dropped_directions": self.n_dropped_directions,
                "probe_calls": self.probe_calls, "full_calls": self.full_calls,
                "mismatch": self.mismatches, "rho": rho,
                "calls_per_accepted_step": (self.calls / self.n_accepted
                                            if self.n_accepted else math.inf)}

    def summary(self) -> dict:
        return {**super().summary(), **self.sketch_stats(),
                "events": self.events}
