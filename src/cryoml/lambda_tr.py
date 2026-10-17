"""Per-curve remainder-weighted corrected model (research node n2b).

``cryoml.trust_region.CorrectedModel`` adds the surrogate's whole
second-order remainder to the NGSpice linearization at the centre ``z_c``::

    r_c(z) = r_c + J_c s + [r_s(z) - r_s(z_c) - J_s(z_c) s],   s = z - z_c.

Here the remainder is weighted per included metric curve ``j``::

    r_lam(z) = r_c + J_c s + D_lam [r_s(z) - r_s(z_c) - J_s(z_c) s],

with ``D_lam`` diagonal over the data rows, ``lambda_j`` on the rows of curve
``j`` (the rows of ``NgspiceResiduals.layout`` entry ``j``, starting at
``curve_starts[j]``).  The prior rows ``sqrt(reg / n_active) z`` are
unchanged.  ``lambda = 0`` is the Gauss-Newton linearization (the
``r_s = 0`` control) and ``lambda = 1`` the full corrected model.

Implementation by composition: ``ScaledSurrogate(base, row_scale)`` returns
``row_scale * base.value(z)`` and ``row_scale[:, None] * base.jacobian(z)``
and is passed as ``surrogate`` to the unchanged ``CorrectedModel``, whose
offset becomes ``r_c - D r_s(z_c)`` and slope ``J_c - D J_s(z_c)``.  In the
sketch family ``SketchTrustRegionRun._accept`` then transports
``J0 = J_c - D J_s(z_c) + D J_s(z_new)`` (plus its secant update).

``LambdaMixin`` overrides two methods of the trust-region run:

* ``solve_subproblem``: if the lambda the mode asks for differs from the one
  in use, it sets the wrapper's ``row_scale`` and drops the model; it builds
  the model (checking centre consistency) and calls the parent.  The weight
  changes only here, so ``_accept`` and ``_reject`` use the ``D`` that
  produced the step.
* ``_residual``: after the parent's NGSpice call, at a trial point it forms
  ``e_f = r(z) - r_c - J_c s`` and ``e_s = r_s(z) - r_s(z_c) - J_s(z_c) s``
  (unscaled base surrogate: one forward at ``z``; ``r_s(z_c)``, ``J_s(z_c)``
  cached per centre) and, if the trial is usable, updates the estimator.
  No NGSpice call is added.

Usable trial: hybrid family (full FD ``J_c``) always; sketch family when
``J_c`` is a fresh full Jacobian or ``|s - V_c V_c^T s|_2 <= unprobed_tol
|s|_2`` (``V_c`` the probed directions at ``z_c``).

Estimator per start, per curve ``j`` (fixed before any DEV run: ``forget``
0.7, ``prior_tau`` 1, ``unprobed_tol`` 0.1, ``cusum_h`` 2.0): at the first
usable trial ``num_j = den_j = tau |e_s,j|^2`` (prior lambda 1); at every
usable trial ``num_j <- f num_j + <e_f,j, e_s,j>``, ``den_j <- f den_j +
|e_s,j|^2``; ``lambda_j = clip(num_j / den_j, 0, 1)`` (unchanged when
``den_j = 0``; 1 before the first usable trial).

Modes: ``curve`` (per-curve lambda), ``scalar`` (one lambda from the sums
over curves), ``zero`` (D = 0), ``one`` (D = I), ``cusum`` (lambda 1 until
``S_t = max(0, S_{t-1} + ln(|e_f - e_s|^2 / |e_f|^2)) >= cusum_h`` on usable
trials, then 0 for the rest of the start), ``radius`` (``lambda_j = min(1,
Delta / delta0_ref)``, set at each subproblem).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from cryoml.sketch_tr import SketchSettings, SketchTrustRegionRun
from cryoml.trust_region import (CorrectedModel, TrustRegionRun,
                                 TrustRegionSettings)

LAMBDA_MODES = ("curve", "scalar", "zero", "one", "cusum", "radius")
TRIAL_MARKS = (1, 3, 10, 30, 100)
CONSISTENCY_TOL = 1e-10


@dataclass
class LambdaSettings:
    mode: str = "curve"
    forget: float = 0.7             # f, applied to the prior as well
    prior_tau: float = 1.0          # tau: prior weight (prior lambda = 1)
    unprobed_tol: float = 0.1       # sketch family usable-trial tolerance
    cusum_h: float = 2.0            # CUSUM threshold
    delta0_ref: float = 0.25        # radius mode: lambda = min(1, Delta / this)

    def __post_init__(self):
        if self.mode not in LAMBDA_MODES:
            raise ValueError(f"unknown lambda mode {self.mode!r}")


class ScaledSurrogate:
    """``row_scale * r_s`` and ``row_scale[:, None] * J_s``.

    An all-zero scale returns zeros without a surrogate call (so ``D = 0``
    does not propagate a non-finite surrogate value) and an all-one scale
    returns the base arrays (``1.0 * x == x``); both equal the general
    product wherever the surrogate is finite."""

    def __init__(self, base, row_scale):
        self.base = base
        self.row_scale = np.asarray(row_scale, dtype=np.float64).copy()

    def _kind(self) -> str:
        if not np.any(self.row_scale):
            return "zero"
        if np.all(self.row_scale == 1.0):
            return "one"
        return "general"

    def value(self, z: np.ndarray) -> np.ndarray:
        kind = self._kind()
        if kind == "zero":
            return np.zeros(len(self.row_scale))
        value = self.base.value(z)
        return value if kind == "one" else self.row_scale * value

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        kind = self._kind()
        if kind == "zero":
            return np.zeros((len(self.row_scale), len(np.asarray(z))))
        jac = self.base.jacobian(z)
        return jac if kind == "one" else self.row_scale[:, None] * jac


def curve_slices(curve_starts, n_rows: int) -> list[slice]:
    """Row slice of each included curve from ``NgspiceResiduals.curve_starts``."""
    starts = [int(s) for s in curve_starts]
    stops = starts[1:] + [int(n_rows)]
    if starts and (starts[0] != 0 or any(b < a for a, b in zip(starts, stops))):
        raise ValueError(f"bad curve starts {starts} for {n_rows} rows")
    return [slice(a, b) for a, b in zip(starts, stops)]


def rows_from_curves(values, slices: list[slice], n_rows: int) -> np.ndarray:
    rows = np.empty(n_rows, dtype=np.float64)
    for value, sl in zip(values, slices):
        rows[sl] = value
    return rows


class LambdaEstimator:
    """Discounted least-squares weight of the surrogate remainder, per curve
    (and pooled), and the CUSUM statistic; one instance per start."""

    def __init__(self, n_curves: int, settings: LambdaSettings):
        self.settings = settings
        self.num = np.zeros(n_curves)
        self.den = np.zeros(n_curves)
        self.curve = np.ones(n_curves)      # lambda_j
        self.scalar = 1.0
        self.started = False
        self.n_updates = 0
        self.cusum = 0.0
        self.switched = False
        self.switch_update = None           # usable-trial index of the switch
        self.switch_call = None

    def update(self, e_f: np.ndarray, e_s: np.ndarray, slices: list[slice],
               calls: int | None = None) -> float:
        """One usable trial; returns the CUSUM increment ``u_t``."""
        s = self.settings
        cross = np.array([float(e_f[sl] @ e_s[sl]) for sl in slices])
        square = np.array([float(e_s[sl] @ e_s[sl]) for sl in slices])
        # A curve with a non-finite remainder (non-finite surrogate output)
        # keeps its accumulators; finite inputs are unaffected.
        finite = np.isfinite(cross) & np.isfinite(square)
        cross, square = np.where(finite, cross, 0.0), np.where(finite, square, 0.0)
        if not self.started:
            self.num = s.prior_tau * square
            self.den = s.prior_tau * square
            self.started = True
        self.num = np.where(finite, s.forget * self.num + cross, self.num)
        self.den = np.where(finite, s.forget * self.den + square, self.den)
        positive = self.den > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.clip(self.num / self.den, 0.0, 1.0)
        self.curve = np.where(positive & np.isfinite(ratio), ratio, self.curve)
        num, den = float(self.num.sum()), float(self.den.sum())
        if den > 0.0 and math.isfinite(num / den):
            self.scalar = float(np.clip(num / den, 0.0, 1.0))
        self.n_updates += 1
        miss_one = float((e_f - e_s) @ (e_f - e_s))   # error of lambda = 1
        miss_zero = float(e_f @ e_f)                  # error of lambda = 0
        if miss_one == 0.0 and miss_zero == 0.0:
            u = 0.0
        else:
            u = (math.log(max(miss_one, 1e-300))
                 - math.log(max(miss_zero, 1e-300)))
        if not math.isfinite(u):
            u = 0.0 if math.isnan(u) else u
        self.cusum = max(0.0, self.cusum + u)
        if not self.switched and self.cusum >= s.cusum_h:
            self.switched = True
            self.switch_update = self.n_updates
            self.switch_call = calls
        return u

    def target(self, mode: str, delta: float) -> np.ndarray:
        """Per-curve lambda the mode asks for now."""
        n = len(self.curve)
        if mode == "curve":
            return self.curve.copy()
        if mode == "scalar":
            return np.full(n, self.scalar)
        if mode == "zero":
            return np.zeros(n)
        if mode == "one":
            return np.ones(n)
        if mode == "cusum":
            return np.full(n, 0.0 if self.switched else 1.0)
        if mode == "radius":
            return np.full(n, min(1.0, float(delta) / self.settings.delta0_ref))
        raise ValueError(mode)


def _round(x: float, digits: int = 6):
    if x is None or not math.isfinite(x):
        return None if x is None else float(x)
    return float(f"{x:.{digits}g}")


class LambdaMixin:
    """Remainder-weighted model for a ``TrustRegionRun`` subclass (module
    doc).  The run's ``surrogate`` is a ``ScaledSurrogate`` around the base
    surrogate."""

    def _lambda_init(self, base_surrogate, curve_starts, n_rows: int,
                     lambda_settings: LambdaSettings) -> None:
        if base_surrogate is None:
            raise ValueError("the lambda model needs the base surrogate")
        self.lam = lambda_settings
        self.slices = curve_slices(curve_starts, n_rows)
        self.n_rows = int(n_rows)
        self.estimator = LambdaEstimator(len(self.slices), lambda_settings)
        self.lambda_in_use = self.estimator.target(lambda_settings.mode,
                                                   self.delta)
        self.surrogate = ScaledSurrogate(
            base_surrogate, rows_from_curves(self.lambda_in_use, self.slices,
                                             self.n_rows))
        self.base_surrogate = base_surrogate
        self._centre_key = None
        self._centre_rs = self._centre_Js = None
        self.trial_log: list[dict] = []
        self.seconds["lambda"] = 0.0        # estimator and consistency work
        self.n_lambda_changes = 0
        self.consistency_stats = {"n_builds": 0, "n_violations": 0,
                                  "max_value_error_rel": 0.0,
                                  "max_jacobian_error_rel": 0.0}

    # lambda changes only here ---------------------------------------
    def solve_subproblem(self):
        target = self.estimator.target(self.lam.mode, self.delta)
        if not np.array_equal(target, self.lambda_in_use):
            self.lambda_in_use = target
            self.surrogate.row_scale = rows_from_curves(target, self.slices,
                                                        self.n_rows)
            self.model = None
            self.n_lambda_changes += 1
        if self.model is None:
            self.model = CorrectedModel(self.surrogate, self.z_c, self.r_c,
                                        self.J_c, self.prior_weight)
            t0 = time.time()
            self._check_consistency()
            self.seconds["lambda"] += time.time() - t0
        return super().solve_subproblem()

    def _check_consistency(self) -> dict:
        row = self.model.consistency(self.r_c, self.J_c)
        value_tol = CONSISTENCY_TOL * max(1.0, row["max_abs_value"])
        jac_tol = CONSISTENCY_TOL * max(1.0, row["max_abs_jacobian"])
        st = self.consistency_stats
        st["n_builds"] += 1
        ok = (row["max_abs_value_error"] <= value_tol
              and row["max_abs_jacobian_error"] <= jac_tol)
        if not ok:
            st["n_violations"] += 1
        st["max_value_error_rel"] = max(
            st["max_value_error_rel"],
            row["max_abs_value_error"] / max(1.0, row["max_abs_value"]))
        st["max_jacobian_error_rel"] = max(
            st["max_jacobian_error_rel"],
            row["max_abs_jacobian_error"] / max(1.0, row["max_abs_jacobian"]))
        return row

    # zero-call estimator update ---------------------------------------
    def _centre_surrogate(self):
        key = self.z_c.tobytes()
        if key != self._centre_key:
            self._centre_rs = np.array(self.base_surrogate.value(self.z_c),
                                       dtype=np.float64, copy=True)
            self._centre_Js = np.array(self.base_surrogate.jacobian(self.z_c),
                                       dtype=np.float64, copy=True)
            self._centre_key = key
        return self._centre_rs, self._centre_Js

    def _usable(self, s: np.ndarray) -> bool:       # family-specific
        raise NotImplementedError

    def _residual(self, z):
        failed_before = self._n_failed()
        r = super()._residual(z)
        z = np.asarray(z, dtype=np.float64)
        if self.model is not None and not np.array_equal(z, self.z_c):
            t0 = time.time()
            self._observe(z, r, self._n_failed() > failed_before)
            self.seconds["lambda"] += time.time() - t0
        return r

    def _n_failed(self) -> int:
        evaluator = getattr(self.expensive, "evaluator", None)
        return int(getattr(evaluator, "n_failed", 0)) if evaluator else 0

    def _observe(self, z: np.ndarray, r: np.ndarray, failed: bool) -> None:
        s = z - self.z_c
        rs_c, Js_c = self._centre_surrogate()
        e_f = np.asarray(r, dtype=np.float64) - self.r_c - self.J_c @ s
        e_s = (np.asarray(self.base_surrogate.value(z), dtype=np.float64)
               - rs_c - Js_c @ s)
        usable = bool(self._usable(s))
        u = None
        if usable:
            u = self.estimator.update(e_f, e_s, self.slices,
                                      calls=self.expensive.n_calls)
        diff = e_f - e_s
        self.trial_log.append({
            "start": self.label, "trial": len(self.trial_log) + 1,
            "calls": int(self.expensive.n_calls), "F": _round(self.F_c),
            "delta": _round(self.delta),
            "lambda": [_round(v) for v in self.lambda_in_use],
            "lambda_curve_est": [_round(v) for v in self.estimator.curve],
            "usable": usable, "failed": bool(failed),
            "ef": _round(float(np.linalg.norm(e_f))),
            "es": _round(float(np.linalg.norm(e_s))),
            "ef_minus_es": _round(float(np.linalg.norm(diff))),
            "cusum_u": _round(u) if u is not None else None,
            "cusum_S": _round(self.estimator.cusum)})

    # reporting ---------------------------------------------------------
    def lambda_trials(self) -> list[dict]:
        """The per-trial log with ``rho`` and ``accepted`` from ``self.log``
        (one ``rho`` row per trial, in order)."""
        trials = [row for row in self.log if "rho" in row]
        out = []
        for entry, row in zip(self.trial_log, trials):
            out.append({**entry, "rho": _round(row["rho"]),
                        "accepted": bool(row["accepted"])})
        return out + self.trial_log[len(trials):]

    def lambda_stats(self) -> dict:
        trials = self.lambda_trials()
        n = len(trials)
        n_usable = sum(1 for t in trials if t["usable"])
        n_failed = sum(1 for t in trials if t["failed"])
        n_failed_usable = sum(1 for t in trials if t["failed"] and t["usable"])
        at = {}
        for mark in TRIAL_MARKS:
            at[str(mark)] = (float(np.mean(trials[mark - 1]["lambda"]))
                             if n >= mark else None)
        mean_lambda = (float(np.mean([np.mean(t["lambda"]) for t in trials]))
                       if n else None)
        return {"mode": self.lam.mode, "settings": asdict(self.lam),
                "n_trials": n, "n_usable": n_usable,
                "usable_fraction": n_usable / n if n else None,
                "n_failed_trials": n_failed,
                "n_failed_usable_trials": n_failed_usable,
                "lambda_at_trial": at, "mean_lambda_over_trials": mean_lambda,
                "lambda_final": [float(v) for v in self.lambda_in_use],
                "lambda_curve_est_final": [float(v)
                                           for v in self.estimator.curve],
                "lambda_scalar_est_final": float(self.estimator.scalar),
                "cusum_final": float(self.estimator.cusum),
                "cusum_switched": bool(self.estimator.switched),
                "cusum_switch_call": self.estimator.switch_call,
                "cusum_switch_usable_trial": self.estimator.switch_update,
                "n_lambda_changes": self.n_lambda_changes,
                "consistency": dict(self.consistency_stats),
                "log_aligned": len([r for r in self.log if "rho" in r]) == n,
                "trials": trials}

    def summary(self) -> dict:
        return {**super().summary(), "lambda": self.lambda_stats()}


class LambdaTrustRegionRun(LambdaMixin, TrustRegionRun):
    """Hybrid family: full FD Jacobian at every centre; every trial usable."""

    def __init__(self, z0, expensive, base_surrogate, prior_weight: float,
                 settings: TrustRegionSettings, curve_starts, n_rows: int,
                 lambda_settings: LambdaSettings, label: str = ""):
        TrustRegionRun.__init__(self, z0, expensive, None, prior_weight,
                                settings, label)
        self._lambda_init(base_surrogate, curve_starts, n_rows,
                          lambda_settings)

    def _usable(self, s: np.ndarray) -> bool:
        return True


class LambdaSketchRun(LambdaMixin, SketchTrustRegionRun):
    """Sketch family: a trial is usable when ``J_c`` is a fresh full FD
    Jacobian or the step lies within ``unprobed_tol`` of span(V_c)."""

    def __init__(self, z0, expensive, base_surrogate, prior_weight: float,
                 settings: TrustRegionSettings, curve_starts, n_rows: int,
                 lambda_settings: LambdaSettings,
                 sketch: SketchSettings | None = None, label: str = "",
                 stream: int = 0):
        SketchTrustRegionRun.__init__(self, z0, expensive, None, prior_weight,
                                      settings, sketch, label, stream)
        self._lambda_init(base_surrogate, curve_starts, n_rows,
                          lambda_settings)

    def _usable(self, s: np.ndarray) -> bool:
        if self.J_exact:
            return True
        V = self.V_c
        norm = float(np.linalg.norm(s))
        if V is None or V.shape[1] == 0 or norm == 0.0:
            return False
        unprobed = s - V @ (V.T @ s)
        return float(np.linalg.norm(unprobed)) <= self.lam.unprobed_tol * norm
