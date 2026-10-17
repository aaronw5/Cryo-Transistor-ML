from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.optimize import lsq_linear

from cryoml.dfo_ls import DfoLsRun, DfoLsSettings, finish_budget
from cryoml.trust_region import run_schedule


class LinearBackend:
    """r(z) = A z - b, call counting, optional box."""

    def __init__(self, m: int = 30, p: int = 6, seed: int = 0, bounds=None):
        rng = np.random.default_rng(seed)
        self.A = rng.standard_normal((m, p))
        self.b = rng.standard_normal(m)
        self.bounds = bounds
        self.n_calls = 0

    def residual(self, z):
        self.n_calls += 1
        return self.A @ z - self.b


class TanhBackend:
    """r(z) = A tanh(B z + c) - y on [-1, 1]^p; calls whose first coordinate
    lies in ``fail_band`` fail (NaN or ``last_failed``, alternating)."""

    def __init__(self, p: int = 8, m: int = 40, seed: int = 1,
                 fail_band=(-0.42, -0.30)):
        rng = np.random.default_rng(seed)
        self.A = rng.standard_normal((m, 16)) / 4.0
        self.B = 1.5 * rng.standard_normal((16, p)) / np.sqrt(p)
        self.c = 0.3 * rng.standard_normal(16)
        self.y = self.A @ np.tanh(self.B @ rng.uniform(-0.5, 0.5, p) + self.c)
        self.bounds = (-1.0, 1.0)
        self.fail_band = fail_band
        self.n_calls = 0
        self.last_failed = False
        self.failed_points: list[np.ndarray] = []

    def residual(self, z):
        self.n_calls += 1
        r = self.A @ np.tanh(self.B @ z + self.c) - self.y
        self.last_failed = False
        if self.fail_band and self.fail_band[0] < z[0] < self.fail_band[1]:
            self.failed_points.append(np.array(z, dtype=np.float64))
            if self.n_calls % 2:
                return np.full_like(r, np.nan)
            self.last_failed = True
            return r + 10.0
        return r


class Rosenbrock:
    bounds = None

    def __init__(self):
        self.n_calls = 0

    def residual(self, z):
        self.n_calls += 1
        return np.array([10.0 * (z[1] - z[0] ** 2), 1.0 - z[0]])


def initialized_run(backend, z0, weight=0.0, settings=None):
    run = DfoLsRun(z0, backend, weight, settings)
    run.advance(len(z0) + 1)
    assert run.initialized and backend.n_calls == len(z0) + 1
    return run


class InterpolationTests(unittest.TestCase):
    def test_linear_residuals_give_the_exact_jacobian_after_p_plus_1_points(self):
        for bounds, z0 in ((None, np.zeros(6)),
                           ((-1.0, 1.0), np.array([0.95, -0.97, 1.0, -1.0, 0.2, 0.0]))):
            backend = LinearBackend(bounds=bounds)
            run = initialized_run(backend, z0, weight=0.03)
            J, D, idx = run.model_jacobian()
            np.testing.assert_allclose(J, backend.A, atol=1e-9)
            # Init points: z0 + rho_beg e_t, sign flipped at the upper face.
            steps = run.Y[1:] - run.z0
            np.testing.assert_allclose(np.abs(np.diag(steps)), 0.1, atol=1e-15)
            if bounds is not None:
                self.assertTrue(np.all(run.Y <= 1.0) and np.all(run.Y >= -1.0))
                self.assertLess(steps[0, 0], 0.0)
            self.assertGreater(run.max_condition, 0.0)

    def test_lagrange_polynomials_are_cardinal(self):
        rng = np.random.default_rng(4)
        p = 7
        run = DfoLsRun(np.zeros(p), LinearBackend(p=p), 0.0)
        run.Y = rng.uniform(-1, 1, (p + 1, p))
        for k in (0, 3, p):
            run.k = k
            D, idx = run.interpolation_matrix()
            for s in range(p + 1):
                values = run.lagrange_values(run.Y[s], D, idx)
                np.testing.assert_allclose(values, np.eye(p + 1)[s], atol=1e-10)
            x = rng.uniform(-1, 1, p)
            values = run.lagrange_values(x, D, idx)
            self.assertAlmostEqual(float(values.sum()), 1.0, places=10)
            for t in idx[:3]:
                g = run.lagrange_gradient(int(t), D, idx)
                self.assertAlmostEqual(float(g @ (x - run.Y[k])),
                                       float(values[t]), places=10)

    def test_step_equals_bounded_gauss_newton_step_when_delta_is_large(self):
        weight = 0.05
        backend = LinearBackend(m=25, p=5, seed=2)
        run = initialized_run(backend, np.full(5, 0.3), weight)
        run.delta = 1e3
        J, _, _ = run.model_jacobian()
        s, predicted = run.step(J)
        z, r = run.z_c, run.R[run.k]
        A = np.vstack([backend.A, weight * np.eye(5)])
        b = np.concatenate([-r, -weight * z])
        gn = np.linalg.lstsq(A, b, rcond=None)[0]
        np.testing.assert_allclose(s, gn, atol=1e-9)
        model = lambda v: 0.5 * float(np.sum((r + backend.A @ v) ** 2)
                                      + np.sum((weight * (z + v)) ** 2))
        self.assertAlmostEqual(predicted, model(np.zeros(5)) - model(gn), places=9)
        # With the box [-1, 1] the step is the box-constrained GN step.
        boxed = LinearBackend(m=25, p=5, seed=2, bounds=(-1.0, 1.0))
        boxed.b = boxed.b * 20.0
        run = initialized_run(boxed, np.full(5, 0.3), weight)
        run.delta = 1e3
        J, _, _ = run.model_jacobian()
        s, _ = run.step(J)
        z, r = run.z_c, run.R[run.k]
        ref = lsq_linear(np.vstack([boxed.A, weight * np.eye(5)]),
                         np.concatenate([-r, -weight * z]),
                         bounds=(-1.0 - z, 1.0 - z), method="trf",
                         tol=1e-14).x
        self.assertTrue(np.any(np.isclose(np.abs(z + s), 1.0)))
        np.testing.assert_allclose(s, ref, atol=1e-6)


class RunTests(unittest.TestCase):
    def test_budget_is_never_exceeded_and_runs_resume(self):
        backend = TanhBackend()
        run = DfoLsRun(np.full(8, 0.1), backend, 0.01, label="a")
        for limit in (0, 1, 5, 9, 9, 10, 23, 60, 61, 150, 400):
            status = run.advance(limit)
            self.assertLessEqual(backend.n_calls, limit)
            self.assertEqual(run.calls, backend.n_calls)
            self.assertIn(status, ("new", "active", "budget", "stationary"))
            if limit == 0:
                self.assertEqual(status, "new")
            elif status != "stationary":
                self.assertEqual(status, "budget")
                self.assertEqual(backend.n_calls, limit)
        # Failed points never enter the interpolation set.
        self.assertGreaterEqual(sum(v > 0 for v in run.n_failed.values()), 2)
        self.assertGreater(run.n_failed_points_skipped, 0)
        self.assertEqual(run.n_failed_points_skipped, len(backend.failed_points))
        for bad in backend.failed_points:
            self.assertFalse(np.any(np.all(run.Y == bad, axis=1)))
        # The centre value never increases.
        F = [row[2] for row in run.log if math.isfinite(row[2])]
        self.assertTrue(all(b <= a for a, b in zip(F, F[1:])))
        self.assertEqual(run.F_c, float(np.min(run.Fy)))
        # A failed initialization point is retried with the opposite sign.
        backend = TanhBackend(fail_band=(0.15, 0.25))
        run = DfoLsRun(np.full(8, 0.1), backend, 0.01)
        run.advance(9)              # p + 1 calls: one init point still missing
        self.assertFalse(run.initialized)
        run.advance(10)
        self.assertTrue(run.initialized)
        self.assertEqual(run.n_failed["init"], 1)
        self.assertAlmostEqual(run.Y[1, 0], 0.0, places=12)

    def test_run_schedule_compatibility(self):
        for schedule in ("equal", "halving"):
            backend = TanhBackend(fail_band=None)
            starts = [np.full(8, v) for v in (-0.5, 0.0, 0.4, 0.8)]
            runs = [DfoLsRun(z, backend, 0.01, label=f"s{i}", stream=i)
                    for i, z in enumerate(starts)]
            run_schedule(runs, backend, 203, schedule)
            self.assertLessEqual(backend.n_calls, 203)
            spent = finish_budget(runs, backend, 203)
            self.assertGreaterEqual(spent, 0)
            self.assertEqual(backend.n_calls, 203)
            self.assertEqual(sum(r.calls for r in runs), 203)
            for run in runs:
                self.assertIn(run.status, ("new", "active", "budget", "stationary"))
                self.assertTrue(math.isfinite(run.F_c))
        # The equal schedule gives the first run its share exactly.
        self.assertEqual(runs[0].summary()["calls"], runs[0].calls)

    def test_rosenbrock_converges_and_soft_restarts(self):
        backend = Rosenbrock()
        settings = DfoLsSettings(rho_beg=0.12, delta_max=1e10)
        run = DfoLsRun(np.array([-1.2, 1.0]), backend, 0.0, settings)
        run.advance(600)
        self.assertLess(run.F_c, 1e-14)
        np.testing.assert_allclose(run.z_c, [1.0, 1.0], atol=1e-6)
        if run.status == "stationary":
            self.assertEqual(run.stop_reason, "zero_objective")
            self.assertEqual(run.F_c, 0.0)
        # A problem with a nonzero minimum goes through every rho level
        # (rho_beg 0.1 -> rho_end 1e-8: 7 reductions) and soft-restarts.
        backend = TanhBackend(fail_band=None)
        run = DfoLsRun(np.full(8, 0.1), backend, 0.01)
        run.advance(400)
        self.assertEqual(run.status, "budget")
        self.assertGreaterEqual(run.n_rho_reductions, 7)
        self.assertGreaterEqual(run.n_restarts, 1)
        self.assertTrue(any(row[1] == "restart" for row in run.log))

    def test_nonzero_residual_problem_restarts_and_keeps_best(self):
        backend = LinearBackend(m=12, p=3, seed=5)
        run = DfoLsRun(np.zeros(3), backend, 0.0,
                       DfoLsSettings(delta_max=1e10, max_restarts=2))
        run.advance(5000)
        best = np.linalg.lstsq(backend.A, backend.b, rcond=None)[0]
        np.testing.assert_allclose(run.z_c, best, atol=1e-6)
        self.assertEqual(run.status, "stationary")
        self.assertEqual(run.stop_reason, "max_restarts")
        self.assertEqual(run.n_restarts, 3)
        self.assertLess(backend.n_calls, 5000)


if __name__ == "__main__":
    unittest.main()
