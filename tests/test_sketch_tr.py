from __future__ import annotations

import unittest

import numpy as np

from cryoml.sketch_tr import (SketchSettings, SketchTrustRegionRun,
                              block_update, choose_directions,
                              projected_broyden)
from cryoml.trust_region import (TrustRegionRun, TrustRegionSettings,
                                 run_schedule)

N, M, HIDDEN = 43, 300, 64


class TanhProblem:
    """r(z) = A tanh(B z + c) - y on [-1, 1]^43, zero residual at z_true."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.A = rng.standard_normal((M, HIDDEN)) / np.sqrt(HIDDEN)
        self.B = 1.5 * rng.standard_normal((HIDDEN, N)) / np.sqrt(N)
        self.c = 0.3 * rng.standard_normal(HIDDEN)
        self.z_true = rng.uniform(-0.6, 0.6, N)
        self.y = self.A @ np.tanh(self.B @ self.z_true + self.c)
        self.z0 = np.clip(self.z_true + rng.uniform(-0.35, 0.35, N), -1, 1)

    def value(self, z, A=None):
        A = self.A if A is None else A
        return A @ np.tanh(self.B @ z + self.c) - self.y

    def jac(self, z, A=None):
        A = self.A if A is None else A
        sech2 = 1.0 - np.tanh(self.B @ z + self.c) ** 2
        return (A * sech2) @ self.B


class FakeBackend:
    """Call-counting expensive residual with an exact Jacobian (n calls)
    and exact directional derivatives (one call per direction)."""

    def __init__(self, problem: TanhProblem):
        self.problem = problem
        self.n_calls = 0
        self.jacobian_cost = N

    def residual(self, z):
        self.n_calls += 1
        return self.problem.value(z)

    def jacobian(self, z, r):
        self.n_calls += self.jacobian_cost
        return self.problem.jac(z)

    def directional(self, z, r, V, probe_step):
        self.n_calls += V.shape[1]
        return self.problem.jac(z) @ V, np.ones(V.shape[1], dtype=bool)


class PerturbedSurrogate:
    """The same network with a perturbed output layer."""

    def __init__(self, problem: TanhProblem, scale: float, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.problem = problem
        self.A = problem.A + scale * rng.standard_normal(problem.A.shape) / np.sqrt(HIDDEN)

    def value(self, z):
        return self.problem.value(z, self.A)

    def jacobian(self, z):
        return self.problem.jac(z, self.A)


def calls_to_reach(run: SketchTrustRegionRun | TrustRegionRun,
                   target: float) -> float:
    for row in run.log:
        if row.get("accepted") and row["F_new"] < target:
            return row["calls"]
    return np.inf


class SketchHelperTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.J0 = rng.standard_normal((M, N))
        self.V, _ = np.linalg.qr(rng.standard_normal((N, 8)))
        self.Y = rng.standard_normal((M, 8))
        self.s = rng.standard_normal(N)
        self.y = rng.standard_normal(M)

    def test_block_update_matches_probes(self):
        J = block_update(self.J0, self.V, self.Y)
        np.testing.assert_allclose(J @ self.V, self.Y, atol=1e-12)
        # The orthogonal complement keeps the prior.
        u = self.s - self.V @ (self.V.T @ self.s)
        np.testing.assert_allclose(J @ u, self.J0 @ u, atol=1e-10)

    def test_projected_broyden_keeps_probes_and_secant(self):
        J = block_update(self.J0, self.V, self.Y)
        J_new = projected_broyden(J, self.s, self.y, self.V)
        np.testing.assert_allclose(J_new @ self.V, self.Y, atol=1e-10)
        np.testing.assert_allclose(J_new @ self.s, self.y, atol=1e-10)
        # A step inside span(V) carries no new information.
        inside = self.V @ np.arange(1.0, 9.0)
        self.assertIs(projected_broyden(J, inside, self.y, self.V), J)

    def test_directions_are_orthonormal_and_free(self):
        problem = TanhProblem()
        z = problem.z0.copy()
        z[:5] = 1.0
        free = np.abs(z) < 1 - 2e-3
        V = choose_directions(problem.jac(z), problem.value(z), z, 1e-3, 0.25,
                              free, 8, np.random.default_rng(0))
        self.assertEqual(V.shape, (N, 8))
        np.testing.assert_allclose(V.T @ V, np.eye(8), atol=1e-12)
        self.assertTrue(np.all(V[~free] == 0.0))


class SketchRunTests(unittest.TestCase):
    def test_always_full_reproduces_trust_region_run(self):
        problem = TanhProblem()
        for surrogate in (None, PerturbedSurrogate(problem, 0.3)):
            for weight in (0.0, 0.05):
                base_backend, sketch_backend = FakeBackend(problem), FakeBackend(problem)
                base = TrustRegionRun(problem.z0, base_backend, surrogate,
                                      weight, TrustRegionSettings())
                sketch = SketchTrustRegionRun(
                    problem.z0, sketch_backend, surrogate, weight,
                    TrustRegionSettings(), SketchSettings(refresh_every=0))
                base.advance(20000)
                sketch.advance(20000)
                self.assertEqual(sketch.status, base.status)
                self.assertEqual(sketch.log, base.log)
                np.testing.assert_array_equal(sketch.z_c, base.z_c)
                self.assertEqual(sketch_backend.n_calls, base_backend.n_calls)
                self.assertEqual(sketch.n_sketch, 0)

    def test_sketch_reaches_zero_residual_in_fewer_calls(self):
        problem = TanhProblem()
        target = 1e-8
        for surrogate in (None, PerturbedSurrogate(problem, 0.3)):
            results = {}
            for name, sketch in (("full", SketchSettings(refresh_every=0)),
                                 ("k8", SketchSettings(k=8))):
                backend = FakeBackend(problem)
                run = SketchTrustRegionRun(problem.z0, backend, surrogate, 0.0,
                                           TrustRegionSettings(), sketch)
                run.advance(5000)
                self.assertLessEqual(backend.n_calls, 5000)
                results[name] = calls_to_reach(run, target)
            label = "none" if surrogate is None else "surrogate"
            self.assertTrue(np.isfinite(results["k8"]), (label, results))
            self.assertLess(results["k8"], results["full"], (label, results))

    def test_budget_is_never_exceeded_and_cost_is_reported(self):
        problem = TanhProblem()
        for budget in (40, 100, 333, 1000):
            backend = FakeBackend(problem)
            runs = [SketchTrustRegionRun(
                        np.clip(problem.z0 + d, -1, 1), backend,
                        PerturbedSurrogate(problem, 0.3), 1e-3,
                        TrustRegionSettings(), SketchSettings(k=8), stream=i)
                    for i, d in enumerate((0.0, 0.1, -0.1, 0.2))]
            self.assertEqual(runs[0].next_iteration_cost(), 2 + N)
            run_schedule(runs, backend, budget, "equal")
            self.assertLessEqual(backend.n_calls, budget)
            for run in runs:
                cost = run.next_iteration_cost()
                if run.status == "active":
                    self.assertIn(cost, (1, 1 + 8, 1 + N))
                self.assertTrue(all(np.isfinite(row["F"]) for row in run.log))


if __name__ == "__main__":
    unittest.main()
