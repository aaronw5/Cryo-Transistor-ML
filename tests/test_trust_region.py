from __future__ import annotations

import unittest

import numpy as np

from cryoml.trust_region import (CorrectedModel, TrustRegionRun,
                                 TrustRegionSettings, run_schedule)


class Rosenbrock:
    """Expensive stand-in: scaled Rosenbrock residuals, minimum (0.5, 0.25)."""

    jacobian_cost = 2

    def __init__(self):
        self.n_calls = 0

    @staticmethod
    def exact(z):
        return np.array([10.0 * (z[1] - z[0] ** 2), 0.5 - z[0]])

    @staticmethod
    def exact_jacobian(z):
        return np.array([[-20.0 * z[0], 10.0], [-1.0, 0.0]])

    def residual(self, z):
        self.n_calls += 1
        return self.exact(z)

    def jacobian(self, z, r):
        self.n_calls += self.jacobian_cost
        return self.exact_jacobian(z)


class Surrogate:
    """Cheap model: the expensive residual plus a smooth bias."""

    def __init__(self, bias: float):
        self.bias = bias

    def value(self, z):
        return Rosenbrock.exact(z) + self.bias * np.array(
            [np.sin(3 * z[0]), np.cos(2 * z[1])])

    def jacobian(self, z):
        return Rosenbrock.exact_jacobian(z) + self.bias * np.array(
            [[3 * np.cos(3 * z[0]), 0.0], [0.0, -2 * np.sin(2 * z[1])]])


class TrustRegionTests(unittest.TestCase):
    def test_corrected_model_matches_value_and_jacobian_at_centre(self):
        z_c = np.array([-0.3, 0.7])
        r_f, J_f = Rosenbrock.exact(z_c), Rosenbrock.exact_jacobian(z_c)
        for surrogate in (None, Surrogate(0.5)):
            model = CorrectedModel(surrogate, z_c, r_f, J_f, 0.0)
            check = model.consistency(r_f, J_f)
            self.assertLess(check["max_abs_value_error"], 1e-12)
            self.assertLess(check["max_abs_jacobian_error"], 1e-12)

    def test_gauss_newton_step_is_zero_at_the_minimum(self):
        expensive = Rosenbrock()
        run = TrustRegionRun(np.array([0.5, 0.25]), expensive, None, 0.0,
                             TrustRegionSettings())
        self.assertEqual(run.advance(100), "stationary")
        self.assertEqual(run.n_accepted + run.n_rejected, 0)
        self.assertEqual(expensive.n_calls, 1 + Rosenbrock.jacobian_cost)

    def test_both_models_converge(self):
        for surrogate in (None, Surrogate(0.3), Surrogate(0.0)):
            expensive = Rosenbrock()
            run = TrustRegionRun(np.array([-0.8, 0.8]), expensive, surrogate,
                                 0.0, TrustRegionSettings())
            run.advance(400)
            self.assertLess(np.max(np.abs(run.z_c - [0.5, 0.25])), 1e-4)
            if surrogate is not None and surrogate.bias == 0.0:
                # An exact model: every step is accepted with rho = 1.
                self.assertEqual(run.n_rejected, 0)
                self.assertLessEqual(run.n_accepted, 3)

    def test_budget_is_never_exceeded_and_runs_resume(self):
        expensive = Rosenbrock()
        runs = [TrustRegionRun(np.array(z), expensive, None, 0.0,
                               TrustRegionSettings())
                for z in ([-0.8, 0.8], [0.9, -0.9], [0.0, 0.0], [-1, -1])]
        run_schedule(runs, expensive, 23, "equal")
        self.assertLessEqual(expensive.n_calls, 23)
        expensive = Rosenbrock()
        runs = [TrustRegionRun(np.array(z), expensive, None, 0.0,
                               TrustRegionSettings())
                for z in ([-0.8, 0.8], [0.9, -0.9], [0.0, 0.0], [-1, -1])]
        run_schedule(runs, expensive, 60, "halving")
        self.assertLessEqual(expensive.n_calls, 60)


if __name__ == "__main__":
    unittest.main()
