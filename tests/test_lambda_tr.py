from __future__ import annotations

import unittest

import numpy as np

from cryoml.lambda_tr import (LambdaEstimator, LambdaSettings,
                              LambdaSketchRun, LambdaTrustRegionRun,
                              ScaledSurrogate, curve_slices,
                              rows_from_curves)
from cryoml.sketch_tr import SketchSettings, SketchTrustRegionRun
from cryoml.trust_region import (CorrectedModel, TrustRegionRun,
                                 TrustRegionSettings, run_schedule)

N, M, HIDDEN = 43, 300, 64
CURVE_STARTS = [0, 50, 120, 200, 260]


class TanhProblem:
    """r(z) = A tanh(B z + c) - y on [-1, 1]^43 with a nonzero residual
    floor (``noise`` added to the target) and a distant start."""

    def __init__(self, seed: int = 0, gain: float = 4.0, noise: float = 0.1,
                 spread: float = 1.0):
        rng = np.random.default_rng(seed)
        self.A = rng.standard_normal((M, HIDDEN)) / np.sqrt(HIDDEN)
        self.B = gain * rng.standard_normal((HIDDEN, N)) / np.sqrt(N)
        self.c = 0.3 * rng.standard_normal(HIDDEN)
        self.z_true = rng.uniform(-0.6, 0.6, N)
        self.y = (self.A @ np.tanh(self.B @ self.z_true + self.c)
                  + noise * rng.standard_normal(M))
        self.z0 = np.clip(self.z_true + rng.uniform(-spread, spread, N), -1, 1)

    def value(self, z, A=None):
        A = self.A if A is None else A
        return A @ np.tanh(self.B @ z + self.c) - self.y

    def jac(self, z, A=None):
        A = self.A if A is None else A
        sech2 = 1.0 - np.tanh(self.B @ z + self.c) ** 2
        return (A * sech2) @ self.B


class FakeBackend:
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
    def __init__(self, problem: TanhProblem, scale: float, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.problem = problem
        self.A = problem.A + scale * rng.standard_normal(problem.A.shape) / np.sqrt(HIDDEN)
        self.n_value = 0

    def value(self, z):
        self.n_value += 1
        return self.problem.value(z, self.A)

    def jacobian(self, z):
        return self.problem.jac(z, self.A)


class RandomSurrogate:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((M, N))
        self.b = rng.standard_normal(M)

    def value(self, z):
        return np.sin(self.W @ z) + self.b

    def jacobian(self, z):
        return np.cos(self.W @ z)[:, None] * self.W


SLICES = curve_slices(CURVE_STARTS, M)


class AlgebraTests(unittest.TestCase):
    def test_scaled_surrogate(self):
        base = RandomSurrogate()
        rng = np.random.default_rng(1)
        z = rng.uniform(-1, 1, N)
        scale = rows_from_curves(rng.uniform(0, 1, len(SLICES)), SLICES, M)
        wrapper = ScaledSurrogate(base, scale)
        np.testing.assert_array_equal(wrapper.value(z), scale * base.value(z))
        np.testing.assert_array_equal(wrapper.jacobian(z),
                                      scale[:, None] * base.jacobian(z))
        zero = ScaledSurrogate(base, np.zeros(M))
        np.testing.assert_array_equal(zero.value(z), np.zeros(M))
        np.testing.assert_array_equal(zero.jacobian(z), np.zeros((M, N)))
        one = ScaledSurrogate(base, np.ones(M))
        np.testing.assert_array_equal(one.value(z), base.value(z))
        np.testing.assert_array_equal(one.jacobian(z), base.jacobian(z))

    def test_rows_from_curves(self):
        rows = rows_from_curves([1, 2, 3, 4, 5], SLICES, M)
        self.assertEqual(rows[0], 1)
        self.assertEqual(rows[49], 1)
        self.assertEqual(rows[50], 2)
        self.assertEqual(rows[299], 5)
        with self.assertRaises(ValueError):
            curve_slices([5, 50], M)

    def test_centre_consistency_and_model_formula(self):
        base = RandomSurrogate()
        rng = np.random.default_rng(2)
        for trial in range(20):
            z_c = rng.uniform(-1, 1, N)
            r_c = rng.standard_normal(M) * 10 ** rng.uniform(-3, 2)
            J_c = rng.standard_normal((M, N)) * 10 ** rng.uniform(-3, 2)
            lam = rng.uniform(0, 1, len(SLICES))
            if trial == 0:
                lam[:] = 0.0
            elif trial == 1:
                lam[:] = 1.0
            D = rows_from_curves(lam, SLICES, M)
            model = CorrectedModel(ScaledSurrogate(base, D), z_c, r_c, J_c,
                                   1e-3)
            row = model.consistency(r_c, J_c)
            self.assertLessEqual(row["max_abs_value_error"],
                                 1e-10 * max(1.0, row["max_abs_value"]))
            self.assertLessEqual(row["max_abs_jacobian_error"],
                                 1e-10 * max(1.0, row["max_abs_jacobian"]))
            z = np.clip(z_c + rng.uniform(-0.2, 0.2, N), -1, 1)
            s = z - z_c
            expected = r_c + J_c @ s + D * (base.value(z) - base.value(z_c)
                                            - base.jacobian(z_c) @ s)
            np.testing.assert_allclose(model.data_residual(z), expected,
                                       rtol=1e-10, atol=1e-10 * np.max(np.abs(expected)))
            expected_jac = J_c + D[:, None] * (base.jacobian(z) - base.jacobian(z_c))
            np.testing.assert_allclose(model.data_jacobian(z), expected_jac,
                                       rtol=1e-10, atol=1e-10 * np.max(np.abs(expected_jac)))


class EstimatorTests(unittest.TestCase):
    def remainders(self, seed=3):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(M)

    def test_half_remainder_gives_half(self):
        e_s = self.remainders()
        # tau = 0: the first usable trial already gives the LS weight.
        est = LambdaEstimator(len(SLICES), LambdaSettings(prior_tau=0.0))
        est.update(0.5 * e_s, e_s, SLICES)
        np.testing.assert_allclose(est.curve, 0.5, rtol=0, atol=1e-15)
        self.assertAlmostEqual(est.scalar, 0.5, places=15)
        # tau = 1, f = 0.7: (0.7 + 0.5) / (0.7 + 1) after one trial, -> 0.5.
        est = LambdaEstimator(len(SLICES), LambdaSettings())
        self.assertTrue(np.all(est.curve == 1.0))      # before any trial
        est.update(0.5 * e_s, e_s, SLICES)
        np.testing.assert_allclose(est.curve, 1.2 / 1.7, rtol=1e-14)
        for _ in range(59):
            est.update(0.5 * e_s, e_s, SLICES)
        np.testing.assert_allclose(est.curve, 0.5, atol=1e-8)
        self.assertAlmostEqual(est.scalar, 0.5, places=8)

    def test_per_curve_weights_and_clipping(self):
        e_s = self.remainders()
        factors = np.array([0.0, 0.25, 0.5, 2.0, -1.0])
        e_f = rows_from_curves(factors, SLICES, M) * e_s
        est = LambdaEstimator(len(SLICES), LambdaSettings(prior_tau=0.0))
        est.update(e_f, e_s, SLICES)
        np.testing.assert_allclose(est.curve, [0.0, 0.25, 0.5, 1.0, 0.0],
                                   atol=1e-15)
        cross = sum(f * float(e_s[sl] @ e_s[sl]) for f, sl in zip(factors, SLICES))
        self.assertAlmostEqual(est.scalar,
                               float(np.clip(cross / float(e_s @ e_s), 0, 1)))
        # A curve with a zero surrogate remainder keeps its lambda.
        e_s0 = e_s.copy()
        e_s0[SLICES[1]] = 0.0
        est = LambdaEstimator(len(SLICES), LambdaSettings())
        est.update(e_f, e_s0, SLICES)
        self.assertEqual(est.curve[1], 1.0)

    def test_cusum_switch(self):
        e_f = self.remainders(4)
        bad = 3.0 * self.remainders(5)       # |e_f - e_s| >> |e_f|
        est = LambdaEstimator(len(SLICES), LambdaSettings())
        increments = []
        for t in range(10):
            increments.append(est.update(e_f, bad, SLICES, calls=100 + t))
            if est.switched:
                break
        u = increments[0]
        self.assertGreater(u, 0.0)
        expected_trials = int(np.ceil(2.0 / u))
        self.assertEqual(est.switch_update, expected_trials)
        self.assertEqual(est.switch_call, 100 + expected_trials - 1)
        self.assertTrue(np.all(est.target("cusum", 0.25) == 0.0))
        # Irreversible: good trials afterwards do not switch back.
        for _ in range(20):
            est.update(e_f, 0.9 * e_f, SLICES)
        self.assertTrue(est.switched)
        self.assertTrue(np.all(est.target("cusum", 0.25) == 0.0))
        # A remainder that halves the linear model's error never switches.
        est = LambdaEstimator(len(SLICES), LambdaSettings())
        for _ in range(50):
            self.assertLess(est.update(e_f, 0.5 * e_f, SLICES), 0.0)
        self.assertFalse(est.switched)
        self.assertEqual(est.cusum, 0.0)
        self.assertTrue(np.all(est.target("cusum", 0.25) == 1.0))

    def test_modes(self):
        est = LambdaEstimator(len(SLICES), LambdaSettings(delta0_ref=0.25))
        self.assertTrue(np.all(est.target("zero", 1.0) == 0.0))
        self.assertTrue(np.all(est.target("one", 1.0) == 1.0))
        self.assertTrue(np.all(est.target("radius", 0.5) == 1.0))
        np.testing.assert_allclose(est.target("radius", 0.05), 0.2)
        with self.assertRaises(ValueError):
            LambdaSettings(mode="bogus")


def make_runs(family, mode, surrogate_scale=0.3, settings=None, seed=0):
    problem = TanhProblem(seed)
    backend = FakeBackend(problem)
    base = PerturbedSurrogate(problem, surrogate_scale)
    settings = settings or TrustRegionSettings(sub_max_nfev=30)
    lam = LambdaSettings(mode=mode)
    if family == "sketch":
        run = LambdaSketchRun(problem.z0, backend, base, 1e-3, settings,
                              CURVE_STARTS, M, lam, SketchSettings(k=8),
                              label="start0", stream=7)
    else:
        run = LambdaTrustRegionRun(problem.z0, backend, base, 1e-3, settings,
                                   CURVE_STARTS, M, lam, label="start0")
    return problem, backend, base, run


def strip(log):
    return [{k: (None if isinstance(v, float) and np.isnan(v) else v)
             for k, v in row.items()} for row in log]


class ParityTests(unittest.TestCase):
    """lambda = 0 is the r_s = 0 control and lambda = 1 the plain corrected
    model, iterate for iterate (same log, same calls)."""

    def check(self, family):
        settings = TrustRegionSettings(sub_max_nfev=30)
        for mode, use_surrogate in (("zero", False), ("one", True)):
            problem, backend, base, run = make_runs(family, mode,
                                                    settings=settings)
            run_schedule([run], backend, 600)
            ref_backend = FakeBackend(problem)
            surrogate = PerturbedSurrogate(problem, 0.3) if use_surrogate else None
            if family == "sketch":
                ref = SketchTrustRegionRun(problem.z0, ref_backend, surrogate,
                                           1e-3, settings, SketchSettings(k=8),
                                           label="start0", stream=7)
            else:
                ref = TrustRegionRun(problem.z0, ref_backend, surrogate, 1e-3,
                                     settings, label="start0")
            run_schedule([ref], ref_backend, 600)
            self.assertGreater(len(ref.log), 5)
            self.assertEqual(strip(run.log), strip(ref.log), (family, mode))
            self.assertEqual(backend.n_calls, ref_backend.n_calls)
            np.testing.assert_array_equal(run.z_c, ref.z_c)
            stats = run.lambda_stats()
            self.assertEqual(stats["consistency"]["n_violations"], 0)
            self.assertGreater(stats["consistency"]["n_builds"], 0)
            self.assertTrue(stats["log_aligned"])
            self.assertEqual(stats["n_lambda_changes"], 0)

    def test_sketch_parity(self):
        self.check("sketch")

    def test_hybrid_parity(self):
        self.check("hybrid")


def watched(cls, names):
    """Subclass that counts row_scale changes inside each named method."""

    class Watched(cls):
        changes: dict

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.changes = {name: 0 for name in names}
            self.changes["solve_subproblem"] = 0
            self.deferred = 0

    def wrap(name):
        def method(self, *args, **kwargs):
            before = self.surrogate.row_scale.copy()
            out = getattr(super(Watched, self), name)(*args, **kwargs)
            if not np.array_equal(before, self.surrogate.row_scale):
                self.changes[name] += 1
            if name == "_residual" and not np.array_equal(
                    self.estimator.target(self.lam.mode, self.delta),
                    self.lambda_in_use):
                self.deferred += 1
            return out
        return method

    for name in list(names) + ["solve_subproblem"]:
        setattr(Watched, name, wrap(name))
    return Watched


class LambdaTimingTests(unittest.TestCase):
    def run_watched(self, family, mode):
        problem = TanhProblem(0)
        backend = FakeBackend(problem)
        base = PerturbedSurrogate(problem, 0.3)
        # delta0 0.1 < delta0_ref 0.25: radius-mode lambda starts at 0.4.
        settings = TrustRegionSettings(delta0=0.1, sub_max_nfev=30)
        lam = LambdaSettings(mode=mode)
        if family == "sketch":
            names = ["_residual", "_jacobian", "_directional", "_accept",
                     "_reject", "_centre_jacobian", "_full_refresh"]
            run = watched(LambdaSketchRun, names)(
                problem.z0, backend, base, 1e-3, settings, CURVE_STARTS, M,
                lam, SketchSettings(k=8), label="start0", stream=3)
        else:
            names = ["_residual", "_jacobian"]
            run = watched(LambdaTrustRegionRun, names)(
                problem.z0, backend, base, 1e-3, settings, CURVE_STARTS, M,
                lam, label="start0")
        run_schedule([run], backend, 800)
        return run, names

    def test_lambda_changes_only_in_subproblem(self):
        for family in ("sketch", "hybrid"):
            for mode in ("curve", "scalar", "radius"):
                run, names = self.run_watched(family, mode)
                for name in names:
                    self.assertEqual(run.changes[name], 0, (family, mode, name))
                self.assertGreater(run.changes["solve_subproblem"], 0,
                                   (family, mode))
                self.assertEqual(run.changes["solve_subproblem"],
                                 run.n_lambda_changes)
                stats = run.lambda_stats()
                self.assertEqual(stats["consistency"]["n_violations"], 0)
                self.assertTrue(stats["log_aligned"])
                if mode in ("curve", "scalar"):
                    # The estimate moved during a trial and waited for the
                    # next subproblem.
                    self.assertGreater(run.deferred, 0, (family, mode))
                lam = [t["lambda"] for t in stats["trials"]]
                self.assertTrue(all(0.0 <= v <= 1.0 for row in lam for v in row))

    def test_accept_uses_step_lambda(self):
        """Sketch transport J0 = J_c - D J_s(z_c) + D J_s(z_new) with the D of
        the step (checked before the secant term, step > secant_max_step)."""
        problem = TanhProblem(0)
        backend = FakeBackend(problem)
        base = PerturbedSurrogate(problem, 0.3)
        run = LambdaSketchRun(problem.z0, backend, base, 1e-3,
                              TrustRegionSettings(sub_max_nfev=30),
                              CURVE_STARTS, M, LambdaSettings(mode="curve"),
                              SketchSettings(k=8, secant_max_step=0.0))
        checked = 0
        original = LambdaSketchRun._accept

        def spy(self, z_new, r_new, F_new):
            D = self.surrogate.row_scale.copy()
            expected = (self.J_c - D[:, None] * base.jacobian(self.z_c)
                        + D[:, None] * base.jacobian(z_new))
            original(self, z_new, r_new, F_new)
            np.testing.assert_allclose(self.J0_pending, expected, rtol=1e-12,
                                       atol=1e-12 * np.max(np.abs(expected)))
            nonlocal checked
            checked += 1

        run._accept = spy.__get__(run)
        run.advance(800)
        self.assertGreater(checked, 0)


class UsableFilterTests(unittest.TestCase):
    def test_sketch_filter(self):
        _problem, _backend, _base, run = make_runs("sketch", "curve")
        rng = np.random.default_rng(0)
        V, _ = np.linalg.qr(rng.standard_normal((N, 8)))
        run.J_exact = False
        run.V_c = V
        inside = V @ rng.standard_normal(8)
        outside = rng.standard_normal(N)
        outside -= V @ (V.T @ outside)
        self.assertTrue(run._usable(inside))
        self.assertFalse(run._usable(outside))
        # 5% of the step unprobed: usable; 20%: not.
        u = outside / np.linalg.norm(outside)
        w = inside / np.linalg.norm(inside)
        self.assertTrue(run._usable(np.sqrt(1 - 0.05 ** 2) * w + 0.05 * u))
        self.assertFalse(run._usable(np.sqrt(1 - 0.2 ** 2) * w + 0.2 * u))
        run.V_c = None
        self.assertFalse(run._usable(inside))
        run.V_c = np.zeros((N, 0))
        self.assertFalse(run._usable(inside))
        run.J_exact = True
        self.assertTrue(run._usable(outside))

    def test_hybrid_always_usable(self):
        _problem, _backend, _base, run = make_runs("hybrid", "curve")
        self.assertTrue(run._usable(np.ones(N)))

    def test_trials_marked_usable(self):
        _problem, backend, _base, run = make_runs("sketch", "curve")
        run_schedule([run], backend, 800)
        stats = run.lambda_stats()
        self.assertGreater(stats["n_trials"], 5)
        self.assertGreater(stats["n_usable"], 0)
        # Trials at a fresh full Jacobian are always usable.
        _problem, backend, _base, run = make_runs("hybrid", "curve")
        run_schedule([run], backend, 800)
        stats = run.lambda_stats()
        self.assertEqual(stats["n_usable"], stats["n_trials"])


if __name__ == "__main__":
    unittest.main()
