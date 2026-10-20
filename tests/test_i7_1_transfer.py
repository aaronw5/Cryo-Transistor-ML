import math
import unittest

import numpy as np
from scipy.stats import spearmanr

from cryoml.explore.e1_correction import CurveRec, DeviceRec
from cryoml.explore.i7_1_transfer import (N_BINS, RandomFunction, device_residual,
                                          injected_log_error, ols, s1, s2, sign_accuracy,
                                          spearman, stats_for)


def make_dev(tag, family, resid_fn, vdd=1.0, n=21):
    """Device with 2 Id-Vd and 2 Id-Vg curves; target = sim * exp(resid_fn(x1, x2))."""
    dev = DeviceRec(tag, family, 1.0, 1.0, vdd, "gf180")
    s = -1.0 if family == "pmos" else 1.0
    sweep = s * np.linspace(0.0, vdd, n)
    for kind, fixed in (("idvd", 0.5), ("idvd", 1.0), ("idvg", 0.3), ("idvg", 0.9)):
        f = np.full(n, s * fixed * vdd)
        vg, vd = (f, sweep) if kind == "idvd" else (sweep, f)
        sim = 1e-4 * (0.1 + np.abs(vg) / vdd) * (0.1 + np.abs(vd) / vdd)
        tgt = sim * np.exp(resid_fn(np.abs(vg) / vdd, np.abs(vd) / vdd))
        den = float(np.mean(np.abs(tgt[2:])))
        dev.curves.append(CurveRec(f"{kind}@{fixed}", kind, vg, vd, sim, tgt, 2, den, True))
    return dev


class TestResidualStats(unittest.TestCase):
    def test_energy_matches_direct_sum(self):
        dev = make_dev("a", "nmos", lambda x1, x2: 0.1 * np.sin(3 * x1) + 0.05 * x2)
        d = device_residual(dev)
        m = np.linspace(-0.1, 0.1, N_BINS * N_BINS)
        direct = float(np.sum(d.w * (d.r - m[d.bins]) ** 2))
        self.assertAlmostEqual(d.energy(m), direct, places=14)
        self.assertAlmostEqual(d.energy(), float(np.sum(d.w * d.r ** 2)), places=14)

    def test_weights_quadratic_form(self):
        dev = make_dev("a", "nmos", lambda x1, x2: 0.0 * x1)
        d = device_residual(dev)
        # sum w = mean over curves of mean (|I|/den)^2 >= 1
        expect = np.mean([np.mean((np.abs(c.target[2:]) / c.den) ** 2) for c in dev.curves])
        self.assertAlmostEqual(float(d.w.sum()), float(expect), places=12)

    def test_clip(self):
        dev = make_dev("a", "nmos", lambda x1, x2: 3.0 + 0.0 * x1)
        d = device_residual(dev)
        self.assertTrue(np.all(d.r <= 1.0 + 1e-15))

    def test_shared_residual_gives_s1_near_one(self):
        fn = lambda x1, x2: 0.2 * np.cos(4 * x1) * x2
        devs = [device_residual(make_dev(f"n{i}", "nmos", fn)) for i in range(5)]
        # identical residuals: only the within-bin variation is unexplained
        self.assertGreater(s1(devs, lodo=True), 0.99)
        self.assertAlmostEqual(s1(devs, lodo=True), s1(devs, lodo=False), places=12)
        self.assertAlmostEqual(s2(devs), 1.0, places=10)

    def test_device_specific_residual_gives_low_s1(self):
        devs = []
        for i in range(8):
            f = RandomFunction.draw(100 + i)
            devs.append(device_residual(make_dev(f"n{i}", "nmos",
                                                 lambda a, b, f=f: 0.1 * f(a, b).reshape(a.shape))))
        lodo, pooled = s1(devs, True), s1(devs, False)
        self.assertLess(lodo, 0.3)
        self.assertGreater(pooled, lodo)

    def test_polarity_separation(self):
        fn_n = lambda x1, x2: 0.1 + 0.0 * x1
        fn_p = lambda x1, x2: -0.1 + 0.0 * x1
        devs = ([device_residual(make_dev(f"n{i}", "nmos", fn_n)) for i in range(3)]
                + [device_residual(make_dev(f"p{i}", "pmos", fn_p)) for i in range(3)])
        self.assertAlmostEqual(s1(devs), 1.0, places=10)
        self.assertAlmostEqual(s2(devs), 1.0, places=10)
        # opposite signs within one polarity: cosine -1
        devs2 = [device_residual(make_dev("n0", "nmos", fn_n)),
                 device_residual(make_dev("n1", "nmos", fn_p))]
        self.assertAlmostEqual(s2(devs2), -1.0, places=10)
        self.assertLess(s1(devs2), 0.0)

    def test_single_device_per_polarity(self):
        fn = lambda x1, x2: 0.1 * x1
        devs = [device_residual(make_dev("n0", "nmos", fn)),
                device_residual(make_dev("p0", "pmos", fn))]
        self.assertAlmostEqual(s1(devs, lodo=True), 0.0, places=12)
        self.assertGreater(s1(devs, lodo=False), 0.99)
        self.assertTrue(math.isnan(s2(devs)))

    def test_stats_for_keys(self):
        fn = lambda x1, x2: 0.1 * x1
        out = stats_for([make_dev(f"n{i}", "nmos", fn) for i in range(3)])
        self.assertEqual(out["n_train"], 3)
        self.assertEqual(out["n_train_pmos"], 0)
        self.assertGreater(out["S1_lodo"], 0.99)


class TestRandomFunction(unittest.TestCase):
    def test_unit_rms_and_determinism(self):
        f = RandomFunction.draw(7)
        g = np.linspace(0, 1, 21)
        x1, x2 = np.meshgrid(g, g, indexing="ij")
        self.assertAlmostEqual(float(np.sqrt(np.mean(f(x1, x2) ** 2))), 1.0, places=12)
        f2 = RandomFunction.draw(7)
        self.assertTrue(np.array_equal(f(x1, x2), f2(x1, x2)))

    def test_injection_mixing(self):
        s, u = RandomFunction.draw(1), RandomFunction.draw(2)
        x = np.linspace(0, 1, 50)
        e1 = injected_log_error(x, x[::-1], 1.0, 0.1, s, u)
        self.assertTrue(np.allclose(e1, 0.1 * s(x, x[::-1])))
        e0 = injected_log_error(x, x[::-1], 0.0, 0.1, s, u)
        self.assertTrue(np.allclose(e0, 0.1 * u(x, x[::-1])))


class TestRankAndRule(unittest.TestCase):
    def test_spearman_matches_scipy(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=30)
        y = np.round(x + rng.normal(size=30), 1)
        self.assertAlmostEqual(spearman(x, y), float(spearmanr(x, y).statistic), places=12)

    def test_ols_and_sign(self):
        x = np.array([0.0, 0.2, 0.4, 0.6])
        y = 0.1 - 0.5 * x
        rule = ols(x, y)
        self.assertAlmostEqual(rule["a"], 0.1, places=12)
        self.assertAlmostEqual(rule["b"], -0.5, places=12)
        self.assertAlmostEqual(rule["threshold_S"], 0.2, places=12)
        acc = sign_accuracy(rule, [0.1, 0.5], [0.02, -0.1])
        self.assertEqual(acc["correct"], 2)


if __name__ == "__main__":
    unittest.main()
