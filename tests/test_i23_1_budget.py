"""Unit tests for I23-1 (src/cryoml/explore/i23_1_budget.py)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from cryoml.explore import i23_1_budget as m
from cryoml.ngspice_fit import polish


class ToyProblem:
    """Two curves y = a * exp(b x) + c on a grid; 3 active parameters of 4."""

    def __init__(self):
        self.box = SimpleNamespace(active=np.array([True, True, False, True]), n_parameters=4)
        self.x = np.linspace(0.0, 1.0, 21)
        self.true = np.array([0.4, -0.3, 0.0, 0.2])
        self.targets = self._curves(self.true)
        self.n_curves = 2
        self.n_sim = 0

    def _curves(self, p):
        a, b, _, c = 1.0 + p[0], p[1], p[2], p[3]
        return [a * np.exp(b * self.x) + c, a * np.exp(0.5 * b * self.x) + 2 * c + 1.0]

    def simulate(self, local):
        self.n_sim += 1
        return self._curves(np.asarray(local, float))

    def curve_rrms(self, sims):
        return np.array([np.sqrt(np.mean((s - t) ** 2)) / np.mean(np.abs(t)) for s, t in zip(sims, self.targets)])

    def residual_vector(self, sims, weights):
        return np.concatenate([(s - t) * w / np.mean(np.abs(t)) for s, t, w in zip(sims, self.targets, weights)])


class PolishCappedTest(unittest.TestCase):
    def test_no_limit_equals_polish(self):
        for nfev, irls in ((60, 2), (200, 3), (10, 1)):
            p1, p2 = ToyProblem(), ToyProblem()
            x0 = np.zeros(4)
            r1 = polish(p1, x0, max_nfev=nfev, irls_rounds=irls, diff_step=0.01)
            r2 = m.polish_capped(p2, x0, max_calls=None, max_nfev=nfev, irls_rounds=irls, diff_step=0.01)
            np.testing.assert_array_equal(r1["local"], r2["local"])
            self.assertEqual(r1["objective"], r2["objective"])
            self.assertEqual(r1["history"], r2["history"])
            self.assertEqual(r1["n_sim"], r2["n_sim"])
            self.assertEqual(r2["calls"], p2.n_sim)
            self.assertFalse(r2["capped"])

    def test_large_limit_not_binding(self):
        p1, p2 = ToyProblem(), ToyProblem()
        r1 = polish(p1, np.zeros(4), max_nfev=60, irls_rounds=2)
        r2 = m.polish_capped(p2, np.zeros(4), max_calls=10 ** 6, max_nfev=60, irls_rounds=2)
        self.assertEqual(r1["objective"], r2["objective"])
        self.assertEqual(r1["n_sim"], r2["n_sim"])

    def test_limit_binds(self):
        full = m.polish_capped(ToyProblem(), np.zeros(4), max_calls=None, max_nfev=400, irls_rounds=2)
        for cap in (1, 2, 5, 13, 37):
            p = ToyProblem()
            r = m.polish_capped(p, np.zeros(4), max_calls=cap, max_nfev=400, irls_rounds=2)
            self.assertLessEqual(p.n_sim, cap)
            self.assertEqual(r["calls"], p.n_sim)
            self.assertTrue(r["capped"])
            self.assertGreaterEqual(r["objective"], full["objective"] - 1e-15)
            self.assertLessEqual(r["objective"], r["history"][0])
            # the kept card reproduces the reported objective
            q = ToyProblem()
            self.assertAlmostEqual(float(np.mean(q.curve_rrms(q.simulate(r["local"])))), r["objective"], places=14)

    def test_monotone_in_limit(self):
        vals = [m.polish_capped(ToyProblem(), np.zeros(4), max_calls=c, max_nfev=400, irls_rounds=2)["objective"]
                for c in (3, 10, 30, 100)]
        self.assertTrue(all(b <= a for a, b in zip(vals, vals[1:])))

    def test_zero_limit(self):
        p = ToyProblem()
        r = m.polish_capped(p, np.zeros(4), max_calls=0, max_nfev=40, irls_rounds=2)
        self.assertEqual(p.n_sim, 0)
        self.assertTrue(r["capped"])


class RuleTest(unittest.TestCase):
    @staticmethod
    def c(lo, hi, r=None):
        return {"ratio": r if r is not None else (lo + hi) / 2, "ci_low": lo, "ci_high": hi}

    def test_partA(self):
        ok = {t: True for t in m.IHP}
        all_gain = {t: self.c(0.7, 0.95) for t in m.IHP}
        self.assertEqual(m.rule_partA(all_gain, ok)["label"], "B1 gain survives the call match")
        none = {t: self.c(0.8, 1.0) for t in m.IHP}
        self.assertEqual(m.rule_partA(none, ok)["label"], "B1 gain not shown at equal calls")
        part = dict(all_gain, ihp300=self.c(0.9, 1.1))
        r = m.rule_partA(part, ok)
        self.assertEqual(r["label"], "partial")
        self.assertEqual(r["gain_temperatures"], ["ihp233", "ihp343", "ihp398"])
        self.assertEqual(m.rule_partA(all_gain, dict(ok, ihp343=False))["label"], "not evaluated")

    def test_partB(self):
        ok = {d: True for d in m.GAIN_SET}
        ni = {d: self.c(0.95, 1.049) for d in m.GAIN_SET}
        self.assertEqual(m.rule_partB(ni, ok)["label"], "fixed lambda 1 V non-inferior to the selected g v2")
        r = m.rule_partB(dict(ni, nistrt=self.c(1.0, 1.05)), ok)
        self.assertEqual(r["label"], "fixed lambda 1 V not shown non-inferior")
        self.assertEqual(r["failing"], ["nistrt"])
        self.assertEqual(m.rule_partB(ni, dict(ok, sky130=False))["label"], "not evaluated")

    def test_call_limit_check(self):
        r = m.call_limit_check({"a": 10, "b": 5}, {"a": 10, "b": 6})
        self.assertTrue(r["ok"])
        self.assertEqual(r["sum_calls"], 15)
        self.assertFalse(m.call_limit_check({"a": 11}, {"a": 10})["ok"])
        self.assertFalse(m.call_limit_check({"a": None}, {"a": 10})["ok"])

    def test_max_abs_diff(self):
        self.assertEqual(m.max_abs_diff({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.5}), 0.5)
        self.assertEqual(m.max_abs_diff({"a": 1.0}, {"a": 1.0, "b": 2.0}), float("inf"))


if __name__ == "__main__":
    unittest.main()
