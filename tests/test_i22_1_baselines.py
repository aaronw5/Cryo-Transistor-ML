"""I22-1: B1 screen selection, labels, outcome rule, pooled statistics,
budget-exact traces, CMA-ES and TuRBO-1 loops."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.explore import i22_1_baselines as m  # noqa: E402


def cmp(ratio, lo, hi):
    return {"ratio": ratio, "ci_low": lo, "ci_high": hi}


class ScreenTest(unittest.TestCase):
    def test_selection_reproduces_stored_i4_1_screens(self):
        p = ROOT / "out/research/explore/I4_1/b1_screen.json"
        if not p.exists():
            self.skipTest("stored I4-1 screen not present")
        st = json.loads(p.read_text())
        for ds in ("sky130_real_dev", "gf180", "ihp"):
            sel = m.screen_select(st[ds]["per_device"])
            self.assertEqual(sel["selected"], st[ds]["selected"], ds)
            self.assertEqual(sel["candidates"], st[ds]["candidates"], ds)

    def test_selection_rules(self):
        rows = {"a": {"x": {"active": True, "response_lo": 2e-4, "response_hi": float("nan")},
                      "y": {"active": True, "response_lo": 5e-5, "response_hi": 9e-5},
                      "z": {"active": False}},
                "b": {"y": {"active": True, "response_lo": float("inf"), "response_hi": 1e-6},
                      "z": {"active": False}}}
        sel = m.screen_select(rows)
        self.assertEqual(sel["selected"], ["x"])
        self.assertEqual(sel["per_parameter"]["z"]["n_active_devices"], 0)
        self.assertEqual(sel["per_parameter"]["y"]["max_response"], 9e-5)   # inf counts as 0
        calls = m.screen_calls(rows)
        self.assertEqual(calls["per_device"], {"a": 5, "b": 3})
        self.assertEqual(calls["total"], 8)


class LabelTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(m.label_vs_control(cmp(0.8, 0.7, 0.95)), "gain")
        self.assertEqual(m.label_vs_control(cmp(1.2, 1.01, 1.4)), "worse")
        self.assertEqual(m.label_vs_control(cmp(1.0, 0.9, 1.1)), "inconclusive")
        self.assertEqual(m.label_vs_control(cmp(1.0, float("nan"), float("nan"))), "inconclusive")
        self.assertEqual(m.label_vs_reference(cmp(0.9, 0.8, 0.99), "B1"), "B1 better than g v2")
        self.assertEqual(m.label_vs_reference(cmp(1.1, 1.01, 1.2), "B1"), "g v2 better than B1")
        self.assertTrue(m.non_inferior(cmp(1.0, 0.95, 1.049)))
        self.assertFalse(m.non_inferior(cmp(1.0, 0.95, 1.05)))

    def test_outcome(self):
        good = {d: {x: cmp(1.2, 1.1, 1.3) for x in m.BASELINES} for d in m.GAIN_SET}
        valid = {d: True for d in m.GAIN_SET}
        out = m.outcome(good, valid)
        self.assertEqual(out["label"], "selected g v2 not matched by a correction-stage baseline")
        self.assertTrue(all(v["g_v2_ahead"] for v in out["per_baseline"].values()))
        vs = json.loads(json.dumps(good))
        for d in m.GAIN_SET:
            vs[d]["B3_d3"] = cmp(1.0, 0.98, 1.03)
        out = m.outcome(vs, valid)
        self.assertEqual(out["matched_by"], ["B3_d3"])
        self.assertEqual(out["label"], "matched by B3_d3")
        vs["sky130"]["B3_d3"] = cmp(1.02, 0.99, 1.06)          # not non-inferior on one dataset
        self.assertEqual(m.outcome(vs, valid)["matched_by"], [])
        out = m.outcome(good, {**valid, "nist4": False})
        self.assertEqual(out["label"], "not evaluated")

    def test_pooled_and_family_means(self):
        self.assertAlmostEqual(m.pooled_gmr([0.5, 2.0, float("nan")]), 1.0)
        fm = m.family_means({"nmos_a": 0.1, "nmos_b": 0.3, "pmos_c": 0.5, "pmos_d": float("nan")},
                            lambda t: t.startswith("pmos"))
        self.assertAlmostEqual(fm["all_device_mean"], 0.3)
        self.assertAlmostEqual(fm["combined"], 0.35)
        self.assertEqual(fm["n_failed"], 1)

    def test_max_abs_diff(self):
        self.assertEqual(m.max_abs_diff({"a": 1.0, "b": float("nan")}, {"a": 1.0, "b": float("nan")}), 0.0)
        self.assertEqual(m.max_abs_diff({"a": 1.0}, {"a": float("nan")}), float("inf"))
        self.assertEqual(m.max_abs_diff({"a": 1.0}, {}), float("inf"))
        self.assertAlmostEqual(m.max_abs_diff({"a": 1.0}, {"a": 1.5}), 0.5)


class TraceTest(unittest.TestCase):
    def test_traced_budget_and_best(self):
        tr = m.Traced(lambda x: float(np.sum(x ** 2)) if x[0] < 5 else float("nan"), 3)
        tr(np.array([2.0])), tr(np.array([1.0])), tr(np.array([7.0]))
        with self.assertRaises(m.BudgetExhausted):
            tr(np.array([0.0]))
        self.assertEqual(tr.values, [4.0, 1.0, m.F_FAIL])
        self.assertEqual(tr.best_f, 1.0)
        np.testing.assert_array_equal(m.best_so_far(tr.values), [4.0, 1.0, 1.0])
        self.assertEqual(m.value_at(tr.values, 1), 4.0)
        self.assertEqual(m.value_at(tr.values, 10), 1.0)
        x, v = m.best_x_at(tr.values, tr.xs, 2)
        self.assertEqual((float(x[0]), v), (1.0, 1.0))
        np.testing.assert_allclose(m.curve_on_grid(tr.values, 4.0, 2, [0.5, 1.0]), [1.0, 0.25])
        self.assertAlmostEqual(m.seed_geomean([0.5, 2.0]), 1.0)


def quad(c):
    return lambda x: 0.05 + float(np.sum((np.asarray(x) - c) ** 2))


class CmaTest(unittest.TestCase):
    def test_exact_budget_start_first_and_improves(self):
        c = np.full(5, 0.2)
        f = quad(c)
        x0 = np.zeros(5)
        r = m.run_cma(f, x0, 37, seed=1)
        self.assertEqual(r["n"], 37)
        self.assertEqual(r["values"][0], f(x0))
        self.assertLess(r["best_f"], f(x0))
        self.assertTrue(all(np.all(np.abs(x) <= 1.0) for x in r["xs"]))
        r2 = m.run_cma(f, x0, 37, seed=1)
        self.assertEqual(r["values"], r2["values"])                     # seeded


class TurboTest(unittest.TestCase):
    def test_state_update(self):
        s = m.TurboState(dim=43, q=4, length_init=0.1)
        self.assertEqual(s.failure_tol, 11)
        s.best = 1.0
        for _ in range(3):
            m.turbo_update(s, [s.best + 1.0])
        self.assertAlmostEqual(s.length, 0.2)
        best = s.best
        for _ in range(11):
            m.turbo_update(s, [best - 1.0])
        self.assertAlmostEqual(s.length, 0.1)
        s.length = m.TURBO["L_min"] * 1.5
        for _ in range(11):
            m.turbo_update(s, [best - 1.0])
        self.assertEqual(s.restarts, 1)
        self.assertAlmostEqual(s.length, 0.1)

    def test_tr_bounds(self):
        lo, hi = m.tr_bounds(np.array([0.5, 0.95]), np.array([1.0, 4.0]), 0.2)
        w = np.array([0.4, 1.6]) / math.sqrt(0.4 * 1.6)
        np.testing.assert_allclose(lo, np.clip([0.5 - w[0] * 0.1, 0.95 - w[1] * 0.1], 0, 1))
        np.testing.assert_allclose(hi, np.clip([0.5 + w[0] * 0.1, 0.95 + w[1] * 0.1], 0, 1))

    def test_run_exact_budget(self):
        try:
            import botorch  # noqa: F401
        except ImportError:
            self.skipTest("botorch not installed")
        import torch
        torch.set_num_threads(1)
        f = quad(np.full(3, 0.2))
        x0 = np.zeros(3)
        r = m.run_turbo(f, x0, 23, seed=1, n_candidates=200)
        self.assertEqual(r["n"], 23)
        self.assertEqual(r["values"][0], f(x0))
        self.assertLessEqual(r["best_f"], f(x0))
        self.assertTrue(all(np.all(np.abs(x) <= 1.0 + 1e-12) for x in r["xs"]))


if __name__ == "__main__":
    unittest.main()
