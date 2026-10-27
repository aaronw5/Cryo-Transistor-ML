"""Unit tests for I24-1 (src/cryoml/explore/i24_1_gate3.py)."""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from cryoml.explore import i24_1_gate3 as m
from cryoml.explore.i22_1_baselines import screen_select
from cryoml.explore.i3_2_bias import paired_ratio

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "out/research/explore"


def _entry(lo, hi):
    return {"active": True, "response_lo": lo, "response_hi": hi}


class TestSelection(unittest.TestCase):
    def test_logo_selection_drops_single_device_parameter(self):
        per = {"a": {"p": _entry(1e-3, 0.0), "q": _entry(0.0, 1e-6)},
               "b": {"p": _entry(2e-4, 0.0), "q": _entry(5e-4, 0.0)},
               "c": {"p": _entry(0.0, 0.0), "q": {"active": False}}}
        self.assertEqual(m.logo_selection(per, ["a", "b", "c"]), ["p", "q"])
        self.assertEqual(m.logo_selection(per, ["a", "c"]), ["p"])
        self.assertEqual(m.logo_selection(per, ["c"]), [])

    def test_stored_screens_reproduce(self):
        """VA2 on stored data: screen_select over all DEV devices = stored selection."""
        for ds in ("sky130", "tud4"):
            p = EXP / "I22_1" / f"screen_{ds}.json"
            if not p.exists():
                self.skipTest("stored screens missing")
            s = json.loads(p.read_text())
            self.assertEqual(screen_select(s["per_device"])["selected"], list(s["selected"]))
        st = json.loads((EXP / "I5_1/start/b1_screen.json").read_text())
        for c in ("lv", "hv"):
            self.assertEqual(screen_select(st[c]["per_device"])["selected"], list(st[c]["selected"]))


class TestStatistic(unittest.TestCase):
    def test_s_b1_mean_and_invalid(self):
        r = m.s_b1({"a": -0.2, "b": 0.0, "c": 0.1})
        self.assertAlmostEqual(r["S_B1"], -0.1 / 3)
        self.assertTrue(r["valid"])
        r = m.s_b1({str(i): (float("nan") if i < 2 else 0.1) for i in range(10)})
        self.assertFalse(r["valid"])
        self.assertTrue(math.isnan(r["S_B1"]))
        r = m.s_b1({str(i): (float("nan") if i < 1 else 0.1) for i in range(10)})
        self.assertTrue(r["valid"])
        self.assertAlmostEqual(r["S_B1"], 0.1)

    def test_log_ratio(self):
        self.assertAlmostEqual(m.log_ratio(0.5, 1.0), math.log(0.5))
        self.assertTrue(math.isnan(m.log_ratio(float("nan"), 1.0)))


class TestChoice(unittest.TestCase):
    def test_no_candidate(self):
        self.assertEqual(m.choose(0.1, 0.05)["arm"], "C0_refit")
        self.assertEqual(m.choose(0.0, 0.0)["arm"], "C0_refit")

    def test_one_candidate(self):
        self.assertEqual(m.choose(-0.1, 0.02)["arm"], "g")
        self.assertEqual(m.choose(0.3, -0.25)["arm"], "B1")

    def test_both_candidates_calibrated(self):
        # yhat_g = -0.021454 + 1.217286 * (-0.1) = -0.14318
        self.assertEqual(m.choose(-0.1, -0.10)["arm"], "g")
        self.assertEqual(m.choose(-0.1, -0.20)["arm"], "B1")
        s_b = m.yhat_g(-0.1)
        self.assertEqual(m.choose(-0.1, s_b)["arm"], "B1")          # tie -> B1

    def test_not_buildable(self):
        self.assertEqual(m.choose(-0.1, None)["arm"], "g")
        self.assertEqual(m.choose(0.1, float("nan"))["arm"], "C0_refit")
        self.assertIn("two-way", m.choose(0.1, None)["reason"])

    def test_raw(self):
        self.assertEqual(m.choose_raw(-0.1, -0.2), "B1")
        self.assertEqual(m.choose_raw(-0.3, -0.2), "g")
        self.assertEqual(m.choose_raw(0.1, 0.2), "C0_refit")
        self.assertEqual(m.choose_raw(-0.2, -0.2), "B1")
        self.assertEqual(m.choose_two_way(-0.01), "g")
        self.assertEqual(m.choose_two_way(0.0), "C0_refit")

    def test_per_device_gate_excludes_own_group(self):
        tags = ["a1", "a2", "b1", "c1"]
        grp = {"a1": "a", "a2": "a", "b1": "b", "c1": "c"}.get
        lr_g = {"a1": -5.0, "a2": -5.0, "b1": 0.1, "c1": 0.1}
        lr_b = {"a1": 0.0, "a2": 0.0, "b1": 0.2, "c1": 0.2}
        out = m.per_device_gate(tags, grp, lr_g, lr_b)
        self.assertEqual(out["a1"]["arm"], "C0_refit")               # group a's own -5 left out
        self.assertAlmostEqual(out["a1"]["S_g"], 0.1)
        self.assertEqual(out["b1"]["arm"], "g")                      # (-5 -5 +0.1) / 3 < 0


class TestPooled(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.data = {}
        for ds, n, cl in (("x", 12, None), ("y", 9, True)):
            tags = [f"{ds}{i}" for i in range(n)]
            c0 = {t: float(rng.uniform(0.01, 0.05)) for t in tags}
            a = {t: c0[t] * float(np.exp(rng.normal(-0.2, 0.1))) for t in tags}
            b = {t: c0[t] * float(np.exp(rng.normal(0.05, 0.1))) for t in tags}
            self.data[ds] = {"tags": tags, "clusters": ({t: f"g{i % 3}" for i, t in enumerate(tags)} if cl else None),
                             "rows": {"A": a, "B": b, "C0": dict(c0)}, "control": c0}

    def test_point_equals_mean_log_gmr(self):
        out = m.pooled_bootstrap(self.data, ["A", "B", "C0"], n_boot=2000)
        for s in ("A", "B"):
            g = [paired_ratio(self.data[d]["rows"][s], self.data[d]["control"], n_boot=10)["ratio"] for d in self.data]
            self.assertAlmostEqual(out[s]["pooled"], float(np.exp(np.mean(np.log(g)))), places=12)
            self.assertLess(out[s]["ci_low"], out[s]["pooled"])
            self.assertGreater(out[s]["ci_high"], out[s]["pooled"])
        self.assertAlmostEqual(out["C0"]["pooled"], 1.0)
        self.assertAlmostEqual(out["C0"]["ci_low"], 1.0)

    def test_paired(self):
        out = m.pooled_bootstrap(self.data, ["A", "B"], n_boot=2000)
        p = m.paired_pooled(out, "A", "B")
        self.assertAlmostEqual(p["ratio"], out["A"]["pooled"] / out["B"]["pooled"])
        self.assertLess(p["ci_high"], 1.0)
        self.assertNotIn("_boot", m.strip_boot(out)["A"])

    def test_nan_rows(self):
        self.data["x"]["rows"]["A"]["x0"] = float("nan")
        out = m.pooled_bootstrap(self.data, ["A"], n_boot=500)
        self.assertTrue(np.isfinite(out["A"]["pooled"]))


class TestRules(unittest.TestCase):
    def test_part_a(self):
        g3 = {"a": 0.8, "b": 1.0, "c": 0.9}
        r = m.part_a_outcome(g3, 0.85, 0.9, 0.95, {"ci_high": 0.97}, True)
        self.assertEqual(r["label"], "three-way gate supported (retrospective)")
        r = m.part_a_outcome({**g3, "d": 1.01}, 0.85, 0.9, 0.95, {"ci_high": 0.97}, True)
        self.assertEqual(r["label"], "partially supported (retrospective)")
        self.assertEqual(r["R1_fail"], ["d"])
        r = m.part_a_outcome(g3, 0.85, 0.84, 0.95, {"ci_high": 0.97}, True)
        self.assertEqual(r["label"], "partially supported (retrospective)")
        r = m.part_a_outcome(g3, 0.85, 0.9, 0.95, {"ci_high": 1.0}, True)
        self.assertEqual(r["label"], "not supported (retrospective)")
        self.assertEqual(m.part_a_outcome(g3, 0.85, 0.9, 0.95, {"ci_high": 0.9}, False)["label"], "invalid")

    def test_part_b(self):
        gain = {"ci_low": 0.8, "ci_high": 0.95}
        worse = {"ci_low": 1.02, "ci_high": 1.2}
        inc = {"ci_low": 0.9, "ci_high": 1.1}
        self.assertEqual(m.part_b_outcome({str(t): gain for t in range(4)}, True)["label"], "g v2 adds to B1 on IHP")
        self.assertEqual(m.part_b_outcome({str(t): worse for t in range(4)}, True)["label"],
                         "g v2 worse than B1 on IHP")
        self.assertEqual(m.part_b_outcome({"0": gain, "1": gain, "2": gain, "3": inc}, True)["label"],
                         "g v2 gain over B1 not shown")
        self.assertEqual(m.label_ci(inc), "inconclusive")


if __name__ == "__main__":
    unittest.main()
