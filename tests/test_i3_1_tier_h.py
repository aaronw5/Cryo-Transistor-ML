"""I3-1 tier H truth generator (out/research/explore/I3_1/prereg.json) and
the driver's study options.  No NGSpice."""

from __future__ import annotations

import argparse
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

from cryoml.explore import e3_temp as E
from cryoml.parameter_sets import PARAMS7

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261019
K_VTH0, K_U0 = PARAMS7.index("vth0"), PARAMS7.index("u0")


def _tags() -> list[str]:
    split = ROOT / "out" / "research" / "explore" / "real_split.json"
    return json.loads(split.read_text())["REAL_DEV"]


def _driver():
    path = ROOT / "scripts" / "research" / "explore" / "e3_temp_misspec.py"
    spec = importlib.util.spec_from_file_location("e3_temp_misspec_i31", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TierHGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.truth = E.draw_truth("nmos_L1_W3", "H", np.random.default_rng([SEED, 2, 0, 3, 0]))

    def test_boundaries(self):
        t = self.truth
        self.assertTrue(np.allclose(E.truth_x(t, 77.0), t["mu"]))
        self.assertTrue(np.allclose(E.truth_x(t, E.T_HOT), t["c"]))
        self.assertEqual(E.truth_rs(t, 150.0), 0.0)

    def test_ranges(self):
        for k, tag in enumerate(_tags()):
            for i in range(6):
                t = E.draw_truth(tag, "H", np.random.default_rng([SEED, 2, 0, k, i]))
                self.assertTrue(90.0 <= t["Tf"] <= 140.0)
                self.assertTrue(all(90.0 <= v <= 260.0 for v in t["Tkink"]))
                self.assertTrue(all(0.10 <= v <= 0.90 for v in t["v_kink"]))
                self.assertTrue(0.02 <= t["freeze"][K_VTH0] <= 0.06)
                self.assertTrue(-0.20 <= t["freeze"][K_U0] <= -0.05)
                others = [f for j, f in enumerate(t["freeze"]) if j not in (K_VTH0, K_U0)]
                self.assertEqual(others, [0.0] * 5)
                c = np.abs(t["c"])
                self.assertLessEqual(c[K_VTH0], 0.05)
                self.assertTrue(np.all(c <= 0.10))

    def test_piecewise_linear_without_freeze(self):
        """For a parameter without a freeze-out step, x is linear in T on
        each side of the kink and passes through v at the kink."""
        t = self.truth
        for k in range(E.N_PAR):
            if k in (K_VTH0, K_U0):
                continue
            tk, v = t["Tkink"][k], t["v_kink"][k]
            mu, c = t["mu"][k], t["c"][k]
            self.assertAlmostEqual(E.truth_x(t, tk)[k], c + (mu - c) * v, places=10)
            for a, b in ((77.0, tk), (tk, E.T_HOT)):
                ts = np.linspace(a, b, 5)
                xs = np.array([E.truth_x(t, s)[k] for s in ts])
                self.assertTrue(np.allclose(np.diff(xs, 2), 0.0, atol=1e-10))

    def test_freeze_step(self):
        self.assertAlmostEqual(E.freeze_step(115.0, 77.0), 1.0)
        self.assertAlmostEqual(E.freeze_step(115.0, E.T_HOT), 0.0)
        self.assertGreater(E.freeze_step(115.0, 105.0), 0.85)
        self.assertLess(E.freeze_step(115.0, 125.0), 0.15)
        self.assertAlmostEqual(E.freeze_step(115.0, 115.0), 0.5, places=2)

    def test_streams_never_cross_u0_switch(self):
        """Every tier-H truth of the I3_1 decision configuration (6 training
        and 3 test draws per geometry)."""
        rejected = 0
        for k, tag in enumerate(_tags()):
            for split, n in ((0, 6), (1, 3)):
                for i in range(n):
                    t = E.draw_truth(tag, "H", np.random.default_rng([SEED, 2, split, k, i]))
                    rejected += t["u0_switch_rejections"]
                    self.assertLess(E.truth_u0_max(t), E.u0_switch_x(tag))
        self.assertEqual(rejected, 10)      # recorded in out/research/explore/I3_1/NOTES.md

    def test_tiers_m_w_unchanged(self):
        """Adding tier H does not change tier M or W draws: a stream gives the
        stored E3b full-run truth (when the run directory exists)."""
        base = ROOT / "out" / "research" / "explore" / "E3b" / "runs" / "full"
        tags = _tags()
        checked = 0
        for tier, code in (("M", 1), ("W", 0)):
            path = base / f"tier_{tier}" / "test" / "pmos_L2_W5__1.json"
            if not path.exists():
                continue
            stored = json.loads(path.read_text())["truth"]
            k = tags.index("pmos_L2_W5")
            t = E.draw_truth("pmos_L2_W5", tier, np.random.default_rng([SEED, code, 1, k, 1]))
            self.assertEqual(json.loads(json.dumps(t)), stored)
            checked += 1
        if not checked:
            self.skipTest("E3b full run not present")


class DriverOptionTests(unittest.TestCase):
    def _args(self, **kw):
        base = dict(study="E3", tier="M", n_train=6, primary_n_train=None, scaling=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_tier_h_only_in_i3_1(self):
        driver = _driver()
        with self.assertRaises(SystemExit):
            driver.check_study_args(self._args(study="E3b", tier="H"))
        with self.assertRaises(SystemExit):
            driver.check_study_args(self._args(study="E3", scaling="3,6"))
        self.assertEqual(driver.check_study_args(self._args(study="I3_1", tier="H")), (6, []))

    def test_scaling_and_primary(self):
        driver = _driver()
        args = self._args(study="I3_1", n_train=12, primary_n_train=6, scaling="3,6,12")
        self.assertEqual(driver.check_study_args(args), (6, [3, 6, 12]))
        with self.assertRaises(SystemExit):
            driver.check_study_args(self._args(study="I3_1", n_train=6, scaling="12"))
        self.assertEqual(driver.check_study_args(self._args()), (6, []))

    def test_study_dirs_and_codes(self):
        driver = _driver()
        self.assertEqual(driver.TIER_CODE, {"W": 0, "M": 1, "H": 2})
        self.assertTrue(str(driver.STUDY_DIRS["I3_1"]).endswith("explore/I3_1"))
        self.assertTrue(str(driver.STUDY_DIRS["E3b"]).endswith("explore/E3b"))


if __name__ == "__main__":
    unittest.main()
