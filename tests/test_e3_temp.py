"""E3 temperature-misspecification world: boundary values, deck parity and
the BSIM4 u0 unit switch (u0 > 1 is read as cm^2/Vs)."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

from cryoml.explore import e3_temp as E
from cryoml.parameter_sets import PARAMS7

ROOT = Path(__file__).resolve().parents[1]
TAG = "nmos_L1_W3"
K_U0 = PARAMS7.index("u0")
E3_SEED = 20261019
TIER_CODE = {"W": 0, "M": 1}


def _e3_tags() -> list[str]:
    """The REAL_DEV geometries used by E3 (the E3 full-run config)."""
    split = ROOT / "out" / "research" / "explore" / "real_split.json"
    return json.loads(split.read_text())["REAL_DEV"]


def _truths():
    rng = np.random.default_rng(3)
    return E.draw_truth(TAG, "W", rng), E.draw_truth(TAG, "M", rng)


def _driver():
    path = ROOT / "scripts" / "research" / "explore" / "e3_temp_misspec.py"
    spec = importlib.util.spec_from_file_location("e3_temp_misspec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LawAndDeckTests(unittest.TestCase):
    def test_law_boundaries(self):
        w_truth, m_truth = _truths()
        for truth in (w_truth, m_truth):
            self.assertTrue(np.allclose(E.truth_x(truth, 77.0), truth["mu"]))
        self.assertTrue(np.allclose(E.truth_x(m_truth, E.T_HOT), m_truth["c"]))
        self.assertTrue(np.allclose(E.truth_x(w_truth, E.T_HOT), 0.0))
        self.assertEqual(E.truth_rs(w_truth, 150.0), 0.0)
        self.assertTrue(0 < E.truth_rs(m_truth, 77.0) < E.truth_rs(m_truth, 300.0))

    def test_phi_and_interp(self):
        for anchors in E.OBS_SETS.values():
            for t in anchors:
                self.assertLess(abs(E.phi(anchors, t)), 1e-12)
            ax = np.arange(len(anchors) * E.N_PAR, dtype=float).reshape(len(anchors), E.N_PAR)
            self.assertTrue(np.allclose(E.linear_interp(anchors, ax, anchors[0]), ax[0]))

    def test_resistor_deck_matches_simulate_pdk(self):
        _, m_truth = _truths()
        zero = E.simulate_truth_at(TAG, dict(m_truth, rs300_ohm=0.0), 150.0)
        tiny = E.simulate_truth_at(TAG, dict(m_truth, rs300_ohm=1e-3), 150.0)
        self.assertLess(np.nanmax(np.abs(zero - tiny)) / np.nanmax(np.abs(zero)), 1e-4)


class U0SwitchTests(unittest.TestCase):
    def test_switch_value(self):
        for tag in _e3_tags():
            stock, _, _ = E.stock_and_box(tag)
            self.assertAlmostEqual(abs(stock[K_U0]) * np.exp(E.u0_switch_x(tag)), 1.0)

    def test_x_bounds_u0_below_switch(self):
        for tag in _e3_tags():
            lo, hi = E.x_bounds(tag)
            switch = E.u0_switch_x(tag)
            self.assertLessEqual(hi[K_U0], switch - E.U0_SWITCH_MARGIN + 1e-12, tag)
            # the forward FD step of the Jacobian stays below the switch
            self.assertLess(hi[K_U0] + E.FD_STEP, switch, tag)
            self.assertLess(lo[K_U0], hi[K_U0], tag)
            self.assertEqual(E.u0_upper_x(tag), hi[K_U0])

    def test_x_bounds_other_parameters_unchanged(self):
        for tag in _e3_tags():
            _, lo0, hi0 = E.stock_and_box(tag)
            lo, hi = E.x_bounds(tag)
            self.assertTrue(np.allclose(lo, np.minimum(lo0, 0.0) - 0.5))
            others = [k for k in range(E.N_PAR) if k != K_U0]
            self.assertTrue(np.allclose(hi[others], (np.maximum(hi0, 0.0) + 0.5)[others]))

    def test_draw_truth_never_crosses_switch(self):
        for tag in _e3_tags():
            for tier in ("M", "W"):
                switch = E.u0_switch_x(tag)
                for seed in range(60):
                    truth = E.draw_truth(tag, tier, np.random.default_rng([7, seed]))
                    self.assertLess(E.truth_u0_max(truth, E.TEMPS_DENSE), switch, (tag, tier))
                    self.assertFalse(E.crosses_u0_switch(truth))
                    for t in E.TEMPS_DENSE:
                        self.assertFalse(E.at_or_above_u0_switch(tag, E.truth_x(truth, t)))

    def test_e3_streams_never_cross_switch(self):
        """Every truth of the E3/E3b decision configuration (seed, tiers M and W,
        6 training and 3 test draws per geometry, driver stream layout)."""
        for k, tag in enumerate(_e3_tags()):
            for tier in ("M", "W"):
                for split, n in ((0, 6), (1, 3)):
                    for i in range(n):
                        rng = np.random.default_rng([E3_SEED, TIER_CODE[tier], split, k, i])
                        truth = E.draw_truth(tag, tier, rng)
                        self.assertLess(E.truth_u0_max(truth), E.u0_switch_x(tag))

    def test_rejections_recorded(self):
        truth = E.draw_truth(TAG, "M", np.random.default_rng(3))
        self.assertIsInstance(truth["u0_switch_rejections"], int)
        self.assertGreaterEqual(truth["u0_switch_rejections"], 0)
        # E3 tier-M test device nmos_L1_W3__2 (I2-0 diagnosis part B): the first
        # draw of its stream crosses the switch, the accepted draw is the fourth.
        index = _e3_tags().index(TAG)
        first = E._draw_truth_once(TAG, "M", np.random.default_rng([E3_SEED, 1, 1, index, 2]))
        self.assertTrue(E.crosses_u0_switch(first))
        truth = E.draw_truth(TAG, "M", np.random.default_rng([E3_SEED, 1, 1, index, 2]))
        self.assertEqual(truth["u0_switch_rejections"], 3)

    def test_unrejected_draw_is_unchanged(self):
        """A draw that does not cross is identical to the pre-fix draw."""
        rng_a, rng_b = np.random.default_rng([E3_SEED, 1, 1, 0, 0]), \
            np.random.default_rng([E3_SEED, 1, 1, 0, 0])
        truth = E.draw_truth("nmos_L0p15_W1p6", "M", rng_a)
        once = E._draw_truth_once("nmos_L0p15_W1p6", "M", rng_b)
        self.assertEqual(truth["u0_switch_rejections"], 0)
        self.assertEqual(truth["mu"], once["mu"])

    def test_draw_truth_raises_after_max_attempts(self):
        original = E.crosses_u0_switch
        try:
            E.crosses_u0_switch = lambda truth, temps=E.TEMPS_DENSE: True
            with self.assertRaises(RuntimeError):
                E.draw_truth(TAG, "M", np.random.default_rng(0), max_attempts=3)
        finally:
            E.crosses_u0_switch = original

    def test_clip_u0(self):
        upper = E.u0_upper_x(TAG)
        x = np.zeros(E.N_PAR)
        x[K_U0] = upper + 0.3
        x[0] = 9.0
        out, clipped = E.clip_u0(TAG, x)
        self.assertTrue(clipped)
        self.assertEqual(out[K_U0], upper)
        self.assertEqual(out[0], 9.0)
        self.assertFalse(E.at_or_above_u0_switch(TAG, out))
        same, clipped = E.clip_u0(TAG, np.zeros(E.N_PAR))
        self.assertFalse(clipped)
        self.assertTrue(np.array_equal(same, np.zeros(E.N_PAR)))

    def test_law_predictions_inside_bounds(self):
        """par3/par14 laws are convex combinations of mu and c in [lo, hi]."""
        lo, hi = E.x_bounds(TAG)
        rng = np.random.default_rng(1)
        for _ in range(200):
            mu, c = rng.uniform(lo, hi), rng.uniform(lo, hi)
            lg = rng.uniform(np.log(1 / 6), np.log(6), E.N_PAR)
            for t in E.TEMPS_DENSE:
                x = E.law_x(mu, c, lg, t)
                self.assertLessEqual(x[K_U0], max(mu[K_U0], c[K_U0]) + 1e-12)
                self.assertLessEqual(x[K_U0], hi[K_U0] + 1e-12)
                self.assertLessEqual(E.law_x(mu, np.zeros(E.N_PAR), lg, t)[K_U0],
                                     hi[K_U0] + 1e-12)


class PairedRatioTests(unittest.TestCase):
    def test_bootstrap_log_ratio(self):
        driver = _driver()
        a = np.array([0.1, 0.2, 0.4, 0.3])
        b = np.array([0.2, 0.2, 0.2, 0.6])
        out = driver.bootstrap_log_ratio(a, b)
        self.assertAlmostEqual(out["geometric_mean_ratio"],
                               float(np.exp(np.mean(np.log(a / b)))))
        self.assertEqual(out["wins"], 2)
        self.assertEqual(out["n_devices"], 4)
        self.assertLessEqual(out["ci95"][0], out["geometric_mean_ratio"])
        self.assertGreaterEqual(out["ci95"][1], out["geometric_mean_ratio"])


if __name__ == "__main__":
    unittest.main()
