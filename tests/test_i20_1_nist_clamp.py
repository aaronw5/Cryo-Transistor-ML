"""I20-1: BSIM4 A1/A2 clamp of the NIST decks (state C) and the pure I20-1
helpers.  Simulation tests are skipped when the NIST data or the cards are
not on disk."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from cryoml.explore import i20_1_nist_clamp as m

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "out" / "research" / "explore"


def _have_nist(src: str) -> bool:
    return (EXP / src / "start" / "bins.json").exists() and (EXP / src / "data").exists()


def _device(src: str, want):
    import importlib.util
    name = {"I8_1": "i8_1_nist", "I9_1": "i9_1_nist_rt"}[src]
    spec = importlib.util.spec_from_file_location(f"{name}_drv", ROOT / "scripts/research/explore" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    devs, _ = mod.dev_devices()
    bins = json.loads((EXP / src / "start" / "bins.json").read_text())["devices"]
    for d in sorted(devs, key=lambda x: x.tag):
        if want(d, bins[d.tag]):
            return d, bins[d.tag]
    raise unittest.SkipTest("no device matches")


class Helpers(unittest.TestCase):
    def test_clamp_class(self):
        self.assertEqual(m.clamp_class(1.0), "none")
        self.assertEqual(m.clamp_class(0.01), "none")
        self.assertEqual(m.clamp_class(1.0000001), "high")
        self.assertEqual(m.clamp_class(0.0099), "low")

    def test_a2_message(self):
        ln = "Warning: A2 = 1.87654 is larger than 1. A2 is set to 1 and A1 is set to 0."
        self.assertAlmostEqual(m.a2_from_messages(["Warning: Eta0 = -1 is negative.", ln]), 1.87654)
        self.assertIsNone(m.a2_from_messages(["Warning: Eta0 = -1 is negative."]))

    def test_selection_blocks(self):
        tags = [f"nfet_{i}" for i in range(7)] + [f"pfet_{i}" for i in range(5)]
        pm = {t: t.startswith("pfet") for t in tags}
        b = m.selection_blocks(list(reversed(tags)), pm, 6)
        self.assertEqual(len(b), 6)
        self.assertEqual(b[0], ["nfet_0", "nfet_6"])
        self.assertEqual(b[1], ["nfet_1", "pfet_0"])
        self.assertEqual(sorted(t for x in b for t in x), sorted(tags))
        lam = m.lambda_by_device(b, {j: float(j) for j in range(6)})
        self.assertEqual(lam["pfet_0"], 1.0)
        self.assertEqual(m.block_of(b)["nfet_6"], 0)

    def test_variant_and_grid(self):
        self.assertEqual(m.variant(2.0), "sym_e1_l2")
        with self.assertRaises(ValueError):
            m.variant(3.0)
        ax = m.pen_axis()
        self.assertEqual(len(ax), 73)
        self.assertAlmostEqual(float(ax[-1]), 1.8)

    def test_amendment_trigger(self):
        committed = {"nist4": -0.10, "nistrt": -0.07}
        self.assertFalse(m.amendment_needed(committed, {"nist4": {"v1_stateC": -0.09, "g_v2": -0.01},
                                                        "nistrt": {"v1_stateC": -0.05, "g_v2": -0.2}})["needed"])
        r = m.amendment_needed(committed, {"nist4": {"v1_stateC": -0.09, "g_v2": 0.0},
                                           "nistrt": {"v1_stateC": -0.05}})
        self.assertTrue(r["needed"])
        self.assertEqual(len(r["changes"]), 1)
        self.assertEqual(r["changes"][0]["new"], "C0_refit")      # S3_D = 0 is not < 0

    def test_frozen_rule(self):
        self.assertEqual(m.frozen_rule_decision(-1e-9), "apply")
        self.assertEqual(m.frozen_rule_decision(0.0), "C0_refit")
        self.assertEqual(m.frozen_rule_decision(float("nan")), "C0_refit")

    def test_max_abs_diff(self):
        self.assertEqual(m.max_abs_diff({"a": 1.0, "b": float("nan")}, {"a": 1.0, "b": float("nan")}), 0.0)
        self.assertEqual(m.max_abs_diff({"a": 1.0}, {"a": float("nan")}), float("inf"))

    def test_validity(self):
        ok = {"v1_max": 0.0, "v2_max": 0.0, "v2_n": 3, "v3_a2_rel_max": 1e-7, "v3_unclamped_msgs": 0,
              "v3_curve1_max": 0.0, "v4_unit_tests_pass": True, "v5_complete": True, "v6_test_closed": True}
        self.assertTrue(m.decide_validity(ok)["valid"])
        self.assertFalse(m.decide_validity({**ok, "v1_max": 1e-6})["valid"])


class Clamp77K(unittest.TestCase):
    """i8_1_nist: 77 K bins, no size terms."""

    def setUp(self):
        if not _have_nist("I8_1"):
            self.skipTest("I8-1 data not on disk")
        from cryoml.explore import i8_1_nist as b8
        self.b8 = b8
        self.dev, self.bi = _device("I8_1", lambda d, b: d.polarity == "nmos" and b["covered"])

    def test_unclamped_identity(self):
        p, ch = self.b8.bsim4_clamp_params(self.dev.polarity, self.bi["bin"], None)
        self.assertEqual((p, ch), ({}, {}))
        p, ch = self.b8.bsim4_clamp_params(self.dev.polarity, self.bi["bin"], {"a2": 0.7, "u0": 0.03})
        self.assertEqual((p, ch), ({"a2": 0.7, "u0": 0.03}, {}))

    def test_high_clamp(self):
        p, ch = self.b8.bsim4_clamp_params(self.dev.polarity, self.bi["bin"], {"a2": 1.5, "a1": 0.3})
        self.assertEqual(p["a2"], 1.0)
        self.assertEqual(p["a1"], 0.0)
        self.assertIn("a2", ch)

    def test_states_equal_on_stock_card(self):
        curves = self.b8.to_pipeline_curves(self.dev)
        out = {s: self.b8.simulate(self.dev.polarity, self.dev.W_um, self.dev.L_um, self.bi["bin"], curves,
                                   None, bsim4_state=s) for s in ("clamped", "legacy", "unclamped")}
        for s in ("legacy", "unclamped"):
            for a, b in zip(out["clamped"], out[s]):
                self.assertTrue(np.array_equal(a, b, equal_nan=True))

    def test_clamped_first_curve_equals_legacy(self):
        curves = self.b8.to_pipeline_curves(self.dev)
        par = {"a2": 1.8}
        c = self.b8.simulate(self.dev.polarity, self.dev.W_um, self.dev.L_um, self.bi["bin"], curves, par,
                             bsim4_state="clamped")
        l_ = self.b8.simulate(self.dev.polarity, self.dev.W_um, self.dev.L_um, self.bi["bin"], curves, par,
                              bsim4_state="legacy")
        u = self.b8.simulate(self.dev.polarity, self.dev.W_um, self.dev.L_um, self.bi["bin"], curves, par,
                             bsim4_state="unclamped")
        self.assertTrue(np.array_equal(c[0], l_[0]))
        for a, b in zip(u[1:], l_[1:]):
            self.assertTrue(np.array_equal(a, b))

    def test_bad_state(self):
        with self.assertRaises(ValueError):
            self.b8.simulate(self.dev.polarity, self.dev.W_um, self.dev.L_um, self.bi["bin"], [], None,
                             bsim4_state="x")


class ClampRT(unittest.TestCase):
    """i9_1_nist_rt: room-temperature bins, pMOS with la2 / wa2 / pa2 terms."""

    def setUp(self):
        if not _have_nist("I9_1"):
            self.skipTest("I9-1 data not on disk")
        from cryoml.explore import i9_1_nist_rt as b9
        self.b9 = b9

        def has_terms(d, b):
            return d.polarity == "pmos" and b9.effective_a1a2(d.polarity, b["bin"], d.W_um, d.L_um)["size_terms"]
        self.dev, self.bi = _device("I9_1", has_terms)

    def _eff(self, params):
        return self.b9.effective_a1a2(self.dev.polarity, self.bi["bin"], self.dev.W_um, self.dev.L_um, params)

    def test_unclamped_identity(self):
        e = self._eff(None)
        self.assertTrue(0.01 <= e["a2"] <= 1.0)
        p, ch = self.b9.bsim4_clamp_params(self.dev.polarity, self.bi["bin"], self.dev.W_um, self.dev.L_um, None)
        self.assertEqual((p, ch), ({}, {}))

    def test_high_and_low_clamp(self):
        delta = self._eff({"a2": 0.0})["a2"]          # la2 / wa2 / pa2 contribution
        for a2_eff, target in ((5.0, 1.0), (-5.0, 0.01)):
            p, ch = self.b9.bsim4_clamp_params(self.dev.polarity, self.bi["bin"], self.dev.W_um, self.dev.L_um,
                                               {"a2": a2_eff - delta})
            self.assertIn("a2_eff", ch)
            self.assertEqual((p["la2"], p["wa2"], p["pa2"]), (0.0, 0.0, 0.0))
            self.assertAlmostEqual(self._eff(p)["a2"], target, places=15)

    def test_effective_a2_equals_b4check(self):
        params = {"a2": 3.0}
        lines = m.check_lines("nistrt", self.dev, self.bi["bin"], params)
        v = m.a2_from_messages(lines)
        self.assertIsNotNone(v)
        self.assertLess(abs(v - self._eff(params)["a2"]) / abs(v), 1e-5)

    def test_states_on_stock_and_clamped_card(self):
        curves = self.b9.base.to_pipeline_curves(self.dev)
        sim = self.b9.simulate
        a = sim(self.dev.polarity, self.dev.W_um, self.dev.L_um, self.bi["bin"], curves, None, bsim4_state="clamped")
        b = sim(self.dev.polarity, self.dev.W_um, self.dev.L_um, self.bi["bin"], curves, None, bsim4_state="legacy")
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x, y, equal_nan=True))
        par = {"a2": 4.0}
        c = sim(self.dev.polarity, self.dev.W_um, self.dev.L_um, self.bi["bin"], curves, par, bsim4_state="clamped")
        l_ = sim(self.dev.polarity, self.dev.W_um, self.dev.L_um, self.bi["bin"], curves, par, bsim4_state="legacy")
        self.assertTrue(np.array_equal(c[0], l_[0]))


if __name__ == "__main__":
    unittest.main()
