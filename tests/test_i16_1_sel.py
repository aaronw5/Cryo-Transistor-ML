"""I16-1: sources of the selected held-out pairs, sign-change counts and the
pre-registered validity / outcome tables."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "out/research/explore"


class SourcesTest(unittest.TestCase):
    def test_sources(self):
        from cryoml.explore.i16_1_sel import DEPLOYED, fold_key, source_for
        self.assertEqual(source_for(1), ("I15_2/lodo_sym_e1_l1", "sym_e1_l1"))
        self.assertEqual(source_for(10.0), ("I13_1/lodo", "M_bias_v2"))
        self.assertEqual(fold_key(2.0, "pmos_L2_W5"), "sym_e1_l2|pmos_L2_W5")
        self.assertEqual(fold_key(0.0, "pmos_L4_W7"), "sym_e1_l0|pmos_L4_W7")
        self.assertEqual(DEPLOYED["key"], "full|sym_e1_l1")
        with self.assertRaises(KeyError):
            source_for(3.0)

    @unittest.skipUnless((EXP / "I15_2/decision.json").exists(), "I15-2 outputs not present")
    def test_every_selected_pair_exists(self):
        from cryoml.explore.i16_1_sel import fold_key, source_for
        lam = json.loads((EXP / "I15_2/decision.json").read_text())["lambda_star"]
        self.assertEqual(float(lam["full"]), 1.0)
        tags = [t for t in lam if t != "full"]
        self.assertEqual(len(tags), 18)
        for t in tags:
            d = EXP / source_for(lam[t])[0]
            self.assertIn(fold_key(lam[t], t), json.loads((d / "corrections.json").read_text()))
            self.assertIn(t, json.loads((d / "heldout_cards.json").read_text()))


class CountTest(unittest.TestCase):
    def test_neg_count(self):
        from cryoml.explore.i16_1_sel import neg_count, neg_range
        x = np.linspace(0.0, 1.0, 101)
        y = x.copy()
        y[60:70] = y[60] - 0.01 * np.arange(10)          # decreasing section
        n = neg_count(x, y, np.ones_like(x, bool))
        self.assertGreater(n, 0)
        self.assertEqual(neg_count(x, y, x < 0.5), 0)
        r = neg_range(x, y, np.ones_like(x, bool))
        self.assertTrue(0.55 <= r[0] <= r[1] <= 0.75)
        self.assertIsNone(neg_range(x, x, np.ones_like(x, bool)))
        # pMOS: x and I negative, sorted by |x| (descending x); monotone -> no negative derivative
        xp = -x
        self.assertEqual(neg_count(xp, -np.abs(xp) ** 2, np.ones_like(x, bool)), 0)


class DecisionTest(unittest.TestCase):
    def ok(self, **kw):
        v = {"v1_max_rrms_diff": 0.0, "v1_n_rows": 18, "v2_unit_tests_pass": True, "v3_n_rows": 18,
             "v3_m2_failed": 0, "v4_deployed_ok": True}
        v.update(kw)
        return v

    def test_validity(self):
        from cryoml.explore.i16_1_sel import decide_validity
        self.assertTrue(decide_validity(self.ok())["valid"])
        self.assertTrue(decide_validity(self.ok(v1_max_rrms_diff=1e-9))["valid"])
        for kw in ({"v1_max_rrms_diff": 2e-9}, {"v1_max_rrms_diff": float("nan")}, {"v1_n_rows": 17},
                   {"v2_unit_tests_pass": False}, {"v3_n_rows": 17}, {"v3_m2_failed": 1},
                   {"v4_deployed_ok": False}):
            self.assertFalse(decide_validity(self.ok(**kw))["valid"], kw)

    def test_outcome(self):
        from cryoml.explore.i16_1_sel import decide_outcome
        good, bad = {"valid": True, "fails": []}, {"valid": False, "fails": ["x"]}
        dr = {"deploy": "deployable with restrictions", "core_fail": [], "restrictions": ["gds sign changes introduced"]}
        ds = {"deploy": "deployable (simulator-level)", "core_fail": [], "restrictions": []}
        nd = {"deploy": "not deployable", "core_fail": ["Q2 gm sign change in strong inversion"],
              "restrictions": []}
        self.assertEqual(decide_outcome(bad, dr)["outcome"], "invalid")
        o = decide_outcome(good, dr)
        self.assertEqual(o["outcome"], "deployable")
        self.assertEqual(o["restrictions"], ["gds sign changes introduced"])
        self.assertEqual(decide_outcome(good, ds)["outcome"], "deployable")
        self.assertEqual(decide_outcome(good, nd)["outcome"], "not deployable")

    def test_gds_is_restriction_gm_is_core(self):
        """The I15-1 classification: gds sign change -> restriction, gm -> core."""
        from cryoml.explore.i12_1_symm import decide_deploy
        s = {"q1_tight_viol": 0, "q1_bad": 0, "q1_vds0_max": 1e-13, "g_text_max": 1e-16,
             "g_sym_text_max": 1e-16, "swap_asym_max": 1e-5, "q2_ac_max": 1e-5, "cap_max": 1e-12,
             "q2_gm_sign": 0, "q3_fail": 0, "q1_default_max": 1e-4, "q2_gds_sign": 7,
             "runtime_ratio_max": 1.5}
        d = decide_deploy(s)
        self.assertEqual(d["deploy"], "deployable with restrictions")
        self.assertEqual(d["restrictions"], ["gds sign changes introduced"])
        d = decide_deploy({**s, "q2_gm_sign": 1, "q2_gds_sign": 0})
        self.assertEqual(d["deploy"], "not deployable")


if __name__ == "__main__":
    unittest.main()
