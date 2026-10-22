import unittest

import numpy as np

from cryoml.explore import i18_2_tud as m


class ClampTests(unittest.TestCase):
    def test_clamp_high_low_and_identity(self):
        p, ch = m.clamp_params({"a2": 2.5, "vth0": 0.3})
        self.assertEqual(p["a2"], 1.0)
        self.assertEqual(p["vth0"], 0.3)
        self.assertEqual(ch, {"a2": (2.5, 1.0)})
        p, ch = m.clamp_params({"a2": 2.5, "a1": 0.3})
        self.assertEqual((p["a2"], p["a1"]), (1.0, 0.0))
        p, ch = m.clamp_params({"a2": 0.005})
        self.assertEqual(p["a2"], 0.01)
        q = {"a2": 0.7, "u0": 0.03}
        p, ch = m.clamp_params(q)
        self.assertEqual(p, q)
        self.assertEqual(ch, {})
        p, ch = m.clamp_params({})          # card a2 = 1: not clamped
        self.assertEqual(ch, {})
        with self.assertRaises(NotImplementedError):
            m.clamp_params({"la2": 1e-9})

    def test_grid(self):
        ax = m.pen_axis()
        self.assertEqual(len(ax), 45)
        self.assertAlmostEqual(ax[-1], 1.1)
        self.assertAlmostEqual(ax[1] - ax[0], 0.025)
        sp = m.grid_curve_specs(True)
        self.assertEqual(len(sp), 45)
        self.assertTrue(np.all(sp[3]["sweep"] <= 0) and sp[3]["fixed"] < 0)

    def test_inner_group_folds(self):
        gs = ["t40_pmos_W1_L1", "t40_nmos_W1_L1", "t40_nmos_W0p12_L0p04", "t40_pmos_W1_L0p04",
              "t40_nmos_W1_L0p04", "t40_nmos_W0p36_L0p04"]
        f = m.inner_group_folds(gs)
        self.assertEqual(f, [["t40_nmos_W0p12_L0p04", "t40_nmos_W1_L1"], ["t40_nmos_W0p36_L0p04", "t40_pmos_W1_L0p04"],
                             ["t40_nmos_W1_L0p04", "t40_pmos_W1_L1"]])
        self.assertEqual(sorted(x for y in m.inner_group_folds(gs + ["t40_nmos_W0p12_L0p05"]) for x in y),
                         sorted(gs + ["t40_nmos_W0p12_L0p05"]))
        with self.assertRaises(ValueError):
            m.inner_group_folds(["a", "a"])

    def test_variant_and_labels(self):
        self.assertEqual(m.variant(2), "sym_e1_l2")
        with self.assertRaises(ValueError):
            m.variant(3)
        self.assertEqual(m.primary_label({"ci_low": 0.7, "ci_high": 0.95}, 0), "gain over C0_refit replicated")
        self.assertEqual(m.primary_label({"ci_low": 0.7, "ci_high": 0.95}, 1), "inconclusive")
        self.assertEqual(m.primary_label({"ci_low": 1.01, "ci_high": 1.2}, 0), "worse than C0_refit")
        self.assertEqual(m.secondary_label({"ci_high": 1.05}, m.NI_MARGIN), "non-inferior to M_bias v1")

    def test_validity(self):
        v = {"v1a_max": 0.0, "v1a_n": 90, "v1b_max": 0.0, "v1b_n": 45, "v1c_max": 0.0, "v1c_n": 4,
             "v2_unit_tests_pass": True, "v3_rows": {"4|selected": 45}, "v3_inner_complete": True,
             "v3_full_complete": True, "v4_grids_ok": True}
        self.assertTrue(m.decide_validity(v)["valid"])
        for k, x in (("v1a_max", 1e-6), ("v1b_max", 1.0), ("v1c_n", 2), ("v4_grids_ok", False),
                     ("v3_rows", {"4|selected": 44}), ("v3_full_complete", False)):
            self.assertFalse(m.decide_validity({**v, k: x})["valid"])
        out = m.decide_outcome({"valid": False, "fails": ["x"]}, {}, 0, {})
        self.assertEqual(out["outcome"], "invalid")


if __name__ == "__main__":
    unittest.main()
