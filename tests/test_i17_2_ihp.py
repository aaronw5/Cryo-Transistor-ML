"""I17-2: penalty grid for another supply, fold structure, clusters, decision
tables, and the vdd / grid options of the I12-1 trainer and penalty."""
from __future__ import annotations

import unittest

import numpy as np

from cryoml.explore import i12_1_symm as v2
from cryoml.explore import i17_2_ihp as m
from tests.test_i12_1_symm import synth_devices

TAGS = ["lv_nmos_W0p15_L0p18", "lv_pmos_W0p15_L0p18", "lv_nmos_W10_L10", "hv_nmos_W0p3_L10",
        "hv_pmos_W0p3_L10", "hv_pmos_W2_L1p2"]


class GridTest(unittest.TestCase):
    def test_axis_and_specs(self):
        ax = m.pen_axis(1.2)
        self.assertEqual(len(ax), 75)
        self.assertEqual(ax[0], 0.0)
        self.assertAlmostEqual(ax[-1], 1.2)
        self.assertAlmostEqual(m.pen_step(3.3), 3.3 / 74)
        sp = m.grid_curve_specs(1.2, True)
        self.assertEqual(len(sp), 75)
        self.assertLessEqual(sp[-1]["fixed"], 0.0)
        self.assertAlmostEqual(sp[-1]["sweep"][-1], -1.2)
        self.assertTrue(all(s["vb"] == 0.0 for s in sp))

    def test_penalty_arrays_grid_option(self):
        UG, UD = np.meshgrid(v2.PEN_GRID, v2.PEN_GRID, indexing="ij")
        i = 1e-6 * np.exp(UG) * (1 - np.exp(-UD / 0.1))
        a = v2.penalty_arrays(i, False)
        b = v2.penalty_arrays(i, False, grid=v2.PEN_GRID)
        for k in ("ug", "ud", "r_gm", "r_gds"):
            np.testing.assert_array_equal(a[k], b[k])
        ax = m.pen_axis(3.3)
        UG, UD = np.meshgrid(ax, ax, indexing="ij")
        i = 1e-6 * np.exp(UG) * (1 - np.exp(-UD / 0.3))
        p = v2.penalty_arrays(-i, True, step=m.pen_step(3.3), grid=ax)
        self.assertAlmostEqual(float(p["ug"].max()), 3.3)
        inner = (p["ug"] > 0.1) & (p["ug"] < 3.2)
        np.testing.assert_allclose(p["r_gm"][inner], 1.0, rtol=2e-3)


class TrainerVddTest(unittest.TestCase):
    def test_default_vdd_unchanged_and_override(self):
        devs = synth_devices(2)
        hp = {"steps": 20, "seeds": (0,)}
        a = v2.fit_v2(devs, None, 0.0, hyper=hp)
        b = v2.fit_v2(devs, None, 0.0, hyper=hp, vdd=v2.VDD_NORM)
        for (wa, ba), (wb, bb) in zip(a.nets[0], b.nets[0]):
            np.testing.assert_array_equal(wa, wb)
            np.testing.assert_array_equal(ba, bb)
        c = v2.fit_v2(devs, None, 0.0, hyper=hp, vdd=1.2)
        self.assertEqual(c.vdd, 1.2)
        self.assertEqual(v2.SymCorrection.from_json(c.to_json()).vdd, 1.2)
        np.testing.assert_allclose(c.mu[1:], a.mu[1:] * v2.VDD_NORM / 1.2, rtol=1e-12)

    def test_penalty_with_vdd(self):
        devs = synth_devices(2)
        ug, ud = np.meshgrid(np.linspace(0.3, 1.2, 5), np.linspace(0.1, 1.2, 5), indexing="ij")
        pens = [{"ug": ug.ravel(), "ud": ud.ravel(), "r_gm": np.full(25, 1.0), "r_gds": np.full(25, 0.02),
                 "is_pmos": d.family == "pmos"} for d in devs]
        c = v2.fit_v2(devs, pens, 1.0, hyper={"steps": 20, "seeds": (0,)}, vdd=1.2)
        self.assertTrue(np.isfinite(v2.penalty_value_numpy(c, pens)["P"]))


class FoldTest(unittest.TestCase):
    def test_fold_train_within_class(self):
        tr = m.fold_train(TAGS, "hv_pmos_W0p3_L10")
        self.assertEqual(tr, ["hv_nmos_W0p3_L10", "hv_pmos_W2_L1p2"])
        self.assertNotIn("lv_nmos_W10_L10", tr)

    def test_clusters(self):
        c = m.clusters(TAGS)
        self.assertEqual(c["lv_nmos_W0p15_L0p18"], c["lv_pmos_W0p15_L0p18"])
        self.assertEqual(c["hv_nmos_W0p3_L10"], c["hv_pmos_W0p3_L10"])
        self.assertNotEqual(c["hv_pmos_W2_L1p2"], c["hv_pmos_W0p3_L10"])
        self.assertEqual(len(set(c.values())), 4)

    def test_variant(self):
        self.assertEqual(m.variant(2), "sym_e1_l2")
        with self.assertRaises(ValueError):
            m.variant(3)


class DecisionTest(unittest.TestCase):
    def ok(self, **kw):
        v = {"v1_c0_max": 0.0, "v1_mb_max": 0.0, "v1_n": 18, "v2_unit_tests_pass": True,
             "v3_rows": {"C0_refit": 18, "selected": 18}, "v3_inner_complete": True, "v4_grids_ok": True}
        v.update(kw)
        return m.decide_validity(v)

    def test_validity(self):
        self.assertTrue(self.ok()["valid"])
        self.assertFalse(self.ok(v1_mb_max=1e-6)["valid"])
        self.assertFalse(self.ok(v3_rows={"C0_refit": 17})["valid"])
        self.assertFalse(self.ok(v4_grids_ok=False)["valid"])

    def test_labels(self):
        g = lambda lo, hi: {"ratio": (lo + hi) / 2, "ci_low": lo, "ci_high": hi}  # noqa: E731
        self.assertEqual(m.primary_label(g(0.8, 0.95), 0), "gain over C0_refit replicated")
        self.assertEqual(m.primary_label(g(0.8, 0.95), 1), "inconclusive")
        self.assertEqual(m.primary_label(g(1.01, 1.4), 0), "worse than C0_refit")
        self.assertEqual(m.primary_label(g(0.9, 1.1), 0), "inconclusive")
        self.assertEqual(m.secondary_label(g(0.9, 1.05)), "non-inferior to M_bias v1")
        self.assertEqual(m.secondary_label(g(0.9, 1.06)), "not shown non-inferior to M_bias v1")
        o = m.decide_outcome({"valid": True}, g(1.01, 1.4), 0, g(0.9, 1.0))
        self.assertEqual((o["outcome"], o["part_B"]), ("worse than C0_refit", "not run"))
        self.assertEqual(m.decide_outcome({"valid": False}, g(0.8, 0.9), 0, g(0.9, 1.0))["outcome"], "invalid")
        self.assertTrue(m.sign_check(1.2, "> 0"))
        self.assertFalse(m.sign_check(1.2, "< 0"))


if __name__ == "__main__":
    unittest.main()
