"""I17-1: V(d,s) = 0 values per tolerance and the re-registered Part B rule."""
from __future__ import annotations

import unittest

import numpy as np


def _values(**kw):
    s = {"q1_tight_viol": 0, "q1_bad": 0, "q1_vds0_max": 0.0, "g_text_max": 1e-16, "g_sym_text_max": 1e-16,
         "swap_asym_max": 1e-5, "q2_ac_max": 1e-5, "cap_max": 1e-14, "q2_gm_sign": 0, "q3_fail": 0,
         "q1_default_max": 1e-4, "q2_gds_sign": 0, "runtime_ratio_max": 1.5}
    s.update(kw)
    return s


def _summary(default, tight, nonfinite=0):
    d = {"elem_max": default, "n_nonfinite": 0}
    return {"default": d, "r4": d, "tight": {"elem_max": tight, "n_nonfinite": nonfinite}}


class Vds0Test(unittest.TestCase):
    def test_column(self):
        from cryoml.explore.i11_1_deploy import GRID_VDS
        from cryoml.explore.i17_1_vds0 import vds0_column
        j = vds0_column(GRID_VDS)
        self.assertEqual(GRID_VDS[j], 0.0)
        with self.assertRaises(ValueError):
            vds0_column([-0.5, 0.25, 1.0])

    def test_values(self):
        from cryoml.explore.i17_1_vds0 import summarize_vds0, vds0_values
        vgs = np.array([0.0, 0.5, 1.0])
        a = {}
        for tol, s in (("default", 1e-12), ("r4", 1e-13), ("tight", 1e-17)):
            a[f"grid_fwd_{tol}_elem"] = np.array([[1.0, 0.0, 1.0], [1.0, -2 * s, 1.0], [1.0, s, 1.0]])
            a[f"grid_fwd_{tol}_card_rep"] = np.array([[1.0, 0.0, 1.0], [1.0, s / 3, 1.0], [1.0, 0.0, 1.0]])
        v = vds0_values(a, 1, vgs)
        self.assertAlmostEqual(v["default"]["elem_max"], 2e-12)
        self.assertEqual(v["default"]["vgs_abs_at_max"], 0.5)
        self.assertEqual(v["default"]["n_above"], 1)
        self.assertEqual(v["tight"]["n_above"], 0)
        self.assertEqual(v["tight"]["n_points"], 3)
        s = summarize_vds0({"a": v, "b": vds0_values({k: x / 10 for k, x in a.items()}, 1, vgs)})
        self.assertEqual(s["default"]["at"], ["a", 0.5])
        self.assertEqual(s["default"]["n_points"], 6)

    def test_rule(self):
        from cryoml.explore.i17_1_vds0 import decide_deploy_i17, decide_outcome
        # I16-1 situation: default above 1e-12, tight below
        d = decide_deploy_i17(_values(q2_gds_sign=6), _summary(1.0006e-12, 4.9e-17))
        self.assertEqual(d["deploy"], "deployable with restrictions")
        self.assertEqual(d["restrictions"], ["gds sign changes introduced"])
        self.assertEqual(d["under_I16_1_rule"]["deploy"], "not deployable")
        self.assertIn("Q1 I_D at Vds = 0", d["under_I16_1_rule"]["core_fail"])
        # tight above threshold fails
        d = decide_deploy_i17(_values(), _summary(1e-13, 2e-12))
        self.assertEqual(d["deploy"], "not deployable")
        # non-finite tight value fails
        d = decide_deploy_i17(_values(), _summary(1e-13, 1e-17, nonfinite=1))
        self.assertEqual(d["deploy"], "not deployable")
        # all pass, no restriction
        d = decide_deploy_i17(_values(), _summary(1e-13, 1e-17))
        self.assertEqual(d["deploy"], "deployable (simulator-level)")
        # another core failure stays a failure
        d = decide_deploy_i17(_values(q2_gm_sign=1), _summary(1e-13, 1e-17))
        self.assertEqual(d["deploy"], "not deployable")
        o = decide_outcome({"valid": True}, decide_deploy_i17(_values(q2_gds_sign=1), _summary(2e-12, 1e-17)))
        self.assertEqual(o["outcome"], "deployable")
        self.assertEqual(decide_outcome({"valid": False}, {"deploy": "x"})["outcome"], "invalid")


if __name__ == "__main__":
    unittest.main()
