"""I15-2: inner folds, inner score, lambda selection, contrast labels and the
primary outcome (pre-registered rules), and the variant table."""
from __future__ import annotations

import math
import unittest

import numpy as np


class InnerFoldsTest(unittest.TestCase):
    def test_assignment(self):
        from cryoml.explore.i15_2_ablate import inner_folds, inner_train
        pm = {"n_a": False, "n_b": False, "n_c": False, "n_d": False, "p_a": True, "p_b": True, "p_c": True}
        load = ["p_b", "n_c", "n_a", "p_a", "n_d", "p_c", "n_b"]
        f = inner_folds(load, pm)
        self.assertEqual(f, [["n_a", "n_d", "p_c"], ["n_b", "p_a"], ["n_c", "p_b"]])
        self.assertEqual(sorted(sum(f, [])), sorted(load))
        self.assertEqual(inner_train(load, f[0]), ["p_b", "n_c", "p_a", "n_b"])   # load order kept
        with self.assertRaises(ValueError):
            inner_folds(load + ["n_a"], pm)

    def test_real_devices(self):
        from cryoml.explore.i15_2_ablate import inner_folds
        tags = [f"nmos_{i}" for i in range(9)] + [f"pmos_{i}" for i in range(8)]
        pm = {t: t.startswith("pmos") for t in tags}
        f = inner_folds(tags, pm)
        self.assertEqual([len(x) for x in f], [6, 6, 5])
        for x in f:                       # both polarities in every inner fold
            self.assertTrue(any(pm[t] for t in x) and any(not pm[t] for t in x))


class SelectionTest(unittest.TestCase):
    def test_score(self):
        from cryoml.explore.i15_2_ablate import inner_score
        self.assertAlmostEqual(inner_score([0.1, 0.01]), 0.5 * (math.log(0.1) + math.log(0.01)))
        self.assertAlmostEqual(inner_score([0.1, float("nan")]), 0.5 * math.log(0.1))
        self.assertAlmostEqual(inner_score([0.1, float("inf")]), 0.5 * math.log(0.1))
        with self.assertRaises(ValueError):
            inner_score([])

    def test_select(self):
        from cryoml.explore.i15_2_ablate import select_lambda
        s = {0.0: -2.0, 1.0: -2.1, 2.0: -2.05, 5.0: -1.9, 10.0: -1.8}
        self.assertEqual(select_lambda(s), 1.0)
        s = {0.0: -2.1, 1.0: -2.1, 2.0: -2.05, 5.0: -2.1 + 5e-13, 10.0: -1.8}
        self.assertEqual(select_lambda(s), 5.0)          # tie within 1e-12 -> larger lambda
        with self.assertRaises(ValueError):
            select_lambda({0.0: 1.0, 1.0: 2.0})

    def test_variant_for_lambda(self):
        from cryoml.explore.i15_2_ablate import VARIANTS, variant_for_lambda
        for lam in (0, 1, 2, 5, 10):
            name = variant_for_lambda(lam)
            self.assertEqual(VARIANTS[name], ("sym_v2", 0.01, float(lam)))
        with self.assertRaises(ValueError):
            variant_for_lambda(3)


class LabelTest(unittest.TestCase):
    def test_labels(self):
        from cryoml.explore.i15_2_ablate import contrast_label, decide_outcome
        self.assertEqual(contrast_label({"ci_low": 1.01, "ci_high": 1.2}), "cost")
        self.assertEqual(contrast_label({"ci_low": 0.9, "ci_high": 1.05}), "no cost beyond the margin")
        self.assertEqual(contrast_label({"ci_low": 0.95, "ci_high": 1.06}), "inconclusive")
        self.assertEqual(contrast_label({"ci_low": float("nan"), "ci_high": 1.0}), "inconclusive")
        self.assertEqual(decide_outcome(False, {"ci_high": 0.9})["outcome"], "invalid")
        o = decide_outcome(True, {"ci_high": 1.05})
        self.assertEqual(o["outcome"], "selected variant non-inferior to M_bias v1")
        self.assertFalse(o["ci_upper_below_1"])
        o = decide_outcome(True, {"ci_high": 1.0500001})
        self.assertEqual(o["outcome"], "selected variant not shown non-inferior to M_bias v1")
        self.assertTrue(decide_outcome(True, {"ci_high": 0.99})["ci_upper_below_1"])

    def test_tables(self):
        from cryoml.explore.i15_2_ablate import CONTRASTS, LAMBDA_CURVE, NEW, STORED, VARIANTS
        self.assertEqual(set(NEW) | set(STORED), set(VARIANTS))
        for pairs in list(CONTRASTS.values()) + [LAMBDA_CURVE]:
            for a, b in pairs:
                self.assertIn(a, VARIANTS)
                self.assertIn(b, VARIANTS)


class InputsTest(unittest.TestCase):
    def test_eps0_equals_abs_inputs_forward(self):
        """EPS = 0 symmetric inputs equal the v1 inputs in forward operation
        (to rounding)."""
        from cryoml.explore.i12_1_symm import abs_inputs, sym_inputs
        rng = np.random.default_rng(1)
        for pm, s in ((False, 1.0), (True, -1.0)):
            vg = s * rng.uniform(0, 1.85, 500)
            vd = s * rng.uniform(0, 1.85, 500)
            a = sym_inputs(pm, vg, vd, eps=0.0)
            b = abs_inputs(pm, vg, vd)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
