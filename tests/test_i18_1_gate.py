import math
import unittest

import numpy as np

from cryoml.explore import i18_1_gate as m


class GateTests(unittest.TestCase):
    def test_threshold_strict_and_nan(self):
        self.assertTrue(m.apply_gate(-1e-12))
        self.assertFalse(m.apply_gate(0.0))
        self.assertFalse(m.apply_gate(0.3))
        self.assertFalse(m.apply_gate(float("nan")))
        self.assertFalse(m.apply_gate(None))

    def test_fold_s3_nanmean(self):
        self.assertAlmostEqual(m.fold_s3([-0.2, float("nan"), 0.1]), -0.05)
        self.assertTrue(math.isnan(m.fold_s3([float("nan")])))

    def test_gated_rows_per_fold(self):
        corr = {"a1": 0.5, "a2": 0.6, "b1": 2.0}
        c0 = {"a1": 1.0, "a2": 1.0, "b1": 1.0}
        fold_of = lambda t: t[0]  # noqa: E731
        rows, dec = m.gated_rows(corr, c0, fold_of, {"a": -0.1, "b": 0.2})
        self.assertEqual(dec, {"a": True, "b": False})
        self.assertEqual(rows, {"a1": 0.5, "a2": 0.6, "b1": 1.0})

    def test_gated_rows_missing_fold(self):
        with self.assertRaises(KeyError):
            m.gated_rows({"a": 1.0}, {"a": 1.0}, lambda t: t, {})

    def test_dataset_gate_weighted(self):
        g = m.dataset_gate({"a": -0.3, "b": 0.2}, {"a": 1, "b": 2})
        self.assertAlmostEqual(g["S3_D"], (-0.3 + 0.4) / 3)
        self.assertFalse(g["apply"])

    def test_fold_y(self):
        y = m.fold_y({"a1": 0.5, "a2": 2.0, "b1": float("nan")}, {"a1": 1.0, "a2": 1.0, "b1": 1.0},
                     lambda t: t[0])
        self.assertAlmostEqual(y["a"], 0.0)
        self.assertNotIn("b", y)

    def test_rule(self):
        ok = {"ci_low": 0.6, "ci_high": 0.9}
        one = {"ci_low": 1.0, "ci_high": 1.0}
        bad = {"ci_low": 1.01, "ci_high": 1.5}
        self.assertEqual(m.decide_rule({m.SKY: ok, "x": one})["outcome"], "supported")
        r = m.decide_rule({m.SKY: ok, "x": bad})
        self.assertFalse(r["C1_never_worse"])
        self.assertIn("x", r["outcome"])
        self.assertEqual(m.decide_rule({m.SKY: one, "x": one})["outcome"], "not supported (no gain on sky130)")

    def test_validity(self):
        base = {"v1_rows": dict(m.EXPECTED_ROWS), "v1_selected_max_diff": 0.0, "v2_max_diff": 0.0, "v2_n": 2,
                "v3_complete": True, "v4_unit_tests_pass": True}
        self.assertTrue(m.decide_validity(base)["valid"])
        for k, v in (("v2_max_diff", 1e-6), ("v3_complete", False), ("v4_unit_tests_pass", False),
                     ("v1_selected_max_diff", 1e-3)):
            self.assertFalse(m.decide_validity({**base, k: v})["valid"])
        rows = dict(m.EXPECTED_ROWS)
        rows["nist_4K"] = 52
        self.assertFalse(m.decide_validity({**base, "v1_rows": rows})["valid"])

    def test_expected_rows_and_primary(self):
        self.assertEqual(len(m.PRIMARY), 9)
        self.assertEqual(sum(m.EXPECTED_ROWS.values()), 18 + 53 + 87 + 4 * 18 + 2 * 45)
        self.assertTrue(np.isfinite(m.THRESHOLD) and m.THRESHOLD == 0.0)


if __name__ == "__main__":
    unittest.main()
