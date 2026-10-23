import math
import unittest

from cryoml.explore import i19_1_dataset_gate as m


class InnerPlan(unittest.TestCase):
    def test_device_groups_all_others(self):
        order = ["a", "b", "c"]
        plan = m.inner_plan(order, lambda t: t, lambda k: [t for t in order if t != k])
        self.assertEqual([p[0] for p in plan], ["a", "b", "c"])
        self.assertEqual(plan[1], ("b", ["a", "c"], ["b"]))

    def test_class_restricted_training(self):
        cls = {"l1": "lv", "l2": "lv", "l3": "lv", "h1": "hv", "h2": "hv"}
        order = list(cls)
        plan = dict((k, (tr, sc)) for k, tr, sc in
                    m.inner_plan(order, lambda t: t, lambda k: [t for t in order if t != k and cls[t] == cls[k]]))
        self.assertEqual(plan["h1"], (["h2"], ["h1"]))
        self.assertEqual(plan["l2"], (["l1", "l3"], ["l2"]))

    def test_geometry_groups_score_all_members(self):
        grp = {"a1": "A", "a2": "A", "b1": "B", "c1": "C"}
        order = list(grp)
        plan = m.inner_plan(order, grp.get, lambda k: [t for t in order if grp[t] != k])
        self.assertEqual(plan[0], ("A", ["b1", "c1"], ["a1", "a2"]))

    def test_own_group_in_training_raises(self):
        with self.assertRaises(ValueError):
            m.inner_plan(["a", "b"], lambda t: t, lambda k: ["a", "b"])


class Gate(unittest.TestCase):
    def test_s3_nanmean(self):
        self.assertAlmostEqual(m.s3_dataset([-0.2, float("nan"), 0.1]), -0.05)
        self.assertTrue(math.isnan(m.s3_dataset([float("nan")])))
        self.assertTrue(math.isnan(m.s3_dataset([])))

    def test_strict_threshold(self):
        self.assertTrue(m.apply_gate(-1e-12))
        self.assertFalse(m.apply_gate(0.0))
        self.assertFalse(m.apply_gate(float("nan")))
        self.assertTrue(m.apply_gate(0.02, m.SENS_THRESHOLD))

    def test_gated_rows_whole_dataset(self):
        corr, c0 = {"a": 0.8, "b": 1.2, "x": 1.0}, {"a": 1.0, "b": 1.0}
        self.assertEqual(m.gated_rows(corr, c0, True), {"a": 0.8, "b": 1.2})
        self.assertEqual(m.gated_rows(corr, c0, False), {"a": 1.0, "b": 1.0})


class Rules(unittest.TestCase):
    def test_part_r_supported(self):
        r = m.part_r_rule({"s": 0.75, "i": 1.0, "t": 1.0}, {"s": 0.75, "i": 1.25, "t": 0.83})
        self.assertTrue(r["R1_never_worse"])
        self.assertTrue(r["R2_pooled_below_correction"])
        self.assertEqual(r["outcome"], "supported (retrospective)")
        self.assertAlmostEqual(r["pooled_gated_D"], math.exp(math.log(0.75) / 3))

    def test_part_r_r1_fails(self):
        r = m.part_r_rule({"s": 0.75, "i": 1.1}, {"s": 0.75, "i": 1.1})
        self.assertFalse(r["R1_never_worse"])
        self.assertEqual(r["R1_fail_datasets"], ["i"])
        self.assertFalse(r["R2_pooled_below_correction"])
        self.assertTrue(r["outcome"].startswith("not supported"))

    def test_part_c(self):
        s3 = {str(i): -0.1 for i in range(18)} | {"18": 0.05, "19": -0.01}
        y = {str(i): -0.2 for i in range(18)} | {"18": 0.07, "19": 0.03}
        c = m.part_c_rule(s3, y)
        self.assertEqual((c["correct"], c["constant_apply_correct"], c["wrong"]), (19, 18, ["19"]))
        self.assertTrue(c["holds"])
        s3["18"] = -0.05
        self.assertFalse(m.part_c_rule(s3, y)["holds"])

    def test_overall(self):
        self.assertEqual(m.overall(True, True, True), "supported")
        self.assertEqual(m.overall(True, False, True), "partially supported")
        self.assertEqual(m.overall(True, False, False), "not supported")
        self.assertEqual(m.overall(False, True, True), "invalid")

    def test_validity(self):
        v = {"v1_rows": dict(m.EXPECTED_ROWS), "v1_rows_max_diff": 0.0, "v2_max_diff": 0.0, "v2_n": 3,
             "v3_complete": True, "v4_unit_tests_pass": True, "v5_test_closed": True}
        self.assertTrue(m.decide_validity(v)["valid"])
        v["v5_test_closed"] = False
        self.assertEqual(m.decide_validity(v)["fails"], ["V5 TEST closed"])


if __name__ == "__main__":
    unittest.main()
