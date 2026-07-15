import unittest

import numpy as np

from cryoml.metrics import (clean_current, device_rrms, device_rrms_new,
                            family_totals)


class PaperExactMetricTests(unittest.TestCase):
    def test_includes_zero_denominator_curve_as_zero_rrms(self):
        sims = [np.array([5.0, 5.0]), np.array([2.0, 4.0])]
        meas = [np.array([0.0, 0.0]), np.array([1.0, 3.0])]

        result = device_rrms(sims, meas)

        self.assertEqual(result["n_curves"], 2)
        self.assertAlmostEqual(result["rrms"], 0.25)


class ConfirmedSetupMetricTests(unittest.TestCase):
    def test_clean_current_zeroes_glitches_before_last_zero(self):
        # 6 uA SMU spike inside an off-curve must be zeroed.
        cur = np.array([0.0, 6e-6, 0.0, 0.0, 1e-6, 2e-6])
        out = clean_current(cur)
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0, 0.0, 1e-6, 2e-6])
        # untouched when no exact zeros are present
        cur2 = np.array([1e-9, 2e-9])
        np.testing.assert_array_equal(clean_current(cur2), cur2)

    @staticmethod
    def _tagged(value_on, value_off=1e-12, n=50):
        """11 metric curves; idvd@0.37 is a near-off curve, rest are on."""
        sim, meas = {}, {}
        for kind, biases in (("idvd", (0.37, 0.74, 1.11, 1.48, 1.85)),
                             ("idvg", (0.01, 0.37, 0.74, 1.11, 1.48, 1.85))):
            for b in biases:
                on = not (kind == "idvd" and b == 0.37)
                lvl = value_on if on else value_off
                meas[(kind, b)] = np.full(n, lvl)
                sim[(kind, b)] = np.full(n, lvl * 1.10)  # 10% high everywhere
        return sim, meas

    def test_nmos_excludes_below_threshold_curves(self):
        sim, meas = self._tagged(value_on=1e-5)
        res = device_rrms_new("nmos", 1.0, 1.6, sim, meas)
        # the 1e-12 idvd@0.37 curve is below the 3e-7 inclusion threshold
        self.assertEqual(res["n_curves"], 10)
        self.assertAlmostEqual(res["rrms"], 0.10, places=6)
        self.assertAlmostEqual(res["sigma"], 0.0, places=9)

    def test_pmos_excludes_when_sim_is_off(self):
        sim, meas = self._tagged(value_on=1e-5)
        # kill the sim on one included curve -> pmos |sim[-1]| rule drops it
        sim[("idvg", 1.85)] = np.full(50, 1e-13)
        res = device_rrms_new("pmos", 2.0, 5.0, sim, meas)
        tags = res["per_curve"]
        self.assertFalse(tags["idvg@1.85"]["included"])
        # idvd@0.37 passes mean>0 but its tiny sim also fails |sim[-1]|>1e-10,
        # so of the 11 curves: 9 kept, idvg@1.85 and idvd@0.37 dropped.
        self.assertEqual(res["n_curves"], 9)

    def test_fixed_inclusion_prevents_candidate_from_hiding_off_curves(self):
        sim, meas = self._tagged(value_on=1e-5)
        sim[("idvg", 1.85)] = np.full(50, 1e-13)
        tags = {
            f"{kind}@{bias:g}"
            for kind, biases in (
                ("idvd", (0.37, 0.74, 1.11, 1.48, 1.85)),
                ("idvg", (0.01, 0.37, 0.74, 1.11, 1.48, 1.85)),
            )
            for bias in biases
        }

        dynamic = device_rrms_new("pmos", 2.0, 5.0, sim, meas)
        fixed = device_rrms_new("pmos", 2.0, 5.0, sim, meas,
                                include_tags=tags)

        self.assertEqual(dynamic["n_curves"], 9)
        self.assertEqual(fixed["n_curves"], 11)
        self.assertTrue(fixed["per_curve"]["idvg@1.85"]["included"])
        self.assertGreater(fixed["rrms"], dynamic["rrms"])

    def test_family_totals_combined_is_average_of_family_means(self):
        res = {
            "nmos_a": {"rrms": 0.1, "sigma": 0.01},
            "nmos_b": {"rrms": 0.3, "sigma": 0.03},
            "pmos_a": {"rrms": 0.5, "sigma": 0.05},
        }
        tot = family_totals(res)
        self.assertAlmostEqual(tot["nmos_rrms"], 0.2)
        self.assertAlmostEqual(tot["pmos_rrms"], 0.5)
        self.assertAlmostEqual(tot["combined_rrms"], 0.35)


if __name__ == "__main__":
    unittest.main()
