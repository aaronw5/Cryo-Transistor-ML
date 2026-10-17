from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml import strict_score as ss  # noqa: E402
from cryoml.metrics import (VD_METRIC_BIASES, VG_METRIC_BIASES,  # noqa: E402
                            device_rrms_new)


def tagged(n_points: int = 20, nan_tags: tuple = ()):
    rng = np.random.default_rng(0)
    meas, sim = {}, {}
    for kind, biases in (("idvd", VD_METRIC_BIASES), ("idvg", VG_METRIC_BIASES)):
        for b in biases:
            m = 1e-4 * (1 + rng.random(n_points))
            s = m * (1 + 0.1 * rng.standard_normal(n_points))
            if f"{kind}@{b:g}" in nan_tags:
                s = s.copy()
                s[n_points // 2] = np.nan
            meas[(kind, b)], sim[(kind, b)] = m, s
    return sim, meas


ALL_TAGS = {f"idvd@{b:g}" for b in VD_METRIC_BIASES} | {
    f"idvg@{b:g}" for b in VG_METRIC_BIASES}


class DeviceStatus(unittest.TestCase):
    def test_metric_drops_nan_and_strict_flags_it(self):
        sim, meas = tagged(nan_tags=("idvd@0.74",))
        res = device_rrms_new("nmos", 1.0, 3.0, sim, meas, include_tags=ALL_TAGS)
        # 10 finite curves (the current port averages them; the proposed
        # core fix makes the device rrms NaN -- the strict status is the same).
        self.assertEqual(res["n_curves"], 10)
        st = ss.device_status(res["per_curve"], ALL_TAGS)
        self.assertEqual(st["status"], ss.FAILED)
        self.assertEqual((st["n_included"], st["n_failed"]), (11, 1))
        self.assertEqual(st["failed_tags"], {"idvd@0.74": "nonfinite"})
        self.assertTrue(math.isnan(st["strict_rrms"]))
        finite = [v["rrms"] for v in res["per_curve"].values()
                  if v["rrms"] is not None and math.isfinite(v["rrms"])]
        self.assertAlmostEqual(st["penalized_rrms"],
                               (sum(finite) + ss.PENALTY) / 11)

    def test_valid_matches_metric(self):
        sim, meas = tagged()
        res = device_rrms_new("nmos", 1.0, 3.0, sim, meas, include_tags=ALL_TAGS)
        st = ss.device_status(res["per_curve"], ALL_TAGS)
        self.assertEqual(st["status"], ss.VALID)
        self.assertAlmostEqual(st["strict_rrms"], res["rrms"])
        self.assertAlmostEqual(st["penalized_rrms"], res["rrms"])

    def test_missing_curve_and_flat_format(self):
        sim, meas = tagged()
        del sim[("idvg", 1.85)]
        res = device_rrms_new("nmos", 1.0, 3.0, sim, meas, include_tags=ALL_TAGS)
        st = ss.device_status(res["per_curve"], ALL_TAGS)
        self.assertEqual(st["failed_tags"], {"idvg@1.85": "not_scored"})
        flat = {"a": 0.1, "b": float("nan")}
        st = ss.device_status(flat, {"a", "b", "c"})
        self.assertEqual(st["failed_tags"], {"b": "nonfinite", "c": "not_scored"})
        self.assertAlmostEqual(st["penalized_rrms"], (0.1 + 2 * ss.PENALTY) / 3)

    def test_zero_target_is_not_a_failure(self):
        per_curve = {"a": {"rrms": 0.2, "included": True, "mean_abs_meas": 1.0},
                     "b": {"rrms": None, "included": False, "mean_abs_meas": 0.0}}
        st = ss.device_status(per_curve, {"a", "b"})
        self.assertEqual(st["status"], ss.VALID)
        self.assertEqual(st["zero_target_tags"], ["b"])
        self.assertEqual(st["n_included"], 1)


class PolishStatus(unittest.TestCase):
    def test_inferred_from_penalized_history(self):
        n, k, finite_mean = 10, 9, 0.0194
        objective = (finite_mean * (n - k) + ss.PENALTY * k) / n
        st = ss.polish_status({"rrms": finite_mean,
                               "history": [9.1, objective]}, n)
        self.assertEqual((st["status"], st["n_failed"]), (ss.FAILED, 9))
        self.assertAlmostEqual(st["penalized_rrms"], objective)
        clean = ss.polish_status({"rrms": 0.3, "history": [0.4, 0.3]}, n)
        self.assertEqual(clean["status"], ss.VALID)
        self.assertAlmostEqual(clean["strict_rrms"], 0.3)

    def test_nonfinite_device_score(self):
        st = ss.polish_status({"rrms": float("inf"), "history": [9.0, 0.95]}, 10)
        self.assertEqual(st["status"], ss.FAILED)
        self.assertIsNone(st["n_failed"])
        self.assertAlmostEqual(st["penalized_rrms"], 0.95)

    def test_explicit_field_wins(self):
        st = ss.polish_status({"rrms": 0.1, "failed_curves": 2,
                               "n_included_curves": 9, "history": [0.1]}, 9)
        self.assertEqual((st["n_failed"], st["basis"]), (2, "failed_curves_field"))


class Aggregate(unittest.TestCase):
    def test_conventions(self):
        statuses = {
            "nmos_a": {"status": ss.VALID, "strict_rrms": 0.1, "penalized_rrms": 0.1,
                       "n_failed": 0},
            "nmos_b": {"status": ss.FAILED, "strict_rrms": math.nan,
                       "penalized_rrms": 1.0, "n_failed": 1},
            "pmos_c": {"status": ss.VALID, "strict_rrms": 0.3, "penalized_rrms": 0.3,
                       "n_failed": 0},
        }
        agg = ss.aggregate(statuses, {"nmos_a": 0.1, "nmos_b": 0.05, "pmos_c": 0.3})
        self.assertEqual(agg["n_failed_devices"], 1)
        self.assertEqual(agg["failed_devices"], ["nmos_b"])
        self.assertAlmostEqual(agg["valid"]["all_device_mean_rrms"], 0.2)
        self.assertAlmostEqual(agg["valid"]["combined_rrms"], 0.2)
        self.assertAlmostEqual(agg["penalized"]["all_device_mean_rrms"], 1.4 / 3)
        self.assertAlmostEqual(agg["penalized"]["combined_rrms"], (0.55 + 0.3) / 2)
        self.assertAlmostEqual(agg["reported"]["all_device_mean_rrms"], 0.45 / 3)

    def test_penalty_matches_optimizers(self):
        from cryoml import ngspice_fit, temp_fit
        self.assertEqual(ss.PENALTY, ngspice_fit.PENALTY)
        self.assertEqual(ss.PENALTY, temp_fit.PENALTY)


if __name__ == "__main__":
    unittest.main()
