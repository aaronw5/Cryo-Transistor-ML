from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_adaptive_parameter_bounds import (  # noqa: E402
    build_bounds, parse_coverage)
from pdk_gen_full_data import sample_local  # noqa: E402
from cryoml.parameter_space import ParameterBox  # noqa: E402


class AdaptiveSamplingTests(unittest.TestCase):
    def test_repeated_evidence_expands_every_device_of_one_polarity(self):
        sensitivity = {
            "bounds_by_device": {
                "nmos_a": {"rdsw": [90.0, 110.0]},
                "nmos_b": {"rdsw": [80.0, 120.0]},
                "pmos_a": {"rdsw": [45.0, 55.0]},
            },
            "published_by_device": {
                "nmos_a": {"rdsw": 100.0},
                "nmos_b": {"rdsw": 100.0},
                "pmos_a": {"rdsw": 50.0},
            },
        }
        probe = {"definition": {}, "recommendations": [{
            "recommend_expansion": True, "parameter": "rdsw",
            "dev_type": "nmos", "direction": "upper",
            "improving_devices": ["nmos_a", "nmos_b"],
            "n_improving_devices": 2,
            "supported_multiplier": 4.0,
            "levels": [
                {"multiplier": 2.0, "improving_devices": ["nmos_a"],
                 "supported": False},
                {"multiplier": 4.0,
                 "improving_devices": ["nmos_a", "nmos_b"], "supported": True},
            ],
            "best_probe": {"multiplier": 4.0},
        }]}
        result = build_bounds(sensitivity, probe, 4.0)
        self.assertEqual(
            result["bounds_by_device"]["nmos_a"]["rdsw"], [90.0, 140.0])
        self.assertEqual(
            result["bounds_by_device"]["nmos_b"]["rdsw"], [80.0, 180.0])
        self.assertEqual(
            result["bounds_by_device"]["pmos_a"]["rdsw"], [45.0, 55.0])

    def test_unsupported_multiplier_is_not_imported_from_best_probe(self):
        """A larger probe that failed the section guard must not widen a box."""
        sensitivity = {
            "bounds_by_device": {"nmos_a": {"rdsw": [90.0, 110.0]},
                                 "nmos_b": {"rdsw": [90.0, 110.0]}},
            "published_by_device": {"nmos_a": {"rdsw": 100.0},
                                    "nmos_b": {"rdsw": 100.0}},
        }
        probe = {"definition": {}, "recommendations": [{
            "recommend_expansion": True, "parameter": "rdsw",
            "dev_type": "nmos", "direction": "upper",
            "improving_devices": ["nmos_a", "nmos_b"],
            "n_improving_devices": 2,
            "supported_multiplier": 2.0,
            "levels": [
                {"multiplier": 2.0,
                 "improving_devices": ["nmos_a", "nmos_b"], "supported": True},
                {"multiplier": 4.0, "improving_devices": ["nmos_a"],
                 "supported": False},
            ],
            "best_probe": {"multiplier": 4.0},
        }]}
        result = build_bounds(sensitivity, probe, 4.0)
        self.assertEqual(
            result["bounds_by_device"]["nmos_a"]["rdsw"], [90.0, 120.0])
        self.assertEqual(
            result["adaptive_groups"][0]["evidence_devices"],
            ["nmos_a", "nmos_b"])

    def test_probe_without_supported_multiplier_is_rejected(self):
        sensitivity = {
            "bounds_by_device": {"nmos_a": {"rdsw": [90.0, 110.0]}},
            "published_by_device": {"nmos_a": {"rdsw": 100.0}},
        }
        probe = {"definition": {}, "recommendations": [{
            "recommend_expansion": True, "parameter": "rdsw",
            "dev_type": "nmos", "direction": "upper",
            "improving_devices": ["nmos_a"], "n_improving_devices": 1,
            "best_probe": {"multiplier": 4.0},
        }]}
        with self.assertRaises(SystemExit):
            build_bounds(sensitivity, probe, 4.0)

    def test_adaptive_mixture_resolves_new_boundary_slab(self):
        box = ParameterBox(
            names=("rdsw", "u0"),
            published={"rdsw": 100.0, "u0": 2.0},
            bounds={"rdsw": [90.0, 140.0], "u0": [1.8, 2.2]})
        summary = {"adaptive_changes": [{
            "device": "nmos_a", "parameter": "rdsw",
            "direction": "upper", "old_bounds": [90.0, 110.0],
        }]}
        local = sample_local(
            box, 1000, 7, summary, "nmos_a", "adaptive_mixture")
        old_upper_local = 2.0 * (110.0 - 90.0) / (140.0 - 90.0) - 1.0
        self.assertTrue(np.isfinite(local).all())
        self.assertGreater(
            np.mean(local[1:, 0] >= old_upper_local), 0.60)
        np.testing.assert_allclose(local[0], box.local_published)


if __name__ == "__main__":
    unittest.main()


class CoverageWideningTests(unittest.TestCase):
    def _sensitivity(self):
        return {
            "bounds_by_device": {
                "nmos_a": {"delta": [0.045, 0.055], "ags": [-0.11, -0.09]},
                "pmos_a": {"delta": [0.09, 0.11], "ags": [3.06, 3.74]},
            },
            "published_by_device": {
                "nmos_a": {"delta": 0.05, "ags": -0.1},
                "pmos_a": {"delta": 0.1, "ags": 3.4},
            },
        }

    def test_coverage_widens_every_polarity_symmetrically(self):
        result = build_bounds(
            self._sensitivity(), {"definition": {}, "recommendations": []},
            4.0, {"delta": 8.0})
        self.assertEqual(
            [round(value, 6) for value in
             result["bounds_by_device"]["nmos_a"]["delta"]], [0.01, 0.09])
        self.assertEqual(
            [round(value, 6) for value in
             result["bounds_by_device"]["pmos_a"]["delta"]], [0.02, 0.18])
        self.assertEqual(
            result["bounds_by_device"]["nmos_a"]["ags"], [-0.11, -0.09])
        self.assertEqual(
            len(result["coverage_expansion"]["changes"]), 4)
        self.assertEqual(
            {row["direction"] for row in result["adaptive_changes"]},
            {"upper", "lower"})

    def test_coverage_handles_negative_published_values(self):
        result = build_bounds(
            self._sensitivity(), {"definition": {}, "recommendations": []},
            4.0, {"ags": 8.0})
        low, high = result["bounds_by_device"]["nmos_a"]["ags"]
        self.assertLess(low, -0.1)
        self.assertGreater(high, -0.1)
        self.assertAlmostEqual(low, -0.18)
        self.assertAlmostEqual(high, -0.02)

    def test_coverage_spec_parsing_rejects_flat_multiplier(self):
        with self.assertRaises(ValueError):
            parse_coverage("delta:1.0")
        self.assertEqual(parse_coverage("delta:8, ags:6"),
                         {"delta": 8.0, "ags": 6.0})
