from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pdk_parameter_sensitivity import (family_scales, initial_half_span,  # noqa: E402
                                       response_metrics)


class _Curve:
    def __init__(self, kind, fixed, values):
        self.kind = kind
        self.fixed = fixed
        self.Id = np.asarray(values, dtype=np.float64)


class ParameterSensitivityTests(unittest.TestCase):
    def test_zero_parameter_uses_family_scale(self):
        self.assertAlmostEqual(initial_half_span(0.0, 4.0, 0.1), 0.4)

    def test_family_scale_uses_nonzero_median(self):
        rows = {
            "a": {"dev_type": "nmos", "published": {"vth0": 0.0}},
            "b": {"dev_type": "nmos", "published": {"vth0": 2.0}},
            "c": {"dev_type": "nmos", "published": {"vth0": 4.0}},
        }
        self.assertAlmostEqual(family_scales(rows)["nmos"]["vth0"], 3.0)

    def test_zero_family_falls_back_to_other_polarity_scale(self):
        rows = {
            "n": {"dev_type": "nmos", "published": {"prwg": 0.0}},
            "p1": {"dev_type": "pmos", "published": {"prwg": 0.04}},
            "p2": {"dev_type": "pmos", "published": {"prwg": 0.06}},
        }
        scales = family_scales(rows)
        self.assertAlmostEqual(scales["nmos"]["prwg"], 0.05)
        self.assertAlmostEqual(scales["pmos"]["prwg"], 0.05)

    def test_high_voltage_response_is_reported_separately(self):
        curves = [
            _Curve("idvd", 0.74, [1.0, 1.0]),
            _Curve("idvd", 1.85, [1.0, 1.0]),
        ]
        metrics = response_metrics(
            [np.ones(2), np.ones(2)],
            [np.ones(2), np.full(2, 1.2)], curves)
        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics["max_high_voltage"], 0.2)
        self.assertAlmostEqual(metrics["max_full_grid"], 0.2)


if __name__ == "__main__":
    unittest.main()
