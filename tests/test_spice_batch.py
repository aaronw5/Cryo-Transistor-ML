from __future__ import annotations

import math
import shutil
import unittest

import numpy as np

from cryoml.spice_batch import (ngspice_jacobian, ngspice_literal,
                                ngspice_value, simulate_pdk_multi)


class LiteralTests(unittest.TestCase):
    def test_integer_mantissa_form(self):
        self.assertEqual(ngspice_literal(-1.59917e-05), "-159917e-10")
        self.assertEqual(ngspice_literal(0.411929), "411929e-6")
        self.assertEqual(ngspice_literal(100.0), "100")
        self.assertEqual(ngspice_literal(0.0), "0")

    def test_value_is_mantissa_times_pow10(self):
        # INPevaluate: integer mantissa of the %.10g digits times pow(10, e).
        for value in (4.148e-09, -0.191127, 23.3385, 7.84243224e4, 1e20):
            text = ngspice_literal(value)
            digits, _, exponent = text.partition("e")
            expected = float(int(digits))
            if exponent:
                expected *= math.pow(10.0, int(exponent))
            self.assertEqual(ngspice_value(value), expected)


def _setup():
    from cryoml.cold_start import device_bins
    from cryoml.data_io import load_device_curves
    from cryoml.devices import PAPER_DEVICES
    from cryoml.spice_pdk import _NGSPICE_BIN, ensure_pdk77k
    if shutil.which(_NGSPICE_BIN) is None or ensure_pdk77k():
        raise unittest.SkipTest("NGSpice or the 77 K PDK is unavailable")
    device = PAPER_DEVICES[0]
    try:
        curves = load_device_curves(device)
        bin_index = device_bins()["nmos_L0p15_W1p6"]
    except Exception as exc:  # pragma: no cover - data not vendored
        raise unittest.SkipTest(f"measured data unavailable: {exc}")
    return device, curves, bin_index


class SimulateMultiTests(unittest.TestCase):
    def test_altermod_batch_is_bit_identical(self):
        from cryoml.spice_pdk import simulate_pdk
        device, curves, bin_index = _setup()
        base = {"vth0": 0.411929, "u0": 0.0740069, "eta0": -0.191127,
                "delta": 0.0533214, "toxe": 4.148e-09}
        cards = [base, {**base, "vth0": 0.42, "toxe": 4.2e-09}, None,
                 {**base, "dsub": -30.0, "u0": 0.07}]
        multi = simulate_pdk_multi(device.dev_type, device.L_um, device.W_um,
                                   curves, cards, bin_index=bin_index,
                                   timeout_s=5)
        for card, got in zip(cards, multi):
            want = simulate_pdk(device.dev_type, device.L_um, device.W_um,
                                curves, params=card, bin_index=bin_index,
                                timeout_s=5)
            for x, y in zip(got, want):
                np.testing.assert_array_equal(x, y)

    def test_jacobian_matches_one_call_per_perturbation(self):
        device, curves, bin_index = _setup()
        params = {"vth0": 0.411929, "u0": 0.0740069, "nfactor": 11.311}
        names = ["vth0", "nfactor"]
        fast = ngspice_jacobian(device, curves, params, names, 1e-2,
                                bin_index=bin_index)
        slow = ngspice_jacobian(device, curves, params, names, 1e-2,
                                bin_index=bin_index, method="separate")
        self.assertEqual(fast.shape, (sum(len(c) for c in curves), 2))
        np.testing.assert_array_equal(fast, slow)


if __name__ == "__main__":
    unittest.main()
