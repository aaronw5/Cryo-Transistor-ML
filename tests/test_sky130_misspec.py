"""Checks of cryoml.sky130_misspec (truth-only simulation of the sky130c tier).

One ngspice process at a time, about 30 NGSpice calls in total.

(a) With every component off, ``simulate_truth`` equals ``simulate_pdk``
    bit for bit (``np.array_equal``) on nmos_L1_W1p6, nmos_L0p15_W1p6,
    pmos_L2_W5 and pmos_L0p35_W1p6 (the last two use the minimal instance
    line), with the tier-a set_0 truth card when present.
(b) ``rs_ext_ohm = 1e-6`` changes every point by at most
    ``1e-6 |I| + 1e-9 A``.  The absolute term is the floating-point floor of
    a 1e-6 ohm branch: its current is G (V_nd - V_ndi) with G = 1e6 S, and
    one ulp of a ~1.8 V node voltage gives ~4e-10 A (measured: 4.7e-10 to
    5.1e-10 A on these devices at every current level).  (b2) The change is
    linear in R near 0: between R = 1e-3 and R = 1e-2 ohm it scales by
    9-11x on every point above 1e-11 A of change, so R -> 0 reproduces the
    stock deck.
(c) ``rs_ext_ohm = 100`` lowers |Id| at the largest-|Vd| point of every
    idvd curve whose current there exceeds 1e-9 A.  Below that current the
    expected change (~(gm/I) I^2 R < 1e-16 A) is under ngspice's absolute
    current tolerance (abstol = 1e-12 A), so those points are only required
    to stay finite and within 1e-3 relative (measured on nmos_L0p15_W1p6
    idvd@0.37 at 1.9e-12 A: +3e-4 relative).

Plus: ``bin_values`` returns numbers for plain values and None for
expressions; the instance rewrite moves drain and source behind RDEXT/RSEXT;
the pre-registered calibration rule of ``sky130c_floor.decide`` (no
simulation).

  PYTHONPATH=src .venv/bin/python -m unittest tests.test_sky130_misspec -v
"""

from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ("nmos_L1_W1p6", "nmos_L0p15_W1p6", "pmos_L2_W5", "pmos_L0p35_W1p6")
TRUTH = ROOT / "out" / "research" / "sim_targets" / "truth.json"


def _setup():
    from cryoml.spice_pdk import _NGSPICE_BIN, ensure_pdk77k
    if shutil.which(_NGSPICE_BIN) is None or ensure_pdk77k():
        raise unittest.SkipTest("NGSpice or the 77 K PDK is unavailable")
    from cryoml.card_eval import device_by_tag
    from cryoml.cold_start import device_bins
    from cryoml.data_io import TARGET_OVERRIDE_ENV, load_device_curves
    import os
    if os.environ.get(TARGET_OVERRIDE_ENV):
        raise unittest.SkipTest(f"{TARGET_OVERRIDE_ENV} is set")
    cards = (json.loads(TRUTH.read_text())["sets"]["0"]
             if TRUTH.exists() else {})
    bins = device_bins()
    rows = {}
    for tag in DEVICES:
        device = device_by_tag(tag)
        try:
            curves = load_device_curves(device)
        except Exception as exc:  # pragma: no cover - data not vendored
            raise unittest.SkipTest(f"measured data unavailable: {exc}")
        rows[tag] = (device, curves, bins[tag],
                     cards.get(tag, {}).get("params", {"vth0": 0.42}))
    return rows


class SimulateTruthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = _setup()
        from cryoml.sky130_misspec import simulate_truth
        from cryoml.spice_pdk import simulate_pdk
        cls.stock, cls.truth = {}, {}
        for tag, (device, curves, bin_index, card) in cls.rows.items():
            cls.stock[tag] = simulate_pdk(device.dev_type, device.L_um,
                                          device.W_um, curves, params=card,
                                          bin_index=bin_index)
            cls.truth[tag] = simulate_truth(device.dev_type, device.L_um,
                                            device.W_um, curves, card,
                                            bin_index=bin_index)

    def _sim(self, tag, **kwargs):
        from cryoml.sky130_misspec import simulate_truth
        device, curves, bin_index, card = self.rows[tag]
        return simulate_truth(device.dev_type, device.L_um, device.W_um,
                              curves, card, bin_index=bin_index, **kwargs)

    def test_a_components_off_equals_simulate_pdk(self):
        for tag in DEVICES:
            stock, truth = self.stock[tag], self.truth[tag]
            self.assertEqual(len(stock), len(truth))
            for i, (a, b) in enumerate(zip(stock, truth)):
                self.assertTrue(np.all(np.isfinite(a)), f"{tag} curve {i}")
                self.assertTrue(np.array_equal(a, b), f"{tag} curve {i}")

    def test_a2_per_curve_processes_equal_batch(self):
        # Distinct per-curve overrides force one process per curve; values
        # that round to the same card text reproduce the batch exactly.
        tag = "nmos_L1_W1p6"
        card = self.rows[tag][3]
        n = len(self.stock[tag])
        sims = self._sim(tag, curve_params=[
            {"vth0": card.get("vth0", 0.42) * (1 + 1e-15 * i)} for i in range(n)])
        for a, b in zip(self.stock[tag], sims):
            self.assertTrue(np.array_equal(a, b))

    def test_b_tiny_resistor_changes_nothing(self):
        for tag in DEVICES:
            ref = np.concatenate(self.stock[tag])
            got = np.concatenate(self._sim(tag, rs_ext_ohm=1e-6))
            self.assertTrue(np.all(np.isfinite(got)), tag)
            excess = np.abs(got - ref) - (1e-6 * np.abs(ref) + 1e-9)
            self.assertLessEqual(float(np.max(excess)), 0.0, tag)

    def test_b2_linear_response_near_zero(self):
        for tag in ("nmos_L0p15_W1p6", "pmos_L0p35_W1p6"):
            ref = np.concatenate(self.stock[tag])
            d3 = np.concatenate(self._sim(tag, rs_ext_ohm=1e-3)) - ref
            d2 = np.concatenate(self._sim(tag, rs_ext_ohm=1e-2)) - ref
            big = np.abs(d2) > 1e-10
            self.assertGreater(int(big.sum()), 100, tag)
            ratio = d2[big] / d3[big]
            self.assertGreaterEqual(float(ratio.min()), 9.0, tag)
            self.assertLessEqual(float(ratio.max()), 11.0, tag)

    def test_c_series_resistance_lowers_saturation_current(self):
        checked = 0
        for tag in DEVICES:
            curves = self.rows[tag][1]
            big = self._sim(tag, rs_ext_ohm=100.0)
            for curve, ref, got in zip(curves, self.stock[tag], big):
                if curve.kind != "idvd":
                    continue
                j = int(np.argmax(np.abs(np.asarray(curve.Vd, dtype=float))))
                self.assertTrue(np.isfinite(got[j]))
                if abs(ref[j]) > 1e-9:
                    self.assertLess(abs(got[j]), abs(ref[j]),
                                    f"{tag} idvd@{curve.fixed}")
                    checked += 1
                else:
                    self.assertLessEqual(abs(got[j] - ref[j]),
                                         1e-3 * abs(ref[j]) + 1e-15)
        self.assertGreaterEqual(checked, 14)


class StaticTests(unittest.TestCase):
    def test_instance_rewrite(self):
        from cryoml.sky130_misspec import instance_lines
        from cryoml.spice_pdk import _instance_line
        for dev_type, L, W in (("nmos", 1.0, 1.6), ("pmos", 2.0, 5.0)):
            stock = _instance_line(dev_type, L, W)
            self.assertEqual(instance_lines(dev_type, L, W, 0.0), stock)
            lines = instance_lines(dev_type, L, W, 12.5).splitlines()
            self.assertEqual(len(lines), 3)
            self.assertTrue(lines[0].startswith("Xm1 ndi ng nsi 0 "))
            self.assertEqual(lines[0][len("Xm1 ndi ng nsi 0 "):],
                             stock.rstrip("\n")[len("Xm1 nd ng 0 0 "):])
            self.assertEqual(lines[1], "RDEXT nd ndi 12.5")
            self.assertEqual(lines[2], "RSEXT nsi 0 12.5")

    def test_bin_values(self):
        from cryoml.spice_pdk import ensure_pdk77k
        if ensure_pdk77k():
            raise unittest.SkipTest("77 K PDK unavailable")
        from cryoml.sky130_misspec import MISSPEC, bin_values
        values = bin_values("nmos", 0, MISSPEC + ("rdsw", "u0", "no_such"))
        self.assertIsNone(values["rdsw"])      # {103.65*0.967479}
        self.assertIsNone(values["u0"])        # expression
        self.assertIsNone(values["no_such"])
        self.assertEqual(values["lint"], 1.2025e-08)
        self.assertEqual(values["pdiblcb"], 0.0)
        self.assertTrue(all(isinstance(values[n], float) for n in
                            ("keta", "etab", "pdiblc1", "dvt2", "wint")))


class CalibrationRuleTests(unittest.TestCase):
    """The pre-registered strength rule of scripts/research/sky130c_floor.py."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts" / "research"))
        import sky130c_floor
        cls.decide = staticmethod(sky130c_floor.decide)

    def test_incomplete(self):
        out = self.decide({0.5: 0.05, 1.0: 0.08})
        self.assertFalse(out["decided"])
        self.assertEqual(out["missing"], [2.0])

    def test_in_range_closest_to_aim(self):
        out = self.decide({0.5: 0.05, 1.0: 0.065, 2.0: 0.11})
        self.assertEqual((out["status"], out["chosen_strength"]), ("in_range", 2.0))
        out = self.decide({0.5: 0.03, 1.0: 0.07, 2.0: 0.15})
        self.assertEqual(out["chosen_strength"], 1.0)

    def test_extra_values(self):
        high = {0.5: 0.13, 1.0: 0.2, 2.0: 0.3}
        self.assertEqual(self.decide(high)["next_strength"], 0.25)
        self.assertEqual(self.decide({**high, 0.25: 0.08})["chosen_strength"], 0.25)
        out = self.decide({**high, 0.25: 0.125})
        self.assertEqual((out["status"], out["chosen_strength"]),
                         ("range_not_reached", 0.25))
        low = {0.5: 0.01, 1.0: 0.02, 2.0: 0.05}
        self.assertEqual(self.decide(low)["next_strength"], 4.0)
        out = self.decide({**low, 4.0: 0.055})
        self.assertEqual((out["status"], out["chosen_strength"]),
                         ("range_not_reached", 4.0))

    def test_straddle_uses_closest(self):
        out = self.decide({0.5: 0.055, 1.0: 0.14, 2.0: 0.3})
        self.assertEqual((out["status"], out["chosen_strength"]),
                         ("range_not_reached", 0.5))


if __name__ == "__main__":
    unittest.main()
