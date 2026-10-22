"""I13-1: BSIM4 model-check clamp state of simulate_pdk.

(a) ``bsim4_clamp_params`` writes A2 = 1 (A1 = 0) for a2 > 1 and A2 = 0.01 for
    a2 < 0.01 and leaves every other card unchanged.
(b) The 77 K bins have no size-dependent A1/A2 terms, a1 = 0 and
    0.01 <= a2 <= 1 (published cards are never clamped).
(c) For the published card and a card with a2 <= 1 the patched card text and
    the simulated curves are identical in the "clamped" and "legacy" states
    (difference 0).
(d) For a card with a2 > 1: the "clamped" first curve equals the "legacy"
    first curve, the "unclamped" later curves equal the "legacy" later curves,
    and the clamped card prints no A2 check warning.
"""
from __future__ import annotations

import re
import shutil
import unittest

import numpy as np


def _need_ngspice():
    from cryoml.spice_pdk import _NGSPICE_BIN, ensure_pdk77k
    if shutil.which(_NGSPICE_BIN) is None or ensure_pdk77k():
        raise unittest.SkipTest("NGSpice or the 77 K PDK is unavailable")


class ClampParamsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _need_ngspice()
        from cryoml.cold_start import device_bins
        cls.bin = device_bins()["nmos_L1_W1p6"]

    def test_a2_above_one_is_clamped(self):
        from cryoml.spice_pdk import bsim4_clamp_params
        p, ch = bsim4_clamp_params("nmos", self.bin, {"a2": 1.8, "vth0": 0.5})
        self.assertEqual(p["a2"], 1.0)
        self.assertEqual(p["vth0"], 0.5)
        self.assertEqual(ch, {"a2": (1.8, 1.0)})          # a1 is 0 in the bin already
        p, ch = bsim4_clamp_params("nmos", self.bin, {"a2": 1.8, "a1": 0.3})
        self.assertEqual((p["a2"], p["a1"]), (1.0, 0.0))
        self.assertEqual(ch["a1"], (0.3, 0.0))

    def test_a2_below_low_is_clamped(self):
        from cryoml.spice_pdk import bsim4_clamp_params
        p, ch = bsim4_clamp_params("nmos", self.bin, {"a2": 0.001})
        self.assertEqual(p["a2"], 0.01)
        self.assertIn("a2", ch)

    def test_inside_range_unchanged(self):
        from cryoml.spice_pdk import bsim4_clamp_params
        for a2 in (0.01, 0.5, 1.0):
            src = {"a2": a2, "u0": 0.03}
            p, ch = bsim4_clamp_params("nmos", self.bin, src)
            self.assertEqual(p, src)
            self.assertEqual(ch, {})
        p, ch = bsim4_clamp_params("nmos", self.bin, None)
        self.assertEqual((p, ch), ({}, {}))

    def test_bins_have_no_size_terms_and_a2_in_range(self):
        from cryoml.spice_pdk import _CORNER_BASENAME, PDK77K_DIR, _bin_spans, _bin_value
        for dev in ("nmos", "pmos"):
            text = (PDK77K_DIR / _CORNER_BASENAME[dev]).read_text(errors="replace")
            spans = _bin_spans(text, dev)
            self.assertGreater(len(spans), 0)
            for _, s, e in spans:
                span = text[s:e]
                for key in ("la1", "wa1", "pa1", "la2", "wa2", "pa2"):
                    self.assertIsNone(_bin_value(span, key))
                self.assertEqual(float(_bin_value(span, "a1")), 0.0)
                a2 = float(_bin_value(span, "a2"))
                self.assertTrue(0.01 <= a2 <= 1.0, a2)

    def test_patched_text_identical_when_unclamped(self):
        from cryoml.spice_pdk import (_CORNER_BASENAME, PDK77K_DIR, _patch_params_in_bin,
                                      bsim4_clamp_params)
        base = (PDK77K_DIR / _CORNER_BASENAME["nmos"]).read_text(errors="replace")
        src = {"a2": 0.888, "vth0": 0.51, "u0": 0.031}
        p, _ = bsim4_clamp_params("nmos", self.bin, src)
        self.assertEqual(_patch_params_in_bin(base, "nmos", self.bin, p),
                         _patch_params_in_bin(base, "nmos", self.bin, src))

    def test_bad_state_raises(self):
        from cryoml.spice_pdk import simulate_pdk
        with self.assertRaises(ValueError):
            simulate_pdk("nmos", 1.0, 1.6, [], bsim4_state="other")


class ClampSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _need_ngspice()
        from cryoml.card_eval import device_by_tag
        from cryoml.cold_start import device_bins
        from cryoml.data_io import load_device_curves
        cls.rows = {}
        for tag in ("nmos_L1_W1p6", "nmos_L0p19_W7", "pmos_L2_W5"):
            d = device_by_tag(tag)
            try:
                curves = load_device_curves(d)
            except Exception as exc:  # pragma: no cover - data not vendored
                raise unittest.SkipTest(f"measured data unavailable: {exc}")
            cls.rows[tag] = (d, curves, device_bins()[tag])

    def sim(self, tag, params, state):
        from cryoml.spice_pdk import simulate_pdk
        d, curves, b = self.rows[tag]
        return simulate_pdk(d.dev_type, d.L_um, d.W_um, curves, params=params,
                            bin_index=b if params else None, bsim4_state=state)

    def test_published_identical(self):
        for tag in ("nmos_L1_W1p6", "pmos_L2_W5"):
            a, b, c = (self.sim(tag, None, s) for s in ("clamped", "legacy", "unclamped"))
            for x, y, z in zip(a, b, c):
                np.testing.assert_array_equal(x, y)
                np.testing.assert_array_equal(z, y)

    def test_card_a2_le_one_identical(self):
        p = {"a2": 0.95, "vth0": 0.40}
        a, b = self.sim("nmos_L1_W1p6", p, "clamped"), self.sim("nmos_L1_W1p6", p, "legacy")
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_card_a2_gt_one_states(self):
        from cryoml.spice_pdk import bsim4_check_messages, bsim4_clamp_params
        p = {"a2": 1.9}
        c, l, u = (self.sim("nmos_L0p19_W7", p, s) for s in ("clamped", "legacy", "unclamped"))
        np.testing.assert_array_equal(c[0], l[0])
        for x, y in zip(u[1:], l[1:]):
            np.testing.assert_array_equal(x, y)
        diff = max(float(np.max(np.abs(x - y))) for x, y in zip(c[1:], l[1:]))
        self.assertGreater(diff, 0.0)
        d, _, b = self.rows["nmos_L0p19_W7"]
        before = bsim4_check_messages(d.dev_type, d.L_um, d.W_um, p, b)
        self.assertTrue(any(re.search(r"A2 = .* larger than 1", m) for m in before))
        p2, _ = bsim4_clamp_params(d.dev_type, b, p)
        after = bsim4_check_messages(d.dev_type, d.L_um, d.W_um, p2, b)
        self.assertFalse(any("A2" in m for m in after))
        # the clamped card simulated in the legacy deck equals the clamped state
        for x, y in zip(self.sim("nmos_L0p19_W7", p2, "legacy"), c):
            np.testing.assert_array_equal(x, y)


if __name__ == "__main__":
    unittest.main()
