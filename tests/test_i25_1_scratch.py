"""I25-1: stock-card template, token patcher, named-multiplier fix, M-rule."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cryoml import spice_pdk as sp  # noqa: E402
from cryoml.explore import i25_1_scratch as m  # noqa: E402

STOCK_BIN = (".model sky130_fd_pr__nfet_01v8_lvt__model.3 nmos\n"
             "* DC IV MOS Parameters\n"
             "+ lmin = 1.0e-06 lmax = 2.0e-06 wmin = 1.5e-06 wmax = 3.0e-06\n"
             "+ vth0 = {0.4+MC_MM_SWITCH*AGAUSS(0,1.0,1)*(sky130_x__vth0_slope/sqrt(l*w*mult))} lvth0 = 1.2e-8\n"
             "+ kvth0 = {0+sky130_x__kvth0_diff}\n"
             "+ k1 = 0.5 lk1 = 2e-9\n")
NEXT_BIN = ".model sky130_fd_pr__nfet_01v8_lvt__model.4 nmos\n+ vth0 = 0.3\n"


class TestI25_1(unittest.TestCase):
    def test_token_patcher_keeps_size_terms_and_adds_missing(self):
        text = STOCK_BIN + NEXT_BIN
        out = sp._patch_params_in_bin_tokens(text, "nmos", 3, {"vth0": 0.45, "k1": 0.6, "eta0": 0.1})
        span = out[:out.index(".model sky130_fd_pr__nfet_01v8_lvt__model.4")]
        assert "+ vth0 = 0.45 lvth0 = 1.2e-8" in span
        assert "+ k1 = 0.6 lk1 = 2e-9" in span
        assert "kvth0 = {0+sky130_x__kvth0_diff}" in span
        assert "+ eta0 = 0.1" in span
        assert out.endswith(NEXT_BIN)                      # other bin untouched


    def test_token_geometry(self):
        g = sp._bin_geometry_tokens(STOCK_BIN)
        assert g == {"lmin": 1e-6, "lmax": 2e-6, "wmin": 1.5e-6, "wmax": 3e-6}


    def test_named_multiplier_fix(self):
        from cryoml.cold_start import multipliers_from_text
        text = ("* x\n.param vth0_nom = 2.08113\n.param u0_nom = 3.41512\n.param ub_nom = 1\n"
                ".model sky130_fd_pr__pfet_01v8_lvt__model.0 pmos\n"
                "+vth0={(-5.4e-01+MC_MM_SWITCH*my_gauss*(s/sqrt(l*w*m)))*vth0_nom}\n"
                "+u0={3.48e-03*u0_nom}\n+ub='2.0e-18*ub_nom'\n+vsat={1.0e+05*1.5}\n")
        mult = multipliers_from_text(text, "pmos", 0)
        assert mult == {"vth0": 2.08113, "u0": 3.41512, "ub": 1.0, "vsat": 1.5}


    def test_named_multiplier_released_card(self):
        """pmos_L0p35_W1p6 bin of the released card: the five tuned named values."""
        from cryoml.cold_start import bin_multipliers, device_bins
        if sp.PDK_TEMPLATE != "paper77k" or sp.ensure_pdk77k() is not None:
            self.skipTest("paper template not available")
        mult = bin_multipliers("pmos", device_bins()["pmos_L0p35_W1p6"])
        for k, v in {"vth0": 2.08113, "u0": 3.41512, "nfactor": 13.5087, "vsat": 9999.92,
                     "rdsw": 1.11407e-05}.items():
            self.assertEqual(mult[k], v)


    def test_m_rule_pmos_uses_measured_last_point(self):
        vd = np.linspace(0, -1.85, 20)
        hi = np.full(20, 1e-6)
        lo = np.concatenate([np.full(19, 1e-6), [5e-11]])
        assert m.m_rule_keep("pmos", 8.0, 5.0, "idvd", hi)
        assert not m.m_rule_keep("pmos", 8.0, 5.0, "idvd", lo)
        assert m.m_rule_keep("nmos", 1.0, 3.0, "idvg", np.full(20, 1e-5))
        assert not m.m_rule_keep("nmos", 1.0, 3.0, "idvg", np.full(20, 1e-8))
        del vd


    def test_stock_round_one_spec_widths_and_zero_fixed(self):
        spec = m.stock_round_one_spec(("vth0", "u0", "k1", "dwg"),
                                      {"t": {"vth0": -0.45, "u0": 0.01, "k1": 0.5, "dwg": 0.0}})
        b = spec["bounds_by_device"]["t"]
        np.testing.assert_allclose(b["vth0"], sorted([-0.45 * 3, -0.45 / 3]))
        np.testing.assert_allclose(b["u0"], [0.0001, 1.0])
        np.testing.assert_allclose(b["k1"], [0.5 / 3, 1.5])
        assert b["dwg"] == [0.0, 0.0] and spec["log_scale_by_device"]["t"]["dwg"] is False


    def test_overlap_label_aggregate(self):
        ov = m.overlap(["a", "b", "c"], ["b", "c", "d", "e"])
        assert ov["n_both"] == 2 and ov["only_old"] == ["a"] and ov["only_new"] == ["d", "e"]
        np.testing.assert_allclose(ov["jaccard"], 2 / 5)
        assert m.gmr_label({"ci_low": 0.5, "ci_high": 0.9}) == "lower"
        assert m.gmr_label({"ci_low": 1.1, "ci_high": 1.3}) == "higher"
        assert m.gmr_label({"ci_low": 0.9, "ci_high": 1.1}) == "inconclusive"
        a = m.aggregate({"nmos_a": 0.1, "nmos_b": 0.3, "pmos_c": 0.5})
        np.testing.assert_allclose([a["all_device"], a["combined"]], [0.3, 0.35])


    def test_edge_fraction(self):
        spec = m.stock_round_one_spec(("vth0", "k1"), {"t": {"vth0": 0.4, "k1": 0.5}})
        e = m.edge_fraction(spec, {"t": [0.95, 0.1]})
        np.testing.assert_allclose(e["pooled"], 0.5)


STOCK_SCRIPT = r"""
import json, numpy as np
from cryoml import spice_pdk as sp
from cryoml.cold_start import device_bins
from cryoml.data_io import load_device_curves
from cryoml.card_eval import device_by_tag
assert sp.PDK_TEMPLATE == "stock" and sp.ensure_pdk77k() is None
bins = device_bins()
d = device_by_tag("nmos_L1_W3")
curves = load_device_curves(d)[:2]
a = sp.simulate_pdk(d.dev_type, d.L_um, d.W_um, curves)
span = sp._bin_text("nmos", bins["nmos_L1_W3"])
val = float(sp._bin_value(span, "k1"))
b = sp.simulate_pdk(d.dev_type, d.L_um, d.W_um, curves, params={"k1": val}, bin_index=bins["nmos_L1_W3"])
c = sp.simulate_pdk(d.dev_type, d.L_um, d.W_um, curves, params={"k1": val * 1.2}, bin_index=bins["nmos_L1_W3"])
p, ch = sp.bsim4_clamp_params("nmos", bins["nmos_L1_W3"], {"k1": val})
print(json.dumps({"bin": bins["nmos_L1_W3"], "same": max(float(np.max(np.abs(x - y))) for x, y in zip(a, b)),
                  "moved": max(float(np.max(np.abs(x - y))) for x, y in zip(a, c)), "clamp": len(ch)}))
"""


class TestI25_2(unittest.TestCase):
    def test_stock_template_end_to_end(self):
        if not Path(sp.resolve_ngspice_bin() if hasattr(sp, "resolve_ngspice_bin") else sp._NGSPICE_BIN).exists():
            self.skipTest("ngspice not available")
        env = {**os.environ, "CRYOML_PDK_TEMPLATE": "stock", "PYTHONPATH": str(ROOT / "src")}
        proc = subprocess.run([sys.executable, "-c", STOCK_SCRIPT], env=env, capture_output=True, text=True,
                              timeout=120, cwd=str(ROOT))
        assert proc.returncode == 0, proc.stderr[-2000:]
        r = json.loads(proc.stdout.strip().splitlines()[-1])
        assert r["bin"] == 35
        assert r["same"] == 0.0
        assert r["moved"] > 0.0 and math.isfinite(r["moved"])
        assert r["clamp"] == 0



class TestMult1(unittest.TestCase):
    def test_multipliers_to_one_text(self):
        text = (".param vth0_nom = 2.08113\n.param rdw_nom = 1\n"
                ".model sky130_fd_pr__pfet_01v8_lvt__model.0 pmos\n"
                "+vth0={(-5.4e-01+MC_MM_SWITCH*my_gauss*(s/sqrt(l*w*m)))*vth0_nom}\n"
                "+ub='2.0e-18*ub_nom'\n+u0={3.48e-03*11.8673}\n+k1=0.5\n")
        with self.assertRaises(ValueError):          # ub_nom has no .param line
            sp.multipliers_to_one(text)
        text = text.replace(".param rdw_nom = 1\n", ".param rdw_nom = 1\n.param ub_nom = 3\n")
        out = sp.multipliers_to_one(text)
        self.assertIn(".param vth0_nom = 1\n", out)
        self.assertIn(".param ub_nom = 1\n", out)
        self.assertIn("+u0={3.48e-03*1}\n", out)
        self.assertIn("+k1=0.5\n", out)
        from cryoml.cold_start import multipliers_from_text
        self.assertEqual(set(multipliers_from_text(out, "pmos", 0).values()), {1.0})

    def test_mult1_template_all_multipliers_one(self):
        script = ("from cryoml import spice_pdk as sp\nfrom cryoml.cold_start import bin_multipliers, device_bins\n"
                  "assert sp.PDK_TEMPLATE == 'mult1' and sp.ensure_pdk77k() is None\n"
                  "v = set()\nfor t, b in device_bins().items():\n    v |= set(bin_multipliers(t[:4], b).values())\n"
                  "print(sorted(v))\n")
        env = {**os.environ, "CRYOML_PDK_TEMPLATE": "mult1", "PYTHONPATH": str(ROOT / "src")}
        proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True,
                              timeout=60, cwd=str(ROOT))
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(proc.stdout.strip().splitlines()[-1], "[1.0]")


if __name__ == "__main__":
    unittest.main()
