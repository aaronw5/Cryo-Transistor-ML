"""Unit tests for cryoml.explore.i11_1_deploy (pure parts)."""

import math
import unittest
from pathlib import Path

import numpy as np

from cryoml.explore import i11_1_deploy as m


class StubCorr:
    """g = 0.1 * x1 - 0.2 * x2 ** 2 + 0.05 * is_pmos on the E1 feature columns."""

    def g(self, phi):
        phi = np.asarray(phi)
        return 0.1 * phi[:, 1] - 0.2 * phi[:, 2] ** 2 + 0.05 * phi[:, 0]


NMOS = m.Dev("nmos_L1_W1p6", "nmos", 1.0, 1.6, 3)
PMOS = m.Dev("pmos_L2_W5", "pmos", 2.0, 5.0, 11)


class TestG(unittest.TestCase):
    def test_abs_folding_and_shape(self):
        c = StubCorr()
        vg = np.array([[0.5, -0.5], [1.0, -1.0]])
        vd = np.array([[0.3, 0.3], [-0.7, 0.7]])
        g = m.g_values(c, False, vg, vd)
        self.assertEqual(g.shape, (2, 2))
        self.assertAlmostEqual(g[0, 0], g[0, 1])
        self.assertAlmostEqual(g[1, 0], g[1, 1])
        self.assertAlmostEqual(float(m.g_values(c, True, 0.0, 0.0)[0]), 0.05)

    def test_partials(self):
        c = StubCorr()
        vg, vd = np.array([0.8, -0.8]), np.array([1.2, -1.2])
        dgg, dgd = m.g_partials(c, False, vg, vd)
        np.testing.assert_allclose(dgg, [0.1 / m.VDD_NORM, -0.1 / m.VDD_NORM], rtol=1e-7)
        exp_d = -0.4 * (1.2 / m.VDD_NORM) / m.VDD_NORM
        np.testing.assert_allclose(dgd, [exp_d, -exp_d], rtol=1e-6)

    def test_python_model(self):
        i = np.array([1e-6, -2e-6])
        g = np.array([0.1, -0.2])
        np.testing.assert_allclose(m.python_current(i, g), i * np.exp(g))
        y = m.python_admittance(i, np.array([1e-5 + 1e-9j, 2e-5]), g, np.array([0.3, 0.0]))
        np.testing.assert_allclose(y[0], np.exp(0.1) * (1e-5 + 1e-9j) + 1e-6 * np.exp(0.1) * 0.3)
        np.testing.assert_allclose(y[1], np.exp(-0.2) * 2e-5)


class TestStats(unittest.TestCase):
    def test_rel_diff(self):
        r = m.rel_diff(np.array([1.1e-6, 5e-10, np.nan]), np.array([1e-6, 4e-10, 1e-6]), 1e-9)
        self.assertAlmostEqual(r[0], 0.1)
        self.assertTrue(np.isnan(r[1]))
        self.assertTrue(np.isinf(r[2]))
        self.assertEqual(m.nanmax0(r), math.inf)
        self.assertEqual(m.nanmax0(np.array([np.nan])), 0.0)

    def test_norm_y_err(self):
        e = m.norm_y_err(np.array([1.001e-6, 2e-12]), np.array([1e-6, 1e-12]))
        np.testing.assert_allclose(e, [1e-3, 1e-12 / 1e-9], rtol=1e-6)

    def test_threshold_voltage(self):
        v = np.linspace(0, 1, 11)
        i = 10.0 ** (-12 + 10 * v)          # one decade per 0.1 V
        self.assertAlmostEqual(m.threshold_voltage(v, i, 1e-7), 0.5, places=6)
        self.assertTrue(math.isnan(m.threshold_voltage(v, i * 1e-9, 1e-7)))

    def test_sign_rules(self):
        card = np.array([1.0, 0.5, 0.1, 0.0])
        corr = np.array([1.0, 0.4, -0.1, -1e-13])
        mask = np.array([True, True, True, True])
        self.assertEqual(m.sign_changes_introduced(corr, card, mask), 1)
        self.assertTrue(m.curve_sign_change_introduced(corr, card, mask))
        card2 = np.array([1.0, -0.5, 0.1, 0.0])
        self.assertFalse(m.curve_sign_change_introduced(corr, card2, mask))
        self.assertFalse(m.curve_sign_change_introduced(corr, card, np.array([1, 1, 0, 1], bool)))


class TestCircuitMetrics(unittest.TestCase):
    def test_vtc(self):
        vin = np.linspace(0, 1.8, 1801)
        vout = 0.9 - 0.9 * np.tanh(20 * (vin - 0.8))
        r = m.vtc_metrics(vin, vout)
        d = vout - vin
        k = np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0][0]
        self.assertTrue(vin[k] <= r["vm"] <= vin[k + 1])
        self.assertAlmostEqual(r["gain_max"], 18.0, delta=0.1)

    def test_oscillation(self):
        t = np.linspace(0, 10e-9, 100001)
        v = 0.9 + 0.9 * np.sin(2 * np.pi * 1e9 * t)
        r = m.oscillation(t, v, 0.9, t_from=2e-9)
        self.assertAlmostEqual(r["freq_hz"] / 1e9, 1.0, places=4)
        self.assertGreaterEqual(r["n_periods"], 6)
        self.assertEqual(m.oscillation(t, np.zeros_like(t), 0.9)["n_periods"], 0)

    def test_interp(self):
        self.assertAlmostEqual(m.interp_at([2.0, 0.0, 1.0], [4.0, 0.0, 2.0], 0.5), 1.0)


class TestNetlist(unittest.TestCase):
    def test_instance(self):
        plain = m.instance("a", NMOS, "out", "in", "0", "0", False)
        self.assertTrue(plain.startswith("Xa out in 0 0 sky130_fd_pr__nfet_01v8_lvt l=1u w=1.6u"))
        self.assertNotIn("Na ", plain)
        el = m.instance("a", PMOS, "out", "in", "vdd", "vdd", True)
        lines = el.strip().splitlines()
        self.assertTrue(lines[0].startswith("Xa out_ai in vdd vdd sky130_fd_pr__pfet_01v8_lvt"))
        self.assertEqual(lines[1], "Na out in vdd out_ai corrp")

    def test_models_and_deck(self):
        mods = m.corr_models()
        self.assertIn(".model corrn i3_2_bias_corr is_pmos=0 VDD=1.85 RS=0", mods)
        self.assertIn(".model corrp i3_2_bias_corr is_pmos=1 VDD=1.85 RS=0", mods)
        d = m.deck("t", Path("/x/lib.spice"), "R1 a 0 1\n", ["op"], Path("/x/m.osdi"), m.TIGHT)
        self.assertIn(".options reltol=1e-9 abstol=1e-18 vntol=1e-12\n", d)
        self.assertIn("pre_osdi /x/m.osdi\n", d)
        self.assertIn(".options temp=-196.1500\n", d)
        self.assertLess(d.index(".model corrn"), d.index(".control"))
        d0 = m.deck("t", Path("/x/lib.spice"), "R1 a 0 1\n", ["op"], None)
        self.assertNotIn("pre_osdi", d0)
        self.assertNotIn("corrn", d0)

    def test_device_body_swap(self):
        b = m.device_body(NMOS, True, swap=True)
        self.assertIn("Xm1 0_m1i ng nd 0 ", b)
        self.assertIn("Nm1 0 ng nd 0_m1i corrn", b)

    def test_ring_and_mirror(self):
        with self.assertRaises(ValueError):
            m.ring_body(NMOS, PMOS, False, stages=4)
        r = m.ring_body(NMOS, PMOS, False, stages=5)
        self.assertIn(".ic v(o1)=1.8 v(o2)=0.0 v(o3)=1.8 v(o4)=0.0 v(o5)=1.8", r)
        self.assertIn("Xs1n o1 o5 0 0", r)
        mp = m.mirror_body(PMOS, True, 1e-5)
        self.assertIn("IREF nref 0 DC 1e-05", mp)
        mn = m.mirror_body(NMOS, False, 1e-5)
        self.assertIn("IREF 0 nref DC 1e-05", mn)

    def test_op_points_and_control(self):
        p = m.op_points(PMOS)
        self.assertEqual(p.shape, (len(m.GM_VDS) * 75 + len(m.GDS_VGS) * 75, 2))
        self.assertTrue(np.all(p <= 0))
        c = m.ac_control(p[:2], "VD")
        self.assertEqual(c[0], "alter VD ac = 1")
        self.assertIn("print q1r q1m q1g", c)
        with self.assertRaises(ValueError):
            m.ac_control(p[:2], "VB")
        g = m.grid_control(PMOS, Path("/x/o.dat"))
        self.assertEqual(g[0], "dc VD -1.85 1.85 0.025 VG 0 -1.85 -0.025")


class TestParsers(unittest.TestCase):
    def test_parse_ac(self):
        text = ("q0i = 1.5e-06\nq0r = 2.0e-05\nq0m = -3.0e-09\nq0g = 4.0e-09\n"
                "junk\nq1i = 7e-07\n")
        r = m.parse_ac(text, 2)
        self.assertEqual(r["i0"][0], 1.5e-6)
        self.assertEqual(r["yr"][0], 2e-5)
        self.assertEqual(r["ym"][0], -3e-9)
        self.assertEqual(r["yg"][0], 4e-9)
        self.assertEqual(r["i0"][1], 7e-7)
        self.assertTrue(np.isnan(r["yr"][1]))

    def test_grid_from_rows(self):
        n = len(m.GRID_VGS) * len(m.GRID_VDS)
        rows = np.column_stack([np.zeros(n), np.arange(n, dtype=float)])
        g = m.grid_from_rows(rows)
        self.assertEqual(g.shape, (75, 149))
        self.assertEqual(g[1, 0], 149.0)
        self.assertTrue(np.all(np.isnan(m.grid_from_rows(rows[:-1]))))

    def test_convergence_messages(self):
        msgs = m.convergence_messages("ok\nWarning: timestep too small; time = 1\nfine")
        self.assertEqual(len(msgs), 1)


class TestDecide(unittest.TestCase):
    BASE = {"q1_tight_max": 1e-7, "q1_default_max": 5e-4, "q1_bad": 0, "q1_vds0_max": 1e-15,
            "q2_ac_max": 1e-6, "q2_gm_sign": 0, "q2_gds_sign": 0, "q3_fail": 0,
            "q3_fail_cardonly": 0, "swap_asym_max": 0.0, "cap_scaled": False,
            "runtime_ratio_max": 1.5}

    def test_outcomes(self):
        self.assertEqual(m.decide(dict(self.BASE))["outcome"], "deployable (simulator-level)")
        s = dict(self.BASE, swap_asym_max=0.1, cap_scaled=True)
        d = m.decide(s)
        self.assertEqual(d["outcome"], "deployable with restrictions")
        self.assertEqual(len(d["restrictions"]), 2)
        for k, v in (("q1_tight_max", 2e-5), ("q1_bad", 1), ("q1_vds0_max", 2e-12),
                     ("q2_ac_max", 2e-3), ("q2_gm_sign", 1), ("q3_fail", 1),
                     ("q1_tight_max", float("nan"))):
            self.assertEqual(m.decide(dict(self.BASE, **{k: v}))["outcome"], "not deployable", k)
        self.assertEqual(m.decide(dict(self.BASE, q3_fail_cardonly=3))["outcome"],
                         "deployable (simulator-level)")


if __name__ == "__main__":
    unittest.main()
