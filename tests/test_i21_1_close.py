"""Unit tests of the I21-1 pure helpers (``cryoml.explore.i21_1_close``)."""

import math
import unittest

import numpy as np

from cryoml.explore import i21_1_close as m


class Dev:
    def __init__(self, polarity, W_um, L_um):
        self.polarity, self.W_um, self.L_um = polarity, W_um, L_um


class TestGrids(unittest.TestCase):
    def test_axes(self):
        self.assertEqual(len(m.GRID_VGS), 73)
        self.assertEqual(len(m.GRID_VDS), 145)
        self.assertEqual(m.GRID_VGS[0], 0.0)
        self.assertAlmostEqual(m.GRID_VGS[-1], 1.8)
        self.assertAlmostEqual(m.GRID_VDS[0], -1.8)
        self.assertAlmostEqual(m.GRID_VDS[-1], 1.8)
        self.assertIn(0.0, list(m.GRID_VDS))
        for v in m.GM_VDS[1:] + m.GDS_VGS:
            self.assertTrue(np.any(np.isclose(m.GRID_VGS, v)), v)

    def test_op_points(self):
        p = m.op_points("pmos")
        self.assertEqual(p.shape, ((len(m.GM_VDS) + len(m.GDS_VGS)) * 73, 2))
        self.assertTrue(np.all(p <= 0))
        self.assertTrue(np.all(m.op_points("nmos") >= 0))

    def test_grid_rows_serpentine(self):
        nv, nd = len(m.GRID_VGS), len(m.GRID_VDS)
        rows = []
        for k in range(nv):
            vd = m.GRID_VDS if k % 2 == 0 else m.GRID_VDS[::-1]
            rows += [[x, 10.0 * k + x] for x in vd]
        g = m.grid_rows(np.array(rows))
        self.assertEqual(g.shape, (nv, nd, 1))
        self.assertTrue(np.allclose(g[3, :, 0], 30.0 + m.GRID_VDS))
        self.assertTrue(np.all(np.isnan(m.grid_rows(np.array(rows[:-1])))))

    def test_serp_control(self):
        c = m.grid_control_serp("pmos", "/tmp/x.dat", True)
        self.assertEqual(c[0], "set appendwrite")
        self.assertEqual(len(c), 1 + 3 * 73)
        self.assertIn("alter VG dc = -0.025", c[4])
        self.assertTrue(c[3].endswith("-i(VD) i(Vm1R)"))


class TestCircuitDevices(unittest.TestCase):
    def test_rule_and_ties(self):
        geoms = {"a_n": ("nmos", 0.18, 1.65), "b_n": ("nmos", 0.15, 1.0), "c_n": ("nmos", 1.0, 1.65),
                 "a_p": ("pmos", 0.35, 1.0), "b_p": ("pmos", 1.5, 5.0), "c_p": ("pmos", 0.35, 1.0)}
        got = m.circuit_devices(geoms)
        self.assertEqual(got["nmos_L0p15_W1p6"], "a_n")
        self.assertEqual(got["nmos_L1_W1p6"], "c_n")
        self.assertEqual(got["pmos_L2_W5"], "b_p")
        # a_p and c_p have the same geometry (exact tie): first tag in sorted order
        self.assertEqual(math.hypot(0.0, math.log(1.0 / 1.6)), math.hypot(0.0, math.log(1.0 / 1.6)))
        self.assertEqual(got["pmos_L0p35_W1p6"], "a_p")

    def test_prereg_devices_cover_roles(self):
        for ds in m.DATASETS:
            self.assertEqual(set(m.PREREG_CIRCUIT_DEVICES[ds]), set(m.CIRCUIT_ROLES))


RT_CARD = """.param sky130_fd_pr__nfet_01v8__toxe_slope_spectre = 0.0
.subckt  sky130_fd_pr__special_nfet_01v8 d g s b
xsky130_fd_pr__nfet_01v8 d g s b sky130_fd_pr__nfet_01v8 l = {l} w = {w}
.ends
.subckt  sky130_fd_pr__nfet_01v8 d g s b
+
.param  l = 1 w = 1 nf = 1.0
msky130_fd_pr__nfet_01v8 d g s b sky130_fd_pr__nfet_01v8__model.44 l = {l} w = {w} nf = {nf}
.model sky130_fd_pr__nfet_01v8__model.44 nmos
+ lmin = 1.5e-07
+ alpha0 = -1e-06
+ lalpha0 = 2e-07
+ igcmod = 0.0
+ vth0 = {0.4 + sky130_fd_pr__nfet_01v8__vth0_diff}
.ends sky130_fd_pr__nfet_01v8
* Well Proximity Effect Parameters
"""


class TestReplicaRT(unittest.TestCase):
    def test_replica_text(self):
        r = m.replica_rt(RT_CARD, "nmos")
        self.assertIn(".subckt  sky130_fd_pr__nfet_01v8_ch d g s b", r)
        self.assertIn(".ends sky130_fd_pr__nfet_01v8_ch", r)
        self.assertIn("sky130_fd_pr__nfet_01v8__model_ch.44 l = {l}", r)
        self.assertIn(".model sky130_fd_pr__nfet_01v8__model_ch.44 nmos", r)
        self.assertNotIn("special_nfet", r)
        self.assertNotIn("toxe_slope_spectre", r)
        self.assertIn("+ alpha0 = 0", r)
        self.assertIn("+ lalpha0 = 0", r)
        self.assertIn("+ xpart = -1", r)
        self.assertIn("vth0 = {0.4 + sky130_fd_pr__nfet_01v8__vth0_diff}", r)
        self.assertTrue(m.replica_values_ok(r, "nistrt", "nmos"))

    def test_replica_check_detects_missing(self):
        r = m.replica_rt(RT_CARD, "nmos").replace("+ agidl = 0\n", "")
        self.assertFalse(m.replica_values_ok(r, "nistrt", "nmos"))

    def test_two_bins_rejected(self):
        two = RT_CARD.replace(".ends sky130_fd_pr__nfet_01v8\n",
                              ".model sky130_fd_pr__nfet_01v8__model.45 nmos\n+ lmin = 1\n.ends sky130_fd_pr__nfet_01v8\n")
        with self.assertRaises(ValueError):
            m.replica_rt(two, "nmos")


class TestNetlist(unittest.TestCase):
    def test_instance_modes(self):
        d = Dev("nmos", 1.65, 0.18)
        c = m.instance("m1", d, "nd", "ng", "0", "0", "card", "nist4")
        self.assertTrue(c.startswith("Xm1 nd ng 0 0 sky130_fd_pr__nfet_01v8_lvt l=0.18u w=1.65u"))
        self.assertEqual(c.count("\n"), 1)
        r = m.instance("m1", d, "nd", "ng", "0", "0", "card_rep", "nist4")
        self.assertIn("Xm1r m1_ri m1_rg m1_rs m1_rb sky130_fd_pr__nfet_01v8_lvt_ch l=0.18u", r)
        self.assertIn("Em1d m1_rx 0 nd 0 1", r)
        self.assertIn("Gm1c m1_ri m1_rb m1_ri m1_rb -9.9999999999999998e-13", r)
        self.assertTrue(r.endswith("Vm1R m1_rx m1_ri DC 0 AC 0\n"))
        e = m.instance("m1", Dev("pmos", 5, 2), "nd", "ng", "0", "0", "elem", "nistrt")
        self.assertIn("sky130_fd_pr__pfet_01v8_ch l=2u", e)
        self.assertTrue(e.endswith("Nm1 nd ng 0 m1_rx m1_ri corr2p\n"))
        with self.assertRaises(ValueError):
            m.instance("m1", d, "nd", "ng", "0", "0", "x", "nist4")

    def test_deck(self):
        t = m.deck("t", "nistrt", "/tmp/lib.spice", "R1 a 0 1\n", ["op"], "/tmp/a.osdi")
        self.assertIn(".options temp=30.0000", t)
        self.assertIn('.lib "/tmp/lib.spice" tt_rt', t)
        self.assertIn("pre_osdi /tmp/a.osdi", t)
        self.assertIn(".model corr2n", t)
        t4 = m.deck("t", "nist4", "/tmp/lib.spice", "R1 a 0 1\n", ["op"], None)
        self.assertIn(".options temp=-196.1500", t4)
        self.assertNotIn("pre_osdi", t4)
        with self.assertRaises(ValueError):
            m.deck("t", "nist4", "/tmp/l", "", [], None, options="gmin=1e-15")

    def test_circuit_bodies(self):
        n, p = Dev("nmos", 1.65, 0.18), Dev("pmos", 1.0, 0.35)
        b = m.ring_body(n, p, "elem", "nist4")
        self.assertEqual(b.count("Ns"), 10)
        self.assertIn("VDD vdd 0 DC 1.8", b)
        mb = m.mirror_body(p, "card", 1e-5, "nist4")
        self.assertIn("IREF nref 0 DC 1e-05", mb)
        with self.assertRaises(ValueError):
            m.ring_body(n, p, "card", "nist4", stages=4)

    def test_nonchannel(self):
        c = m.nonchannel_control(np.array([[0.5, 0.9]]), "nistrt", "pmos")
        self.assertIn("print @m.xm1r.msky130_fd_pr__pfet_01v8[isub]", c)
        txt = "@m.xm1r.msky130_fd_pr__pfet_01v8[isub] = 0.000000e+00\n@m.xm1r.msky130_fd_pr__pfet_01v8[igb] = -2e-20\n"
        self.assertEqual(m.parse_nonchannel(txt), {"isub": 0.0, "igb": 2e-20})


GOOD = {"q1_tight_viol": 0, "q1_bad": 0, "q1_vds0_max": 1e-17, "g_text_max": 1e-16, "g_sym_text_max": 1e-16,
        "swap_asym_max": 1e-5, "q2_ac_max": 1e-5, "cap_max": 1e-14, "q2_gm_sign": 0, "q3_fail": 0,
        "q1_default_max": 1e-4, "q2_gds_sign": 0, "runtime_ratio_max": 1.5}


class TestDecisions(unittest.TestCase):
    def test_part_a_labels(self):
        self.assertEqual(m.decide_part_a(GOOD)["deploy"], "deployable (simulator-level)")
        self.assertEqual(m.decide_part_a({**GOOD, "q2_gds_sign": 3})["deploy"], "deployable with restrictions")
        self.assertEqual(m.decide_part_a({**GOOD, "q1_vds0_max": 2e-12})["deploy"], "not deployable")
        self.assertEqual(m.decide_part_a({**GOOD, "q3_fail": 1, "q2_gds_sign": 2})["deploy"], "not deployable")
        self.assertEqual(m.decide_part_a({**GOOD, "q2_gm_sign": 1})["deploy"], "not deployable")

    def test_validity_a(self):
        ok = {"va1_max_diff": 0.0, "m2_failed": 0, "m2_n": 53, "unit_tests_pass": True, "n_rows": 53, "n_q3": 24,
              "dev_ok": True}
        self.assertTrue(m.validity_part_a(ok, "nist4")["valid"])
        self.assertFalse(m.validity_part_a(ok, "nistrt")["valid"])
        self.assertFalse(m.validity_part_a({**ok, "va1_max_diff": 1e-8}, "nist4")["valid"])
        self.assertFalse(m.validity_part_a({**ok, "n_q3": 23}, "nist4")["valid"])

    def test_part_b(self):
        r = m.part_b_report(0.01, -0.029)
        self.assertTrue(r["sign_differs"])
        self.assertEqual(r["decision_state_C"], "C0_refit")
        self.assertEqual(r["decision_state_L"], "apply")
        self.assertFalse(m.part_b_report(-0.1, -0.05)["sign_differs"])
        self.assertFalse(m.validity_part_b({"vb1_max_diff": 0.0, "dev_ok": True, "n_finite": 44})["valid"])
        self.assertTrue(m.validity_part_b({"vb1_max_diff": 0.0, "dev_ok": True, "n_finite": 45})["valid"])

    def test_part_c(self):
        ok = {"vc1_c0_max": 0.0, "vc1_mb_max": 0.0, "vc1_n": 18, "unit_tests_pass": True,
              "rows": {"C0_refit": 18, "M_bias_v1": 18, "selected": 18}, "inner_complete": True, "grids_ok": True,
              "dev_ok": True}
        self.assertTrue(m.validity_part_c(ok)["valid"])
        self.assertFalse(m.validity_part_c({**ok, "vc1_mb_max": 2e-9})["valid"])
        self.assertFalse(m.validity_part_c({**ok, "rows": {"selected": 17}})["valid"])


class TestVds0(unittest.TestCase):
    def arrays(self, nan_tight=False):
        a = {}
        for tol in ("default", "r4", "tight"):
            el = np.zeros((4, 5))
            el[:, 2] = [1e-18, -3e-13, 2e-19, 5e-14]
            if nan_tight and tol == "tight":
                el[:, 2] = np.nan
            a[f"grid_fwd_{tol}_elem"] = el
            a[f"grid_fwd_{tol}_card_rep"] = np.full((4, 5), 1e-19)
        return a

    def test_values_and_rule(self):
        v = m.vds0_values(self.arrays(), 2, [0.0, 0.1, 0.2, 0.3])
        self.assertEqual(v["tight"]["elem_max"], 3e-13)
        self.assertEqual(v["tight"]["vgs_abs_at_max"], 0.1)
        self.assertEqual(v["tight"]["n_nonfinite"], 0)
        s = m.summarize_vds0({"a": v})
        self.assertEqual(m.vds0_rule_value(s), 3e-13)

    def test_all_nan_column(self):
        v = m.vds0_values(self.arrays(nan_tight=True), 2, [0.0, 0.1, 0.2, 0.3])
        self.assertTrue(math.isnan(v["tight"]["elem_max"]))
        self.assertEqual(v["tight"]["n_nonfinite"], 4)
        ok = m.vds0_values(self.arrays(), 2, [0.0, 0.1, 0.2, 0.3])
        s = m.summarize_vds0({"a": v, "b": ok})
        self.assertEqual(s["tight"]["devices_nonfinite"], ["a"])
        self.assertEqual(m.vds0_rule_value(s), float("inf"))


if __name__ == "__main__":
    unittest.main()
