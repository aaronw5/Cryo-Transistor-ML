import unittest
from types import SimpleNamespace

import numpy as np

from cryoml.explore import i18_2_partb as m


class PartBTests(unittest.TestCase):
    def test_card_and_replica_text(self):
        card = m.card_text("nmos", {"a2": 2.0, "vth0": 0.3})
        self.assertIn("+ a2 = 1\n", card)
        self.assertIn(".model i6card nmos", card)
        rep = m.card_text("nmos", {"a2": 2.0}, replica=True)
        self.assertTrue(m.replica_ok(rep))
        self.assertIn(".model i6card_ch nmos", rep)
        self.assertIn("+ igcmod = 0\n", rep)
        self.assertIn("+ xpart = -1\n", rep)
        self.assertFalse(m.replica_ok(card))

    def test_instance_modes(self):
        dev = SimpleNamespace(dev_type="pmos", W_um=1.0, L_um=0.04)
        a = m.instance("m1", dev, "nd", "ng", "0", "0", "card")
        self.assertEqual(a, "Mm1 nd ng 0 0 i6card w=1u l=0.04u nf=1\n")
        b = m.instance("m1", dev, "nd", "ng", "0", "0", "card_rep")
        self.assertIn("Mm1r m1_ri m1_rg m1_rs m1_rb i6card_ch", b)
        self.assertIn("Vm1R m1_rx m1_ri DC 0 AC 0", b)
        self.assertIn("Gm1c m1_ri m1_rb m1_ri m1_rb -9.9999999999999998e-13", b)
        c = m.instance("m1", dev, "nd", "ng", "0", "0", "elem")
        self.assertIn("Nm1 nd ng 0 m1_rx m1_ri corr2p", c)
        self.assertNotIn("Vm1R", c)

    def test_grid_rows_serpentine(self):
        nv, nd = len(m.GRID_VGS), len(m.GRID_VDS)
        self.assertEqual((nv, nd), (45, 89))
        rows = []
        for k in range(nv):
            vds = m.GRID_VDS if k % 2 == 0 else m.GRID_VDS[::-1]
            rows += [[v, 10 * k + v] for v in vds]
        g = m.grid_rows(np.array(rows))
        self.assertEqual(g.shape, (nv, nd, 1))
        self.assertTrue(np.allclose(g[3, :, 0], 30 + m.GRID_VDS))
        self.assertTrue(np.all(np.isnan(m.grid_rows(np.array(rows[:-1])))))

    def test_op_points(self):
        p = m.op_points("pmos")
        self.assertEqual(p.shape, ((3 + 5) * 45, 2))
        self.assertTrue(np.all(p <= 0))

    def test_parse_nonchannel(self):
        txt = "@mm1r[isub] = 0.000000e+00\n@mm1r[igd] = -2.0e-15\n@mm1r[isub] = 1e-20\n"
        self.assertEqual(m.parse_nonchannel(txt), {"isub": 1e-20, "igd": 2e-15})

    def test_decide(self):
        vals = {"q1_tight_viol": 0, "q1_bad": 0, "q1_vds0_max": 1e-17, "g_text_max": 1e-16,
                "g_sym_text_max": 1e-16, "swap_asym_max": 1e-5, "q2_ac_max": 1e-5, "cap_max": 1e-13,
                "q2_gm_sign": 0, "q1_default_max": 2e-3, "q2_gds_sign": 1, "runtime_ratio_max": 1.5}
        d = m.decide_partb(vals)
        self.assertEqual(d["deploy"], "deployable with restrictions")
        self.assertIn("Q3 not run", d["label_full"])
        self.assertEqual(m.decide_partb({**vals, "q1_vds0_max": 2e-12})["deploy"], "not deployable")
        self.assertFalse(m.decide_validity_b({"m2_failed": 1, "m2_n": 45, "unit_tests_pass": True,
                                              "n_rows": 45})["valid"])


if __name__ == "__main__":
    unittest.main()
