"""I3-2 follow-up: OSDI text change, corrected netlist, difference statistics, F2 rules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.explore.e1_correction import CurveRec, DeviceRec  # noqa: E402
from cryoml.explore.i3_2_bias import (BiasCorrection, eval_verilog_a_g,  # noqa: E402
                                      to_verilog_a_bias)
from cryoml.explore.i3_2_followup import (F2_GROUPS, F2_REAL_VARIANTS,  # noqa: E402
                                          RS_DECL, SENSE_EXPORTED, SENSE_OSDI,
                                          convergence_messages, insert_correction,
                                          max_rel_diff, network_dims, osdi_va_text,
                                          real_variant_fails, scored_masks,
                                          tier_b_pdk_passes)


def tiny_corr(seed: int = 0) -> BiasCorrection:
    rng = np.random.default_rng(seed)
    nets = [[(rng.normal(size=(3, 4)), rng.normal(size=4)),
             (rng.normal(size=(4, 4)), rng.normal(size=4)),
             (rng.normal(size=(4, 1)), rng.normal(size=1))] for _ in range(2)]
    return BiasCorrection("M_bias", np.array([0.0, 0.6, 0.5]), np.array([1.0, 0.2, 0.3]),
                          nets=nets, cols=(0, 1, 2))


DECK = ("* corrected-repository batched DC sweeps\n.options scale=1.0\n"
        ".options temp=-196.1500\n.lib \"/x/lib77k.spice\" tt_77k\n"
        "Xm1 nd ng 0 0 sky130_fd_pr__pfet_01v8_lvt l=2u w=5u mult=1 m=1\n"
        "VG ng 0 DC 0\nVD nd 0 DC 0\n.control\nset filetype=ascii\n"
        "alter VD dc = -0.01\ndc VG 0 -1.85 -0.05\nwrdata /x/raw_0.dat -i(VD)\n"
        "quit\n.endc\n.end\n")


class TestOsdiText(unittest.TestCase):
    def test_one_line_changed_and_g_unchanged(self):
        corr = tiny_corr()
        text = to_verilog_a_bias(corr, vdd=1.85)
        new = osdi_va_text(text)
        self.assertNotIn(SENSE_EXPORTED, new)
        self.assertEqual(new.count(SENSE_OSDI), 1)
        self.assertEqual(new.count(RS_DECL), 1)
        diff = set(new.splitlines()) ^ set(text.splitlines())
        self.assertEqual(diff, {SENSE_EXPORTED, SENSE_OSDI, RS_DECL})
        for pmos, vgs, vds in ((0, 0.9, 1.2), (1, -1.1, -0.3), (0, 0.0, 0.0)):
            self.assertEqual(eval_verilog_a_g(new, pmos, vgs, vds),
                             eval_verilog_a_g(text, pmos, vgs, vds))

    def test_rejects_unexpected_text(self):
        with self.assertRaises(ValueError):
            osdi_va_text("module x; endmodule")


class TestDeck(unittest.TestCase):
    def test_insert_correction(self):
        deck = insert_correction(DECK, "/o/m.osdi", pmos=True, vdd=1.85)
        lines = deck.splitlines()
        self.assertIn("Xm1 ndi ng 0 0 sky130_fd_pr__pfet_01v8_lvt l=2u w=5u mult=1 m=1", lines)
        self.assertNotIn("Xm1 nd ng 0 0 sky130_fd_pr__pfet_01v8_lvt l=2u w=5u mult=1 m=1",
                         lines)
        i = lines.index("NCORR nd ng 0 ndi i32corr")
        self.assertEqual(lines[i + 1], ".model i32corr i3_2_bias_corr pmos=1 VDD=1.85 RS=0")
        c = lines.index(".control")
        self.assertEqual(lines[c + 1], "pre_osdi /o/m.osdi")
        # sources and sweep lines unchanged
        for ln in ("VG ng 0 DC 0", "VD nd 0 DC 0", "wrdata /x/raw_0.dat -i(VD)"):
            self.assertIn(ln, lines)
        self.assertEqual(len(lines), len(DECK.splitlines()) + 3)
        nm = insert_correction(DECK, "/o/m.osdi", pmos=False, vdd=1.85)
        self.assertIn("pmos=0 ", nm)

    def test_rejects_two_instances(self):
        with self.assertRaises(ValueError):
            insert_correction(DECK + "Xm1 nd ng 0 0 x\n", "/o", True, 1.85)


class TestStats(unittest.TestCase):
    def test_max_rel_diff(self):
        b = [np.array([1.0, -2.0, 1e-14]), np.array([4.0])]
        a = [np.array([1.001, -2.0, 2e-14]), np.array([4.0])]
        self.assertAlmostEqual(max_rel_diff(a, b), 1.0)            # the 1e-14 point
        self.assertAlmostEqual(max_rel_diff(a, b, floor=1e-12), 1e-3)
        masks = [np.array([False, True, False]), np.array([True])]
        self.assertEqual(max_rel_diff(a, b, masks), 0.0)
        a_nan = [np.array([np.nan, -2.0, 2e-14]), np.array([4.0])]
        self.assertEqual(max_rel_diff(a_nan, b, masks), 0.0)        # NaN outside the mask
        self.assertEqual(max_rel_diff(a_nan, b), float("inf"))
        before = [x.copy() for x in masks]
        max_rel_diff(a, b, masks, floor=1.0)                        # must not modify masks
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(masks, before)))

    def test_scored_masks(self):
        rec = DeviceRec("t", "nmos", 1.0, 1.0, 1.85, "sky130")
        z = np.zeros(5)
        rec.curves = [CurveRec("idvg@0.1", "idvg", z, z, z, z, 2, 1.0, True),
                      CurveRec("idvd@1.85", "idvd", z, z, z, z, 0, 1.0, False),
                      CurveRec("idvd@0.37", "idvd", z, z, z, z, 1, 0.0, True)]
        m = scored_masks(rec)
        self.assertEqual(m[0].tolist(), [False, False, True, True, True])
        self.assertFalse(m[1].any())
        self.assertFalse(m[2].any())

    def test_convergence_messages(self):
        text = ("Circuit: x\nWarning: singular matrix:  check node nd\n"
                "doAnalyses: TRAN:  Timestep too small\nNo. of Data Rows : 7\n")
        self.assertEqual(len(convergence_messages(text)), 2)
        self.assertEqual(convergence_messages("No. of Data Rows : 37\nngspice-46 done\n"), [])


class TestF2Rules(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(set(F2_REAL_VARIANTS), {v for vs in F2_GROUPS.values() for v in vs})
        seeds = [F2_REAL_VARIANTS[v]["hyper"]["seeds"] for v in F2_GROUPS["a"]]
        flat = [s for ss in seeds for s in ss]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertFalse(set(flat) & {0, 1, 2, 3, 4})
        self.assertEqual(network_dims(3, F2_REAL_VARIANTS["b_width8"]["hyper"]), [3, 8, 8, 1])
        self.assertEqual(network_dims(3, F2_REAL_VARIANTS["b_width32"]["hyper"]), [3, 32, 32, 1])
        self.assertEqual(network_dims(3, F2_REAL_VARIANTS["b_depth1"]["hyper"]), [3, 16, 1])
        self.assertEqual(network_dims(3, {}), [3, 16, 16, 1])

    def test_failure_rules(self):
        self.assertFalse(real_variant_fails({"ci_high": 0.95}, 0))
        self.assertTrue(real_variant_fails({"ci_high": 1.0}, 0))
        self.assertTrue(real_variant_fails({"ci_high": 0.9}, 1))
        self.assertTrue(real_variant_fails({"ci_high": float("nan")}, 0))
        self.assertTrue(tier_b_pdk_passes({"ratio": 0.90}, 0))
        self.assertFalse(tier_b_pdk_passes({"ratio": 0.91}, 0))
        self.assertFalse(tier_b_pdk_passes({"ratio": 0.5}, 2))


if __name__ == "__main__":
    unittest.main()
