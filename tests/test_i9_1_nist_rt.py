"""I9-1 card-text helpers and the A1 rule in the I9-1 driver (no downloaded data needed;
the card test is skipped when the open_pdks files are not on disk)."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from cryoml.explore import i9_1_nist_rt as m

ROOT = Path(__file__).resolve().parents[1]

BLOCK = (".model sky130_fd_pr__nfet_01v8__model.7 nmos\n"
         "* comment line\n"
         "+ lmin = 1.5e-07 lmax = 1.8e-07 wmin = 1.0e-06 wmax = 1.26e-06\n"
         "+ vth0 = {0.51+MC_MM_SWITCH*AGAUSS(0,1.0,1)*(x/sqrt(l*w*mult))} lvth0 = -3.6e-09 "
         "wvth0 = -4.6e-08 pvth0 = 3.7e-13\n"
         "+ u0 = 0.0318\n"
         "+ k2 = 'a + b' k1 = 0.54\n")


def _driver():
    spec = importlib.util.spec_from_file_location(
        "i9_1_driver", ROOT / "scripts" / "research" / "explore" / "i9_1_nist_rt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CardText(unittest.TestCase):
    def test_parse_assignments(self):
        p = m.parse_assignments(" vth0 = {a+b*(c/d)} lvth0 = -3.6e-09 k2 = 'x + y'")
        self.assertEqual(p, [("vth0", "{a+b*(c/d)}"), ("lvth0", "-3.6e-09"), ("k2", "'x + y'")])
        self.assertEqual(m.parse_assignments(" "), [])
        self.assertIsNone(m.parse_assignments(" vth0 = {a"))          # unbalanced brace
        self.assertIsNone(m.parse_assignments(" 0.5 1.0"))            # not name = value
        self.assertIsNone(m.parse_assignments(" a = b=c"))

    def test_normalize_and_patch(self):
        norm, kept = m.normalize_block(BLOCK)
        self.assertEqual(kept, 0)
        self.assertIn("+ lvth0 = -3.6e-09\n", norm)
        self.assertIn("+ wmax = 1.26e-06\n", norm)
        self.assertEqual(m.block_geometry(norm), {"lmin": 1.5e-07, "lmax": 1.8e-07,
                                                  "wmin": 1.0e-06, "wmax": 1.26e-06})
        out = m.patch_block(norm, {"vth0": 0.7, "vfb": -0.9})
        self.assertIn("+ vth0 = 0.7\n", out)
        self.assertIn("+ lvth0 = -3.6e-09\n", out)                     # binning terms kept
        self.assertIn("+ pvth0 = 3.7e-13\n", out)
        self.assertEqual(out.split("\n")[1], "+ vfb = -0.9")           # added after .model
        self.assertNotIn("MC_MM_SWITCH", out)

    def test_force_bin(self):
        pre = ("msky130_fd_pr__nfet_01v8 d g s b sky130_fd_pr__nfet_01v8__model l = {l} w = {w}\n"
               "xsky130_fd_pr__special_nfet_01v8 d g s b sky130_fd_pr__special_nfet_01v8__model l = {l}\n")
        out = m.force_bin(pre, "nmos", 42)
        self.assertIn("sky130_fd_pr__nfet_01v8__model.42 l = {l}", out)
        self.assertIn("sky130_fd_pr__special_nfet_01v8__model l = {l}", out)

    def test_min_swing(self):
        vg = np.linspace(0, 1.0, 101)
        i = 1e-12 * 10 ** (vg / 0.06)                                  # 60 mV/decade
        self.assertAlmostEqual(m.min_swing_mV_dec(vg, i), 60.0, places=6)
        self.assertTrue(np.isnan(m.min_swing_mV_dec(vg, np.full(101, 1e-13))))

    @unittest.skipUnless(m.card_sources()["nmos_pm3"].exists(), "open_pdks sky130A not on disk")
    def test_card_parts(self):
        for pol, n in (("nmos", 180), ("pmos", 108)):
            p = m.card_parts(pol)
            self.assertEqual(len(p["blocks"]), n)
            self.assertEqual(p["n_unparsed"], 0)
            self.assertTrue(p["suffix"].startswith(".ends"))
            text = m.card_text(pol, 3, {"u0": 0.05})
            self.assertEqual(text.count(".model "), 1)
            self.assertIn(f"{m.MODEL[pol]}.3 l", text)
        self.assertEqual(m.bin_candidates("nmos", 0.36, 0.15), [179])


class DriverA1(unittest.TestCase):
    def test_s1_rows_exclude_nonfinite(self):
        from cryoml.explore.e1_correction import CurveRec, DeviceRec
        d = _driver()

        def rec(tag, fam, bad=False):
            vg = np.linspace(0, 1.8, 40)
            r = DeviceRec(tag, fam, 1.0, 1.0, 1.8, "nist4k")
            for k, vd in enumerate((0.1, 1.8)):
                tgt = 1e-5 * (1 + vg) * (1.0 + 0.1 * k)
                sim = tgt * (1.2 + 0.05 * k)
                if bad and k == 0:
                    sim = np.full_like(sim, np.nan)
                r.curves.append(CurveRec(f"idvg@d{k}", "idvg", vg, np.full_like(vg, vd), sim, tgt,
                                         0, float(np.mean(tgt)), True))
            return r
        tags = {"nfet_a": "nmos", "nfet_b": "nmos", "pfet_c": "pmos", "pfet_d": "pmos"}
        recs = [rec(t, f, bad=(t == "nfet_b")) for t, f in tags.items()]
        rows = d.s1_rows(recs, d.family_of(list(tags)))
        self.assertEqual(sorted(rows), ["nfet_a", "pfet_c", "pfet_d"])
        self.assertTrue(all(r["n_train"] == 2 for r in rows.values()))


if __name__ == "__main__":
    unittest.main()
