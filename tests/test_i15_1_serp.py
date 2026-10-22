"""I15-1: serpentine Q1 grid sweep (control lines, row reordering) and one
ngspice check that the serpentine card-only grid equals the nested grid
within the tight solver tolerance with every grid value present once."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class _Dev:
    def __init__(self, dev_type):
        self.dev_type = dev_type


class SerpControlTest(unittest.TestCase):
    def test_lines(self):
        from cryoml.explore.i11_1_deploy import GRID_VDS, GRID_VGS
        from cryoml.explore.i15_1_serp import grid_control_serp
        for dt, sg in (("nmos", 1.0), ("pmos", -1.0)):
            lines = grid_control_serp(_Dev(dt), Path("/x/o.dat"), False, True)
            self.assertEqual(lines[0], "set appendwrite")
            dcs = [ln for ln in lines if ln.startswith("dc VD ")]
            alts = [ln for ln in lines if ln.startswith("alter VG dc = ")]
            wrs = [ln for ln in lines if ln.startswith("wrdata ")]
            self.assertEqual(len(dcs), len(GRID_VGS))
            self.assertEqual(len(alts), len(GRID_VGS))
            self.assertEqual(len(wrs), len(GRID_VGS))
            for k, (a, d) in enumerate(zip(alts, dcs)):
                self.assertAlmostEqual(float(a.split("=")[1]), sg * GRID_VGS[k], places=9)
                _, _, lo, hi, st = d.split()
                if k % 2 == 0:
                    self.assertEqual((float(lo), float(hi)), (GRID_VDS[0], GRID_VDS[-1]))
                    self.assertGreater(float(st), 0)
                else:
                    self.assertEqual((float(lo), float(hi)), (GRID_VDS[-1], GRID_VDS[0]))
                    self.assertLess(float(st), 0)
            self.assertTrue(all(w.endswith("-i(VD) i(Vm1R)") for w in wrs))
            lines = grid_control_serp(_Dev(dt), Path("/x/o.dat"), True, False)
            self.assertTrue(all(w.endswith("-i(VD)") for w in lines if w.startswith("wrdata ")))

    def test_grid_rows(self):
        from cryoml.explore.i11_1_deploy import GRID_VDS, GRID_VGS, grid_from_rows
        from cryoml.explore.i15_1_serp import grid_rows
        nv, nd = len(GRID_VGS), len(GRID_VDS)
        val = np.arange(nv * nd, dtype=float).reshape(nv, nd)       # nested order values
        rows = []
        for k in range(nv):
            idx = range(nd) if k % 2 == 0 else range(nd - 1, -1, -1)
            rows += [[GRID_VDS[j], val[k, j], GRID_VDS[j], -val[k, j]] for j in idx]
        rows = np.array(rows)
        out = grid_rows(rows)
        np.testing.assert_array_equal(grid_from_rows(out[:, :2]), val)
        np.testing.assert_array_equal(grid_from_rows(out[:, 2:4]), -val)
        self.assertEqual(grid_rows(rows[:-1]).shape[0], 0)              # missing row
        bad = rows.copy()
        bad[3, 0] = bad[4, 0]                                           # duplicated V(d) value
        self.assertEqual(grid_rows(bad).shape[0], 0)


class SerpNgspiceTest(unittest.TestCase):
    def test_card_grid_serp_equals_nested(self):
        from cryoml.spice_pdk import ensure_pdk77k
        from cryoml.explore import i11_1_deploy as m1
        from cryoml.explore import i15_1_serp as m
        if not Path(m1.NG_OSDI).exists() or ensure_pdk77k():
            raise unittest.SkipTest("ngspice OSDI build or the 77 K PDK is unavailable")
        from cryoml.card_eval import device_by_tag
        from cryoml.cold_start import device_bins
        tag = "nmos_L1_W1p6"
        d = device_by_tag(tag)
        dev = m1.Dev(tag, d.dev_type, float(d.L_um), float(d.W_um), int(device_bins()[tag]))
        with tempfile.TemporaryDirectory() as td_s:
            td = Path(td_s)
            lib = m.patched_library(td, {dev.dev_type: [(dev.bin, None)]})
            grids = {}
            for order in ("nested", "serp"):
                f = td / f"{order}.dat"
                ctrl = (m.grid_control_serp(dev, f, False, True) if order == "serp"
                        else m.grid_control(dev, f, False, True))
                r = m1.run_deck(m.deck(order, lib, m.device_body(dev, "card_rep"), ctrl, None,
                                       options=m1.TIGHT), td, f"{order}.sp", timeout_s=300)
                self.assertEqual(r["msgs"], [])
                rows = m1.read_wrdata(f, 4)
                self.assertEqual(rows.shape[0], len(m1.GRID_VGS) * len(m1.GRID_VDS))
                if order == "serp":
                    rows = m.grid_rows(rows)
                    self.assertEqual(rows.shape[0], len(m1.GRID_VGS) * len(m1.GRID_VDS))
                grids[order] = (m1.grid_from_rows(rows[:, :2]), m1.grid_from_rows(rows[:, 2:4]))
        for i in (0, 1):
            a, b = grids["serp"][i], grids["nested"][i]
            ratio = np.abs(a - b) / (1e-6 * np.abs(b) + 1e-13)
            self.assertLess(float(np.max(ratio)), 10.0)


if __name__ == "__main__":
    unittest.main()
