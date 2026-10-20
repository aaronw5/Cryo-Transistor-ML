"""I5-1 IHP measured data: MDM parsing, curve keys, inclusion, split, box,
scoring, records, correction in the loop (synthetic data; the simulator
tests run only when the PDK model files and ngspice-46 are present)."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.explore import i5_1_ihp_real as m  # noqa: E402
from cryoml.explore.e1_correction import Correction, score_device  # noqa: E402
from cryoml.pdk2 import metric as pdk2_metric  # noqa: E402

MDM = """! VERSION = 6.00
BEGIN_HEADER
 ICCAP_INPUTS
  vd         V  D GROUND SMU1 0.1 LIN        1    0          1.35       4   0.45
  vg         V  G GROUND SMU2 0.001 LIST       2 2 0.3 0.6
  vb         V  B GROUND SMU4 0.1 CON        0
  vs         V  S GROUND SMU3 0.1 CON        0
 ICCAP_OUTPUTS
  id         I  D GROUND SMU1 B
  ig         I  G GROUND SMU2 B
  ib         I  B GROUND SMU4 B
  is         I  S GROUND SMU3 B
 ICCAP_VALUES
  TYPE "1"
  TEMP " 27.0000 "
  MAIN.W "10.00u"
  MAIN.L "130.0n"
END_HEADER

BEGIN_DB
 ICCAP_VAR vg         0.3
 ICCAP_VAR vb         0
 ICCAP_VAR vs         0

 #vd                    id                    ig                    ib                    is
  0                     1e-12                 0                     0                     -1e-12
  0.45                  2e-06                 0                     0                     -2e-06
  0.9                   3e-06                 0                     0                     -3e-06
  1.35                  3.5e-06               0                     0                     -3.5e-06
END_DB

BEGIN_DB
 ICCAP_VAR vg         0.6
 ICCAP_VAR vb         0
 ICCAP_VAR vs         0

 #vd                    id                    ig                    ib                    is
  0                     0                     0                     0                     0
  0.45                  2e-05                 0                     0                     -2e-05
  0.9                   3e-05                 0                     0                     -3e-05
  1.35                  3.5e-05               0                     0                     -3.5e-05
END_DB
"""


def _curve(setup, kind, fixed, vb, n=20, scale=1e-5, sign=1.0):
    sweep = sign * np.linspace(0, 1.2, n)
    i = sign * scale * np.linspace(0, 1, n) ** 2
    return m.MeasCurve(setup, kind, sign * fixed, vb, sweep, i, 0 * i, 0 * i, -i)


def _device(tag="lv_nmos_W1_L1", dev_type="nmos", scale=1e-5, sign=1.0):
    dev = m.MeasDevice(tag, "lv", dev_type, 1.0, 1.0, "W1u0_L1u0_S1_1",
                       {"AD": 1e-12, "AS": 1e-12, "PD": 1e-6, "PS": 1e-6, "NF": 1.0, "M": 1.0},
                       {300: 27.0})
    curves = []
    for k, vd in enumerate((0.05, 0.6, 1.2)):
        for vb in (0.0, -0.3):
            curves.append(_curve("dc_idvg", "idvg", vd, vb, scale=scale * (k + 1), sign=sign))
    for k, vg in enumerate((1.2, 0.5, 0.7, 0.9, 1.1)):     # deliberately unsorted
        curves.append(_curve("dc_idvd", "idvd", vg, 0.0, scale=scale * vg, sign=sign))
    for vg in (0.6, 0.2, 0.4):
        curves.append(_curve("dc_idvd_vth", "idvd", vg, 0.0, scale=scale * vg * 0.01, sign=sign))
        curves.append(_curve("dc_idvd_vth_vbmin", "idvd", vg, -1.2, scale=scale, sign=sign))
    m.assign_keys(curves)
    dev.curves[300] = curves
    return dev


class TestParse(unittest.TestCase):
    def test_parse_mdm(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.mdm"
            p.write_text(MDM)
            f = m.parse_mdm(p)
        self.assertEqual(f.inputs["vd"].mode, "LIN")
        self.assertEqual(f.inputs["vd"].values, [0.0, 1.35, 4, 0.45])
        self.assertEqual(f.inputs["vg"].values, [0.3, 0.6])
        self.assertEqual(f.inputs["vb"].values, [0.0])
        self.assertAlmostEqual(f.inputs["vg"].compliance, 1e-3)
        self.assertEqual(f.values["MAIN.L"], "130.0n")
        self.assertEqual(len(f.blocks), 2)
        curves = m.file_curves(f, "dc_idvd")
        self.assertEqual([c.fixed for c in curves], [0.3, 0.6])
        self.assertTrue(all(c.vb == 0.0 and c.kind == "idvd" for c in curves))
        np.testing.assert_allclose(curves[1].id, [0, 2e-5, 3e-5, 3.5e-5])
        np.testing.assert_allclose(curves[1].vd, [0, 0.45, 0.9, 1.35])
        np.testing.assert_allclose(curves[1].vg, 0.6)

    def test_geometry_and_si(self):
        self.assertEqual(m.dut_geometry("W02u0_L1u2_S563_3"), (2.0, 1.2))
        self.assertEqual(m.dut_geometry("W0u15_L0u13_S546_3"), (0.15, 0.13))
        self.assertEqual(m.dut_geometry("W10u0_L10u0_S541_5"), (10.0, 10.0))
        self.assertAlmostEqual(m.si_value("130.0n"), 1.3e-7)
        self.assertAlmostEqual(m.si_value("3.400p"), 3.4e-12)
        self.assertAlmostEqual(m.si_value("20.68u"), 2.068e-5)
        self.assertEqual(m.device_tag("hv", "pmos", 0.3, 0.35), "hv_pmos_W0p3_L0p35")


class TestCurves(unittest.TestCase):
    def test_keys_and_fit_curves(self):
        dev = _device()
        fc = m.fit_curves(dev, 300)
        self.assertEqual([c.key for c in fc],
                         ["idvg@d0", "idvg@d1", "idvg@d2", "idvd@g0", "idvd@g1", "idvd@g2",
                          "idvd@g3", "idvd@g4", "idvd@t0", "idvd@t1", "idvd@t2"])
        self.assertTrue(all(c.vb == 0.0 for c in fc))
        self.assertEqual([round(c.fixed, 3) for c in fc[3:8]], [0.5, 0.7, 0.9, 1.1, 1.2])
        self.assertEqual([round(c.fixed, 3) for c in fc[8:]], [0.2, 0.4, 0.6])
        # body-bias curves carry no key
        self.assertTrue(all(c.key == "" for c in dev.curves[300] if c.vb != 0.0))

    def test_pmos_keys_by_magnitude(self):
        dev = _device("lv_pmos_W1_L1", "pmos", sign=-1.0)
        fc = m.fit_curves(dev, 300)
        self.assertEqual([round(c.fixed, 3) for c in fc[:3]], [-0.05, -0.6, -1.2])

    def test_inclusion_matches_pdk2_metric(self):
        dev = _device()
        for c in m.fit_curves(dev, 300):
            start, _, den = pdk2_metric.curve_trim(c.kind, c.id)
            inc = m.inclusion(c)
            self.assertEqual(inc["start"], start)
            self.assertAlmostEqual(inc["den"], den)
            self.assertEqual(inc["included"], den > pdk2_metric.INCLUDE_MIN[c.kind])
        # the vth curves are scaled below the threshold, the others above
        incl = {c.key: m.inclusion(c)["included"] for c in m.fit_curves(dev, 300)}
        self.assertFalse(any(incl[f"idvd@t{k}"] for k in range(3)))
        self.assertTrue(all(incl[f"idvg@d{k}"] for k in range(3)))

    def test_aliases(self):
        c = _curve("dc_idvg", "idvg", 0.05, 0.0)
        np.testing.assert_array_equal(c.Vg, c.sweep)
        np.testing.assert_allclose(c.Vd, 0.05)
        np.testing.assert_array_equal(c.Id, c.id)


class TestScore(unittest.TestCase):
    def test_score_and_record(self):
        dev = _device()
        fc = m.fit_curves(dev, 300)
        sims = [c.id * 1.1 for c in fc]
        sc = m.score(fc, sims)
        # every included curve has RRMS 0.1 * rms / mean over the trimmed points
        vals = []
        for c in fc:
            inc = m.inclusion(c)
            if inc["included"]:
                t = c.id[inc["start"]:]
                vals.append(math.sqrt(np.mean((0.1 * t) ** 2)) / inc["den"])
        self.assertAlmostEqual(sc["rrms"], float(np.mean(vals)))
        rec = m.to_record(dev, 300, sims, {"vfbo": -0.9})
        self.assertEqual(rec.source, "ihp_meas")
        self.assertAlmostEqual(rec.vdd, 1.2)
        self.assertAlmostEqual(score_device(rec, [c.sim for c in rec.curves])["rrms"], sc["rrms"])
        bad = list(sims)
        bad[0] = np.full_like(bad[0], np.nan)
        self.assertTrue(math.isnan(m.score(fc, bad)["rrms"]))
        self.assertEqual(m.score(fc, bad)["failed"], ["idvg@d0"])

    def test_aggregate(self):
        a = m.aggregate({"lv_nmos_W1_L1": 0.1, "lv_nmos_W2_L1": 0.3, "hv_pmos_W1_L1": 0.2})
        self.assertAlmostEqual(a["all_device_mean"], 0.2)
        self.assertAlmostEqual(a["combined"], 0.2)
        a = m.aggregate({"lv_nmos_W1_L1": 0.1, "lv_pmos_W1_L1": float("nan")})
        self.assertEqual(a["n_failed"], 1)
        self.assertTrue(math.isnan(a["all_device_mean"]))


class TestSplit(unittest.TestCase):
    def _devices(self):
        out = []
        for pol in ("nmos", "pmos"):
            for W, L in ((0.15, 0.13), (0.15, 0.18), (0.15, 10), (0.3, 0.13), (0.3, 10),
                         (10, 0.13), (10, 0.18), (10, 10)):
                out.append(m.MeasDevice(m.device_tag("lv", pol, W, L), "lv", pol, W, L, "", {}))
        for pol, geoms in (("nmos", ((2, 3), (0.3, 0.5), (0.3, 10), (0.6, 1.2), (10, 0.5), (10, 10))),
                           ("pmos", ((2, 1.2), (0.3, 0.35), (0.3, 10), (0.6, 0.5), (10, 0.35), (10, 10)))):
            for W, L in geoms:
                out.append(m.MeasDevice(m.device_tag("hv", pol, W, L), "hv", pol, W, L, "", {}))
        return sorted(out, key=lambda d: d.tag)

    def test_split(self):
        devs = self._devices()
        s = m.make_split(devs, 20261020)
        self.assertEqual((s["n_dev"], s["n_test"]), (18, 10))
        self.assertEqual(s, m.make_split(devs, 20261020))
        self.assertFalse(set(s["DEV"]) & set(s["TEST"]))
        test_lv = [t for t in s["TEST"] if t.startswith("lv")]
        geo = lambda t: t.split("_", 2)[2]
        self.assertEqual(sorted(geo(t) for t in test_lv if "nmos" in t),
                         sorted(geo(t) for t in test_lv if "pmos" in t))
        for cls, pol, n in (("hv", "nmos", 2), ("hv", "pmos", 2), ("lv", "nmos", 3), ("lv", "pmos", 3)):
            self.assertEqual(sum(t.startswith(f"{cls}_{pol}") for t in s["TEST"]), n)


class TestCorrectedProblem(unittest.TestCase):
    def test_keys_and_gain(self):
        dev = _device()

        class Fake:
            def __init__(self):
                self.dev = dev
                self.curves = m.fit_curves(dev, 300)
                self.box = None
                self.tag = dev.tag

            def simulate(self, local):
                return [c.id.copy() for c in self.curves]
        prob = Fake()
        corr = Correction("T1", table={"nmos|idvg@d1": 0.5})
        cp = m.corrected_problem(prob, corr)
        self.assertEqual(cp._keys[1], "idvg@d1")
        out = cp.simulate(None)
        np.testing.assert_allclose(out[1], prob.curves[1].id * math.exp(0.5))
        np.testing.assert_allclose(out[0], prob.curves[0].id)


def _sim_ready() -> bool:
    if not (m.PARM_LIBS["lv"].exists() and m.OSDI.exists()):
        return False
    try:
        m.ngspice46_bin()
        return True
    except Exception:
        return False


SIM_READY = _sim_ready()


@unittest.skipUnless(SIM_READY, "PDK model files / ngspice-46 not present")
class TestSimulator(unittest.TestCase):
    def test_card_path_matches_pdk_subckt_and_box(self):
        dev = _device()
        dev.W_um, dev.L_um = 1.0, 0.5
        curves = m.fit_curves(dev, 300)
        a = m.simulate(dev, m.stock_card(dev, m.BASE_SET), curves, 27.0)
        b = m.simulate_pdk_subckt(dev, curves, 27.0)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-15)
        self.assertGreater(a[2][-1], 0)                 # nMOS on-current into the drain
        box = m.base_box(dev)
        self.assertEqual(box.n_active, 12)
        np.testing.assert_allclose(box.params_to_local(box.published), 0.0, atol=1e-9)
        i = list(box.names).index("vfbo")
        self.assertAlmostEqual(box.hi[i] - box.published["vfbo"], 0.2)
        j = list(box.names).index("uo")
        self.assertAlmostEqual(box.hi[j] / box.lo[j], 4.0)


if __name__ == "__main__":
    unittest.main()
