"""I10-1 helpers (cryoml.explore.i10_1_rt4k) and decision logic of the driver.
No downloaded data and no simulator needed."""
from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cryoml.explore import i8_1_nist as b
from cryoml.explore import i10_1_rt4k as m
from cryoml.explore.i3_2_bias import paired_ratio
from cryoml.parameter_space import ParameterBox

ROOT = Path(__file__).resolve().parents[1]
T = "nfet_g5v0d10v5"

PM3 = (".param sky130_fd_pr__nfet_g5v0d10v5__vth0_slope_spectre = 0.0\n"
       ".subckt  sky130_fd_pr__nfet_g5v0d10v5 d g s b\n"
       "msky130_fd_pr__nfet_g5v0d10v5 d g s b sky130_fd_pr__nfet_g5v0d10v5__model l = {l} w = {w}\n"
       ".model sky130_fd_pr__nfet_g5v0d10v5__model.0 nmos\n"
       "+ lmin = 2.0e-05 lmax = 1.0e-04 wmin = 2.0e-05 wmax = 1.0e-4\n"
       "+ vth0 = 0.8 lvth0 = 1e-8\n"
       ".model sky130_fd_pr__nfet_g5v0d10v5__model.1 nmos\n"
       "+ lmin = 5.0e-07 lmax = 6.0e-07 wmin = 4.2e-07 wmax = 7.0e-07\n"
       "+ vth0 = 0.9 lvth0 = 2e-8 pclm = 0.5\n"
       ".ends sky130_fd_pr__nfet_g5v0d10v5\n")
CORNER = ".param\n+ x_mult = 1.0\n.include \"sky130_fd_pr__nfet_g5v0d10v5__tt.pm3.spice\"\n"


def _driver():
    spec = importlib.util.spec_from_file_location(
        "i10_1_driver", ROOT / "scripts" / "research" / "explore" / "i10_1_rt4k.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _box():
    names = ("vth0", "u0", "pclm", "cit")
    pub = {"vth0": 0.8, "u0": 0.03, "pclm": 0.5, "cit": 0.0}
    bounds = {"vth0": [0.8 / 3, 2.4], "u0": [0.003, 0.3], "pclm": [0.5 / 3, 1.5], "cit": [0.0, 0.0]}
    return ParameterBox(names=names, published=pub, bounds=bounds, log_scale=(True, True, True, False))


class Cards(unittest.TestCase):
    def test_split_force_patch(self):
        p = m.split_card(PM3, T, CORNER)
        self.assertEqual(sorted(p["blocks"]), [0, 1])
        self.assertEqual(p["n_unparsed"], 0)
        self.assertNotIn(".include", p["pre_param"])
        self.assertIn("x_mult", p["pre_param"])
        g = p["geometry"][1]
        self.assertAlmostEqual(g["lmin"], 5e-7)
        self.assertIn("+ lvth0 = 2e-8\n", p["blocks"][1])
        forced = m.force_bin(p["prefix"], T, 1)
        self.assertIn("sky130_fd_pr__nfet_g5v0d10v5__model.1 l = {l}", forced)
        patched = m.rt9.patch_block(p["blocks"][1], {"vth0": 1.1})
        self.assertIn("+ vth0 = 1.1\n", patched)
        self.assertIn("+ lvth0 = 2e-8\n", patched)

    def test_type_info(self):
        self.assertIsNone(m.TYPE_INFO["nfet_01v8"].corner)
        self.assertIsNone(m.TYPE_INFO["pfet_01v8_hvt"].corner)
        self.assertEqual(m.TYPE_INFO["pfet_g5v0d10v5"].polarity, "pmos")
        self.assertEqual(m.TYPE_INFO[T].corner, "sky130_fd_pr__nfet_g5v0d10v5__tt.corner.spice")
        self.assertEqual(m.VDD[T], 3.3)


class Data(unittest.TestCase):
    def _dev(self):
        d = b.NistDevice("x", "ETest", T, "nmos", 1, 1, 1.0, 0.5)
        v = np.round(np.arange(0, 5.0001, 0.05), 10)
        for fixed in (0.1, 5.0):
            d.curves.append(b.NistCurve("idvg", fixed, 0.0, v.copy(), 1e-6 * v, np.zeros(0)))
        d.curves.append(b.NistCurve("idvg", 0.1, -2.5, v.copy(), 1e-6 * v, np.zeros(0)))
        for fixed in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
            d.curves.append(b.NistCurve("idvd", fixed, 0.0, v.copy(), 1e-6 * v * fixed, np.zeros(0)))
        b.assign_keys(d.curves)
        return d

    def test_restrict(self):
        r = m.restrict(self._dev(), 3.3)
        keys = [(c.key, c.fixed, len(c.sweep)) for c in b.fit_curves(r)]
        self.assertEqual(keys, [("idvg@d0", 0.1, 67), ("idvd@g0", 0.0, 67), ("idvd@g1", 1.0, 67),
                                ("idvd@g2", 2.0, 67), ("idvd@g3", 3.0, 67)])
        self.assertTrue(all(np.max(np.abs(c.sweep)) <= 3.3 + 1e-9 for c in r.curves))
        self.assertTrue(all(c.vb == 0.0 for c in r.curves))

    def test_descriptors(self):
        vg = np.linspace(0, 2, 201)
        i = 1e-12 * 10 ** (vg / 0.1)                      # 100 mV / decade
        c1 = SimpleNamespace(kind="idvg", fixed=0.1, Vg=vg, Id=i)
        c2 = SimpleNamespace(kind="idvg", fixed=3.3, Vg=vg, Id=10 * i)
        c3 = SimpleNamespace(kind="idvd", fixed=3.0, Vg=None, Id=np.array([0.0, 2e-3]))
        d = m.descriptors([c2, c1, c3], W_um=1.0, L_um=1.0)
        self.assertAlmostEqual(d["vth_cc_V"], 0.5, places=6)           # 1e-7 A at 0.5 V
        self.assertAlmostEqual(d["log10_ion_lin"], math.log10(i[-1]))
        self.assertAlmostEqual(d["log10_ion_sat"], math.log10(2e-3))
        # I6-1 attribute names (vg / id) and the 1e-8 A floor of the criterion
        c4 = SimpleNamespace(kind="idvg", fixed=0.05, vg=-vg, id=-i)
        d = m.descriptors([c4], W_um=0.1, L_um=10.0)
        self.assertAlmostEqual(d["vth_cc_V"], 0.4, places=6)           # max(1e-9, 1e-8) A at 0.4 V
        self.assertTrue(math.isnan(d["log10_ion_sat"]))

    def test_discover_types_skips_other_types(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "Sky130-4k").mkdir()
            # a core (TEST) file with content that cannot be parsed: must not be opened
            (root / "Sky130-4k" / "sky130_fd_pr__nfet_01v8_w1p0u_l1p0u_m1(8392_1_2_IDVG)_4K.mdm").write_text(
                "not an MDM file")
            self.assertEqual(m.discover_types(m.DEV_TYPES, root=root), [])


class Shift(unittest.TestCase):
    def test_local_and_apply(self):
        box = _box()
        z0 = m.local(box, box.published)
        self.assertTrue(np.allclose(z0, 0.0))
        dz = np.array([0.5, -0.25, 0.1, 0.9])
        p = m.apply_shift(box, z0, dz)
        self.assertAlmostEqual(p["vth0"], 0.8 * math.exp(0.5 * math.log(3)))
        self.assertAlmostEqual(p["u0"], 0.03 * math.exp(-0.25 * math.log(10)))
        self.assertEqual(p["cit"], 0.0)                                   # inactive coordinate kept
        q = m.apply_shift(box, z0, dz, subset=("vth0",))
        self.assertAlmostEqual(q["u0"], 0.03)
        self.assertAlmostEqual(q["vth0"], p["vth0"])
        r = m.apply_shift(box, z0, np.array([3.0, 0, 0, 0]))              # clipped to the box
        self.assertAlmostEqual(r["vth0"], 2.4)
        D = np.array([[0, 1.0], [1, 2.0], [10, 3.0]])
        self.assertTrue(np.allclose(m.shift_stat(D), [1.0, 2.0]))
        self.assertTrue(np.allclose(m.shift_stat(D, "mean"), [11 / 3, 2.0]))

    def test_ridge(self):
        rng = np.random.default_rng(0)
        n = 30
        X = rng.normal(size=(n, 3))
        Y = np.column_stack([0.4 * X[:, 0] + 0.1, rng.normal(scale=0.05, size=n), np.full(n, 0.2)])
        r = m.RidgeShift().fit(X, Y)
        x = np.array([1.0, 0.0, 0.0])
        pred = r.predict(x)
        s0 = X[:, 0].std()
        expect = np.median(Y[:, 0]) + 0.4 * (1.0 - X[:, 0].mean())
        self.assertAlmostEqual(pred[0], expect, delta=0.05 * s0 + 0.02)
        self.assertIsNone(r.summary(["a", "b", "c"])["lambda_over_n"]["c"])   # constant target: median
        self.assertAlmostEqual(pred[2], 0.2)
        # fewer than 3 rows: the median shift (arm A1)
        r2 = m.RidgeShift().fit(X[:2], Y[:2])
        self.assertTrue(np.allclose(r2.predict(x), np.median(Y[:2], axis=0)))
        # NaN features imputed with the training median; constant columns dropped
        Xn = X.copy()
        Xn[0, 1] = np.nan
        Xn[:, 2] = 1.0
        self.assertTrue(np.all(np.isfinite(m.RidgeShift().fit(Xn, Y).predict([np.nan, 0.0, 1.0]))))


class Stats(unittest.TestCase):
    def test_gmr_ci_matches_paired_ratio(self):
        rng = np.random.default_rng(1)
        a = {f"d{i}": float(x) for i, x in enumerate(rng.uniform(0.05, 0.5, 20))}
        c = {k: v * float(rng.uniform(0.8, 1.3)) for k, v in a.items()}
        c["d3"] = float("nan")
        p, g = paired_ratio(a, c), m.gmr_ci(a, c, level=0.95)
        for k in ("ratio", "ci_low", "ci_high", "n_used", "wins"):
            self.assertAlmostEqual(p[k], g[k])
        g99 = m.gmr_ci(a, c, level=0.99)
        self.assertLessEqual(g99["ci_low"], g["ci_low"])
        self.assertGreaterEqual(g99["ci_high"], g["ci_high"])
        cl = {k: k[:2] for k in a}
        self.assertTrue(m.gmr_ci(a, c, clusters=cl)["resample_unit"].startswith("cluster"))
        self.assertEqual(m.ci_label({"ci_low": 0.8, "ci_high": 0.95}), "lower")
        self.assertEqual(m.ci_label({"ci_low": 1.01, "ci_high": 1.2}), "higher")
        self.assertEqual(m.ci_label({"ci_low": 0.9, "ci_high": 1.1}), "inconclusive")


class Decision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = _driver()

    def test_outcome(self):
        lo, inc = "lower", "inconclusive"
        o = self.d.outcome_of({"K1": lo, "K2": lo, "K3": lo, "K4": inc, "K5": inc})
        self.assertEqual(o[0], "supported")
        self.assertEqual(o[2], ["K1", "K2", "K3"])
        o = self.d.outcome_of({"K1": lo, "K2": inc, "K3": lo, "K4": inc, "K5": inc})
        self.assertEqual((o[0], o[2]), ("partly supported", ["K1", "K3"]))
        o = self.d.outcome_of({"K1": inc, "K2": lo, "K3": lo, "K4": lo, "K5": lo})
        self.assertEqual((o[0], o[2]), ("not supported", []))
        self.assertTrue(o[1].startswith("stop"))

    def test_test_arm_and_variant(self):
        comps = {"K4": {"bonferroni": {"ratio": 0.9}}, "K5": {"bonferroni": {"ratio": 0.8}},
                 "V_V_mean": {"level95": {"ratio": 0.97, "ci_low": 0.95, "ci_high": 0.99, "n_used": 60}},
                 "V_V_ind": {"level95": {"ratio": 0.90, "ci_low": 0.8, "ci_high": 1.01, "n_used": 60}}}
        self.assertEqual(self.d.test_arm({"K4": "lower", "K5": "lower"}, comps), "A3")
        self.assertEqual(self.d.test_arm({"K4": "lower", "K5": "inconclusive"}, comps), "A2")
        self.assertEqual(self.d.test_arm({"K4": "higher", "K5": "inconclusive"}, comps), "A1")
        v = self.d.variant_choice(comps, ("V_ind", "V_mean", "V_sub7"))
        self.assertEqual(v["adopted"], "V_mean")

    def test_bonferroni_level(self):
        self.assertAlmostEqual(self.d.BONF_LEVEL, 1 - 0.05 / 5)

    def test_guard_refuses_without_prereg(self):
        d = self.d
        old = d.PREREG
        try:
            d.PREREG = ROOT / "out" / "research" / "explore" / "I10_1" / "no_such_prereg.json"
            with self.assertRaises(SystemExit):
                d.require_prereg(SimpleNamespace(smoke=False))
            self.assertTrue(d.require_prereg(SimpleNamespace(smoke=True))["smoke"])
        finally:
            d.PREREG = old


if __name__ == "__main__":
    unittest.main()
