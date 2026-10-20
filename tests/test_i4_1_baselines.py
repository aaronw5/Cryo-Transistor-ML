"""I4-1 baselines: B1 boxes, budget matching, B2 elements, B3 polynomial,
B4 GP, decision rule, I4-2 helpers, sky130 deck with extrinsic resistance."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.explore.e1_correction import (Correction, CurveRec, DeviceRec,  # noqa: E402
                                          make_batch, _loss_fn, raw_features)
from cryoml.explore.i4_1_baselines import (  # noqa: E402
    B1_FACTOR, AugProblem, GPCorrection, PolyCorrection, b1_extra_box, b1_param_box,
    b2_box, correction_from_json_any, decide_rule, edge_response, extend_box, fit_gp,
    fit_poly, g_grid, gp_training_data, heat_factor, linear_vth, matched_nfev,
    measured_pmax, poly_exponents, poly_features, region_masks, region_stats,
    t1_comparison)
from cryoml.parameter_space import ParameterBox  # noqa: E402

VDD = 1.85
GRID = np.linspace(0.05, VDD, 30)
BIASES = (("idvg", 0.1), ("idvg", 1.85), ("idvd", 0.74), ("idvd", 1.85))


def shared(vg, vd):
    return 0.3 * np.abs(vd) / VDD - 0.2 * np.abs(vg) / VDD


def make_device(i, family, L, W):
    sgn = -1.0 if family == "pmos" else 1.0
    rec = DeviceRec(f"{family}_toy{i}", family, L, W, VDD, "toy")
    for kind, bias in BIASES:
        if kind == "idvg":
            vg, vd = sgn * GRID, np.full_like(GRID, sgn * bias)
        else:
            vg, vd = np.full_like(GRID, sgn * bias), sgn * GRID
        sim = sgn * (W / L) * 1e-4 * np.log1p(np.exp(8 * (np.abs(vg) - 0.4))) * np.tanh(3 * np.abs(vd))
        tgt = sim * np.exp(shared(vg, vd))
        rec.curves.append(CurveRec(f"{kind}@{bias:g}", kind, vg, vd, sim, tgt, 0,
                                   float(np.mean(np.abs(tgt))), True))
    return rec


def toy_devices(n=4):
    return [make_device(i, "pmos" if i % 2 else "nmos", 1.0 + i, 2.0) for i in range(n)]


def mean_loss(corr, recs):
    batch = make_batch(recs)
    import torch
    loss = _loss_fn(batch)
    g = torch.tensor(corr.g(batch.phi) if corr.arm != "C0" else np.zeros(len(batch.phi)))
    return float(loss(g))


class TestB1Box(unittest.TestCase):
    def test_zero_range_and_log(self):
        zr = {"ud": (0.0, 1e17), "rdswmin": (0.0, "rdsw"), "voffl": (-1e-8, 1e-8)}
        self.assertEqual(b1_param_box("ud", 0.0, zr), ([0.0, 1e17], False))
        self.assertEqual(b1_param_box("rdswmin", 0.0, zr, rdsw=250.0), ([0.0, 250.0], False))
        # small nonzero value inside the zero-range rule is kept inside the box
        b, log = b1_param_box("voffl", -1e-12, zr)
        self.assertEqual((b, log), ([-1e-8, 1e-8], False))
        # nonzero value: log box with factor B1_FACTOR, sign kept
        b, log = b1_param_box("ud", 4e16, zr)
        self.assertTrue(log)
        self.assertAlmostEqual(b[0], 4e16 / B1_FACTOR)
        b, log = b1_param_box("k", -0.4, zr)
        self.assertTrue(log)
        self.assertEqual(b, [-0.4 * B1_FACTOR, -0.4 / B1_FACTOR])
        # zero without a range: inactive
        self.assertEqual(b1_param_box("dvt2", 0.0, zr), ([0.0, 0.0], False))

    def test_extend_box_keeps_start(self):
        base = ParameterBox(names=("a", "b"), published={"a": 1.0, "b": -2.0},
                            bounds={"a": [0.5, 2.0], "b": [-3.0, -1.0]},
                            log_scale=(True, False))
        extras = b1_extra_box(["ud", "pdits", "dvt2"], {"ud": 0.0, "pdits": 0.3, "dvt2": 0.0},
                              {"ud": (0.0, 1e17)})
        box = extend_box(base, extras)
        self.assertEqual(box.names, ("a", "b", "ud", "pdits", "dvt2"))
        self.assertEqual(box.n_active, 4)
        start = {"a": 1.3, "b": -1.5, "ud": 0.0, "pdits": 0.3, "dvt2": 0.0}
        back = box.local_to_params(box.params_to_local(start))
        for k, v in start.items():
            self.assertAlmostEqual(back[k], v, delta=1e-12 * max(1.0, abs(v)))

    def test_matched_nfev(self):
        for n0, n1 in ((41, 41), (41, 60), (21, 56), (21, 23), (12, 83)):
            nfev = matched_nfev(n0, n1)
            self.assertEqual(max(2, nfev // 2 // (n1 + 2)), max(2, 400 // 2 // (n0 + 2)))
        self.assertEqual(matched_nfev(41, 41), 4 * 2 * 43)

    def test_edge_response(self):
        layout = [{"index": 0, "start": 1, "denominator": 2.0},
                  {"index": 1, "start": 0, "denominator": 1.0}]
        ref = [np.array([9.0, 1.0, 1.0]), np.array([1.0, 1.0])]
        cand = [np.array([0.0, 3.0, 3.0]), np.array([1.0, 1.5])]
        self.assertAlmostEqual(edge_response(ref, cand, layout), 1.0)
        cand[1] = np.array([np.nan, 1.0])
        self.assertTrue(math.isnan(edge_response(ref, cand, layout)))


class FakeCurve:
    def __init__(self, vd):
        self.Vd = vd


class FakeBase:
    def __init__(self):
        self.curves = [FakeCurve(np.array([0.5, 1.0])), FakeCurve(np.array([1.0, 2.0]))]
        self.layout = []

    def curve_rrms(self, sims):
        return np.array([float(np.sum(s)) for s in sims])

    def residual_vector(self, sims, weights):
        return np.concatenate(sims)


class TestB2(unittest.TestCase):
    def test_heat_and_pmax(self):
        f = heat_factor(np.array([1.0, -2.0]), np.array([1.0, -1.0]), 0.5, 2.0)
        np.testing.assert_allclose(f, np.exp([0.25, 0.5]))
        rec = make_device(0, "nmos", 1.0, 2.0)
        best = max(float(np.max(np.abs(c.vd * c.target))) for c in rec.curves)
        self.assertAlmostEqual(measured_pmax(rec), best)

    def test_aug_problem(self):
        base_box = ParameterBox(names=("a",), published={"a": 1.0}, bounds={"a": [0.0, 2.0]})
        box = b2_box(base_box, 100.0)
        self.assertEqual(box.names, ("a", "rext_ohm", "heat_kappa"))
        seen = {}

        def sim_fn(card, rext):
            seen["card"], seen["rext"] = card, rext
            return [np.array([1.0, 2.0]) * card["a"], np.array([1.0, 1.0])]
        aug = AugProblem(FakeBase(), box, sim_fn, pmax=2.0)
        start = box.params_to_local({"a": 1.0, "rext_ohm": 0.0, "heat_kappa": 0.0})
        sims = aug.simulate(start)
        self.assertEqual(seen["rext"], 0.0)
        self.assertEqual(seen["card"], {"a": 1.0})
        np.testing.assert_allclose(sims[0], [1.0, 2.0])
        local = box.params_to_local({"a": 1.0, "rext_ohm": 50.0, "heat_kappa": 0.4})
        sims = aug.simulate(local)
        self.assertAlmostEqual(seen["rext"], 50.0)
        p0 = np.abs(np.array([0.5, 1.0]) * np.array([1.0, 2.0]))
        np.testing.assert_allclose(sims[0], np.array([1.0, 2.0]) * np.exp(0.4 * p0 / 2.0))
        self.assertEqual(aug.n_sim, 2)
        np.testing.assert_allclose(aug.curve_rrms(sims), [np.sum(sims[0]), np.sum(sims[1])])


class TestB3Poly(unittest.TestCase):
    def test_exponents(self):
        self.assertEqual(len(poly_exponents(2)), 6)
        self.assertEqual(len(poly_exponents(3)), 10)
        f = poly_features(np.array([2.0]), np.array([3.0]), 2)
        np.testing.assert_allclose(f[0], [1, 2, 3, 4, 6, 9])

    def test_fit_reduces_loss_and_roundtrip(self):
        recs = toy_devices()
        corr = fit_poly("B3_d2", 2, recs, hyper={"steps": 600, "lr": 2e-2})
        self.assertLess(mean_loss(corr, recs), 0.5 * mean_loss(Correction("C0"), recs))
        back = correction_from_json_any(json.loads(json.dumps(corr.to_json())))
        phi = raw_features(recs[1], recs[1].curves[0].vg, recs[1].curves[0].vd,
                           recs[1].curves[0].sim)
        np.testing.assert_allclose(back.g(phi), corr.g(phi))
        self.assertIsInstance(back, PolyCorrection)
        # polarity-specific coefficients
        p = np.column_stack([np.array([0.0, 1.0]), [0.5, 0.5], [0.5, 0.5], np.zeros((2, 3))])
        coef = corr.coef.copy()
        corr.coef = np.stack([coef[0], coef[0] + 1.0])
        g = corr.g(p)
        self.assertNotAlmostEqual(g[0], g[1])


class TestB4GP(unittest.TestCase):
    def test_training_data_and_fit(self):
        recs = toy_devices()
        x, y = gp_training_data(recs, total=80, seed=0)
        self.assertEqual(len(y), 80)
        np.testing.assert_allclose(y, 0.3 * x[:, 2] - 0.2 * x[:, 1], atol=1e-12)
        corr = fit_gp("B4", recs, total=80, maxiter=100)
        phi = np.column_stack([x, np.zeros((len(x), 3))])
        self.assertLess(float(np.max(np.abs(corr.g(phi) - y))), 0.02)
        back = correction_from_json_any(json.loads(json.dumps(corr.to_json())))
        self.assertIsInstance(back, GPCorrection)
        np.testing.assert_allclose(back.g(phi), corr.g(phi))
        self.assertIn("ell", corr.info)


class TestDecision(unittest.TestCase):
    def test_rule(self):
        good_real = {"ci_high": 0.9, "ratio": 0.8}
        bad_real = {"ci_high": 1.1, "ratio": 0.95}
        sim_good = {"ratio": 0.9}
        sim_bad = {"ratio": 0.97}
        names = ("B1", "B2", "B3_d2", "B3_d3", "B4")
        out = decide_rule({b: good_real for b in names}, {b: sim_bad for b in names},
                          {b: sim_bad for b in names})
        self.assertEqual(out["verdict"], "supported against baselines")
        real = {b: good_real for b in names}
        real["B2"] = bad_real
        real["B4"] = bad_real
        gf = {b: sim_bad for b in names}
        ihp = {b: sim_bad for b in names}
        gf["B2"], ihp["B2"] = sim_good, sim_good          # B2 beaten on simulated data
        gf["B4"] = sim_good                                # B4 only on one PDK
        out = decide_rule(real, gf, ihp)
        self.assertEqual(out["matched_by"], ["B4"])
        self.assertTrue(out["verdict"].startswith("matched by"))
        out = decide_rule({}, {}, {})
        self.assertEqual(out["matched_by"], list(names))


class TestI42Helpers(unittest.TestCase):
    def test_grid_and_regions(self):
        corr = PolyCorrection("B3_d2", np.zeros(2), np.ones(2), degree=2,
                              coef=np.array([[0, 0, 1.0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]]))
        z = g_grid(corr, 0.0, n=5)
        np.testing.assert_allclose(z[0], np.tanh(np.linspace(0, 1, 5)))   # nMOS: g = tanh(|Vds|)
        z = g_grid(corr, 1.0, n=5)
        np.testing.assert_allclose(z[:, 0], np.tanh(np.linspace(0, 1, 5)))  # pMOS: tanh(|Vgs|)
        m = region_masks(np.array([0.2, 0.8, 0.8]), np.array([1.0, 1.0, 0.1]), 0.5, 1.0)
        self.assertEqual(m["subthreshold"].tolist(), [True, False, False])
        self.assertEqual(m["saturation"].tolist(), [False, True, False])
        self.assertEqual(m["linear"].tolist(), [False, False, True])
        self.assertEqual(m["high_vds"].tolist(), [True, True, False])
        st = region_stats(np.array([1.0, 2.0, 3.0]), m, np.array([1.0, 2.0, 3.1]))
        self.assertAlmostEqual(st["subthreshold"]["mean_g"], 1.0)
        self.assertGreater(st["pearson_g_resid"], 0.99)
        t = t1_comparison([{"group": "nmos|idvd@1", "t1": 0.1, "g_mean": 0.12},
                           {"group": "nmos|idvg@1", "t1": 0.2, "g_mean": 0.18},
                           {"group": "nmos|idvg@2", "t1": 0.3, "g_mean": 0.33}])
        self.assertAlmostEqual(t["mean_abs_diff"], (0.02 + 0.02 + 0.03) / 3)

    def test_linear_vth(self):
        rec = DeviceRec("nmos_t", "nmos", 1.0, 1.0, VDD, "toy")
        vg = np.linspace(0, 1.85, 200)
        i = np.where(vg > 0.5, 1e-4 * (vg - 0.5 - 0.005), 0.0)
        rec.curves.append(CurveRec("idvg@0.01", "idvg", vg, np.full_like(vg, 0.01), i, i, 0,
                                   1.0, True))
        self.assertAlmostEqual(linear_vth(rec), 0.5, delta=0.02)


class TestSkyDeck(unittest.TestCase):
    def test_instance_line_rext(self):
        from cryoml.spice_pdk import _instance_line
        plain = _instance_line("nmos", 1.0, 3.0)
        self.assertEqual(plain, _instance_line("nmos", 1.0, 3.0, 0.0))
        self.assertTrue(plain.startswith("Xm1 nd ng 0 0 "))
        self.assertNotIn("RDEXT", plain)
        ext = _instance_line("nmos", 1.0, 3.0, 25.0)
        self.assertTrue(ext.startswith("Xm1 ndi ng nsi 0 "))
        self.assertIn("RDEXT nd ndi 25\n", ext)
        self.assertIn("RSEXT nsi 0 25\n", ext)
        self.assertEqual(ext.split("\n")[0].replace("ndi ng nsi", "nd ng 0"), plain.rstrip("\n"))
        mini = _instance_line("pmos", 2.0, 5.0, 10.0)
        self.assertTrue(mini.startswith("Xm1 ndi ng nsi 0 "))


if __name__ == "__main__":
    unittest.main()
