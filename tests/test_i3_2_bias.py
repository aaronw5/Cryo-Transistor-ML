"""I3-2 bias-only correction: training recipe, inputs, statistic, Verilog-A text."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.explore.e1_correction import (Correction, CurveRec, DeviceRec,  # noqa: E402
                                          fit_arm, raw_features)
from cryoml.explore.i2_1_joint import CorrectedProblem, corrected_score  # noqa: E402
from cryoml.explore.i3_2_bias import (ARM_COLUMNS, BiasCorrection,  # noqa: E402
                                      correction_from_json, eval_verilog_a_g, fit_bias,
                                      merge_seed_results, paired_ratio, split_seed_tasks,
                                      to_verilog_a_bias)

VDD = 1.85
GRID = np.linspace(0.05, VDD, 30)
BIASES = (("idvg", 0.1), ("idvg", 1.85), ("idvd", 0.74), ("idvd", 1.85))


def shared(vg, vd):
    """Bias-only model-form error shared by every device."""
    return 0.3 * np.abs(vd) / VDD - 0.2 * np.abs(vg) / VDD


def make_device(i, family, L, W, rng, device_noise=0.0):
    sgn = -1.0 if family == "pmos" else 1.0
    rec = DeviceRec(f"{family}_toy{i}", family, L, W, VDD, "toy")
    own = rng.normal(scale=device_noise) if device_noise else 0.0
    for kind, bias in BIASES:
        if kind == "idvg":
            vg, vd = sgn * GRID, np.full_like(GRID, sgn * bias)
        else:
            vg, vd = np.full_like(GRID, sgn * bias), sgn * GRID
        sim = sgn * (W / L) * 1e-4 * np.log1p(np.exp(8 * (np.abs(vg) - 0.4))) * np.tanh(3 * np.abs(vd))
        tgt = sim * np.exp(shared(vg, vd) + own * np.log10(L))
        rec.curves.append(CurveRec(f"{kind}@{bias:g}", kind, vg, vd, sim, tgt, 0,
                                   float(np.mean(np.abs(tgt))), True))
    return rec


def toy_devices(n=6, seed=0, device_noise=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        fam = "pmos" if i % 2 else "nmos"
        out.append(make_device(i, fam, float(rng.choice([0.5, 1.0, 8.0])),
                               float(rng.choice([1.0, 5.0])), rng, device_noise))
    return out


def random_phi(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.integers(0, 2, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n),
                            rng.uniform(-1, 2, n), rng.uniform(-1, 2, n),
                            rng.uniform(-12, -3, n)]).astype(float)


class FitTest(unittest.TestCase):

    def test_all_columns_hidden32_reproduces_e1_fit_arm_m(self):
        devs = toy_devices(4)
        hp = {"seeds": (0, 1), "steps": 40, "hidden": 32}
        mine = fit_bias("M_bias", devs, hyper=hp, cols=(0, 1, 2, 3, 4, 5))
        ref = fit_arm("M", devs, hyper={"seeds": (0, 1), "steps": 40}, threads=1)
        phi = random_phi()
        self.assertTrue(np.allclose(mine.g(phi), ref.g(phi), rtol=0, atol=1e-12))

    def test_inputs_are_the_declared_columns(self):
        devs = toy_devices(4)
        hp = {"seeds": (0,), "steps": 50}
        bias = fit_bias("M_bias", devs, hyper=hp)
        geom = fit_bias("M_bias_geom", devs, hyper=hp)
        self.assertEqual(bias.cols, ARM_COLUMNS["M_bias"])
        self.assertEqual(geom.cols, ARM_COLUMNS["M_bias_geom"])
        self.assertEqual(bias.nets[0][0][0].shape, (3, 16))
        self.assertEqual(geom.nets[0][0][0].shape, (5, 16))
        self.assertEqual(bias.mu[0], 0.0)
        self.assertEqual(bias.sd[0], 1.0)
        phi = random_phi()
        moved = phi.copy()
        moved[:, 3:] += 0.7                            # geometry and current
        self.assertTrue(np.array_equal(bias.g(phi), bias.g(moved)))
        self.assertFalse(np.allclose(geom.g(phi), geom.g(moved)))
        cur = phi.copy()
        cur[:, 5] += 2.0                               # current only
        self.assertTrue(np.array_equal(geom.g(phi), geom.g(cur)))

    def test_learns_shared_bias_error_and_transfers(self):
        devs = toy_devices(7)
        held, train = devs[0], devs[1:]
        corr = fit_bias("M_bias", train, hyper={"seeds": (0, 1), "steps": 600})
        c0 = corrected_score(Correction("C0"), held)
        self.assertGreater(c0, 0.05)
        self.assertLess(corrected_score(corr, held), 0.3 * c0)
        c = held.curves[1]
        g = corr.g(raw_features(held, c.vg, c.vd, c.sim))
        self.assertGreater(np.corrcoef(g, shared(c.vg, c.vd))[0, 1], 0.9)

    def test_per_seed_split_matches_full_fit(self):
        devs = toy_devices(4)
        hyper = {"seeds": (0, 1, 2), "steps": 30}
        tasks = [("M_bias", devs), ("M_bias_geom", devs[:3])]
        expanded, owner = split_seed_tasks(tasks, hyper)
        self.assertEqual(owner, [0, 0, 0, 1, 1, 1])
        merged = merge_seed_results(2, owner, [fit_bias(a, r, hyper=h) for a, r, h in expanded])
        ref = fit_bias("M_bias", devs, hyper=hyper)
        phi = random_phi()
        self.assertEqual(len(merged[0].nets), 3)
        self.assertTrue(np.allclose(merged[0].g(phi), ref.g(phi), rtol=0, atol=1e-12))
        self.assertEqual(merged[1].cols, ARM_COLUMNS["M_bias_geom"])

    def test_json_round_trip_and_corrected_problem(self):
        devs = toy_devices(3)
        corr = fit_bias("M_bias", devs, hyper={"seeds": (0,), "steps": 40})
        back = correction_from_json(corr.to_json())
        self.assertIsInstance(back, BiasCorrection)
        phi = random_phi()
        self.assertTrue(np.array_equal(back.g(phi), corr.g(phi)))
        e1 = correction_from_json(Correction("T1", table={"a": 0.1}).to_json())
        self.assertNotIsInstance(e1, BiasCorrection)

        rec = devs[0]

        class Curve:
            def __init__(self, c):
                self.kind, self.fixed = c.kind, float(c.key.split("@")[1])
                self.Vg, self.Vd, self.Id, self.meta = c.vg, c.vd, c.target, {}

        class Base:
            tag = rec.tag
            n_sim = 0
            curves = [Curve(c) for c in rec.curves]

            def simulate(self, local):
                return [c.sim * float(local[0]) for c in rec.curves]

        prob = CorrectedProblem(Base(), corr, rec.family, rec.L_um, rec.W_um, VDD, "toy")
        out = prob.simulate(np.array([2.0]))
        for s, c in zip(out, rec.curves):
            g = corr.g(raw_features(rec, c.vg, c.vd, 2.0 * c.sim))
            self.assertTrue(np.allclose(s, 2.0 * c.sim * np.exp(g), rtol=1e-14, atol=0))


class PairedRatioTest(unittest.TestCase):

    def test_ratio_wins_and_failed(self):
        ctrl = {"a": 0.1, "b": 0.2, "c": 0.4, "d": 0.3}
        meth = {"a": 0.05, "b": 0.2, "c": 0.8, "d": float("nan")}
        st = paired_ratio(meth, ctrl, n_boot=200)
        self.assertEqual(st["n"], 4)
        self.assertEqual(st["n_used"], 3)
        self.assertEqual(st["n_method_failed"], 1)
        self.assertEqual(st["wins"], 1)
        self.assertEqual(st["ties"], 1)
        self.assertAlmostEqual(st["ratio"], math.exp((math.log(0.5) + 0 + math.log(2)) / 3))
        self.assertLessEqual(st["ci_low"], st["ratio"])
        self.assertGreaterEqual(st["ci_high"], st["ratio"])

    def test_constant_ratio_has_degenerate_ci(self):
        ctrl = {f"d{i}": 0.1 * (i + 1) for i in range(12)}
        meth = {k: 0.8 * v for k, v in ctrl.items()}
        st = paired_ratio(meth, ctrl)
        for k in ("ratio", "ci_low", "ci_high"):
            self.assertAlmostEqual(st[k], 0.8, places=12)
        self.assertEqual(st["wins"], 12)
        self.assertEqual(st["n_boot"], 10000)

    def test_cluster_bootstrap_is_reproducible_and_uses_clusters(self):
        rng = np.random.default_rng(3)
        ctrl = {f"g{g}_s{s}": float(rng.uniform(0.05, 0.2)) for g in range(16) for s in range(3)}
        meth = {k: v * float(np.exp(rng.normal(-0.1, 0.2))) for k, v in ctrl.items()}
        clusters = {k: k.split("_")[0] for k in ctrl}
        a = paired_ratio(meth, ctrl, clusters)
        b = paired_ratio(meth, ctrl, clusters)
        self.assertEqual(a, b)
        self.assertEqual(a["resample_unit"], "cluster (16)")
        d = paired_ratio(meth, ctrl)
        self.assertAlmostEqual(a["ratio"], d["ratio"], places=14)
        self.assertNotEqual(a["ci_high"], d["ci_high"])


class VerilogATest(unittest.TestCase):

    def test_emitted_arithmetic_matches_g(self):
        devs = toy_devices(4)
        for arm in ("M_bias", "M_bias_geom"):
            corr = fit_bias(arm, devs, hyper={"seeds": (0, 1), "steps": 80})
            text = to_verilog_a_bias(corr, vdd=VDD)
            self.assertIn("module i3_2_bias_corr(d, g, s, di);", text)
            self.assertIn("I(d, s) <+ (exp(gsum / 2) - 1.0) * ib;", text)
            if arm == "M_bias":
                self.assertNotIn("log(L)", text)
            rng = np.random.default_rng(0)
            for _ in range(40):
                pmos = float(rng.integers(0, 2))
                sgn = -1.0 if pmos else 1.0
                vgs, vds = sgn * rng.uniform(0, VDD), sgn * rng.uniform(0, VDD)
                L, W = 10 ** rng.uniform(-1, 2), 10 ** rng.uniform(-1, 2)
                phi = np.array([[pmos, abs(vgs) / VDD, abs(vds) / VDD,
                                 math.log10(L), math.log10(W), -6.0]])
                self.assertAlmostEqual(eval_verilog_a_g(text, pmos, vgs, vds, L, W),
                                       float(corr.g(phi)[0]), places=9)

    def test_current_feature_is_rejected(self):
        corr = fit_bias("M_bias", toy_devices(2), hyper={"seeds": (0,), "steps": 5},
                        cols=(0, 1, 2, 5))
        with self.assertRaises(ValueError):
            to_verilog_a_bias(corr)


if __name__ == "__main__":
    unittest.main()
