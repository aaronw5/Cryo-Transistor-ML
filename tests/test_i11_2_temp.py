"""I11-2 temperature-input correction: features, training, folds, statistics."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.explore.e1_correction import CurveRec, DeviceRec, score_device  # noqa: E402
from cryoml.explore.i11_2_temp import (TempBiasCorrection, TempLinCorrection,  # noqa: E402
                                       check_training, correction_from_json_i11, decide,
                                       fit_bias_t, fit_lin_t, gmr_ci, label_ihp, label_tud,
                                       merge_temp_seeds, nearest_temp, s1_loto, s3_loto_tasks,
                                       t_feature, temp_kelvin, train_mlp, training_set,
                                       unit_log_ratios)

VDD = 1.2
GRID = np.linspace(0.05, VDD, 24)
BIASES = (("idvg", 0.1), ("idvg", 1.2), ("idvd", 0.6), ("idvd", 1.2))
TEMPS = (233, 300, 343, 398)


def shared(vg, vd, temp_k):
    """Bias error whose amplitude depends on temperature."""
    return (0.3 * np.abs(vd) / VDD - 0.2 * np.abs(vg) / VDD) * (1.0 + 2.0 * math.log(temp_k / 300.0))


def make_rec(i, family, temp_k, rng):
    sgn = -1.0 if family == "pmos" else 1.0
    L, W = float(rng.choice([0.5, 1.0])), float(rng.choice([1.0, 5.0]))
    rec = DeviceRec(f"{family}_d{i}", family, L, W, VDD, "toy",
                    meta={"T_K": int(temp_k), "temp_C": float(temp_k) - 273.15 + 0.15, "cls": "lv"})
    for kind, bias in BIASES:
        if kind == "idvg":
            vg, vd = sgn * GRID, np.full_like(GRID, sgn * bias)
        else:
            vg, vd = np.full_like(GRID, sgn * bias), sgn * GRID
        sim = sgn * (W / L) * 1e-4 * np.log1p(np.exp(8 * (np.abs(vg) - 0.4))) * np.tanh(3 * np.abs(vd))
        tgt = sim * np.exp(shared(vg, vd, temp_k))
        rec.curves.append(CurveRec(f"{kind}@{bias:g}", kind, vg, vd, sim, tgt, 0,
                                   float(np.mean(np.abs(tgt))), True))
    return rec


def toy_by_t(n=4, seed=0):
    out = {}
    for t in TEMPS:
        rng = np.random.default_rng(seed)        # same geometry per device at every T
        out[t] = [make_rec(i, "pmos" if i % 2 else "nmos", t, rng) for i in range(n)]
    return out


def unit(rec):
    return rec.tag


class TestTemperature(unittest.TestCase):
    def test_temp_kelvin_sources(self):
        r = DeviceRec("a", "nmos", 1, 1, 1.1, "x", meta={"T_K": 4, "temp_stage_K": 4.134})
        self.assertAlmostEqual(temp_kelvin(r), 4.134)
        r = DeviceRec("a", "nmos", 1, 1, 1.1, "x", meta={"T_K": 233, "temp_C": -40.0})
        self.assertAlmostEqual(temp_kelvin(r), 233.15)
        r = DeviceRec("a", "nmos", 1, 1, 1.1, "x", meta={"T_K": 295})
        self.assertEqual(temp_kelvin(r), 295.0)
        self.assertEqual(t_feature(300.0), 0.0)

    def test_nearest_temp(self):
        self.assertEqual(nearest_temp(233, TEMPS), 300)
        self.assertEqual(nearest_temp(300, TEMPS), 343)
        self.assertEqual(nearest_temp(343, TEMPS), 300)
        self.assertEqual(nearest_temp(398, TEMPS), 343)
        self.assertEqual(nearest_temp(4, (4, 295)), 295)
        self.assertEqual(nearest_temp(295, (4, 295)), 4)


class TestTraining(unittest.TestCase):
    def test_train_mlp_equals_fit_bias(self):
        """The shared loop reproduces i3_2_bias.fit_bias on the same columns."""
        from cryoml.explore.e1_correction import _loss_fn, make_batch
        from cryoml.explore.i3_2_bias import HYPER_BIAS, fit_bias
        recs = toy_by_t()[300]
        hp = {**HYPER_BIAS, "steps": 60, "seeds": (0, 1)}
        ref = fit_bias("M_bias", recs, hyper=hp)
        batch = make_batch(recs)
        mu, sd, nets = train_mlp(batch.phi[:, [0, 1, 2]], _loss_fn(batch), hp)
        np.testing.assert_array_equal(mu, ref.mu)
        np.testing.assert_array_equal(sd, ref.sd)
        for a, b in zip(nets, ref.nets):
            for (wa, ba), (wb, bb) in zip(a, b):
                np.testing.assert_array_equal(wa, wb)
                np.testing.assert_array_equal(ba, bb)

    def test_unbound_refuses_and_bind_changes_g(self):
        by_t = toy_by_t()
        train = by_t[233] + by_t[343] + by_t[398]
        corr = fit_bias_t("M_bias_T", train, hyper={"steps": 300, "seeds": (0,)})
        rec = by_t[300][0]
        with self.assertRaises(RuntimeError):
            corr.apply(rec)
        g_lo = corr.bind(233.15).apply(rec)
        g_hi = corr.bind(398.15).apply(rec)
        self.assertFalse(np.allclose(g_lo[1], g_hi[1]))
        self.assertEqual(corr.mu.shape, (4,))
        self.assertEqual((corr.mu[0], corr.sd[0]), (0.0, 1.0))

    def test_temperature_input_learns_temperature_dependence(self):
        """With a temperature-dependent error the T input lowers the training
        loss below the pooled M_bias and interpolates to the held-out T."""
        from cryoml.explore.i3_2_bias import fit_bias
        by_t = toy_by_t()
        train = by_t[233] + by_t[343] + by_t[398]
        hp = {"steps": 1500, "seeds": (0,)}
        ct = fit_bias_t("M_bias_T", train, hyper=hp)
        cp = fit_bias("M_bias", train, hyper=hp)
        held = by_t[300]
        start = np.mean([score_device(r, [c.sim for c in r.curves])["rrms"] for r in held])
        rt = np.mean([score_device(r, ct.bind(temp_kelvin(r)).apply(r))["rrms"] for r in held])
        rp = np.mean([score_device(r, cp.apply(r))["rrms"] for r in held])
        self.assertLess(rt, rp)
        self.assertLess(rt, start)

    def test_seed_merge_equals_joint_fit(self):
        by_t = toy_by_t()
        train = by_t[233] + by_t[398]
        hp = {"steps": 40}
        joint = fit_bias_t("M_bias_T", train, hyper={**hp, "seeds": (0, 1)})
        parts = [fit_bias_t("M_bias_T", train, hyper={**hp, "seeds": (s,)}) for s in (0, 1)]
        merged = merge_temp_seeds(parts)
        rec = by_t[300][1]
        np.testing.assert_array_equal(np.concatenate(joint.bind(300.15).apply(rec)),
                                      np.concatenate(merged.bind(300.15).apply(rec)))

    def test_lin_t_and_json_roundtrip(self):
        by_t = toy_by_t()
        train = by_t[233] + by_t[343] + by_t[398]
        lin = fit_lin_t("lin_T", train, hyper={"steps": 400})
        self.assertEqual(lin.coef.shape, (2, 4))
        rec = by_t[300][0]
        for corr in (lin, fit_bias_t("M_bias_T", train, hyper={"steps": 50, "seeds": (0,)})):
            bound = corr.bind(300.15)
            back = correction_from_json_i11(bound.to_json())
            np.testing.assert_allclose(np.concatenate(back.apply(rec)),
                                       np.concatenate(bound.apply(rec)), rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            TempLinCorrection("x", lin.mu, lin.sd, coef=lin.coef).apply(rec)
        self.assertIsInstance(correction_from_json_i11(
            fit_bias_t("a", train, hyper={"steps": 5, "seeds": (0,)}).to_json()), TempBiasCorrection)


class TestFolds(unittest.TestCase):
    def test_training_sets(self):
        by_t = toy_by_t()
        held = by_t[300][0].tag
        for arm, temps in (("M_bias_T", {233, 343, 398}), ("M_bias_pool", {233, 343, 398}),
                           ("lin_T", {233, 343, 398}), ("M_bias_near", {343}),
                           ("M_bias_in", {233, 300, 343, 398})):
            tr = training_set(by_t, held, 300, arm, unit)
            self.assertEqual({r.meta["T_K"] for r in tr}, temps, arm)
            self.assertNotIn(held, {r.tag for r in tr})
            self.assertEqual(len(tr), 3 * len(temps))

    def test_eligible_filter(self):
        by_t = toy_by_t()
        tr = training_set(by_t, "nmos_d0", 233, "M_bias_T", unit,
                          eligible=lambda r: r.family == "nmos")
        self.assertEqual({r.tag for r in tr}, {"nmos_d2"})

    def test_check_training_refuses_leaks(self):
        by_t = toy_by_t()
        with self.assertRaises(RuntimeError):
            check_training(by_t[233], "nmos_d0", 300, "M_bias_T", unit)
        with self.assertRaises(RuntimeError):
            check_training(by_t[300][1:], "nmos_d0", 300, "M_bias_T", unit)
        check_training(by_t[300][1:], "nmos_d0", 300, "M_bias_in", unit)

    def test_s3_tasks_exclude_scored(self):
        by_t = toy_by_t()
        train = [r for t in (233, 343, 398) for r in by_t[t] if r.tag != "nmos_d0"]
        tasks = s3_loto_tasks(train, unit)
        self.assertEqual(len(tasks), 6)
        for t, k, inner, scored in tasks:
            self.assertTrue(all(r.meta["T_K"] == t for r in scored))
            self.assertFalse({r.tag for r in inner} & {r.tag for r in scored})
            self.assertNotIn(t, {r.meta["T_K"] for r in inner})


class TestStatistics(unittest.TestCase):
    def test_s1_loto(self):
        from cryoml.explore.i7_1_transfer import device_residual
        by_t = toy_by_t()
        recs = [r for t in (233, 343, 398) for r in by_t[t]]
        res = [device_residual(r) for r in recs]
        s = s1_loto(res, [r.meta["T_K"] for r in recs], [r.tag for r in recs])
        self.assertTrue(-5 < s < 1)
        # temperature-independent error: S1_loto equals the shared-error case near 1
        flat = toy_by_t()
        for t in TEMPS:
            for r in flat[t]:
                for c in r.curves:
                    c.target = c.sim * np.exp(shared(c.vg, c.vd, 300.0))
        recs2 = [r for t in (233, 343, 398) for r in flat[t]]
        s2 = s1_loto([device_residual(r) for r in recs2], [r.meta["T_K"] for r in recs2],
                     [r.tag for r in recs2])
        self.assertGreater(s2, s)
        self.assertGreater(s2, 0.9)

    def test_unit_log_ratios_and_gmr(self):
        m = {("a", 300): 1.0, ("a", 343): 4.0, ("b", 300): 2.0, ("b", 343): float("nan"),
             ("a", 233): 9.0}
        c = {("a", 300): 2.0, ("a", 343): 2.0, ("b", 300): 1.0, ("b", 343): 1.0,
             ("a", 233): 1.0}
        lr, counts = unit_log_ratios(m, c, (300, 343))
        self.assertAlmostEqual(lr["a"], 0.0)
        self.assertAlmostEqual(lr["b"], math.log(2.0))
        self.assertEqual(counts["pairs"], 4)
        self.assertEqual(counts["used"], 3)
        self.assertEqual(counts["method_failed"], 1)
        g = gmr_ci(lr, n_boot=200)
        self.assertAlmostEqual(g["ratio"], math.sqrt(2.0))
        gc = gmr_ci({"a": -0.1, "b": -0.2, "c": -0.3}, clusters={"a": 1, "b": 1, "c": 2},
                    level=0.99, n_boot=500)
        self.assertLess(gc["ci_high"], 1.0)
        self.assertEqual(gc["resample_unit"], "cluster (2)")

    def test_gmr_level_widens(self):
        rng = np.random.default_rng(1)
        lr = {f"u{i}": float(v) for i, v in enumerate(rng.normal(-0.1, 0.2, 20))}
        a, b = gmr_ci(lr, level=0.95), gmr_ci(lr, level=0.99375)
        self.assertLess(b["ci_low"], a["ci_low"])
        self.assertGreater(b["ci_high"], a["ci_high"])

    def test_labels_and_decision(self):
        lo = {"ratio": 0.8, "ci_low": 0.7, "ci_high": 0.9}
        inc = {"ratio": 1.0, "ci_low": 0.9, "ci_high": 1.1}
        hi = {"ratio": 1.2, "ci_low": 1.1, "ci_high": 1.3}
        self.assertEqual(label_ihp(lo, lo, lo), "supported")
        self.assertEqual(label_ihp(lo, inc, lo), "transfer without temperature input")
        self.assertEqual(label_ihp(inc, lo, lo), "no gain")
        self.assertEqual(label_ihp(hi, lo, lo), "loss")
        self.assertEqual(label_tud(lo), "transfer gain")
        self.assertEqual(label_tud(hi), "transfer loss")
        self.assertEqual(decide({"interp": "supported", "extrap": "loss"}, True)["outcome"],
                         "supported")
        self.assertEqual(decide({"interp": "transfer without temperature input",
                                 "extrap": "no gain"}, True)["outcome"],
                         "transfer without temperature input")
        self.assertEqual(decide({"interp": "no gain", "extrap": "loss"}, True)["outcome"],
                         "not supported")
        self.assertEqual(decide({"interp": "supported", "extrap": "supported"}, False)["outcome"],
                         "invalid")


if __name__ == "__main__":
    unittest.main()
