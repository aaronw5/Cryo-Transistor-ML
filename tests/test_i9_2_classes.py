"""I9-2 function-class ablation: class registry, spline basis, GAM class,
fitting dispatch, ensemble members, pooled statistic, decision rule,
learning-curve subsets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.explore.e1_correction import (Correction, CurveRec, DeviceRec,  # noqa: E402
                                          _loss_fn, make_batch, raw_features)
from cryoml.explore.i3_2_bias import fit_bias, paired_ratio  # noqa: E402
from cryoml.explore.i9_2_classes import (  # noqa: E402
    CLASS_SPECS, CONTRASTS, REFERENCE, REPLICATE, GAMCorrection, LegendreCorrection,
    bonferroni_alpha, bspline_basis, correction_from_json_i9, decide, ensemble_member,
    fit_class, fit_gam, label_class, lc_classes, legendre_1d, legendre_features, n_params,
    n_params_member, pooled_ratio, select_lc_classes, training_loss, training_subsets)

VDD = 1.85
GRID = np.linspace(0.05, VDD, 30)
BIASES = (("idvg", 0.1), ("idvg", 1.85), ("idvd", 0.74), ("idvd", 1.85))


def shared(vg, vd, family):
    u, v = np.abs(vg) / VDD, np.abs(vd) / VDD
    base = 0.3 * v - 0.2 * u + 0.15 * np.sin(3 * u) * v
    return base + (0.1 if family == "pmos" else -0.05)


def make_device(i, family, L, W):
    sgn = -1.0 if family == "pmos" else 1.0
    rec = DeviceRec(f"{family}_toy{i}", family, L, W, VDD, "toy")
    for kind, bias in BIASES:
        if kind == "idvg":
            vg, vd = sgn * GRID, np.full_like(GRID, sgn * bias)
        else:
            vg, vd = np.full_like(GRID, sgn * bias), sgn * GRID
        sim = sgn * (W / L) * 1e-4 * np.log1p(np.exp(8 * (np.abs(vg) - 0.4))) * np.tanh(3 * np.abs(vd))
        tgt = sim * np.exp(shared(vg, vd, family))
        rec.curves.append(CurveRec(f"{kind}@{bias:g}", kind, vg, vd, sim, tgt, 0,
                                   float(np.mean(np.abs(tgt))), True))
    return rec


def toy_devices(n=4):
    return [make_device(i, "pmos" if i % 2 else "nmos", 1.0 + i, 2.0) for i in range(n)]


def mean_loss(corr, recs):
    import torch
    batch = make_batch(recs)
    g = corr.g(batch.phi) if corr.arm != "C0" else np.zeros(len(batch.phi))
    return float(_loss_fn(batch)(torch.tensor(g)))


class TestRegistry(unittest.TestCase):
    def test_param_counts(self):
        want = {"const": 2, "lin": 6, "poly2": 12, "poly3": 20, "poly4": 30, "poly5": 42,
                "gam": 66, "mlp8": 565, "mlp16": 1765, "mlp16_s1": 353, "mlp32": 6085,
                "gp": 2006, "mlp16_rep": 1765}
        self.assertEqual({c: n_params(c) for c in CLASS_SPECS}, want)
        self.assertEqual(n_params_member("mlp16"), 353)
        self.assertEqual(n_params_member("mlp8"), 113)
        self.assertEqual(n_params_member("poly5"), 42)

    def test_contrasts(self):
        self.assertEqual(REFERENCE, "mlp16")
        self.assertEqual(REPLICATE, "mlp16_rep")
        self.assertEqual(len(CONTRASTS), 11)
        self.assertNotIn(REFERENCE, CONTRASTS)
        self.assertNotIn(REPLICATE, CONTRASTS)
        self.assertNotIn("mlp64", CLASS_SPECS)
        self.assertEqual(CLASS_SPECS[REPLICATE]["seeds"], (5, 6, 7, 8, 9))
        self.assertAlmostEqual(bonferroni_alpha(), 0.05 / 11)


class TestLegendre(unittest.TestCase):
    def test_polynomials(self):
        x = np.linspace(-1, 1, 7)
        p = legendre_1d(x, 5)
        np.testing.assert_allclose(p[2], (3 * x ** 2 - 1) / 2, atol=1e-14)
        np.testing.assert_allclose(p[5], (63 * x ** 5 - 70 * x ** 3 + 15 * x) / 8, atol=1e-14)
        # orthogonality on [-1, 1] (Gauss-Legendre quadrature)
        g, w = np.polynomial.legendre.leggauss(10)
        q = legendre_1d(g, 5)
        gram = np.array([[np.sum(w * a * b) for b in q] for a in q])
        np.testing.assert_allclose(gram, np.diag(2.0 / (2 * np.arange(6) + 1)), atol=1e-13)

    def test_features_and_fit(self):
        f = legendre_features(np.array([0.3]), np.array([-0.5]), 4)
        self.assertEqual(f.shape, (1, 15))
        self.assertAlmostEqual(f[0, 0], 1.0)
        recs = toy_devices(4)
        for name in ("poly4", "poly5"):
            corr = fit_class(name, recs, {"steps": 150})
            self.assertIsInstance(corr, LegendreCorrection)
            self.assertEqual(corr.coef.size, n_params(name))
            self.assertLess(mean_loss(corr, recs), mean_loss(Correction("C0"), recs))
            self.assertAlmostEqual(training_loss(corr, recs), mean_loss(corr, recs), places=12)
            back = correction_from_json_i9(corr.to_json())
            self.assertIsInstance(back, LegendreCorrection)
            batch = make_batch(recs)
            np.testing.assert_array_equal(back.g(batch.phi), corr.g(batch.phi))


class TestSpline(unittest.TestCase):
    def test_partition_of_unity_and_clip(self):
        x = np.linspace(-0.5, 1.5, 401)
        B = bspline_basis(x, 0.0, 1.0, 8)
        self.assertEqual(B.shape, (401, 8))
        self.assertTrue(np.all(B >= -1e-15))
        np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(B[x <= 0.0], np.tile(B[x <= 0.0][0], (np.sum(x <= 0.0), 1)))
        np.testing.assert_allclose(B[x >= 1.0][:, -1], 1.0)
        np.testing.assert_allclose(B[x <= 0.0][:, 0], 1.0)

    def test_reproduces_cubic(self):
        # a cubic polynomial lies in the span of the cubic B-spline basis
        x = np.linspace(0.0, 1.0, 200)
        B = bspline_basis(x, 0.0, 1.0, 8)
        y = 1.0 - 2.0 * x + 0.5 * x ** 3
        coef, *_ = np.linalg.lstsq(B, y, rcond=None)
        np.testing.assert_allclose(B @ coef, y, atol=1e-10)


class TestGAM(unittest.TestCase):
    def setUp(self):
        self.recs = toy_devices(4)

    def test_zero_and_polarity(self):
        k, r = 8, 1
        z = GAMCorrection("gam", lo=np.zeros(2), hi=np.ones(2), n_basis=k, rank=r,
                          c=np.zeros(2), a=np.zeros((2, k)), b=np.zeros((2, k)),
                          p=np.zeros((2, r, k)), q=np.zeros((2, r, k)))
        phi = raw_features(self.recs[0], self.recs[0].curves[0].vg, self.recs[0].curves[0].vd,
                           self.recs[0].curves[0].sim)
        np.testing.assert_array_equal(z.g(phi), 0.0)
        z.c = np.array([0.0, 0.4])           # pMOS-only offset
        np.testing.assert_array_equal(z.g(phi), 0.0)    # recs[0] is nMOS
        phi_p = phi.copy()
        phi_p[:, 0] = 1.0
        np.testing.assert_allclose(z.g(phi_p), np.tanh(0.4))

    def test_fit_reduces_loss_and_json(self):
        corr = fit_gam("gam", self.recs, hyper={"steps": 300})
        self.assertLess(mean_loss(corr, self.recs), 0.5 * mean_loss(Correction("C0"), self.recs))
        back = correction_from_json_i9(corr.to_json())
        self.assertIsInstance(back, GAMCorrection)
        batch = make_batch(self.recs)
        np.testing.assert_allclose(back.g(batch.phi), corr.g(batch.phi), rtol=0, atol=1e-14)


class TestFitClass(unittest.TestCase):
    def setUp(self):
        self.recs = toy_devices(4)

    def test_const_is_constant_per_polarity(self):
        corr = fit_class("const", self.recs, {"steps": 200})
        batch = make_batch(self.recs)
        g = corr.g(batch.phi)
        for p in (0.0, 1.0):
            sel = batch.phi[:, 0] == p
            self.assertLess(np.ptp(g[sel]), 1e-12)
        self.assertNotAlmostEqual(g[batch.phi[:, 0] == 0][0], g[batch.phi[:, 0] == 1][0])

    def test_lin_and_poly_degrees(self):
        for name in ("const", "lin", "poly2", "poly3"):
            corr = fit_class(name, self.recs, {"steps": 100})
            self.assertEqual(corr.degree, CLASS_SPECS[name]["degree"])
            self.assertEqual(corr.coef.size, n_params(name))
            self.assertLess(mean_loss(corr, self.recs), mean_loss(Correction("C0"), self.recs))

    def test_mlp_width_and_seeds(self):
        corr = fit_class("mlp8", self.recs, {"steps": 30, "seeds": (0, 1)})
        self.assertEqual(len(corr.nets), 2)
        self.assertEqual(corr.nets[0][0][0].shape, (3, 8))
        one = fit_class("mlp16_s1", self.recs, {"steps": 30, "seeds": (0, 1, 2)})
        self.assertEqual(len(one.nets), 1)             # 1-seed class ignores the override
        total = sum(w.size + b.size for w, b in one.nets[0])
        self.assertEqual(total, n_params("mlp16_s1"))

    def test_member_zero_equals_one_seed_fit(self):
        hyper = {"steps": 40}
        ens = fit_bias("M_bias", self.recs, hyper={**hyper, "seeds": (0, 1)})
        one = fit_bias("M_bias", self.recs, hyper={**hyper, "seeds": (0,)})
        mem = ensemble_member(ens, 0, arm="mlp16_s1")
        batch = make_batch(self.recs)
        np.testing.assert_array_equal(mem.g(batch.phi), one.g(batch.phi))
        self.assertEqual(mem.arm, "mlp16_s1")
        self.assertEqual(len(mem.nets), 1)

    def test_trace_does_not_change_fit(self):
        for name, ov in (("poly3", {"steps": 500}), ("poly4", {"steps": 500}),
                         ("gam", {"steps": 500}), ("mlp8", {"steps": 500, "seeds": (0, 1)})):
            tr = []
            a = fit_class(name, self.recs, ov, trace=tr)
            b = fit_class(name, self.recs, ov)
            batch = make_batch(self.recs)
            np.testing.assert_array_equal(a.g(batch.phi), b.g(batch.phi))
            n_mem = len(ov.get("seeds", (0,)))
            self.assertEqual(len(tr), 2 * n_mem)
            self.assertEqual([t[1] for t in tr[:2]], [250, 500])

    def test_gp_reduced(self):
        corr = fit_class("gp", self.recs, {"gp_total": 120, "gp_maxiter": 15})
        self.assertEqual(corr.info["n_points"], 120)
        back = correction_from_json_i9(corr.to_json())
        batch = make_batch(self.recs)
        np.testing.assert_allclose(back.g(batch.phi[:50]), corr.g(batch.phi[:50]))


class TestStatistics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.c = {f"d{i}": float(x) for i, x in enumerate(rng.uniform(0.05, 0.2, 30))}
        self.m = {t: v * float(np.exp(rng.normal(-0.1, 0.05))) for t, v in self.c.items()}

    def test_single_dataset_matches_paired_ratio(self):
        p = pooled_ratio({"A": (self.m, self.c)}, n_boot=4000)
        q = paired_ratio(self.m, self.c, n_boot=4000)
        self.assertAlmostEqual(p["ratio"], q["ratio"], places=12)
        self.assertLess(p["ci_low"], p["ratio"])
        self.assertGreater(p["ci_high"], p["ratio"])

    def test_equal_weight_and_alpha(self):
        c2 = {k: v for k, v in list(self.c.items())[:6]}
        m2 = {k: 1.2 * v for k, v in c2.items()}
        p = pooled_ratio({"A": (self.m, self.c), "B": (m2, c2)}, n_boot=4000)
        a = np.mean(np.log([self.m[t] / self.c[t] for t in self.c]))
        self.assertAlmostEqual(p["ratio"], float(np.exp((a + np.log(1.2)) / 2)), places=12)
        wide = pooled_ratio({"A": (self.m, self.c), "B": (m2, c2)}, n_boot=4000, alpha=0.05 / 12)
        self.assertLessEqual(wide["ci_low"], p["ci_low"])
        self.assertGreaterEqual(wide["ci_high"], p["ci_high"])

    def test_nan_excluded_and_counted(self):
        m = dict(self.m)
        m["d0"] = float("nan")
        p = pooled_ratio({"A": (m, self.c)}, n_boot=500)
        self.assertEqual(p["per_dataset"]["A"]["n_method_failed"], 1)
        self.assertEqual(p["per_dataset"]["A"]["n_used"], 29)

    def test_clusters(self):
        cl = {"A": {t: f"g{i // 3}" for i, t in enumerate(sorted(self.c))}}
        p = pooled_ratio({"A": (self.m, self.c)}, clusters=cl, n_boot=2000)
        self.assertTrue(np.isfinite(p["ci_low"]) and np.isfinite(p["ci_high"]))

    def test_labels(self):
        f = lambda lo, hi: {"ci_low": lo, "ci_high": hi}
        self.assertEqual(label_class(f(0.9, 0.99)), "beats")
        self.assertEqual(label_class(f(1.01, 1.2)), "worse")
        self.assertEqual(label_class(f(0.98, 1.02)), "equivalent")
        self.assertEqual(label_class(f(0.95, 1.02)), "non-inferior")
        self.assertEqual(label_class(f(0.95, 1.06)), "inconclusive")
        self.assertEqual(label_class(f(float("nan"), 1.0)), "invalid")

    def test_decide(self):
        def cmp(r, lo, hi, per=(1.0, 1.0)):
            return {"ratio": r, "ci_low": lo, "ci_high": hi,
                    "per_dataset": {f"D{i}": {"ratio": x} for i, x in enumerate(per)}}
        pooled = {"const": cmp(1.2, 1.1, 1.3), "lin": cmp(1.08, 1.02, 1.12),
                  "poly3": cmp(1.0, 0.98, 1.02, (1.01, 1.04)),
                  "gam": cmp(0.99, 0.97, 1.01, (0.97, 1.06)),      # fails per-dataset
                  "mlp32": cmp(0.99, 0.98, 1.0)}
        out = decide(pooled, {"poly3": {"D0": 0, "D1": 0}})
        self.assertEqual(out["Q1"], "class-dependent")
        self.assertEqual(out["Q2_smallest_matching_class"], "poly3")
        self.assertNotIn("gam", out["Q2_candidates"])
        self.assertNotIn("mlp32", out["Q2_candidates"])   # more parameters than M_bias
        out2 = decide(pooled, {"poly3": {"D0": 1}})
        self.assertEqual(out2["Q2_smallest_matching_class"], "none")
        eq = decide({"poly3": cmp(1.0, 0.99, 1.01)}, {})
        self.assertEqual(eq["Q1"], "no class dependence detected")
        rep = {"ratio": 1.0, "ci_low": 0.95, "ci_high": 1.10}
        out3 = decide(pooled, {}, replicate=rep)
        self.assertEqual(out3["labels"]["lin"], "worse")
        self.assertIn("lin", out3["notes"])             # 1.08 inside [0.95, 1.10]
        self.assertNotIn("const", out3["notes"])        # 1.2 outside
        self.assertNotIn("poly3", out3["notes"])        # not beats / worse
        self.assertEqual(out3["n_params_per_member"]["mlp32"], 1217)


class TestLearningCurve(unittest.TestCase):
    def test_subsets(self):
        pool = {f"n{i}": "nmos" for i in range(10)}
        pool.update({f"p{i}": "pmos" for i in range(3)})
        for n in (2, 4, 8):
            subs = training_subsets(pool, n, reps=3, key=(5,))
            self.assertEqual(len(subs), 3)
            for s in subs:
                self.assertEqual(len(s), n)
                self.assertEqual(len(set(s)), n)
                self.assertTrue(set(s) <= set(pool))
            self.assertEqual(subs, training_subsets(pool, n, reps=3, key=(5,)))
        s8 = training_subsets(pool, 8, reps=1)[0]
        self.assertEqual(sum(t.startswith("p") for t in s8), 3)     # pMOS short: all 3
        s2 = training_subsets(pool, 2, reps=1)[0]
        self.assertEqual(sum(t.startswith("p") for t in s2), 1)
        self.assertEqual(training_subsets(pool, 13), [sorted(pool)])
        self.assertNotEqual(training_subsets(pool, 4, key=(1,)), training_subsets(pool, 4, key=(2,)))

    def test_select(self):
        v = lambda r: {"ratio": r}
        out = select_lc_classes({"mlp16": v(0.800), "poly3": v(0.804), "gam": v(0.81),
                                 "const": v(0.95)})
        self.assertEqual(out, ["poly3", "mlp16"])
        out = select_lc_classes({"mlp16": v(0.80), "poly3": v(0.83), "gam": v(0.81)})
        self.assertEqual(out, ["mlp16", "gam"])
        self.assertEqual(lc_classes({"mlp16": v(0.80), "poly3": v(0.83), "gam": v(0.81)}),
                         ["mlp16", "gam"])
        self.assertEqual(lc_classes({"mlp16": v(0.85), "poly3": v(0.80), "gam": v(0.81),
                                     "mlp16_rep": v(0.70)}), ["mlp16", "poly3", "gam"])


if __name__ == "__main__":
    unittest.main()
