"""I7-2 hierarchical correction: basis, split rule, delta fits, prior, joint
fit problem, cross-curve control, aggregation and the decision rule."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.explore import i7_2_hier as h  # noqa: E402
from cryoml.explore.e1_correction import Correction, CurveRec, DeviceRec  # noqa: E402
from cryoml.parameter_space import ParameterBox  # noqa: E402

VDD = 1.2
SWEEP = np.round(np.linspace(0.0, VDD, 25), 6)          # 0.05 V steps
G_BIAS = (0.05, 0.6, 1.2)
D_BIAS = (0.3, 0.6, 0.9, 1.2)


def model_current(vg, vd, k=1e-4, vt=0.3):
    vov = np.maximum(np.abs(vg) - vt, 0.0) + 0.02
    vds = np.abs(vd)
    return k * vov * np.tanh(3 * vds / (vov + 0.1)) + 1e-9


def make_rec(tag="nmos_toy", family="nmos", truth_delta=None, sim_scale=1.0):
    sgn = -1.0 if family == "pmos" else 1.0
    rec = DeviceRec(tag, family, 1.0, 1.0, VDD, "toy")
    for kind, biases in (("idvg", G_BIAS), ("idvd", D_BIAS)):
        for b in biases:
            vg = SWEEP if kind == "idvg" else np.full(len(SWEEP), b)
            vd = np.full(len(SWEEP), b) if kind == "idvg" else SWEEP
            sim = model_current(vg, vd) * sim_scale
            tgt = model_current(vg, vd)
            if truth_delta is not None:
                phi = np.column_stack([np.zeros(len(vg)), np.abs(vg) / VDD, np.abs(vd) / VDD,
                                       np.zeros(len(vg)), np.zeros(len(vg)), np.zeros(len(vg))])
                tgt = tgt * np.exp(h.basis(phi) @ truth_delta)
            den = float(np.mean(np.abs(tgt[1:])))
            rec.curves.append(CurveRec(f"{kind}@{b:g}", kind, sgn * vg, sgn * vd, sgn * sim,
                                       sgn * tgt, 1, den, True))
    return rec


class TestBasisAndShift(unittest.TestCase):
    def test_basis_values(self):
        phi = np.array([[0, 0.5, 1.0, 0, 0, 0], [1, 0.0, 0.0, 0, 0, 0]], float)
        B = h.basis(phi)
        np.testing.assert_allclose(B[0], [1, 0, 1, 0, 0, 1])
        np.testing.assert_allclose(B[1], [1, -1, -1, 1, 1, 1])

    def test_shifted_correction_adds_basis(self):
        rec = make_rec()
        delta = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        out = h.ShiftedCorrection("x", base=Correction("C0"), delta=delta).apply(rec)
        for c, o in zip(rec.curves, out):
            np.testing.assert_allclose(o, c.sim * math.exp(0.1))


class TestSplitRule(unittest.TestCase):
    def test_draws_positions(self):
        rec = make_rec()
        d = h.draws(rec)
        self.assertEqual(d["K2a"]["adapt"], ["idvg@1.2", "idvd@1.2"])
        self.assertEqual(d["K2b"]["adapt"], ["idvg@0.05", "idvd@0.6"])     # (4-1)//2 = 1
        self.assertEqual(d["K4a"]["adapt"], ["idvg@0.05", "idvg@1.2", "idvd@0.6", "idvd@1.2"])
        # nG = 3: floor(2/3) = 0, ceil(4/3) = 2; nD = 4: floor(3/3) = 1, ceil(6/3) = 2
        self.assertEqual(d["K4b"]["adapt"], ["idvg@0.05", "idvg@1.2", "idvd@0.6", "idvd@0.9"])
        for v in d.values():
            self.assertEqual(len(set(v["adapt"]) & set(v["heldout"])), 0)
            self.assertEqual(len(v["adapt"]) + len(v["heldout"]), 7)

    def test_undefined_draws(self):
        rec = make_rec()
        for c in rec.curves:                       # leave one Id-Vd and two Id-Vg included
            if c.key in ("idvd@0.3", "idvd@0.6", "idvd@0.9", "idvg@0.6"):
                c.included = False
        d = h.draws(rec)
        self.assertIsNone(d["K4a"])                # two Id-Vd picks from one curve
        self.assertIsNone(d["K2a"])                # 3 included - 2 = 1 held-out curve < 2
        rec2 = make_rec()
        rec2.curves[0].included = False            # sort order ignores excluded curves
        self.assertEqual(h.ordered_keys(rec2)[0], ["idvg@0.6", "idvg@1.2"])

    def test_restrict_and_score_keys(self):
        rec = make_rec(sim_scale=1.1)
        r = h.restrict(rec, {"idvg@1.2"})
        self.assertEqual([c.key for c in r.curves if c.included], ["idvg@1.2"])
        self.assertTrue(all(c.included for c in rec.curves))      # original unchanged
        sc = h.score_keys(rec, [c.sim for c in rec.curves], {"idvg@1.2", "idvd@0.3"})
        self.assertEqual(sc["n_curves"], 2)
        self.assertGreater(sc["rrms"], 0.05)


class TestDeltaFits(unittest.TestCase):
    def test_fit_delta_full_recovers_truth(self):
        truth = np.array([0.2, -0.1, 0.15, 0.0, 0.05, -0.05])
        rec = make_rec(truth_delta=truth)
        ps = h.point_set(rec, Correction("C0"))
        d = h.fit_delta_full(ps)
        self.assertLess(h.mean_rrms(ps, d), 1e-3)
        self.assertLess(h.mean_rrms(ps, d), 0.05 * h.mean_rrms(ps, np.zeros(6)))

    def test_gradient_matches_finite_difference(self):
        rec = make_rec(truth_delta=np.array([0.1, 0.2, -0.1, 0.0, 0.0, 0.1]))
        ps = h.point_set(rec, Correction("C0"))
        d0 = np.array([0.01, -0.02, 0.03, 0.0, 0.01, 0.0])
        _, g = h.mean_rrms_and_grad(ps, d0)
        for k in range(6):
            e = np.zeros(6)
            e[k] = 1e-6
            fd = (h.mean_rrms_and_grad(ps, d0 + e)[0] - h.mean_rrms_and_grad(ps, d0 - e)[0]) / 2e-6
            self.assertAlmostEqual(g[k], fd, places=5)

    def test_prior_and_fit_z(self):
        rng = np.random.default_rng(0)
        deltas = rng.normal(scale=[0.2, 0.1, 0.1, 0.05, 0.05, 0.05], size=(20, 6)) + 0.1
        pr = h.prior_from_deltas(deltas)
        np.testing.assert_allclose(pr.m, deltas.mean(0))
        sig = np.cov(deltas, rowvar=False)
        np.testing.assert_allclose(np.diag(pr.S), np.diag(sig) + h.JITTER)
        np.testing.assert_allclose(pr.S[0, 1], 0.5 * sig[0, 1])
        np.testing.assert_allclose(pr.Lc @ pr.Lc.T, pr.S)
        rec = make_rec(truth_delta=deltas[3])
        ps = h.point_set(rec, Correction("C0"))
        z_free = h.fit_z(ps, pr, 0.0)
        self.assertLess(h.mean_rrms(ps, pr.delta(z_free)), 1e-3)
        z_big = h.fit_z(ps, pr, 1e3)
        self.assertLess(np.linalg.norm(z_big), np.linalg.norm(z_free))
        np.testing.assert_array_equal(h.fit_z(ps, pr, math.inf), np.zeros(6))

    def test_select_mu_runs(self):
        rng = np.random.default_rng(1)
        train = [make_rec(f"nmos_t{i}", truth_delta=rng.normal(scale=0.1, size=6))
                 for i in range(5)]
        out = h.select_mu(train, Correction("C0"), 2)
        self.assertIn(out["mu"], h.MU_GRID)
        self.assertEqual(out["n_train_devices"], 5)
        self.assertEqual(out["n_tasks"], 10)
        self.assertEqual(len(out["criterion"]), len(h.MU_GRID))


class FakeProblem:
    """ngspice_fit.polish interface: current = k * f(V) with k = 10^(local)."""

    def __init__(self, rec):
        self.tag = rec.tag
        self.rec = rec
        self.box = ParameterBox(names=("k",), published={"k": 1.0}, bounds={"k": [0.1, 10.0]},
                                log_scale=(True,))
        self.curves = list(rec.curves)
        self.layout = [{"index": i, "start": c.start, "tag": c.key,
                        "target": np.asarray(c.target, float)[c.start:], "denominator": c.den}
                       for i, c in enumerate(rec.curves)]
        self.n_curves = len(self.layout)
        self.n_sim = 0

    def simulate(self, local):
        self.n_sim += 1
        k = self.box.local_to_params(np.asarray(local, float))["k"]
        return [k * np.asarray(c.sim, float) for c in self.curves]

    def curve_rrms(self, sims):
        return np.array([math.sqrt(float(np.mean((np.asarray(sims[r["index"]])[r["start"]:]
                                                  - r["target"]) ** 2))) / r["denominator"]
                         for r in self.layout])

    def residual_vector(self, sims, weights):
        s = 1.0 / math.sqrt(self.n_curves)
        return np.concatenate([(np.asarray(sims[r["index"]])[r["start"]:] - r["target"]) * w * s
                               / (r["denominator"] * math.sqrt(len(r["target"])))
                               for r, w in zip(self.layout, weights)])


class FakeCorrected:
    def __init__(self, base, corr):
        self.base, self.corr = base, corr

    def record(self, sims):
        rec = DeviceRec(self.base.tag, "nmos", 1.0, 1.0, VDD, "toy")
        rec.curves = [CurveRec(c.key, c.kind, c.vg, c.vd, np.asarray(s, float), c.target,
                               0, 0.0, False) for c, s in zip(self.base.curves, sims)]
        return rec

    def simulate(self, local):
        return self.corr.apply(self.record(self.base.simulate(local)))


class TestJointProblem(unittest.TestCase):
    def test_subset_problem(self):
        prob = FakeProblem(make_rec())
        sub = h.subset_problem(prob, ["idvd@0.9", "idvg@0.05"])
        self.assertEqual(sub.n_curves, 2)
        self.assertEqual([c.key for c in sub.curves], ["idvg@0.05", "idvd@0.9"])
        self.assertEqual([r["index"] for r in sub.layout], [0, 1])
        self.assertEqual(len(sub.simulate(np.zeros(1))), 2)
        self.assertEqual(prob.n_curves, 7)
        with self.assertRaises(KeyError):
            h.subset_problem(prob, ["idvd@5"])

    def test_polish_prior_recovers_card_and_delta(self):
        truth = np.array([0.15, 0.0, 0.1, 0.0, 0.0, 0.0])
        rec = make_rec(truth_delta=truth, sim_scale=2.0)          # card k = 0.5 is correct
        sub = h.subset_problem(FakeProblem(rec), ["idvg@1.2", "idvd@1.2", "idvd@0.6"])
        prior = h.Prior(np.zeros(6), 0.04 * np.eye(6), 0.2 * np.eye(6), 10)
        hp = h.HierProblem(FakeCorrected(sub, Correction("C0")), Correction("C0"), prior, sub.box)
        self.assertEqual(hp.box.n_parameters, 7)
        res = h.polish_prior(hp, np.zeros(7), mu=0.0, max_nfev=400, irls_rounds=2)
        card, z = hp.split(res["local"])
        self.assertLess(res["objective"], 0.02)
        self.assertTrue(np.all(np.abs(z) <= h.Z_MAX))
        self.assertGreater(hp.n_cache_hits, 0)
        self.assertLess(hp.base.n_sim, 400)
        res_mu = h.polish_prior(hp, np.zeros(7), mu=10.0, max_nfev=400, irls_rounds=2)
        _, z_mu = hp.split(res_mu["local"])
        self.assertLessEqual(float(z_mu @ z_mu), float(z @ z) + 1e-9)


class TestCrossInterp(unittest.TestCase):
    def test_crossing_residual(self):
        rec = make_rec(sim_scale=1.0)
        held = rec.curves[4]                                       # idvd@0.6
        for c in rec.curves:                                       # measured = sim + 1e-6 everywhere
            c.target = np.asarray(c.target) + 1e-6
        sims, cross = h.crossinterp_sims(rec, ["idvg@1.2", "idvd@1.2"], ["idvd@0.6", "idvg@0.6"])
        self.assertEqual(cross["idvd@0.6"], 1)                     # idvg@1.2 crosses at Vd = 1.2
        self.assertEqual(cross["idvg@0.6"], 1)                     # idvd@1.2 crosses at Vg = 1.2
        corr = sims[4] - held.sim
        np.testing.assert_allclose(corr[-1], 1e-6, rtol=1e-6)      # at the crossing (Vd = 1.2)
        np.testing.assert_allclose(corr[0], 0.0, atol=1e-15)       # zero anchor at 0 V
        np.testing.assert_allclose(corr[12], 0.5e-6, rtol=1e-6)    # linear in between
        np.testing.assert_array_equal(sims[0], rec.curves[0].sim)  # adaptation curve unchanged


class TestAggregationAndDecision(unittest.TestCase):
    def test_device_values_geometric_mean(self):
        rows = [{"tag": "a", "draw": "K2a", "arms": {"c_hier": {"heldout_rrms": 0.1}}},
                {"tag": "a", "draw": "K4a", "arms": {"c_hier": {"heldout_rrms": 0.4}}},
                {"tag": "b", "draw": "K2a", "arms": {"c_hier": {"heldout_rrms": math.nan}}}]
        v = h.device_values(rows, "c_hier")
        self.assertAlmostEqual(v["a"], 0.2)
        self.assertTrue(math.isnan(v["b"]))
        self.assertAlmostEqual(h.device_values(rows, "c_hier", 4)["a"], 0.4)

    def test_decide_outcomes(self):
        def r(p, hi):
            return {"ratio": p, "ci_high": hi}
        good = {"c_vs_b": r(0.8, 0.9), "c_vs_a": r(0.7, 0.8), "c_vs_d": r(0.8, 0.95)}
        ok = {"n": 12, "n_failed": 0, "n_paired": 12}
        ds = h.RULE_DATASETS
        out = h.decide({d: good for d in ds}, {d: ok for d in ds})
        self.assertEqual(out["outcome"], "supported_continue")
        nog = {"c_vs_b": r(0.99, 1.1), "c_vs_a": r(0.7, 0.8), "c_vs_d": r(0.8, 0.95)}
        out = h.decide({ds[0]: nog, ds[1]: nog, ds[2]: good, ds[3]: good}, {d: ok for d in ds})
        self.assertEqual(out["outcome"], "delta_no_gain_stop")
        share = {"c_vs_b": r(0.8, 0.9), "c_vs_a": r(0.7, 0.8), "c_vs_d": r(1.05, 1.2)}
        out = h.decide({ds[0]: share, ds[1]: good, ds[2]: good, ds[3]: good}, {d: ok for d in ds})
        self.assertEqual(out["outcome"], "sharing_no_gain")
        bad = {"n": 12, "n_failed": 5, "n_paired": 7}
        out = h.decide({d: good for d in ds}, {ds[0]: bad, ds[1]: bad, ds[2]: ok, ds[3]: ok})
        self.assertEqual(out["outcome"], "invalid")
        out = h.decide({ds[0]: good}, {ds[0]: ok})
        self.assertEqual(out["outcome"], "incomplete")


if __name__ == "__main__":
    unittest.main()
