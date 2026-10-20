"""I3-1 hybrid law, GMR statistic and decision rule on synthetic data (no
NGSpice)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from cryoml.explore import e3_temp as E
from cryoml.explore import i3_hybrid as I

ANCHORS = E.OBS_SETS["O4"]
TEMPS = E.TEMPS_DENSE


def _residual(temp):
    w = float(E.w_of(temp))
    return 0.15 * math.sin(math.pi * w) * np.array([1.0, -1.0, 0.5, 0.0, 0.0, 0.8, 0.0])


def _device(rng, kind, shared_residual):
    law = {"mu": rng.uniform(-0.5, 0.5, E.N_PAR).tolist(),
           "c": rng.uniform(-0.1, 0.1, E.N_PAR).tolist(),
           "lg": rng.uniform(-0.8, 0.8, E.N_PAR).tolist()}
    fits = {}
    for t in TEMPS:
        x = I.law_base(law, t) + (_residual(t) if shared_residual else 0.0)
        fits[f"{t:g}"] = {"x": x.tolist(), "H": np.eye(E.N_PAR).tolist(), "rrms": 0.01}
    rec = {"geometry": [kind, rng.uniform(-1, 1), rng.uniform(-1, 1)], "fits": fits}
    return rec, law


def _population(n, shared_residual, seed=0):
    rng = np.random.default_rng(seed)
    pairs = [_device(rng, (-1.0, 1.0)[i % 2], shared_residual) for i in range(n)]
    return [p[0] for p in pairs], [p[1] for p in pairs]


class HybridLawTests(unittest.TestCase):
    def test_recovers_par3_when_residual_is_zero(self):
        recs, laws = _population(10, False)
        model = I.HybridLaw(recs, laws, ANCHORS, TEMPS, seeds=2, steps=200)
        test_rec, test_law = _device(np.random.default_rng(99), 1.0, False)
        ax = np.array([test_rec["fits"][f"{t:g}"]["x"] for t in ANCHORS])
        pred = model.predict(test_rec["geometry"], ax, test_law, list(E.TEMPS_TEST))
        base = np.array([I.law_base(test_law, t) for t in E.TEMPS_TEST])
        self.assertTrue(np.allclose(pred, base, atol=1e-7))

    def test_learns_shared_residual(self):
        recs, laws = _population(20, True)
        model = I.HybridLaw(recs, laws, ANCHORS, TEMPS, seeds=2, steps=800)
        test_rec, test_law = _device(np.random.default_rng(7), -1.0, True)
        ax = np.array([test_rec["fits"][f"{t:g}"]["x"] for t in ANCHORS])
        temps = list(E.TEMPS_HELDOUT)
        pred = model.predict(test_rec["geometry"], ax, test_law, temps)
        truth = np.array([test_rec["fits"][f"{t:g}"]["x"] for t in temps])
        base = np.array([I.law_base(test_law, t) for t in temps])
        err_h = np.sqrt(np.mean((pred - truth) ** 2))
        err_p = np.sqrt(np.mean((base - truth) ** 2))
        self.assertLess(err_h, 0.5 * err_p)
        self.assertEqual(len(model.history), 2)

    def test_learns_geometry_dependent_residual(self):
        """Residual amplitude 1 + 0.8 g1 (second geometry feature)."""
        recs, laws = _population(40, False, seed=3)
        for rec, law in zip(recs, laws):
            amp = 1.0 + 0.8 * rec["geometry"][1]
            for t in TEMPS:
                rec["fits"][f"{t:g}"]["x"] = (I.law_base(law, t) + amp * _residual(t)).tolist()
        model = I.HybridLaw(recs, laws, ANCHORS, TEMPS, seeds=2, steps=1500)
        for h in model.history:
            self.assertIn(h["feature_set"], I.FEATURE_SETS)
            self.assertEqual(set(h["candidates"]), set(I.FEATURE_SETS))
        errs_h, errs_p = [], []
        for seed in range(5):
            rec, law = _device(np.random.default_rng(100 + seed), 1.0, False)
            amp = 1.0 + 0.8 * rec["geometry"][1]
            temps = list(E.TEMPS_HELDOUT)
            truth = np.array([I.law_base(law, t) + amp * _residual(t) for t in temps])
            ax = np.array([I.law_base(law, t) + amp * _residual(t) for t in ANCHORS])
            pred = model.predict(rec["geometry"], ax, law, temps)
            base = np.array([I.law_base(law, t) for t in temps])
            errs_h.append(np.mean((pred - truth) ** 2))
            errs_p.append(np.mean((base - truth) ** 2))
        self.assertLess(np.sqrt(np.mean(errs_h)), 0.3 * np.sqrt(np.mean(errs_p)))

    def test_anchor_residuals(self):
        _, law = _device(np.random.default_rng(1), 1.0, False)
        ax = np.array([I.law_base(law, t) for t in ANCHORS]) + 0.5
        self.assertTrue(np.allclose(I.anchor_residuals(ax, law, ANCHORS), 0.5))

    def test_mean_law_exact_on_grid(self):
        recs, laws = _population(6, True)
        model = I.HybridMeanLaw(recs, laws, ANCHORS, TEMPS)
        test_rec, test_law = _device(np.random.default_rng(5), 1.0, True)
        pred = model.predict(test_rec["geometry"], None, test_law, list(TEMPS))
        truth = np.array([test_rec["fits"][f"{t:g}"]["x"] for t in TEMPS])
        self.assertTrue(np.allclose(pred, truth, atol=1e-12))

    def test_rows_skip_nonfinite(self):
        recs, laws = _population(2, False)
        recs[0]["fits"]["110"]["rrms"] = float("nan")
        recs[1]["fits"]["190"]["H"] = np.full((E.N_PAR, E.N_PAR), np.nan).tolist()
        rows = I.hybrid_rows(recs, laws, ANCHORS, TEMPS)
        self.assertEqual(len(rows), 2 * len(TEMPS) - 2)

    def test_sigma_floor(self):
        recs, laws = _population(4, False)
        sigma = I.residual_sigma(I.hybrid_rows(recs, laws, ANCHORS, TEMPS))
        self.assertTrue(np.allclose(sigma, 0.02))


class GmrTests(unittest.TestCase):
    def test_values(self):
        a = np.array([0.0, 0.1, 0.4, np.nan])
        b = np.array([0.0, 0.2, 0.2, 0.3])
        out = I.gmr(a, b)
        expect = math.exp(np.mean(np.log((a[:3] + 0.01) / (b[:3] + 0.01))))
        self.assertAlmostEqual(out["gmr"], expect)
        self.assertEqual(out["n_devices"], 3)
        self.assertEqual(out["wins"], 1)
        self.assertLessEqual(out["ci95"][0], out["gmr"] + 1e-12)
        self.assertGreaterEqual(out["ci95"][1], out["gmr"] - 1e-12)

    def test_identity(self):
        a = np.array([0.1, 0.0, 0.3])
        out = I.gmr(a, a)
        self.assertAlmostEqual(out["gmr"], 1.0)
        self.assertEqual(out["wins"], 0)
        self.assertEqual(out["ci95"], [1.0, 1.0])

    def test_empty(self):
        out = I.gmr([np.nan], [0.1])
        self.assertEqual(out["n_devices"], 0)
        self.assertTrue(math.isnan(out["gmr"]))


def _g(value, lo, hi):
    return {"gmr": value, "ci95": [lo, hi], "wins": 0, "n_devices": 36}


def _tier(par3, lrn_mean=(0.8, 0.7, 0.9), interp=(0.8, 0.7, 0.9), valid=True, fail=0.0):
    return {"valid": valid,
            "failed_fraction": {m: fail for m in ("hyb", "par3", "lrn_mean", "pt_interp")},
            "gmr": {"hyb_vs_par3": _g(*par3), "hyb_vs_lrn_mean": _g(*lrn_mean),
                    "hyb_vs_pt_interp": _g(*interp)}}


class DecideTests(unittest.TestCase):
    def _tiers(self, **kw):
        tiers = {"M": _tier((0.7, 0.6, 0.85)), "W": _tier((1.05, 0.98, 1.2)),
                 "H": _tier((0.8, 0.7, 0.95))}
        tiers.update(kw)
        return tiers

    def test_continue(self):
        out = I.decide(self._tiers())
        self.assertEqual(out["verdict_rule_output"], "CONTINUE")
        self.assertTrue(all(out["continue_conditions"].values()))

    def test_kill_on_w(self):
        out = I.decide(self._tiers(W=_tier((1.6, 1.3, 2.0))))
        self.assertEqual(out["verdict_rule_output"], "KILL")
        self.assertFalse(out["continue_conditions"]["W1"])

    def test_kill_on_m(self):
        out = I.decide(self._tiers(M=_tier((0.96, 0.8, 1.1))))
        self.assertEqual(out["verdict_rule_output"], "KILL")

    def test_kill_on_h(self):
        out = I.decide(self._tiers(H=_tier((1.0, 0.9, 1.1))))
        self.assertEqual(out["verdict_rule_output"], "KILL")

    def test_inconclusive_when_lrn_mean_not_beaten(self):
        out = I.decide(self._tiers(M=_tier((0.7, 0.6, 0.85), lrn_mean=(0.95, 0.85, 1.05))))
        self.assertEqual(out["verdict_rule_output"], "INCONCLUSIVE")
        self.assertFalse(out["continue_conditions"]["M2"])

    def test_w_margin_boundary(self):
        out = I.decide(self._tiers(W=_tier((1.1, 1.0, 1.25))))
        self.assertTrue(out["continue_conditions"]["W1"])
        out = I.decide(self._tiers(W=_tier((1.1, 1.0, 1.2501))))
        self.assertFalse(out["continue_conditions"]["W1"])
        self.assertEqual(out["verdict_rule_output"], "INCONCLUSIVE")

    def test_failed_cells(self):
        out = I.decide(self._tiers(H=_tier((0.8, 0.7, 0.95), fail=0.11)))
        self.assertFalse(out["continue_conditions"]["F"])
        self.assertEqual(out["verdict_rule_output"], "INCONCLUSIVE")

    def test_invalid(self):
        out = I.decide(self._tiers(H=_tier((0.8, 0.7, 0.95), valid=False)))
        self.assertEqual(out["verdict_rule_output"], "INVALID")


if __name__ == "__main__":
    unittest.main()
