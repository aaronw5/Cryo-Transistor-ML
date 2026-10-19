"""Alternating refit/retrain loop of I2-1 on a toy two-parameter device model."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.optimize import least_squares

from cryoml.explore.e1_correction import Correction, CurveRec, DeviceRec, fit_arm
from cryoml.explore.i2_1_joint import (CorrectedProblem, Fold, correlation_stats,
                                       joint_rounds, log_ratio, merge_train_results,
                                       split_train_tasks, with_sims)

VDD = 1.0
GRID = np.linspace(0.05, 1.0, 25)
BIASES = (("idvg", 0.3), ("idvg", 1.0), ("idvd", 0.6), ("idvd", 1.0))


def _sp(x):
    return np.log1p(np.exp(10.0 * x)) / 10.0


def model(vg, vd, params, L, W):
    return np.exp(params[0]) * (W / L) * _sp(vg - params[1]) ** 2 * np.tanh(3.0 * vd)


def shared_error(vg, vd):
    return 0.25 * vd - 0.2 * vg


def grids():
    for kind, bias in BIASES:
        if kind == "idvg":
            yield kind, bias, GRID, np.full_like(GRID, bias)
        else:
            yield kind, bias, np.full_like(GRID, bias), GRID


def make_device(i, true_params, L, W):
    rec = DeviceRec(f"toy{i}", "nmos", L, W, VDD, "toy", meta={"true": true_params})
    for kind, bias, vg, vd in grids():
        tgt = model(vg, vd, true_params, L, W) * np.exp(shared_error(vg, vd))
        rec.curves.append(CurveRec(f"{kind}@{bias:g}", kind, vg, vd, np.zeros_like(vg),
                                   tgt, 0, float(np.mean(np.abs(tgt))), True))
    return rec


def simulate(rec, params):
    return [model(c.vg, c.vd, params, rec.L_um, rec.W_um) for c in rec.curves]


def refit(corr, rec, start=None):
    """Least-squares refit of (ln a, vt) with ``corr`` applied to the model."""
    x0 = np.asarray(rec.meta["params"] if start is None else start, float)

    def resid(x):
        trial = with_sims(rec, simulate(rec, x))
        sims = corr.apply(trial)
        return np.concatenate([(s - c.target) / (c.den * np.sqrt(len(s)))
                               for s, c in zip(sims, rec.curves)])
    x = least_squares(resid, x0, method="lm", max_nfev=400).x
    if np.sum(resid(x) ** 2) > np.sum(resid(x0) ** 2):   # best seen, as ngspice_fit.polish
        x = x0
    return with_sims(rec, simulate(rec, x), params=[float(v) for v in x], refit_n_sim=0)


def toy_devices(n=7, seed=0):
    rng = np.random.default_rng(seed)
    devs = []
    for i in range(n):
        true = np.array([rng.uniform(-0.5, 0.5), rng.uniform(0.1, 0.5)])
        L, W = float(rng.choice([0.5, 1.0, 2.0])), float(rng.choice([1.0, 4.0]))
        dev = make_device(i, true, L, W)
        dev.meta["params"] = [float(true[0]) + 0.3, float(true[1]) - 0.1]
        devs.append(refit(Correction("C0"), dev))       # card fit without correction
    return devs


def serial_refit(tasks):
    return [refit(corr, rec) for corr, rec in tasks]


def serial_train(tasks):
    return [fit_arm(arm, recs, threads=1) for arm, recs in tasks]


class JointRoundsTest(unittest.TestCase):

    def test_k0_keeps_initial_correction_and_records(self):
        devs = toy_devices(4)
        g0 = Correction("T1", table={"nmos|idvg@0.3": 0.1})
        calls = []
        fold = Fold("f", "T1", devs, corr=g0)
        joint_rounds([fold], 0, lambda t: calls.append(t) or [],
                     lambda t: calls.append(t) or [])
        assert calls == []
        assert fold.corrs == [g0] and fold.train == devs
        assert len(fold.history) == 1 and len(fold.cards) == 1

    def test_round_counts_and_order(self):
        devs = toy_devices(5)
        folds = [Fold("a", "T1", devs[:3]), Fold("b", "T1", devs[2:])]
        n_refit, n_train = [], []

        def refit_map(tasks):
            n_refit.append(len(tasks))
            return [with_sims(rec, [c.sim for c in rec.curves]) for _, rec in tasks]

        def train_map(tasks):
            n_train.append(len(tasks))
            return serial_train(tasks)

        joint_rounds(folds, 2, refit_map, train_map)
        assert n_refit == [6, 6]                 # 3 + 3 training records per round
        assert n_train == [2, 2, 2]              # round 0 (no g_0 given) + 2 rounds
        for f in folds:
            assert len(f.corrs) == 3 and len(f.history) == 3 and len(f.cards) == 3
            assert [r.tag for r in f.train] == [d.tag for d in (devs[:3] if f.name == "a" else devs[2:])]

    def test_refit_step_descends_and_final_correction_transfers(self):
        devs = toy_devices(7)
        held, train = devs[0], devs[1:]
        fold = Fold("joint", "LIN", list(train))
        joint_rounds([fold], 3, serial_refit, serial_train)
        h = fold.history
        # the warm-started refit with g_{r-1} never raises the training objective
        # of g_{r-1} (a descent step from c_{r-1})
        for r in range(1, 4):
            self.assertLessEqual(h[r]["train_rrms_prev_corr_on_refit_cards"],
                                 h[r - 1]["train_rrms_corrected"] + 1e-9)
        # the loop refits the cards
        self.assertNotEqual(fold.cards[1], fold.cards[0])
        # held-out device: refit with frozen g_K is far below the uncorrected refit
        from cryoml.explore.i2_1_joint import corrected_score
        c0 = corrected_score(Correction("C0"), refit(Correction("C0"), held))
        gk = corrected_score(fold.corr, refit(fold.corr, held))
        self.assertLess(gk, 0.5 * c0)
        # the correction reproduces the shared error up to what the card absorbs
        rec = refit(fold.corr, held)
        g = np.concatenate([np.log(s / c.sim) for s, c in zip(fold.corr.apply(rec), rec.curves)])
        s_true = np.concatenate([shared_error(c.vg, c.vd) for c in rec.curves])
        self.assertGreater(np.corrcoef(g, s_true)[0, 1], 0.2)

    def test_per_seed_training_split_matches_fit_arm(self):
        devs = toy_devices(4)
        hyper = {"M": {"seeds": (0, 1, 2), "steps": 60}}
        tasks = [("M", devs), ("LIN", devs[:2])]
        expanded, owner = split_train_tasks(tasks, hyper)
        self.assertEqual(owner, [0, 0, 0, 1])
        merged = merge_train_results(2, owner, [
            fit_arm(a, r, hyper=h.get(a), threads=1) for a, r, h in expanded])
        ref = fit_arm("M", devs, hyper=hyper["M"], threads=1)
        phi = np.random.default_rng(0).normal(size=(40, 6)) * ref.sd + ref.mu
        self.assertEqual(len(merged[0].nets), 3)
        self.assertTrue(np.allclose(merged[0].g(phi), ref.g(phi), rtol=0, atol=1e-12))
        self.assertEqual(merged[1].arm, "LIN")

    def test_corrected_problem_applies_correction(self):
        class Base:
            tag = "toy"
            n_sim = 0

            class _C:
                def __init__(self, kind, fixed, vg, vd):
                    self.kind, self.fixed, self.Vg, self.Vd = kind, fixed, vg, vd
                    self.Id = np.ones_like(vg)
                    self.meta = {}
            curves = [_C("idvg", 0.3, GRID, np.full_like(GRID, 0.3))]

            def simulate(self, local):
                self.n_sim += 1
                return [np.full_like(GRID, float(local[0]))]

        corr = Correction("T1", table={"nmos|idvg@0.3": 0.2})
        prob = CorrectedProblem(Base(), corr, "nmos", 1.0, 1.0, VDD, "toy")
        out = prob.simulate(np.array([2.0]))
        assert np.allclose(out[0], 2.0 * np.exp(0.2))
        assert prob.n_sim == 1 and prob.tag == "toy"
        plain = CorrectedProblem(Base(), Correction("C0"), "nmos", 1.0, 1.0, VDD, "toy")
        assert np.allclose(plain.simulate(np.array([2.0]))[0], 2.0)

    def test_correlation_stats_recovers_linear_mix(self):
        rng = np.random.default_rng(1)
        rows = []
        for _ in range(3):
            comp = {c: rng.normal(size=200) for c in ("c_param", "c_rs", "c_heat", "c_gain")}
            g = 0.5 * comp["c_rs"] + 1.0 * comp["c_heat"] + 0.1
            rows.append({"g": g, "resid": g + 0.01 * rng.normal(size=200), **comp})
        st = correlation_stats(rows)
        assert st["n_points"] == 600
        assert abs(st["ols"]["c_rs"] - 0.5) < 1e-9 and abs(st["ols"]["c_heat"] - 1.0) < 1e-9
        assert st["ols"]["r2"] > 0.999999
        assert st["pearson"]["resid"] > 0.99
        lr = log_ratio(np.array([1.0, 1e-20, 2.0]), np.array([np.e, 1.0, np.nan]))
        assert np.isclose(lr[0], -1.0) and np.isnan(lr[1]) and np.isnan(lr[2])


if __name__ == "__main__":
    unittest.main()
