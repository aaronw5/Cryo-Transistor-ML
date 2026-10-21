"""I10-2 class selection: inner folds, fold scores, selection rule, gate,
stored-row selector, assembly of the selected series, labels and the overall
rule."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cryoml.explore.i10_2_select import (  # noqa: E402
    CONTROL, FULL_MENU, MENU, NONE, TIE, agreement, assemble, bonferroni_alpha, class_size,
    cv_logs, fold_scores, gate_select, gmr, inner_folds, oracle_classes, overall_decision,
    refit_inner_index, row_key, select_class, selection_counts, selector_label)
from cryoml.explore.i9_2_classes import fit_class  # noqa: E402


class TestInnerFolds(unittest.TestCase):
    def test_logo_small(self):
        train = ["nmos_b", "pmos_a", "nmos_a"]
        fam = {"nmos_a": "nmos", "nmos_b": "nmos", "pmos_a": "pmos"}
        out = inner_folds(train, {t: t for t in train}, fam)
        self.assertEqual(out, [["nmos_a"], ["nmos_b"], ["pmos_a"]])

    def test_groups_kept_and_order(self):
        train = ["g1a", "g2a", "g1b", "g3a"]
        grp = {"g1a": "g1", "g1b": "g1", "g2a": "g2", "g3a": "g3"}
        fam = {t: "nmos" for t in train}
        out = inner_folds(train, grp, fam)
        self.assertEqual(out, [["g1a", "g1b"], ["g2a"], ["g3a"]])

    def test_kfold_large(self):
        train = [f"nfet{i:02d}" for i in range(10)] + [f"pfet{i:02d}" for i in range(10)]
        fam = {t: "nmos" if t.startswith("n") else "pmos" for t in train}
        out = inner_folds(train, {t: t for t in train}, fam, k=4, logo_max=12)
        self.assertEqual(len(out), 4)
        self.assertEqual(sorted(t for p in out for t in p), sorted(train))
        for p in out:          # both polarities in every inner fold
            self.assertTrue(any(fam[t] == "nmos" for t in p) and any(fam[t] == "pmos" for t in p))
        self.assertEqual(out[0][:3], ["nfet00", "nfet04", "nfet08"])

    def test_refit_index(self):
        self.assertEqual([refit_inner_index(i, 4) for i in range(6)], [0, 1, 2, 3, 0, 1])


class TestSelection(unittest.TestCase):
    def test_argmin_and_none(self):
        self.assertEqual(select_class({"lin": -0.02, "gam": -0.10, "mlp8": -0.05}), "gam")
        self.assertEqual(select_class({"lin": 0.02, "gam": 0.1}), NONE)

    def test_tie_prefers_fewer_values(self):
        s = {"lin": -0.050, "gam": -0.052, "mlp16": -0.054}
        self.assertEqual(select_class(s), "lin")
        self.assertEqual(select_class(s, tie=0.0), "mlp16")
        self.assertEqual(select_class({"lin": -0.003}), NONE)       # within TIE of none (0)
        self.assertEqual(select_class({"lin": -0.003}, tie=0.0), "lin")

    def test_blocked_and_nan(self):
        s = {"lin": -0.02, "gam": -0.10, "mlp8": float("nan")}
        self.assertEqual(select_class(s, blocked={"gam": 1}), "lin")
        self.assertEqual(select_class({"gam": float("nan")}), NONE)

    def test_fold_scores(self):
        logs = {"lin": {"a": -0.1, "b": 0.1, "c": float("nan")},
                "gam": {"a": -0.2, "b": float("nan"), "c": 0.0}}
        sc, bl = fold_scores(logs, ["a", "b", "c"], {"a": True, "b": True, "c": False})
        self.assertAlmostEqual(sc["lin"], 0.0)
        self.assertAlmostEqual(sc["gam"], -0.2)
        self.assertEqual(bl, {"gam": 1})

    def test_gate(self):
        self.assertEqual(gate_select(-0.01), "mlp16")
        self.assertEqual(gate_select(0.0), NONE)
        self.assertEqual(gate_select(float("nan")), NONE)

    def test_sizes(self):
        self.assertEqual(class_size(NONE), 0)
        self.assertEqual([class_size(c) for c in MENU], [0, 6, 66, 565, 1765])
        self.assertEqual(row_key(NONE), CONTROL)
        self.assertEqual(len(FULL_MENU), 13)


class TestStoredSelector(unittest.TestCase):
    def setUp(self):
        self.rows = {CONTROL: {"a": 0.10, "b": 0.20, "c": 0.30, "x": float("nan")},
                     "lin": {"a": 0.09, "b": 0.21, "c": 0.30, "x": float("nan")},
                     "gam": {"a": 0.05, "b": 0.10, "c": float("nan"), "x": float("nan")},
                     "mlp8": {"a": 0.2, "b": 0.2, "c": 0.3, "x": float("nan")},
                     "mlp16": {"a": 0.11, "b": 0.19, "c": 0.29, "x": float("nan")}}

    def test_cv_logs(self):
        logs, ok = cv_logs(self.rows, MENU, ["a", "c", "x"])
        self.assertAlmostEqual(logs["lin"]["a"], math.log(0.9))
        self.assertTrue(math.isnan(logs["gam"]["c"]))
        self.assertFalse(ok["x"])
        sc, bl = fold_scores(logs, ["a", "c", "x"], ok)
        self.assertEqual(bl, {"gam": 1})
        self.assertEqual(select_class(sc, bl), "lin")

    def test_assemble(self):
        sel = {"f1": "gam", "f2": NONE}
        fold = {"a": "f1", "b": "f1", "c": "f2"}
        out = assemble(sel, self.rows, fold, extra_nan=["x"])
        self.assertEqual(out["a"], 0.05)
        self.assertEqual(out["c"], 0.30)
        self.assertTrue(math.isnan(out["x"]))

    def test_counts_agreement(self):
        self.assertEqual(selection_counts({"f1": "gam", "f2": "gam", "f3": NONE})["gam"], 2)
        a = agreement({"f1": "gam", "f2": NONE}, {"f1": "gam", "f2": "lin"})
        self.assertEqual((a["n"], a["same"]), (2, 1))


class TestRule(unittest.TestCase):
    @staticmethod
    def ci(r, lo, hi):
        return {"ratio": r, "ci_low": lo, "ci_high": hi}

    def test_selector_label(self):
        per_ok = {"d1": self.ci(0.9, 0.8, 1.0), "d2": self.ci(1.0, 1.0, 1.0)}
        per_bad = {"d1": self.ci(0.9, 0.8, 1.0), "d2": self.ci(1.1, 1.0, 1.2)}
        lab = selector_label(per_ok, self.ci(0.95, 0.9, 0.99))
        self.assertEqual(lab["label"], "supported")
        self.assertTrue(lab["beats_mlp16_pooled"])
        lab = selector_label(per_bad, self.ci(0.95, 0.9, 1.02))
        self.assertEqual(lab["label"], "partly supported")
        self.assertEqual(lab["C1_failing_datasets"], ["d2"])
        self.assertEqual(selector_label(per_bad, self.ci(1.1, 1.0, 1.2))["label"], "not supported")

    def test_overall(self):
        sup = {"label": "supported"}
        no = {"label": "not supported"}
        c0 = {"S_gate": self.ci(0.9, 0.8, 1.0), "S_cv": self.ci(0.85, 0.8, 0.9),
              "S_loss": self.ci(0.86, 0.8, 0.9), "S_refit": self.ci(0.9, 0.8, 1.0)}
        vg = {"S_cv": self.ci(0.95, 0.9, 0.99), "S_loss": self.ci(0.97, 0.9, 1.01),
              "S_refit": self.ci(0.93, 0.9, 1.01)}
        r = overall_decision({"S_gate": sup, "S_cv": sup, "S_loss": sup, "S_refit": no}, c0, vg)
        self.assertEqual(r["decision"], "continue: class selection adds to the gate")
        self.assertEqual(r["frozen_selector"], "S_cv")
        vg2 = {k: self.ci(0.99, 0.95, 1.02) for k in vg}
        r = overall_decision({"S_gate": sup, "S_cv": sup, "S_loss": sup, "S_refit": no}, c0, vg2)
        self.assertEqual((r["decision"], r["frozen_selector"]), ("continue: gate only", "S_gate"))
        r = overall_decision({"S_gate": no, "S_cv": sup, "S_loss": sup, "S_refit": no}, c0, vg2)
        self.assertEqual((r["decision"], r["frozen_selector"]), ("continue: multi-class selector", "S_cv"))
        c0b = dict(c0, S_loss=self.ci(0.853, 0.8, 0.9), S_cv=self.ci(0.855, 0.8, 0.9))
        r = overall_decision({"S_gate": no, "S_cv": sup, "S_loss": sup, "S_refit": no}, c0b, vg2)
        self.assertEqual(r["frozen_selector"], "S_cv")       # tie -> cheaper selector
        r = overall_decision({s: no for s in c0}, c0, vg)
        self.assertEqual(r["decision"], "stop")

    def test_bonferroni(self):
        self.assertAlmostEqual(bonferroni_alpha(), 0.05 / 11)

    def test_identity_ratio(self):
        a = {"t1": 0.1, "t2": 0.2, "t3": 0.3}
        r = gmr({"d": (a, dict(a))})
        self.assertEqual((r["ratio"], r["ci_low"], r["ci_high"]), (1.0, 1.0, 1.0))

    def test_oracle(self):
        pooled = {"lin": {"ratio": 1.01}, "gam": {"ratio": 0.9}, "mlp16": {"ratio": 0.92}}
        per = {"d1": {"lin": 1.05, "gam": 1.2, "mlp16": 1.1}, "d2": {"lin": 0.9, "gam": 0.8, "mlp16": 0.85}}
        o = oracle_classes(per, pooled)
        self.assertEqual(o["pooled"], "gam")
        self.assertEqual(o["per_dataset"], {"d1": NONE, "d2": "gam"})


class TestInnerFit(unittest.TestCase):
    def test_single_seed_mlp_and_logs(self):
        from cryoml.explore.i7_1_transfer import s3_fold_value
        from test_i9_2_classes import toy_devices
        recs = toy_devices(6)
        corr = fit_class("mlp16", recs[:4], {"seeds": (0,), "steps": 60})
        self.assertEqual(len(corr.nets), 1)
        self.assertEqual(corr.arm, "mlp16")
        ref = fit_class("mlp16_s1", recs[:4], {"steps": 60})
        phi = np.random.default_rng(0).uniform(0, 1, (20, 6))
        phi[:, 0] = (phi[:, 0] > 0.5).astype(float)
        np.testing.assert_allclose(corr.g(phi), ref.g(phi), rtol=0, atol=0)
        logs = s3_fold_value(corr, recs[4:])
        self.assertTrue(all(np.isfinite(logs)))
        self.assertLess(np.mean(logs), 0.0)             # shared toy error is learned
        lin = fit_class("lin", recs[:4], {"seeds": (0,), "steps": 60})
        self.assertEqual(np.asarray(lin.coef).shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
