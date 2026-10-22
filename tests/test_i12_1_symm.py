"""Unit tests for cryoml.explore.i12_1_symm (pure parts, small synthetic data)."""

import math
import unittest

import numpy as np

from cryoml.explore import i12_1_symm as m
from cryoml.explore.e1_correction import CurveRec, DeviceRec, make_batch


def synth_devices(n=3, seed=0):
    rng = np.random.default_rng(seed)
    devs = []
    for d in range(n):
        pm = d % 2 == 1
        s = -1.0 if pm else 1.0
        dev = DeviceRec(f"{'pmos' if pm else 'nmos'}_L{d + 1}_W1", "pmos" if pm else "nmos",
                        float(d + 1), 1.0, 1.85, "sky130")
        for k, vfix in enumerate((0.05, 1.2)):
            vg = s * np.linspace(0, 1.85, 30)
            vd = s * np.full(30, vfix)
            sim = s * 1e-5 * np.log1p(np.exp((np.abs(vg) - 0.4) / 0.05)) * (1 + 0.1 * vfix)
            tgt = sim * (1 + 0.1 * rng.standard_normal(30) * 0.1 + 0.05 * np.abs(vg))
            dev.curves.append(CurveRec(f"idvg@{vfix:g}", "idvg", vg, vd, sim, tgt, 3,
                                       float(np.mean(np.abs(tgt[3:]))), True))
        vd = s * np.linspace(0, 1.85, 30)
        vg = s * np.full(30, 1.5)
        sim = s * 1e-4 * np.tanh(np.abs(vd) / 0.3) * (1 + 0.05 * np.abs(vd))
        dev.curves.append(CurveRec("idvd@1.5", "idvd", vg, vd, sim, sim * 1.1, 0,
                                   float(np.mean(np.abs(sim * 1.1))), True))
        dev.curves.append(CurveRec("idvd@0.9", "idvd", vg, vd, sim, sim, 0, 1.0, False))
        devs.append(dev)
    return devs


def random_corr(seed=0, inputs="sym_v2"):
    rng = np.random.default_rng(seed)
    nets = []
    for _ in range(2):
        nets.append([(rng.normal(size=(3, 16)) * 0.5, rng.normal(size=16) * 0.1),
                     (rng.normal(size=(16, 16)) * 0.3, rng.normal(size=16) * 0.1),
                     (rng.normal(size=(16, 1)) * 0.3, rng.normal(size=1) * 0.1)])
    return m.SymCorrection("M_bias_v2", np.array([0.0, 0.4, 0.5]), np.array([1.0, 0.3, 0.3]),
                           nets=nets, inputs=inputs)


class TestInputs(unittest.TestCase):
    def test_swap_invariance(self):
        rng = np.random.default_rng(1)
        vgs, vds = rng.uniform(-1.85, 1.85, (2, 500))
        for pm in (False, True):
            a = m.sym_inputs(pm, vgs, vds)
            b = m.sym_inputs(pm, vgs - vds, -vds)       # exchanged d and s
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)

    def test_forward_close_to_abs(self):
        vgs = np.linspace(0, 1.85, 20)
        vds = np.linspace(0, 1.85, 20)
        for pm, s in ((False, 1.0), (True, -1.0)):
            x = m.sym_inputs(pm, s * vgs, s * vds)
            self.assertTrue(np.all(np.abs(x[:, 1] - vgs / 1.85) <= m.EPS / (2 * 1.85) + 1e-15))
            np.testing.assert_allclose(x[:, 2], np.sqrt(vds ** 2 + m.EPS ** 2) / 1.85)
            self.assertTrue(np.all(x[:, 0] == (1.0 if pm else 0.0)))

    def test_pmos_source_is_higher_terminal(self):
        # pMOS with V(d,s) = +0.5 (reverse): effective source is d; drive = Vd - Vg
        x = m.sym_inputs(True, -1.0, 0.5)
        # a = -(-1.0) = 1.0, b = -(-1.0 - 0.5) = 1.5 -> smooth max ~ 1.5
        self.assertAlmostEqual(x[0, 1] * 1.85, (1.0 + 1.5 + math.sqrt(0.25 + 1e-4)) / 2, places=12)

    def test_torch_forward_matches(self):
        import torch
        ug = np.linspace(0, 1.85, 11)
        ud = np.linspace(0.025, 1.85, 11)
        x1, x2 = m.torch_sym_forward(torch.tensor(ug), torch.tensor(ud))
        x = m.sym_inputs(True, -ug, -ud)
        np.testing.assert_allclose(x1.numpy(), x[:, 1], rtol=0, atol=1e-15)
        np.testing.assert_allclose(x2.numpy(), x[:, 2], rtol=0, atol=1e-15)


class TestCorrection(unittest.TestCase):
    def test_json_roundtrip_and_apply(self):
        c = random_corr()
        c2 = m.SymCorrection.from_json(c.to_json())
        dev = synth_devices(2)[1]
        for a, b in zip(c.apply(dev), c2.apply(dev)):
            np.testing.assert_array_equal(a, b)
        cur = dev.curves[0]
        np.testing.assert_allclose(c.apply(dev)[0],
                                   cur.sim * np.exp(c.g_v(True, cur.vg, cur.vd)))
        with self.assertRaises(TypeError):
            c.g(np.zeros((2, 6)))

    def test_g_terminal_swap(self):
        c = random_corr(3)
        vg, vd = np.meshgrid(np.linspace(0, 1.85, 9), np.linspace(-1.85, 1.85, 13), indexing="ij")
        g_f = m.g_terminal(c, False, vg, vd, 0.0)
        g_s = m.g_terminal(c, False, vg, 0.0, vd)
        np.testing.assert_allclose(g_f, g_s, rtol=0, atol=1e-14)

    def test_batch_inputs_order(self):
        devs = synth_devices(3)
        b = make_batch(devs)
        x = m.batch_inputs(devs, "abs_v1")
        np.testing.assert_array_equal(x, b.phi[:, :3])
        self.assertEqual(len(m.batch_inputs(devs)), len(b.sim))


class TestTraining(unittest.TestCase):
    def test_abs_lambda0_reproduces_fit_bias(self):
        from cryoml.explore.i3_2_bias import fit_bias
        devs = synth_devices(3)
        hyper = {"steps": 30, "seeds": (0, 1)}
        old = fit_bias("M_bias", devs, hyper=hyper)
        new = m.fit_v2(devs, None, 0.0, hyper=hyper, inputs="abs_v1")
        phi = make_batch(devs).phi
        np.testing.assert_array_equal(old.g(phi), new.g_x(m.batch_inputs(devs, "abs_v1")))

    def test_penalty_reduces_negative_slope(self):
        devs = synth_devices(2)
        # card with gds/I = 0.02 /V everywhere (small positive) on a few points
        ug, ud = np.meshgrid(np.linspace(0.5, 1.85, 6), np.linspace(0.1, 1.85, 6), indexing="ij")
        pens = [{"ug": ug.ravel(), "ud": ud.ravel(), "r_gm": np.full(36, 1.0),
                 "r_gds": np.full(36, 0.02), "is_pmos": d.family == "pmos"} for d in devs]
        trace = []
        c = m.fit_v2(devs, pens, 10.0, hyper={"steps": 60, "seeds": (0,)}, trace=trace)
        pv = m.penalty_value_numpy(c, pens)
        self.assertTrue(np.isfinite(pv["P"]))
        self.assertEqual(len(c.nets), 1)

    def test_merge(self):
        a, b = random_corr(0), random_corr(0)
        b.nets = b.nets[:1]
        mm = m.merge_seed_corrections([a, b])
        self.assertEqual(len(mm.nets), 3)
        b.mu = b.mu + 1
        with self.assertRaises(RuntimeError):
            m.merge_seed_corrections([a, b])


class TestPenalty(unittest.TestCase):
    def test_penalty_arrays(self):
        n = m.PEN_N
        UG, UD = np.meshgrid(m.PEN_GRID, m.PEN_GRID, indexing="ij")
        i = 1e-6 * np.exp(UG) * (1 - np.exp(-UD / 0.1))      # forward-positive nMOS
        p = m.penalty_arrays(i, False)
        self.assertEqual(p["n_grid"], n * n)
        self.assertTrue(np.all(p["ud"] > 0))
        inner = (p["ug"] > 0.03) & (p["ug"] < 1.82)
        np.testing.assert_allclose(p["r_gm"][inner], 1.0, rtol=2e-4)
        pp = m.penalty_arrays(-i, True)          # pMOS: signed current negative
        np.testing.assert_allclose(pp["r_gm"], p["r_gm"])
        low = m.penalty_arrays(i * 1e-6, False)  # below the 1 nA floor except large u
        self.assertLess(low["n_kept"], p["n_kept"])

    def test_penalty_value_zero_for_g0(self):
        c = random_corr()
        for layers in c.nets:
            layers[-1] = (np.zeros_like(layers[-1][0]), np.zeros_like(layers[-1][1]))
        pens = [{"ug": np.array([1.0]), "ud": np.array([1.0]), "r_gm": np.array([0.5]),
                 "r_gds": np.array([-0.2]), "is_pmos": False}]
        pv = m.penalty_value_numpy(c, pens)
        self.assertAlmostEqual(pv["P"], 0.2)
        self.assertEqual(pv["frac_neg_gds"], 1.0)
        self.assertEqual(m.card_negative_fraction(pens)["frac_neg_gds_card"], 1.0)


class TestVerilogA(unittest.TestCase):
    def test_text_matches_python_and_is_symmetric(self):
        c = random_corr(5)
        text = m.to_verilog_a_v2(c)
        self.assertIn("V(sense) <+ RS * I(sense);", text)
        self.assertIn("I(d, s) <+ (exp(gsum / 2) - 1.0) * ib;", text)
        self.assertNotIn("abs(", text)
        rng = np.random.default_rng(0)
        for _ in range(50):
            p = int(rng.integers(0, 2))
            vgs, vds = rng.uniform(-1.85, 1.85, 2)
            gt = m.eval_verilog_a_v2(text, p, vgs, vds)
            self.assertLess(abs(gt - float(c.g_v(bool(p), vgs, vds))), 1e-12)
            self.assertLess(abs(m.eval_verilog_a_v2(text, p, vgs - vds, -vds) - gt), 1e-12)

    def test_rejects_abs_inputs(self):
        with self.assertRaises(ValueError):
            m.to_verilog_a_v2(random_corr(inputs="abs_v1"))


class TestNetlist(unittest.TestCase):
    def test_replica_corner(self):
        from cryoml import spice_pdk as sp
        sp.ensure_pdk77k()
        for dev_type in ("nmos", "pmos"):
            text = (sp.PDK77K_DIR / sp._CORNER_BASENAME[dev_type]).read_text(errors="replace")
            rep = m.replica_corner(text, dev_type)
            sub = sp._SUBCKT[dev_type]
            self.assertIn(f".subckt {sub}_rep ", rep)
            self.assertIn(f".ends {sub}_rep", rep)
            self.assertNotIn(f".subckt {sub} ", rep)
            self.assertTrue(m.replica_bins_ok(rep, dev_type))
            self.assertFalse(m.replica_bins_ok(text, dev_type))
            # DC parameters unchanged (vth0 / u0 lines identical in order)
            for key in ("vth0", "u0", "vsat", "rdsw"):
                pat = rf"^\+\s*{key}\s*=.*$"
                import re
                a = re.findall(pat, text.split(f".subckt {sub} ", 1)[1], re.M | re.I)
                b = re.findall(pat, rep, re.M | re.I)
                self.assertEqual(a, b)

    def test_instance_v2(self):
        body = "sky130_fd_pr__nfet_01v8_lvt l=1u w=1.6u mult=1 m=1\n"
        t = m.instance_v2("q", body, "sky130_fd_pr__nfet_01v8_lvt", "nmos", "d1", "g1", "s1", "b1", True)
        lines = t.strip().splitlines()
        self.assertEqual(lines[0], "Xq d1 g1 s1 b1 " + body.strip())
        self.assertEqual(lines[1], "Xqr q_ri q_rg q_rs q_rb sky130_fd_pr__nfet_01v8_lvt_rep l=1u w=1.6u mult=1 m=1")
        self.assertEqual(lines[2:6], ["Eqd q_rx 0 d1 0 1", "Eqg q_rg 0 g1 0 1",
                                      "Eqs q_rs 0 s1 0 1", "Eqb q_rb 0 b1 0 1"])
        self.assertEqual(lines[6], "Nq d1 g1 s1 q_rx q_ri corr2n")
        self.assertEqual(m.instance_v2("q", body, "sky130_fd_pr__nfet_01v8_lvt", "nmos",
                                       "d1", "g1", "s1", "b1", False).strip(), lines[0])


class TestReference(unittest.TestCase):
    def test_python_models(self):
        np.testing.assert_allclose(m.python_current_fwd([1e-6], [0.1]), [1e-6 * math.exp(0.1)])
        # exchanged bench, channel current only: 2e-6 A into the card source (nd);
        # card drain-terminal current -2e-6 A, i(VZ) = minus that = 2e-6 A
        np.testing.assert_allclose(m.python_current_swap([2e-6], [2e-6], [0.1]),
                                   [2e-6 * math.exp(0.1)])
        y = m.python_admittance_v2([1e-6], [1e-5 + 2e-9j], [0.2], [0.3])
        self.assertAlmostEqual(y[0].real, math.exp(0.2) * 1e-5 + 1e-6 * math.exp(0.2) * 0.3)
        self.assertEqual(y[0].imag, 2e-9)

    def test_eq_violation(self):
        r = m.eq_violation([1.0 + 1e-7, 1e-13, np.nan], [1.0, 0.0, 1.0])
        self.assertEqual(r["n"], 3)
        self.assertEqual(r["n_viol"], 1)              # the NaN point
        self.assertTrue(math.isinf(r["max_ratio"]))
        r = m.eq_violation([1.0 + 2e-6], [1.0])
        self.assertEqual(r["n_viol"], 1)
        r = m.eq_violation([1e-9 + 9e-14], [1e-9])
        self.assertEqual(r["n_viol"], 0)

    def test_cap_err(self):
        e = m.cap_err([1e-9, 1e-16], [1e-9 * (1 + 1e-4), 0.0])
        self.assertLess(e[0], 1e-3)
        self.assertAlmostEqual(e[1], 0.1)


class TestSelections(unittest.TestCase):
    """Row selections against the prereg.json wording."""

    def test_v1_keys(self):
        import json
        from pathlib import Path
        pre = json.loads((Path(__file__).resolve().parents[1]
                          / "out/research/explore/I12_1/prereg.json").read_text())
        keys = ["M_bias|" + t for t in pre["validity"]["V1"]["folds"]]
        self.assertEqual(keys, pre["validity"]["V1"]["stored_keys"])
        stored = keys + ["M_bias|nmos_L1_W3", "full|M_bias"]
        self.assertEqual(m.select_v1_keys(stored), keys)
        with self.assertRaises(KeyError):
            m.select_v1_keys(keys[:1])
        self.assertEqual(list(m.V2_DEVICES), pre["validity"]["V2"]["devices"])

    def test_v2_rows(self):
        rows = {t: 0.1 for t in ("nmos_L1_W1p6", "pmos_L8_W5", "nmos_L1_W3")}
        self.assertEqual(sorted(m.select_v2_rows(rows)), sorted(m.V2_DEVICES))
        with self.assertRaises(KeyError):
            m.select_v2_rows({"nmos_L1_W1p6": 0.1})

    def test_primary_rows(self):
        tags = [f"nmos_d{i}" for i in range(9)] + [f"pmos_d{i}" for i in range(9)]
        rows = {t: 0.1 for t in tags}
        self.assertEqual(len(m.select_primary_rows(rows, tags)), 18)
        with self.assertRaises(KeyError):
            m.select_primary_rows({**rows, "extra": 0.2}, tags)
        with self.assertRaises(KeyError):
            m.select_primary_rows({t: 0.1 for t in tags[:-1]}, tags)
        with self.assertRaises(ValueError):
            m.select_primary_rows(rows, tags[:-1])


class TestDecision(unittest.TestCase):
    def good(self):
        return {"q1_tight_viol": 0, "q1_bad": 0, "q1_vds0_max": 1e-13, "g_text_max": 1e-12,
                "g_sym_text_max": 0.0, "swap_asym_max": 5e-4, "q2_ac_max": 1e-5, "cap_max": 1e-8,
                "q2_gm_sign": 0, "q3_fail": 0, "q1_default_max": 1e-4, "q2_gds_sign": 0,
                "runtime_ratio_max": 1.5}

    def test_deploy(self):
        self.assertEqual(m.decide_deploy(self.good())["deploy"], "deployable (simulator-level)")
        s = {**self.good(), "runtime_ratio_max": 2.5, "q2_gds_sign": 2}
        d = m.decide_deploy(s)
        self.assertEqual(d["deploy"], "deployable with restrictions")
        self.assertEqual(len(d["restrictions"]), 2)
        for k, v in (("q1_tight_viol", 1), ("cap_max", 2e-3), ("swap_asym_max", 2e-3),
                     ("g_sym_text_max", 1e-11), ("q2_gm_sign", 1), ("q3_fail", 1),
                     ("q1_vds0_max", float("nan"))):
            self.assertEqual(m.decide_deploy({**self.good(), k: v})["deploy"], "not deployable", k)

    def test_accuracy_and_outcome(self):
        a1 = {"ci_high": 0.9}
        a2 = {"ci_high": 1.04}
        acc = m.decide_accuracy(a1, a2, 0)
        self.assertEqual(acc, {"A1_gain_vs_C0_refit": True, "A2_noninferior_vs_v1": True})
        self.assertFalse(m.decide_accuracy(a1, a2, 1)["A1_gain_vs_C0_refit"])
        self.assertFalse(m.decide_accuracy(a1, {"ci_high": 1.051}, 0)["A2_noninferior_vs_v1"])
        val_ok = m.decide_validity({"v1_max_g_diff": 0.0, "v2_max_rrms_diff": 0.0, "v3_n_rows": 18})
        self.assertTrue(val_ok["valid"])
        self.assertFalse(m.decide_validity({"v1_max_g_diff": 0.0, "v2_max_rrms_diff": 1e-6,
                                            "v3_n_rows": 18})["valid"])
        dep_ok = m.decide_deploy(self.good())
        dep_bad = m.decide_deploy({**self.good(), "q1_tight_viol": 3})
        self.assertEqual(m.decide_outcome(val_ok, acc, dep_ok)["outcome"], "supported")
        self.assertEqual(m.decide_outcome(val_ok, acc, dep_bad)["outcome"], "not deployable")
        acc2 = {"A1_gain_vs_C0_refit": True, "A2_noninferior_vs_v1": False}
        self.assertTrue(m.decide_outcome(val_ok, acc2, dep_ok)["outcome"].startswith("deployable, accuracy"))
        acc3 = {"A1_gain_vs_C0_refit": False, "A2_noninferior_vs_v1": True}
        self.assertEqual(m.decide_outcome(val_ok, acc3, dep_ok)["outcome"], "no gain over C0_refit")
        self.assertEqual(m.decide_outcome({"valid": False, "fails": ["x"]}, acc, dep_ok)["outcome"],
                         "invalid")


class TestQ3Selections(unittest.TestCase):
    def rows(self):
        r = {}
        for n, card_ok, full_ok in (("a", True, True), ("b", False, False), ("c", True, False),
                                    ("d", False, True)):
            r[f"{n}|M2_card"] = {"ok": card_ok, "wall": [1.0, 1.0, 1.0] if card_ok else [0.07]}
            r[f"{n}|M2_full"] = {"ok": full_ok, "wall": [2.0, 3.0, 2.5] if full_ok else [120.0]}
            r[f"{n}|published"] = {"ok": False, "wall": [1.0]}
        return r

    def test_q3_fail_count(self):
        # only 'c' fails with the element while the card-only run passes
        self.assertEqual(m.q3_fail_count(self.rows(), ["a", "b", "c", "d"]), 1)

    def test_runtime_ratios_amendment_a1(self):
        r = m.circuit_runtime_ratios(self.rows(), ["a", "b", "c", "d"])
        self.assertEqual(sorted(r), ["a"])
        self.assertAlmostEqual(r["a"], 2.5)


if __name__ == "__main__":
    unittest.main()
