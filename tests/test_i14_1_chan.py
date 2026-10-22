"""I14-1: clamp state in every multi-analysis deck builder; channel-current
replica, element netlist and reference model.

Clamp (task fix 1): for a card with a2 > 1, ``spice_batch.simulate_pdk_multi``
(altermod and separate), ``sky130_misspec.simulate_truth`` and
``i3_2_followup.simulate_sky130`` (canonical binary, no element) equal
``simulate_pdk`` in state C (difference 0); ``patched_library`` writes the
clamp into the card and leaves a2 <= 1 cards unchanged.

Element (fix 2 / 3): replica parameter set, netlist text, Verilog-A text of
the v1 reference arm, the element-identity reference formulas.
"""
from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

A2_HIGH = {"a2": 1.9, "vth0": 0.36}


def _need_ngspice():
    from cryoml.spice_pdk import _NGSPICE_BIN, ensure_pdk77k
    if shutil.which(_NGSPICE_BIN) is None or ensure_pdk77k():
        raise unittest.SkipTest("NGSpice or the 77 K PDK is unavailable")


def _device(tag):
    from cryoml.card_eval import device_by_tag
    from cryoml.cold_start import device_bins
    from cryoml.data_io import load_device_curves
    d = device_by_tag(tag)
    try:
        curves = load_device_curves(d)
    except Exception as exc:  # pragma: no cover - data not vendored
        raise unittest.SkipTest(f"measured data unavailable: {exc}")
    return d, curves, device_bins()[tag]


def _dev(tag):
    from cryoml.explore.i11_1_deploy import Dev
    d, _, b = _device(tag)
    return Dev(tag, d.dev_type, float(d.L_um), float(d.W_um), int(b))


def _equal(test, a, b):
    test.assertEqual(len(a), len(b))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


class ClampMultiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _need_ngspice()
        cls.d, cls.curves, cls.bin = _device("nmos_L0p19_W7")

    def ref(self, params, state="clamped"):
        from cryoml.spice_pdk import simulate_pdk
        d = self.d
        return simulate_pdk(d.dev_type, d.L_um, d.W_um, self.curves, params=params,
                            bin_index=self.bin if params else None, bsim4_state=state)

    def test_multi_equals_simulate_pdk_state_c(self):
        from cryoml.spice_batch import simulate_pdk_multi
        d = self.d
        cards = [A2_HIGH, {**A2_HIGH, "vth0": 0.38}, {"a2": 0.5, "vth0": 0.38}, None]
        want = [self.ref(c) for c in cards]
        for method in ("altermod", "separate"):
            got = simulate_pdk_multi(d.dev_type, d.L_um, d.W_um, self.curves, cards,
                                     bin_index=self.bin, method=method)
            for g, w in zip(got, want):
                _equal(self, g, w)

    def test_multi_parallel_near_exact_state_c(self):
        from cryoml.spice_batch import simulate_pdk_multi
        d = self.d
        cards = [A2_HIGH, {**A2_HIGH, "vth0": 0.38}]
        got = simulate_pdk_multi(d.dev_type, d.L_um, d.W_um, self.curves, cards,
                                 bin_index=self.bin, method="parallel")
        for g, c in zip(got, cards):
            for x, y in zip(g, self.ref(c)):
                big = np.abs(y) > 1e-9
                self.assertLess(float(np.max(np.abs(x - y)[big] / np.abs(y[big]))), 1e-6)

    def test_multi_legacy_differs_for_a2_above_one(self):
        from cryoml.spice_batch import simulate_pdk_multi
        d = self.d
        leg = simulate_pdk_multi(d.dev_type, d.L_um, d.W_um, self.curves, [A2_HIGH],
                                 bin_index=self.bin, bsim4_state="legacy")[0]
        _equal(self, leg, self.ref(A2_HIGH, "legacy"))
        diff = max(float(np.max(np.abs(x - y))) for x, y in zip(leg, self.ref(A2_HIGH)))
        self.assertGreater(diff, 0.0)

    def test_misspec_truth_equals_simulate_pdk_state_c(self):
        from cryoml.sky130_misspec import simulate_truth
        d = self.d
        got = simulate_truth(d.dev_type, d.L_um, d.W_um, self.curves, A2_HIGH, bin_index=self.bin)
        _equal(self, got, self.ref(A2_HIGH))

    def test_simulate_sky130_equals_simulate_pdk_state_c(self):
        from cryoml.explore.i3_2_followup import simulate_sky130
        from cryoml.spice_pdk import _NGSPICE_BIN
        d = self.d
        got, msgs = simulate_sky130(d.dev_type, d.L_um, d.W_um, self.curves, A2_HIGH, self.bin,
                                    _NGSPICE_BIN)
        _equal(self, got, self.ref(A2_HIGH))

    def test_patched_library_writes_clamp(self):
        from cryoml import spice_pdk as sp
        from cryoml.explore.i11_1_deploy import patched_library

        def bin_a2(lib_dir: Path, params, state):
            lib = patched_library(lib_dir, {"nmos": [(self.bin, params)]}, state)
            text = (lib.parent / sp._CORNER_BASENAME["nmos"]).read_text()
            span = next(text[s:e] for i, s, e in sp._bin_spans(text, "nmos") if i == self.bin)
            return text, float(re.search(r"^\+\s*a2\s*=\s*([^\s]+)", span, re.M | re.I).group(1))

        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            _, v = bin_a2(Path(a), A2_HIGH, "clamped")
            self.assertEqual(v, 1.0)
            _, v = bin_a2(Path(b), A2_HIGH, "legacy")
            self.assertEqual(v, 1.9)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            t1, _ = bin_a2(Path(a), {"a2": 0.5}, "clamped")
            t2, _ = bin_a2(Path(b), {"a2": 0.5}, "legacy")
            self.assertEqual(t1, t2)


class ReplicaTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _need_ngspice()

    def test_replica_bins_carry_every_value(self):
        from cryoml import spice_pdk as sp
        from cryoml.explore import i14_1_chan as m
        for dev_type in ("nmos", "pmos"):
            text = (sp.PDK77K_DIR / sp._CORNER_BASENAME[dev_type]).read_text()
            rep = m.replica_corner_chan(text, dev_type)
            self.assertTrue(m.replica_bins_ok(rep, dev_type))
            self.assertIn(f".subckt {sp._SUBCKT[dev_type]}{m.REPLICA_SUFFIX} ", rep)
            # every bin line not in REPLICA_CHAN is unchanged
            base_spans = {i: text[s:e] for i, s, e in sp._bin_spans(text, dev_type)}
            mod = sp._MODEL[dev_type] + m.REPLICA_SUFFIX
            hits = list(re.finditer(rf"^\.model\s+{re.escape(mod)}\.(\d+)\s", rep, re.M | re.I))
            self.assertEqual(len(hits), len(base_spans))
            keys = {k.lower() for k in m.REPLICA_CHAN}
            for i, h in enumerate(hits):
                idx = int(h.group(1))
                span = rep[h.start(): hits[i + 1].start() if i + 1 < len(hits) else len(rep)]

                def kv(s):
                    return {mm.group(1).lower(): mm.group(2) for mm in
                            re.finditer(r"^\+\s*([a-z0-9_]+)\s*=\s*([^\s]+)", s, re.M | re.I)}
                a, b = kv(base_spans[idx]), kv(span)
                for k, v in a.items():
                    if k not in keys:
                        self.assertEqual(b.get(k), v, f"{dev_type} bin {idx} {k}")

    def test_size_dependent_isub_terms_zero(self):
        from cryoml.explore import i14_1_chan as m
        for k in ("lalpha0", "walpha0", "palpha0", "lbeta0", "wbeta0", "pbeta0"):
            self.assertEqual(m.REPLICA_CHAN[k], 0)

    def test_instance_modes(self):
        from cryoml.explore import i14_1_chan as m
        dev = _dev("nmos_L1_W1p6")
        card = m.instance("m1", dev, "nd", "ng", "0", "0", "card")
        rep = m.instance("m1", dev, "nd", "ng", "0", "0", "card_rep")
        el = m.instance("m1", dev, "nd", "ng", "0", "0", "elem")
        self.assertTrue(rep.startswith(card) and el.startswith(card))
        self.assertIn("Vm1R m1_rx m1_ri DC 0 AC 0", rep)
        self.assertNotIn("Nm1", rep)
        self.assertIn("Nm1 nd ng 0 m1_rx m1_ri corr2n", el)
        self.assertIn("Gm1c m1_ri m1_rb m1_ri m1_rb -9.9999999999999998e-13", el)
        self.assertNotIn("Vm1R", el)
        pel = m.instance("m1", _dev("pmos_L2_W5"), "nd", "ng", "0", "0", "elem", m.VA_MODULE_V1)
        self.assertIn("corr1p", pel)
        with self.assertRaises(ValueError):
            m.deck("x", Path("lib"), "", [], None, options="gmin=1e-15")


class ElementModelTests(unittest.TestCase):
    def test_reference_formulas(self):
        from cryoml.explore import i14_1_chan as m
        i_card, i_rep, g = 2e-6, 1.5e-6, 0.2
        self.assertAlmostEqual(m.current_fwd(i_card, i_rep, g), i_card + (np.exp(g) - 1) * i_rep, delta=1e-20)
        self.assertAlmostEqual(m.current_swap(i_card, i_rep, g), i_card - (np.exp(g) - 1) * i_rep, delta=1e-20)
        y =m.admittance(1e-5 + 2e-7j, 0.8e-5, i_rep, g, 0.3)
        self.assertAlmostEqual(y.real, 1e-5 + (np.exp(g) - 1) * 0.8e-5 + i_rep * np.exp(g) * 0.3, delta=1e-20)
        self.assertEqual(y.imag, 2e-7)

    def test_swap_sign(self):
        from cryoml.explore import i14_1_chan as m
        # exchanged bench in forward operation: the card current into nd (card
        # source) is +I, the replica drain (copy of nz, the card drain) carries
        # -I_rep; the element adds (exp(g)-1) I_rep leaving nd, so
        # -i(VD) = I + (exp(g)-1) I_rep, the forward-bench value
        i_card, i_rep, g = 3e-6, 2.9e-6, -0.1
        got = m.current_swap(i_card, -i_rep, g)
        self.assertAlmostEqual(got, float(m.current_fwd(i_card, i_rep, g)), delta=1e-20)

    def test_drain_channel_current_sign(self):
        from cryoml.explore import i14_1_chan as m
        idv = np.array([1e-6, 1e-6])
        np.testing.assert_array_equal(m.drain_channel_current(idv, np.array([0.5, -0.5]), False),
                                      [1e-6, -1e-6])
        np.testing.assert_array_equal(m.drain_channel_current(idv, np.array([-0.5, 0.5]), True),
                                      [-1e-6, 1e-6])

    def test_v1_text_and_conversion(self):
        from cryoml.explore import i14_1_chan as m
        from cryoml.explore.i12_1_symm import abs_inputs
        from cryoml.explore.i3_2_bias import BiasCorrection
        rng = np.random.default_rng(0)
        nets = [[(rng.normal(size=(3, 4)), rng.normal(size=4)), (rng.normal(size=(4, 1)), rng.normal(size=1))]
                for _ in range(2)]
        mu, sd = np.array([0.0, 0.4, 0.5]), np.array([1.0, 0.3, 0.25])
        bc = BiasCorrection("M_bias", mu, sd, nets=nets, cols=(0, 1, 2))
        sym = m.v1_as_sym(bc)
        text = m.to_verilog_a(sym)
        self.assertIn(f"module {m.VA_MODULE_V1}(d, g, s, rx, ri);", text)
        for _ in range(50):
            p = int(rng.integers(0, 2))
            vgs, vds = rng.uniform(-1.85, 1.85, size=2)
            phi = np.zeros((1, 6))
            phi[0, :3] = abs_inputs(bool(p), vgs, vds)[0]
            want = float(bc.g(phi)[0])
            self.assertAlmostEqual(float(sym.g_v(bool(p), vgs, vds)), want, delta=1e-14)
            self.assertAlmostEqual(m.eval_verilog_a(text, p, vgs, vds), want, delta=1e-12)

    def test_v2_text_is_i12_text(self):
        from cryoml.explore import i14_1_chan as m
        from cryoml.explore import i12_1_symm as v2
        rng = np.random.default_rng(1)
        nets = [[(rng.normal(size=(3, 4)), rng.normal(size=4)), (rng.normal(size=(4, 1)), rng.normal(size=1))]]
        c = v2.SymCorrection("M_bias_v2", np.array([0.0, 0.4, 0.5]), np.array([1.0, 0.3, 0.25]), nets=nets)
        self.assertEqual(m.to_verilog_a(c), v2.to_verilog_a_v2(c))
        self.assertEqual(m.module_name(c), v2.VA_MODULE_V2)

    def test_decide_validity(self):
        from cryoml.explore import i14_1_chan as m
        ok = {"v1_max_rrms_diff": 0.0, "v2_unit_tests_pass": True, "v3_n_rows_min": 18, "v3_m2_failed": 0}
        self.assertTrue(m.decide_validity(ok)["valid"])
        self.assertFalse(m.decide_validity({**ok, "v1_max_rrms_diff": 1e-6})["valid"])
        self.assertFalse(m.decide_validity({**ok, "v2_unit_tests_pass": False})["valid"])
        self.assertFalse(m.decide_validity({**ok, "v3_n_rows_min": 17})["valid"])
        self.assertFalse(m.decide_validity({**ok, "v3_m2_failed": 1})["valid"])


class ReplicaSimulationTests(unittest.TestCase):
    """card + replica deck: the card current equals the card-only deck (the
    replica draws no circuit current) and the replica has no substrate
    current."""

    @classmethod
    def setUpClass(cls):
        _need_ngspice()

    def test_card_rep_equals_card(self):
        from cryoml import spice_pdk as sp
        from cryoml.explore import i11_1_deploy as m1
        from cryoml.explore import i14_1_chan as m
        dev = _dev("nmos_L100_W100")
        sub = sp._SUBCKT["nmos"]
        out = {}
        for mode in ("card", "card_rep"):
            with tempfile.TemporaryDirectory() as td_s:
                td = Path(td_s)
                lib = m.patched_library(td, {})
                ctrl = ["alter VG dc = 0.6", "alter VD dc = 1.85", "op", "print -i(VD)"]
                if mode == "card_rep":
                    ctrl.append(f"print @m.xm1r.m{sub}[isub] i(Vm1R) @m.xm1r.m{sub}[id]")
                r = m1.run_deck(m.deck("t", lib, m.device_body(dev, mode), ctrl, None, options=m1.TIGHT),
                                td, "t.sp", binary=sp._NGSPICE_BIN)
                out[mode] = {ln.split("=")[0].strip().lower(): float(ln.split("=")[1])
                             for ln in r["text"].splitlines() if " = " in ln and ln.strip()[0] in "-@i"}
        self.assertEqual(out["card"]["-i(vd)"], out["card_rep"]["-i(vd)"])
        self.assertEqual(out["card_rep"][f"@m.xm1r.m{sub}[isub]"], 0.0)
        a, b = out["card_rep"]["i(vm1r)"], out["card_rep"][f"@m.xm1r.m{sub}[id]"]
        self.assertLess(abs(a - b), 1e-7 * abs(b) + 1e-13)     # solver tolerance level


if __name__ == "__main__":
    unittest.main()
