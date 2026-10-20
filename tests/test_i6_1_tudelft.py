"""I6-1 TU Delft 40-nm data: restricted .npy reading, curve keys, discovery,
split, card handling, box, scoring, records, correction in the loop
(synthetic files; the simulator tests run only when ngspice-41 and the card
file are present)."""

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.explore import i6_1_tudelft as m  # noqa: E402
from cryoml.explore.e1_correction import Correction, score_device  # noqa: E402

N = 111


def fake_obj(pol: str, W: int, L: int, temp: int, sample: int = 3, scale: float = 1.0) -> dict:
    s = -1.0 if pol == "p" else 1.0
    vg_f = s * np.linspace(0.45, 1.1, N)
    vd_f = s * np.linspace(0.0, 1.1, N)
    vd_m = s * np.concatenate([[0.05], np.geomspace(0.5, 1.1, 11)])
    vg_m = s * np.geomspace(0.5, 1.1, 11)
    i_vg = [s * scale * 1e-4 * (np.abs(vg_f) - 0.4) * min(1.0, abs(v) / 0.3) for v in vd_m]
    i_vd = [s * scale * 1e-4 * (abs(v) - 0.4) * np.tanh(np.abs(vd_f) / 0.2) for v in vg_m]
    # shuffle one list so the key order is checked
    order = np.arange(len(vd_m))[::-1]
    return {"setup": {"sample": sample, "temperature": temp, "campaingn": 1,
                      "dev_dict": {"type": pol, "vth": "l", "NF": np.int64(1), "W": np.int64(W),
                                   "L": np.int64(L), "sub_block": "LC"}},
            "dc_sett": {"v_g_fine_sweep": vg_f, "v_d_fine_sweep": vd_f,
                        "v_d_matrix_sweep": vd_m[order], "v_g_matrix_sweep": vg_m},
            "dc_results": {"i_d_const_d": [i_vg[k] for k in order], "i_d_const_g": i_vd,
                           "i_d_diode": np.zeros(11), "i_d_triode": np.zeros(11),
                           "temperature_meas_start": np.float64(4.13 if temp == 4 else 295.0),
                           "temperature_meas_end": np.float64(4.15 if temp == 4 else 295.0)}}


def write_npy(path: Path, obj: dict) -> None:
    arr = np.empty(1, dtype=object)
    arr[0] = obj
    np.save(path, arr, allow_pickle=True)


class ReadTests(unittest.TestCase):
    def test_roundtrip_and_restriction(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.npy"
            write_npy(p, fake_obj("n", 1000, 40, 4))
            obj = m.read_npy_dict(p)
            self.assertEqual(obj["setup"]["dev_dict"]["type"], "n")
            self.assertEqual(len(obj["dc_results"]["i_d_const_d"]), 12)

            class Evil:
                def __reduce__(self):
                    return (print, ("not allowed",))
            arr = np.empty(1, dtype=object)
            arr[0] = {"a": Evil()}
            q = Path(td) / "evil.npy"
            np.save(q, arr, allow_pickle=True)
            with self.assertRaises(pickle.UnpicklingError):
                m.read_npy_dict(q)

    def test_keys_by_abs_bias(self):
        curves = m.file_curves(fake_obj("p", 1000, 1000, 300))
        self.assertEqual(len(curves), 23)
        idvg = [c for c in curves if c.kind == "idvg"]
        self.assertEqual([c.key for c in idvg], [f"idvg@d{k}" for k in range(12)])
        self.assertTrue(np.all(np.diff([abs(c.fixed) for c in idvg]) > 0))
        self.assertAlmostEqual(idvg[0].fixed, -0.05)
        # current follows its bias after reordering: |I| grows with |Vd| up to 0.3 V
        self.assertLess(abs(idvg[0].id[-1]), abs(idvg[1].id[-1]))
        idvd = [c for c in curves if c.kind == "idvd"]
        self.assertEqual(idvd[-1].key, "idvd@g10")
        np.testing.assert_allclose(idvd[0].vd, idvd[0].sweep)
        np.testing.assert_allclose(idvd[0].vg, idvd[0].fixed)


class DiscoverSplitTests(unittest.TestCase):
    def make_dir(self, td: Path) -> None:
        devs = [("n", 3, 152, 120, 40), ("n", 3, 153, 120, 40), ("n", 3, 128, 1000, 1000),
                ("p", 5, 328, 1000, 1000), ("n", 3, 192, 1000, 40), ("p", 5, 392, 1000, 40),
                ("n", 3, 160, 120, 50)]
        for pol, sample, dev, W, L in devs:
            for temp in (4, 300):
                write_npy(td / f"{pol}_dc_camp_1_sample_{sample}_device_{dev}_temp_{temp}.npy",
                          fake_obj(pol, W, L, temp, sample))
        # 4 K only, and another campaign: both skipped
        write_npy(td / "n_dc_camp_1_sample_3_device_999_temp_4.npy", fake_obj("n", 120, 40, 4))
        write_npy(td / "n_dc_camp_2_sample_3_device_152_temp_4.npy", fake_obj("n", 120, 40, 4))

    def test_discover(self):
        with tempfile.TemporaryDirectory() as td_s:
            td = Path(td_s)
            self.make_dir(td)
            devs = m.discover(td)
            self.assertEqual(len(devs), 7)
            d = [x for x in devs if x.device == 152][0]
            self.assertEqual(d.tag, "t40_nmos_W0p12_L0p04_d152")
            self.assertEqual(d.group, "t40_nmos_W0p12_L0p04")
            self.assertAlmostEqual(d.temps_K[4], 4.14)
            self.assertEqual(sorted(d.curves), [4, 295])
            # dataset round trip
            man = {"devices": [m.device_row(x) for x in devs]}
            man.update(m.save_dataset(devs, td / "ds.npz"))
            back = m.load_dataset(td / "ds.npz", man, [d.tag])
            self.assertEqual(len(back), 1)
            for a, b in zip(m.fit_curves(d, 295), m.fit_curves(back[0], 295)):
                self.assertEqual(a.key, b.key)
                np.testing.assert_array_equal(a.id, b.id)
                self.assertEqual(a.fixed, b.fixed)

    def test_split(self):
        with tempfile.TemporaryDirectory() as td_s:
            td = Path(td_s)
            self.make_dir(td)
            devs = m.discover(td)
            s1, s2 = m.make_split(devs, 1), m.make_split(devs, 1)
            self.assertEqual(s1, s2)
            self.assertEqual(len(s1["TEST_pairs"]), 2)
            shared = [p for p in s1["TEST_pairs"] if tuple(p) in {(1.0, 0.04), (1.0, 1.0)}]
            self.assertEqual(len(shared), 1)
            self.assertFalse(set(s1["DEV"]) & set(s1["TEST"]))
            self.assertEqual(len(s1["DEV"]) + len(s1["TEST"]), 7)
            # both polarities of the shared TEST pair are in TEST
            test_pmos = [t for t in s1["TEST"] if "_pmos_" in t]
            self.assertEqual(len(test_pmos), 1)


@unittest.skipUnless(m.CARD_FILE.exists(), "card file missing")
class CardBoxTests(unittest.TestCase):
    def test_card_and_tc_zero(self):
        for t in ("nmos", "pmos"):
            block = m.model_block(t)
            self.assertTrue(block.startswith(f".model {m.CARD_MODEL} {t} level = 54"))
            for k in m.TC_ZERO:
                self.assertEqual(m.card_value(t, k, block), 0.0)
            self.assertNotEqual(float(m.card_params(t)["ute"]), 0.0)
        self.assertAlmostEqual(m.card_value("nmos", "vth0"), 0.22)
        self.assertAlmostEqual(m.card_value("pmos", "vth0"), -0.22)

    def test_box(self):
        d = m.MeasDevice("t40_nmos_W1_L1_d1", "nmos", 1.0, 1.0, 3, 1, "LC")
        box = m.base_box(d)
        self.assertEqual(len(box.names), 43)
        i = box.names.index("vth0")
        self.assertAlmostEqual(box.lo[i], 0.22 - 0.5)
        i = box.names.index("u0")
        self.assertAlmostEqual(box.hi[i] / box.center[i], 10.0)
        self.assertTrue(box.active[box.names.index("cit")])
        self.assertFalse(box.active[box.names.index("k3")])
        self.assertFalse(box.active[box.names.index("ags")])
        p = m.MeasDevice("t40_pmos_W1_L1_d2", "pmos", 1.0, 1.0, 5, 2, "LC")
        pb = m.base_box(p)
        i = pb.names.index("vth0")
        self.assertLess(pb.hi[i], 0.3)
        self.assertNotIn("kt1", m.b1_universe())
        self.assertNotIn("vth0", m.b1_universe())
        self.assertNotIn("dvt0w", m.b1_universe())
        self.assertEqual(len(m.b1_universe()), 54)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.dev = m.MeasDevice("t40_nmos_W1_L1_d1", "nmos", 1.0, 1.0, 3, 1, "LC", {4: 4.14},
                                {4: "x"}, {4: m.file_curves(fake_obj("n", 1000, 1000, 4))})

    def test_score_and_record(self):
        curves = m.fit_curves(self.dev, 4)
        sims = [1.1 * c.id for c in curves]
        sc = m.score(curves, sims)
        expect = []
        for c in curves:
            inc = m.inclusion(c)
            if inc["included"]:
                t = c.id[inc["start"]:]
                expect.append(np.sqrt(np.mean((0.1 * t) ** 2)) / np.mean(np.abs(t)))
        self.assertAlmostEqual(sc["rrms"], float(np.mean(expect)), places=12)
        self.assertGreater(sc["n_curves"], 15)
        rec = m.to_record(self.dev, 4, sims)
        self.assertEqual(rec.source, "tudelft_meas")
        self.assertEqual(rec.vdd, m.VDD)
        self.assertAlmostEqual(score_device(rec, [c.sim for c in rec.curves])["rrms"], sc["rrms"])
        sims[0] = np.full(N, np.nan)
        self.assertTrue(np.isnan(m.score(curves, sims)["rrms"]))

    def test_corrected_problem_keys(self):
        box = m.base_box(self.dev) if m.CARD_FILE.exists() else None
        if box is None:
            self.skipTest("card file missing")
        prob = m.MeasProblem(self.dev, 4, box)
        cp = m.corrected_problem(prob, Correction("C0"))
        self.assertEqual(cp._keys[0], "idvg@d0")
        self.assertEqual(prob.n_curves, sum(m.inclusion(c)["included"] for c in prob.curves))

    def test_aggregate(self):
        a = m.aggregate({"t40_nmos_W1_L1_d1": 0.1, "t40_nmos_W1_L1_d2": 0.3, "t40_pmos_W1_L1_d3": 0.2})
        self.assertAlmostEqual(a["all_device_mean"], 0.2)
        self.assertAlmostEqual(a["combined"], 0.2)
        self.assertEqual(m.group_of("t40_pmos_W1_L0p04_d392"), "t40_pmos_W1_L0p04")


@unittest.skipUnless(m.NGSPICE.exists() and m.CARD_FILE.exists(), "ngspice-41 or card missing")
class SimulatorTests(unittest.TestCase):
    def test_simulate_sign_and_resistor(self):
        for pol, t in (("n", 4), ("p", 300)):
            obj = fake_obj(pol, 1000, 1000, t)
            dev_type = {"n": "nmos", "p": "pmos"}[pol]
            label = 4 if t == 4 else 295
            dev = m.MeasDevice(f"t40_{dev_type}_W1_L1_d1", dev_type, 1.0, 1.0, 3, 1, "LC",
                               {label: 4.14 if t == 4 else 295.0}, {label: "x"},
                               {label: m.file_curves(obj)})
            curves = m.fit_curves(dev, label)
            sims = m.simulate(dev, m.model_block(dev_type), curves, dev.temps_K[label])
            self.assertTrue(all(np.all(np.isfinite(s)) for s in sims))
            sgn = 1.0 if pol == "n" else -1.0
            self.assertGreater(sgn * sims[-1][-1], 0)          # Id-Vd at the largest |Vg|, |Vd|
            with_r = m.simulate(dev, m.model_block(dev_type), curves[-1:], dev.temps_K[label],
                                rs_ext_ohm=1000.0)
            self.assertLess(abs(with_r[0][-1]), abs(sims[-1][-1]))


if __name__ == "__main__":
    unittest.main()
