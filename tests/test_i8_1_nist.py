"""I8-1 NIST ingest helpers (no downloaded data needed)."""
from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from cryoml.explore import i8_1_nist as m


def _xlsx(path, rows):
    """Minimal xlsx with one sheet 'FETModules' and shared strings."""
    strings, cells = [], []
    for r, row in enumerate(rows, 1):
        for c, text in enumerate(row, 1):
            if text is None:
                continue
            ref = f"{chr(64 + c)}{r}"
            strings.append(text)
            cells.append((r, f'<c r="{ref}" t="s"><v>{len(strings) - 1}</v></c>'))
    by_row = {}
    for r, x in cells:
        by_row.setdefault(r, []).append(x)
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    sheet = (f'<worksheet xmlns="{ns}"><sheetData>'
             + "".join(f'<row r="{r}">{"".join(v)}</row>' for r, v in sorted(by_row.items()))
             + "</sheetData></worksheet>")
    sst = (f'<sst xmlns="{ns}">' + "".join(f"<si><t>{s}</t></si>" for s in strings) + "</sst>")
    wb = (f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/'
          f'2006/relationships"><sheets><sheet name="FETModules" sheetId="2" r:id="rId1"/>'
          f"</sheets></workbook>")
    rels = ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", wb)
        z.writestr("xl/_rels/workbook.xml.rels", rels)
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        z.writestr("xl/sharedStrings.xml", sst)


def _dev(tag, fam, W, L):
    return m.NistDevice(tag, "f", fam, "nmos", 1, 1, W, L)


class TestI81(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_col_index(self):
        assert m._col_index("A1") == (1, 1)
        assert m._col_index("M414") == (414, 13)
        assert m._col_index("AA2") == (2, 27)

    def test_layout_map_offsets(self):
        tmp_path = self.tmp
        rows = [["Device Name", "sky130_fd_pr__nfet_01v8"], ["Column / Row #", "devices"],
                ["1", "W=0.36/L=0.15"], [None, "W=0.39/L=0.15"],
                ["Device Name", "sky130_fd_pr__pfet_01v8_lvt"], ["Column # +26", "devices"],
                ["1", "W=0.42/L=0.35"], [None, "W=0.8//L=2"], ["2", "W=5/L=8"]]
        p = tmp_path / "x.xlsx"
        _xlsx(p, rows)
        wl = m.layout_map(p)
        assert wl[(1, 1)]["device"] == "nfet_01v8" and wl[(1, 2)]["W_um"] == 0.39
        assert wl[(27, 1)] == {**wl[(27, 1)], "device": "pfet_01v8_lvt", "W_um": 0.42, "L_um": 0.35}
        assert wl[(27, 2)]["L_um"] == 2.0          # 'W=0.8//L=2' typo form
        assert wl[(28, 1)]["W_um"] == 5.0


    def test_xlsx_rejects_dtd(self):
        tmp_path = self.tmp
        p = tmp_path / "bad.xlsx"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("xl/workbook.xml", '<!DOCTYPE x [<!ENTITY a "b">]><workbook/>')
        with self.assertRaises(ValueError):
            m.read_xlsx(p, "FETModules")

    def test_split_deterministic_and_stratified(self):
        devs = [_dev(f"n{i}", "a", 1.0, 0.1 * (i + 1)) for i in range(12)] + \
               [_dev(f"p{i}", "b", 2.0, 0.1 * (i + 1)) for i in range(7)]
        s1, s2 = m.make_split(devs, 7), m.make_split(devs, 7)
        assert s1["DEV"] == s2["DEV"] and s1["TEST"] == s2["TEST"]
        assert set(s1["DEV"]).isdisjoint(s1["TEST"])
        assert s1["n_dev"] + s1["n_test"] == 19
        # family a: blocks 5, 5, 2 -> 3 + 3 + 1 DEV; family b: 5, 2 -> 3 + 1
        assert sum(t.startswith("n") for t in s1["DEV"]) == 7
        assert sum(t.startswith("p") for t in s1["DEV"]) == 4
        for fam, blocks in s1["blocks"].items():
            for b in blocks[:-1]:
                assert len(b["DEV"]) == 3 and len(b["TEST"]) == 2

    def test_keys_and_pipeline_sign(self):
        dev = m.NistDevice("t", "pthin_lvt", "pfet_01v8_lvt", "pmos", 27, 1, 1.0, 1.0)
        vg = -np.linspace(0, 1.8, 5)
        vd = -np.linspace(0, 1.8, 4)
        dev.curves = [m.NistCurve("idvg", -1.8, 0.0, vg, -np.arange(5.0), np.zeros(0)),
                      m.NistCurve("idvg", -0.1, 0.0, vg, -np.arange(5.0), np.zeros(0)),
                      m.NistCurve("idvg", -0.1, 0.4, vg, -np.arange(5.0), np.zeros(0)),
                      m.NistCurve("idvd", -0.6, 0.0, vd, -np.arange(4.0), np.zeros(0)),
                      m.NistCurve("idvd", 0.0, 0.0, vd, -np.arange(4.0), np.zeros(0))]
        m.assign_keys(dev.curves)
        assert [c.key for c in dev.curves] == ["idvg@d1", "idvg@d0", "", "idvd@g1", "idvd@g0"]
        pc = m.to_pipeline_curves(dev)
        assert [c.meta["key"] for c in pc] == ["idvg@d0", "idvg@d1", "idvd@g0", "idvd@g1"]
        assert all(np.all(c.Id >= 0) for c in pc)          # pMOS magnitude convention
        assert pc[0].fixed == -0.1 and np.all(pc[0].Vd == -0.1)

    def test_stretch_and_box(self):
        box = {"lmin": 0.5, "lmax": 1.0, "wmin": 1.0, "wmax": 3.0}
        self.assertEqual(m.stretch(box, 2.0, 0.7), (1.0, 1.0))
        sl, sw = m.stretch(box, 6.0, 0.25)
        self.assertAlmostEqual(sl, 2.0)
        self.assertAlmostEqual(sw, 2.0)
        names = m._base_set()
        stock = {n: 1.0 for n in names}
        stock.update(vth0=-0.6, u0=0.02, b0=0.0)
        b = m.base_box(stock)
        self.assertAlmostEqual(b.bounds["vth0"][0], -1.8)        # x/ 3, sign kept
        self.assertAlmostEqual(b.bounds["u0"][1], 2.0)           # x/ 100
        self.assertAlmostEqual(b.bounds["k1"][1], 3.0)           # x/ 3
        self.assertEqual(list(b.bounds["b0"]), [0.0, 0.0])       # zero -> inactive
        self.assertEqual(b.n_active, len(names) - 1)

    def test_changed_params_and_instance(self):
        stock = {"vth0": 0.5, "u0": 0.03}
        self.assertEqual(m.changed_params({"vth0": 0.5, "u0": 0.031}, stock), {"u0": 0.031})
        self.assertEqual(m.changed_params(None, stock), {})
        line = m.instance_line("pmos", 5.0, 0.35)
        self.assertIn("mult=1 m=1", line)                          # no paper-device mult table
        self.assertIn("ad=1.45p", line)

    def test_prediction_guard(self):
        import subprocess
        repo = self.tmp / "repo"
        repo.mkdir()
        run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, check=True)
        run("init", "-q")
        run("config", "user.email", "t@example.com")
        run("config", "user.name", "t")
        f = repo / "prediction.json"
        self.assertFalse(m.prediction_guard(f, cwd=repo)["ok"])         # missing
        f.write_text("{}")
        self.assertFalse(m.prediction_guard(f, cwd=repo)["ok"])         # untracked
        run("add", "prediction.json")
        run("commit", "-q", "-m", "p")
        g = m.prediction_guard(f, cwd=repo)
        self.assertTrue(g["ok"])
        self.assertTrue(g["commit"])
        f.write_text('{"x": 1}')
        self.assertFalse(m.prediction_guard(f, cwd=repo)["ok"])         # modified

    def test_nan_record_excluded(self):
        from cryoml.explore.e1_correction import CurveRec, DeviceRec
        from cryoml.explore.i3_2_bias import fit_bias

        def rec(tag, fam, bad=False):
            vg = np.linspace(0, 1.8, 40)
            r = DeviceRec(tag, fam, 1.0, 1.0, 1.8, "nist4k")
            for k, vd in enumerate((0.1, 1.8)):
                tgt = 1e-5 * (1 + vg) * (1.0 + 0.1 * k)
                sim = tgt * 1.2
                if bad and k == 0:
                    sim = np.full_like(sim, np.nan)
                r.curves.append(CurveRec(f"idvg@d{k}", "idvg", vg, np.full_like(vg, vd), sim, tgt,
                                         0, float(np.mean(tgt)), True))
            return r
        recs = [rec("a", "nmos"), rec("b", "nmos", bad=True), rec("c", "pmos"), rec("d", "pmos")]
        good, bad = m.split_finite(recs)
        self.assertEqual(bad, ["b"])
        self.assertEqual([r.tag for r in good], ["a", "c", "d"])
        hyper = {"seeds": (0,), "steps": 20}
        g_all = fit_bias("M_bias", recs, hyper=hyper, threads=1).apply(recs[0])
        g_good = fit_bias("M_bias", good, hyper=hyper, threads=1).apply(recs[0])
        self.assertFalse(all(np.all(np.isfinite(x)) for x in g_all))   # the failure mode
        self.assertTrue(all(np.all(np.isfinite(x)) for x in g_good))


if __name__ == "__main__":
    unittest.main()
