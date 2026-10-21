"""I10-1 ETest 4 K / room-temperature pairing helpers (synthetic files; no
downloaded data and no network)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.explore import i10_1_etest_rt as m  # noqa: E402


def _mdm(swept: str, fixed: str, fixed_vals, sweep, scale: float) -> str:
    """Minimal ETest-style MDM text: one block per (fixed value, VB = 0)."""
    other = "VD" if swept == "VG" else "VG"
    head = ("! VERSION = 6.00\nBEGIN_HEADER\n ICCAP_INPUTS\n"
            f"  {swept}         V  S GROUND SMU1 0.1 LIN        1    0          {sweep[-1]}"
            f"       {len(sweep)}    {sweep[1] - sweep[0]}\n"
            "  VS         V  E GROUND SMU2 0.1 CON        0\n"
            f"  {other}         V  B GROUND SMU4 0.1 LIN        2    {fixed_vals[0]}  "
            f"{fixed_vals[-1]}  {len(fixed_vals)}  1\n"
            "  VB         V  C GROUND SMU3 0.1 CON        0\n"
            " ICCAP_OUTPUTS\n  ID         I  S GROUND SMU1 B\n  IB         I  C GROUND SMU3 B\n"
            "  IG         I  B GROUND SMU4 B\nEND_HEADER\n\n")
    body = ""
    for fv in fixed_vals:
        body += (f"BEGIN_DB\n ICCAP_VAR {fixed}         {fv}\n ICCAP_VAR VS         0\n"
                 f" ICCAP_VAR VB         0\n\n #{swept}              ID              IB"
                 "              IG\n")
        for x in sweep:
            body += f"  {x}   {scale * (fv + 1) * x:.6g}   0   0\n"
        body += "END_DB\n\n"
    return head + body


class TestI101(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_names(self):
        n4 = "sky130_fd_pr__nfet_01v8_w7u_l8u_m1(8392_5_6_IDVG)_4K.mdm"
        assert m.rt_name(n4) == "sky130_fd_pr__nfet_01v8_w7u_l8u_m1(8392_5_6_IDVG).mdm"
        assert m.rt_relpath("nfet_01v8", n4) == (
            "sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8_w7u_l8u_m1(8392_5_6_IDVG).mdm")
        with self.assertRaises(ValueError):
            m.rt_name("x.mdm")
        url = m.raw_url(m.rt_relpath("nfet_01v8", n4), "abc")
        assert url.startswith("https://raw.githubusercontent.com/google/"
                              "skywater-pdk-sky130-raw-data/abc/sky130_fd_pr/cells/")
        assert "(8392_5_6_IDVG).mdm" in url

    def test_git_blob_sha1(self):
        # `git hash-object` of an empty file and of "hello\n"
        assert m.git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
        assert m.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"

    def test_pair(self):
        root4 = self.tmp / "etest"
        (root4 / "Sky130-4k").mkdir(parents=True)
        rtroot = self.tmp / "rt"
        stem = "sky130_fd_pr__nfet_01v8_w7u_l8u_m1(8392_5_6_{})"
        sweep4 = [round(0.1 * i, 2) for i in range(19)]
        sweepr = [round(0.3 * i, 2) for i in range(7)]
        (root4 / "Sky130-4k" / (stem.format("IDVG") + "_4K.mdm")).write_text(
            _mdm("VG", "VD", [0.1, 1.8], sweep4, 1e-4))
        (root4 / "Sky130-4k" / (stem.format("IDVD") + "_4K.mdm")).write_text(
            _mdm("VD", "VG", [0.0, 0.9, 1.8], sweep4, 1e-4))
        # second 4 K device without RT files
        stem2 = "sky130_fd_pr__pfet_01v8_w7u_l8u_m1(8397_6_5_{})"
        for k, (sw, fx) in {"IDVG": ("VG", "VD"), "IDVD": ("VD", "VG")}.items():
            (root4 / "Sky130-4k" / (stem2.format(k) + "_4K.mdm")).write_text(
                _mdm(sw, fx, [-0.1, -1.8], [-x for x in sweep4], 1e-5))
        d = rtroot / "sky130_fd_pr" / "cells" / "nfet_01v8"
        d.mkdir(parents=True)
        (d / (stem.format("IDVG") + ".mdm")).write_text(_mdm("VG", "VD", [0.1, 1.8], sweepr, 5e-5))
        (d / (stem.format("IDVD") + ".mdm")).write_text(
            _mdm("VD", "VG", [0.0, 0.9, 1.8], sweepr, 5e-5))

        pairs = m.pair_etest(root_4k=root4, root_rt=rtroot)
        assert len(pairs) == 1
        d4, drt = pairs[0]
        assert d4.tag == drt.tag and d4.dev_type == drt.dev_type == "nfet_01v8"
        assert (drt.W_um, drt.L_um, drt.module) == (7.0, 8.0, 8392)
        assert sorted(c.key for c in drt.curves if c.key) == [
            "idvd@g0", "idvd@g1", "idvd@g2", "idvg@d0", "idvg@d1"]
        assert all(len(c.sweep) == 7 for c in drt.curves)
        assert all(len(c.sweep) == 19 for c in d4.curves)
        assert m.rt_relpaths([d4]) == [
            "sky130_fd_pr/cells/nfet_01v8/" + stem.format("IDVG") + ".mdm",
            "sky130_fd_pr/cells/nfet_01v8/" + stem.format("IDVD") + ".mdm"]
        assert m.pair_etest(types=("pfet_01v8",), root_4k=root4, root_rt=rtroot) == []
        rt_missing = m.load_rt([p for p in m.base.discover_etest(root4)
                                if p.dev_type == "pfet_01v8"][0], rtroot)
        assert rt_missing.flags == ["missing RT IDVG", "missing RT IDVD"]


if __name__ == "__main__":
    unittest.main()
