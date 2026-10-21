"""I8-1: NIST mds2-2997 sky130 transistors at 4 K (MPW-5 test tile), ingest.

Data: doi:10.18434/mds2-2997 (NIST public data), "CMOS Sky130 Primitives
measured at cryogenic temperatures", files ``MPW-5 Test Tile.zip`` and
``ETest Tile.zip``, downloaded to ``out/research/data_raw/nist_mds2_2997/``
(not committed).  Files are read as text or as XML from a ZIP container
(``read_xlsx``); nothing in the download is executed.

MPW-5 transistor module (``transistormodule/{nthin,pthin,nthin_lvt,
pthin_lvt,nnative}``): one ``<module>_<position>_IDVG.mdm`` and one
``..._IDVD.mdm`` per transistor, Keysight IC-CAP MDM text.  W/L per
(module, position) are in ``Module Layout Coding.xlsx``, sheet
``FETModules``.  Chuck temperature 4 K (tile README); the MDM headers have no
temperature field.

This module parses the files, maps W/L, builds an inventory and converts the
Vb = 0 curves of the chosen families to ``MeasCurve`` lists keyed by position
(``idvg@d0..``, by |Vd|; ``idvd@g0..``, by |Vg|), as in ``i5_1_ihp_real`` and
``i6_1_tudelft``.  It fits nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree as ET

import numpy as np

from ..config import ROOT
from ..pdk2 import metric as pdk2_metric
from .i5_1_ihp_real import MdmFile, parse_mdm

DOI = "10.18434/mds2-2997"
RAW = ROOT / "out" / "research" / "data_raw" / "nist_mds2_2997"
MPW5_ROOT = RAW / "ext_mpw5" / "MPW-5 Test Tile"
ETEST_ROOT = RAW / "ext_etest" / "ETest Tile"
LAYOUT_XLSX = MPW5_ROOT / "Module Layout Coding.xlsx"
OUT = ROOT / "out" / "research" / "explore" / "I8_1"
DOWNLOADS = {
    "MPW-5 Test Tile.zip": {
        "url": "https://data.nist.gov/od/ds/mds2-2997/MPW-5%20Test%20Tile.zip",
        "local": "dl_mpw5/MPW-5 Test Tile.zip"},
    "ETest Tile.zip": {
        "url": "https://data.nist.gov/od/ds/mds2-2997/ETest%20Tile.zip",
        "local": "dl_etest/ETest Tile.zip"},
}
# folder -> (sky130 device type, polarity)
FOLDERS = {"nthin": ("nfet_01v8", "nmos"), "pthin": ("pfet_01v8", "pmos"),
           "nthin_lvt": ("nfet_01v8_lvt", "nmos"), "pthin_lvt": ("pfet_01v8_lvt", "pmos"),
           "nnative": ("nfet_03v3_nvt", "nmos")}
# I8-1 families: the two device types of the 77 K paper cards
# (spice_pdk._SUBCKT: sky130_fd_pr__nfet_01v8_lvt, sky130_fd_pr__pfet_01v8_lvt)
CHOSEN = ("nthin_lvt", "pthin_lvt")
FLAG_README = {"nthin_lvt": "readme.txt", "pthin": "readme.txt"}
VDD = 1.8
_FILE_RE = re.compile(r"^(\d+)_(\d+)_(IDVG|IDVD)\.mdm$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# xlsx (stdlib reader: ZIP + XML; openpyxl is not installed in .venv)
# ---------------------------------------------------------------------------
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
       "pr": "http://schemas.openxmlformats.org/package/2006/relationships"}


def _xml(z: zipfile.ZipFile, name: str) -> ET.Element:
    data = z.read(name)
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise ValueError(f"{name}: DTD or entity declaration in xlsx part")
    return ET.fromstring(data)


def _col_index(ref: str) -> tuple[int, int]:
    m = re.match(r"^([A-Z]+)(\d+)$", ref)
    if not m:
        raise ValueError(ref)
    col = 0
    for ch in m.group(1):
        col = col * 26 + (ord(ch) - 64)
    return int(m.group(2)), col


def read_xlsx(path: Path, sheet: str) -> dict[tuple[int, int], str]:
    """Cell values of one sheet as {(row, col): text} (1-based; shared strings
    resolved, numbers as their XML text, formulas as their cached value)."""
    with zipfile.ZipFile(path) as z:
        wb = _xml(z, "xl/workbook.xml")
        rels = _xml(z, "xl/_rels/workbook.xml.rels")
        target = {r.get("Id"): r.get("Target") for r in rels.findall("pr:Relationship", _NS)}
        rid = None
        for s in wb.find("m:sheets", _NS):
            if s.get("name") == sheet:
                rid = s.get(f"{{{_NS['r']}}}id")
        if rid is None:
            raise KeyError(sheet)
        part = target[rid].lstrip("/")
        part = part if part.startswith("xl/") else "xl/" + part
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in _xml(z, "xl/sharedStrings.xml").findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        cells = {}
        for c in _xml(z, part).iter(f"{{{_NS['m']}}}c"):
            v = c.find("m:v", _NS)
            t = c.get("t")
            if t == "inlineStr":
                text = "".join(x.text or "" for x in c.iter(f"{{{_NS['m']}}}t"))
            elif v is None:
                continue
            elif t == "s":
                text = shared[int(v.text)]
            else:
                text = v.text or ""
            cells[_col_index(c.get("r"))] = text
    return cells


# ---------------------------------------------------------------------------
# W/L map (sheet FETModules)
# ---------------------------------------------------------------------------
_WL_RE = re.compile(r"^\s*W\s*=\s*([0-9.]+)\s*/+\s*L\s*=\s*([0-9.]+)\s*$")
_OFFSET_RE = re.compile(r"\+\s*(\d+)")


def layout_map(path: Path = LAYOUT_XLSX) -> dict[tuple[int, int], dict]:
    """{(global module, position): {device, W_um, L_um, row}} from sheet
    ``FETModules``.  A block starts at a 'Device Name' / 'Device' row (column
    B = device name); its next row carries the module offset in column A
    ('Column / Row #' = 0, 'Column # +N' = N); a row with a number in column A
    starts a module (local number), rows 1-8 of the module are positions 1-8."""
    cells = read_xlsx(path, "FETModules")
    rows = sorted({r for r, _ in cells})
    out: dict[tuple[int, int], dict] = {}
    device, offset, module, pos = None, 0, None, 0
    for r in rows:
        a, b = cells.get((r, 1), "").strip(), cells.get((r, 2), "").strip()
        if a in ("Device Name", "Device"):
            device, module, pos = b.replace("sky130_fd_pr__", ""), None, 0
            continue
        if a.startswith("Column"):
            m = _OFFSET_RE.search(a)
            offset = int(m.group(1)) if m else 0
            continue
        wl = _WL_RE.match(b)
        if not wl or device is None:
            continue
        if a:
            module, pos = int(float(a)) + offset, 0
        pos += 1
        if pos > 8:
            raise ValueError(f"row {r}: more than 8 positions in module {module}")
        key = (module, pos)
        if key in out:
            raise ValueError(f"row {r}: module {module} position {pos} twice")
        out[key] = {"device": device, "W_um": float(wl.group(1)), "L_um": float(wl.group(2)),
                    "row": r, "text": b}
    return out


# ---------------------------------------------------------------------------
# Curves and devices
# ---------------------------------------------------------------------------
@dataclass
class NistCurve:
    kind: str                  # idvg | idvd
    fixed: float               # Vd (idvg) or Vg (idvd), signed, V (block ICCAP_VAR)
    vb: float                  # bulk bias, V (block ICCAP_VAR)
    sweep: np.ndarray          # swept voltage, signed
    id: np.ndarray             # drain current as stored (pMOS negative), A
    aux: np.ndarray            # IG (idvg files) or IB (idvd files), A
    key: str = ""              # idvg@d<k> / idvd@g<k> for the Vb = 0 curves

    @property
    def Vg(self) -> np.ndarray:
        return self.sweep if self.kind == "idvg" else np.full(len(self.sweep), self.fixed)

    @property
    def Vd(self) -> np.ndarray:
        return self.sweep if self.kind == "idvd" else np.full(len(self.sweep), self.fixed)

    @property
    def Id(self) -> np.ndarray:
        return self.id


@dataclass
class NistDevice:
    tag: str
    folder: str
    dev_type: str              # sky130 device type, e.g. nfet_01v8_lvt
    polarity: str              # nmos | pmos
    module: int
    position: int
    W_um: float
    L_um: float
    files: dict = field(default_factory=dict)        # IDVG / IDVD -> relative path
    inputs: dict = field(default_factory=dict)       # IDVG / IDVD -> header inputs
    curves: list = field(default_factory=list)       # every block of both files
    flags: list = field(default_factory=list)

    @property
    def vdd(self) -> float:
        return VDD


def device_tag(dev_type: str, W_um: float, L_um: float, module: int, position: int) -> str:
    f = lambda x: f"{x:g}".replace(".", "p")
    return f"{dev_type}_W{f(W_um)}_L{f(L_um)}_m{module}p{position}"


def _inputs_row(mdm: MdmFile) -> dict:
    return {n: {"smu": i.smu, "compliance": i.compliance, "mode": i.mode, "values": i.values}
            for n, i in mdm.inputs.items()}


def file_curves(mdm: MdmFile, kind: str) -> list[NistCurve]:
    swept, fixed_name = ("VG", "VD") if kind == "idvg" else ("VD", "VG")
    aux_name = "IG" if kind == "idvg" else "IB"
    out = []
    for b in mdm.blocks:
        col = {c.upper(): j for j, c in enumerate(b.columns)}
        if b.columns[0].upper() != swept:
            raise ValueError(f"{kind}: first column {b.columns[0]} != {swept}")
        fv = {k.upper(): v for k, v in b.fixed.items()}
        if abs(fv.get("VS", 0.0)) > 0:
            raise ValueError(f"{kind}: nonzero source bias")
        out.append(NistCurve(kind, float(fv[fixed_name]), float(fv["VB"]), b.data[:, 0].copy(),
                             b.data[:, col["ID"]].copy(),
                             b.data[:, col[aux_name]].copy() if aux_name in col
                             else np.zeros(0)))
    return out


def assign_keys(curves: list[NistCurve]) -> None:
    """Keys of the Vb = 0 curves by ascending |fixed bias|."""
    for kind, prefix in (("idvg", "idvg@d"), ("idvd", "idvd@g")):
        sel = [c for c in curves if c.kind == kind and c.vb == 0.0]
        for k, c in enumerate(sorted(sel, key=lambda c: abs(c.fixed))):
            c.key = f"{prefix}{k}"


def read_flags(root: Path = MPW5_ROOT) -> dict[tuple[str, int, int], str]:
    """Folder readme lines '<module>_<position> <text>' -> {(folder, m, p): text}."""
    out = {}
    for folder, name in FLAG_README.items():
        path = root / "transistormodule" / folder / name
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            m = re.match(r"^\s*(\d+)_(\d+)\s+(.*\S)\s*$", line)
            if m:
                out[(folder, int(m.group(1)), int(m.group(2)))] = m.group(3)
    return out


def discover_mpw5(root: Path = MPW5_ROOT, folders: Sequence[str] = tuple(FOLDERS)
                  ) -> list[NistDevice]:
    wl = layout_map(root / "Module Layout Coding.xlsx")
    flags = read_flags(root)
    devices = []
    for folder in folders:
        dev_type, pol = FOLDERS[folder]
        found: dict[tuple[int, int], dict] = {}
        for path in sorted((root / "transistormodule" / folder).iterdir()):
            if path.name == FLAG_README.get(folder):
                continue
            m = _FILE_RE.match(path.name)
            if not m:
                raise ValueError(f"unexpected file {path.name}")
            found.setdefault((int(m.group(1)), int(m.group(2))), {})[m.group(3)] = path
        for (module, pos), by_kind in sorted(found.items()):
            info = wl.get((module, pos))
            if info is None:
                raise KeyError(f"{folder} {module}_{pos}: not in the layout sheet")
            if info["device"] != dev_type:
                raise ValueError(f"{folder} {module}_{pos}: sheet device {info['device']}")
            dev = NistDevice(device_tag(dev_type, info["W_um"], info["L_um"], module, pos),
                             folder, dev_type, pol, module, pos, info["W_um"], info["L_um"])
            for name, kind in (("IDVG", "idvg"), ("IDVD", "idvd")):
                if name not in by_kind:
                    dev.flags.append(f"missing {name}")
                    continue
                mdm = parse_mdm(by_kind[name])
                dev.files[name] = by_kind[name].relative_to(root).as_posix()
                dev.inputs[name] = _inputs_row(mdm)
                dev.curves.extend(file_curves(mdm, kind))
            if (folder, module, pos) in flags:
                dev.flags.append(f"readme: {flags[(folder, module, pos)]}")
            assign_keys(dev.curves)
            devices.append(dev)
    return devices


# ---------------------------------------------------------------------------
# ETest tile (inventory only)
# ---------------------------------------------------------------------------
_ETEST_RE = re.compile(r"^sky130_fd_pr__(\w+?)_w([0-9p]+)u_l([0-9p]+)u_m(\d+)"
                       r"\((\d+)_(\d+)_(\d+)_(IDVG|IDVD)\)_4K\.mdm$")


def discover_etest(root: Path = ETEST_ROOT) -> list[NistDevice]:
    found: dict[tuple, dict] = {}
    for path in sorted((root / "Sky130-4k").iterdir()):
        m = _ETEST_RE.match(path.name)
        if not m:
            raise ValueError(f"unexpected file {path.name}")
        dev_type = m.group(1)
        W, L = (float(m.group(i).replace("p", ".")) for i in (2, 3))
        key = (dev_type, W, L, int(m.group(4)), int(m.group(5)), m.group(6), m.group(7))
        found.setdefault(key, {})[m.group(8)] = path
    devices = []
    for (dev_type, W, L, mult, module, pa, pb), by_kind in sorted(found.items()):
        pol = "pmos" if dev_type.startswith("pfet") else "nmos"
        dev = NistDevice(f"etest_{device_tag(dev_type, W, L, module, int(pa))}", "ETest",
                         dev_type, pol, module, int(pa), W, L)
        if mult != 1:
            dev.flags.append(f"m={mult}")
        for name, kind in (("IDVG", "idvg"), ("IDVD", "idvd")):
            if name not in by_kind:
                dev.flags.append(f"missing {name}")
                continue
            mdm = parse_mdm(by_kind[name])
            dev.files[name] = by_kind[name].relative_to(root).as_posix()
            dev.inputs[name] = _inputs_row(mdm)
            dev.curves.extend(file_curves(mdm, kind))
        assign_keys(dev.curves)
        devices.append(dev)
    return devices


# ---------------------------------------------------------------------------
# Checks (inventory)
# ---------------------------------------------------------------------------
GATE_CONTROL_MIN = 100.0     # |Id|(Vg = max, Vd = max) / |Id|(Vg = 0, Vd = max), Id-Vd at Vb = 0
COMPLIANCE_FRAC = 0.98


def curve_checks(c: NistCurve, compliance_d: float) -> dict:
    a = np.abs(c.id)
    finite = np.isfinite(c.id)
    sgn = np.sign(c.fixed if c.kind == "idvg" else c.sweep[-1])
    return {"n_points": int(len(c.id)), "n_nonfinite": int(np.sum(~finite)),
            "all_zero": bool(np.all(c.id[finite] == 0.0)) if finite.any() else True,
            "max_abs_id": float(np.nanmax(a)) if finite.any() else float("nan"),
            "at_compliance": bool(np.nanmax(a) >= COMPLIANCE_FRAC * compliance_d)
            if finite.any() else False,
            "n_sign_wrong": int(np.sum(np.sign(c.id) == -sgn)),
            "tail_run_at_max": int(_tail_run(a))}


def _tail_run(a: np.ndarray) -> int:
    if not len(a):
        return 0
    top, run = a.max(), 0
    for x in a[::-1]:
        if x != top:
            break
        run += 1
    return run


def data_digest(dev: NistDevice) -> str:
    h = hashlib.sha256()
    for c in dev.curves:
        h.update(np.ascontiguousarray(c.id, dtype=np.float64).tobytes())
    return h.hexdigest()


def gate_control(dev: NistDevice) -> float | None:
    vd0 = [c for c in dev.curves if c.kind == "idvd" and c.vb == 0.0]
    if not vd0:
        return None
    lo = min(vd0, key=lambda c: abs(c.fixed))
    hi = max(vd0, key=lambda c: abs(c.fixed))
    den = abs(lo.id[-1])
    return float(abs(hi.id[-1]) / den) if den > 0 else float("inf")


def device_checks(dev: NistDevice) -> dict:
    comp = {k: float(v["VD"]["compliance"]) for k, v in dev.inputs.items()}
    per = [curve_checks(c, comp["IDVG" if c.kind == "idvg" else "IDVD"]) for c in dev.curves]
    gc = gate_control(dev)
    bad = []
    if any(p["n_nonfinite"] for p in per):
        bad.append("nonfinite")
    if any(p["all_zero"] for p in per):
        bad.append("all_zero_curve")
    if any(p["at_compliance"] for p in per):
        bad.append("drain_compliance")
    if gc is not None and gc < GATE_CONTROL_MIN:
        bad.append(f"gate_control {gc:.3g} < {GATE_CONTROL_MIN:g}")
    if len(dev.files) < 2:
        bad.append("missing file")
    return {"gate_control_ratio": gc, "bad": bad,
            "n_nonfinite": sum(p["n_nonfinite"] for p in per),
            "max_abs_id": max((p["max_abs_id"] for p in per), default=float("nan")),
            "n_sign_wrong_vb0": sum(p["n_sign_wrong"] for p, c in zip(per, dev.curves)
                                    if c.vb == 0.0),
            "max_tail_run_at_max": max((p["tail_run_at_max"] for p in per), default=0)}


def sweep_summary(devs: Sequence[NistDevice]) -> dict:
    """Bias grid of a device group: per kind the fixed-bias values, Vb values
    and point counts seen (union over devices)."""
    out = {}
    for kind in ("idvg", "idvd"):
        cs = [c for d in devs for c in d.curves if c.kind == kind]
        out[kind] = {
            "fixed_V": sorted({round(c.fixed, 6) for c in cs}, key=abs),
            "vb_V": sorted({round(c.vb, 6) for c in cs}, key=abs),
            "n_points": sorted({len(c.sweep) for c in cs}),
            "sweep_V": sorted({(round(float(c.sweep[0]), 6), round(float(c.sweep[-1]), 6))
                               for c in cs}),
            "step_V": sorted({round(float(np.median(np.diff(c.sweep))), 6) for c in cs
                              if len(c.sweep) > 1}),
            "curves_per_device": sorted({sum(1 for c in d.curves if c.kind == kind)
                                         for d in devs})}
    return out


def inventory(devices: Sequence[NistDevice], tile: str) -> dict:
    by_type: dict[str, list[NistDevice]] = {}
    for d in devices:
        by_type.setdefault(d.dev_type, []).append(d)
    digests: dict[str, list[str]] = {}
    for d in devices:
        digests.setdefault(data_digest(d), []).append(d.tag)
    rows = {}
    for t, devs in sorted(by_type.items()):
        wl = [(d.W_um, d.L_um) for d in devs]
        dup_wl = sorted({f"W{w:g}/L{l:g}" for w, l in wl if wl.count((w, l)) > 1})
        checks = {d.tag: device_checks(d) for d in devs}
        rows[t] = {
            "tile": tile, "folder": sorted({d.folder for d in devs}),
            "polarity": devs[0].polarity, "n_devices": len(devs), "n_wl": len(set(wl)),
            "W_um": [min(w for w, _ in wl), max(w for w, _ in wl)],
            "L_um": [min(l for _, l in wl), max(l for _, l in wl)],
            "sweeps": sweep_summary(devs),
            "flagged_readme": {d.tag: [f for f in d.flags if f.startswith("readme")]
                               for d in devs if any(f.startswith("readme") for f in d.flags)},
            "other_flags": {d.tag: [f for f in d.flags if not f.startswith("readme")]
                            for d in devs if any(not f.startswith("readme") for f in d.flags)},
            "duplicate_wl": dup_wl,
            "bad": {k: v["bad"] for k, v in checks.items() if v["bad"]},
            "max_abs_id_A": max(v["max_abs_id"] for v in checks.values()),
            "gate_control_ratio_min": min((v["gate_control_ratio"] for v in checks.values()
                                           if v["gate_control_ratio"] is not None), default=None),
            "devices": [{"tag": d.tag, "module": d.module, "position": d.position,
                         "W_um": d.W_um, "L_um": d.L_um, "files": d.files, "flags": d.flags,
                         **checks[d.tag]} for d in devs]}
    return {"tile": tile, "n_devices": len(devices), "types": rows,
            "identical_current_data": [v for v in digests.values() if len(v) > 1]}


# ---------------------------------------------------------------------------
# Pipeline format (77 K curve format, Vb = 0 curves)
# ---------------------------------------------------------------------------
def fit_curves(dev: NistDevice) -> list[NistCurve]:
    """The Vb = 0 curves in key order (Id-Vg by |Vd|, then Id-Vd by |Vg|)."""
    sel = [c for c in dev.curves if c.key]
    return sorted(sel, key=lambda c: (c.kind != "idvg", int(c.key.split("@")[1][1:])))


def to_pipeline_curves(dev: NistDevice):
    """``cryoml.data_io.Curve`` list in the 77 K convention: signed voltages,
    nMOS Id > 0, pMOS Id as a positive magnitude (spice_pdk.simulate_pdk
    returns pMOS currents negated, as the upstream 77 K curves)."""
    from ..data_io import Curve
    sign = -1.0 if dev.polarity == "pmos" else 1.0
    out = []
    for c in fit_curves(dev):
        out.append(Curve(c.kind, float(c.fixed), np.asarray(c.Vg, float), np.asarray(c.Vd, float),
                         sign * np.asarray(c.id, float), path=dev.files.get(
                             "IDVG" if c.kind == "idvg" else "IDVD", ""),
                         source="nist_mds2_2997_4K", meta={"key": c.key, "vb": c.vb}))
    return out


def inclusion_measured(kind: str, i_meas: np.ndarray) -> dict:
    """pdk2.metric rule (I5-1, I6-1): clean, trim to the first |I| > 1e-10 A,
    include iff trimmed mean |I| > 3e-7 A (Id-Vd) / 2e-7 A (Id-Vg)."""
    start, _, den = pdk2_metric.curve_trim(kind, i_meas)
    inc = bool(np.isfinite(den) and den > pdk2_metric.INCLUDE_MIN[kind])
    return {"start": int(start), "den": float(den), "included": inc}


def inclusion_rrmscalc_pmos(kind: str, i_meas: np.ndarray, i_sim: np.ndarray) -> dict:
    """rrmsCalc pMOS rule with the 1e-10 A trim floor: include iff trimmed mean
    |I_meas| > 0 and |I_sim| at the last trimmed point > 1e-10 A (Id-Vd) /
    1e-11 A (Id-Vg).  Depends on the simulation; frozen from one card."""
    start, _, den = pdk2_metric.curve_trim(kind, i_meas)
    last = float(abs(np.asarray(i_sim, float)[-1])) if len(i_sim) else 0.0
    lim = 1e-10 if kind == "idvd" else 1e-11
    return {"start": int(start), "den": float(den),
            "included": bool(den > 0 and np.isfinite(last) and last > lim)}


INDEX_COLUMNS = "device|i|key|kind|fixed_V|vb_V|n_points (npz prefix d<device order>_c<i>)"


def save_dataset(devices: Sequence[NistDevice], npz_path: Path) -> dict:
    """npz of the Vb = 0 curves of ``devices`` (raw sign); returns index rows."""
    arrays, index = {}, []
    for d_i, dev in enumerate(devices):
        for c_i, c in enumerate(fit_curves(dev)):
            p = f"d{d_i}_c{c_i}"
            arrays[f"{p}_sweep"] = c.sweep
            arrays[f"{p}_id"] = c.id
            index.append("|".join([dev.tag, str(c_i), c.key, c.kind, f"{c.fixed:.17g}",
                                   f"{c.vb:.17g}", str(len(c.sweep))]))
    Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    return {"index_columns": INDEX_COLUMNS, "index": index}


def device_row(dev: NistDevice) -> dict:
    return {"tag": dev.tag, "folder": dev.folder, "dev_type": dev.dev_type,
            "polarity": dev.polarity, "module": dev.module, "position": dev.position,
            "W_um": dev.W_um, "L_um": dev.L_um, "files": dev.files, "flags": dev.flags}


def load_dataset(npz_path: Path, manifest: dict, tags: Sequence[str] | None = None
                 ) -> list[NistDevice]:
    keep = None if tags is None else set(tags)
    devices, order = {}, {}
    for i, row in enumerate(manifest["devices"]):
        order[row["tag"]] = i
        if keep is not None and row["tag"] not in keep:
            continue
        devices[row["tag"]] = NistDevice(row["tag"], row["folder"], row["dev_type"],
                                         row["polarity"], row["module"], row["position"],
                                         row["W_um"], row["L_um"], dict(row["files"]),
                                         flags=list(row["flags"]))
    with np.load(npz_path, allow_pickle=False) as data:
        for line in manifest["index"]:
            tag, c_i, key, kind, fixed, vb, _ = line.split("|")
            if tag not in devices:
                continue
            p = f"d{order[tag]}_c{c_i}"
            devices[tag].curves.append(NistCurve(kind, float(fixed), float(vb),
                                                 np.asarray(data[f"{p}_sweep"]),
                                                 np.asarray(data[f"{p}_id"]), np.zeros(0), key))
    return [devices[t] for t in sorted(devices)]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def make_split(devices: Sequence[NistDevice], seed: int, block: int = 5,
               n_dev_per_block: int = 3) -> dict:
    """Per family, devices sorted by (L, W) and cut into consecutive blocks of
    ``block``; in each block a seeded permutation puts ``n_dev_per_block`` in
    DEV and the rest in TEST (a short last block: round(0.6 n) DEV).  This
    stratifies DEV and TEST over the L and W range of each family.
    ``numpy.random.default_rng(seed)``, families in sorted order."""
    rng = np.random.default_rng(seed)
    fams: dict[str, list[NistDevice]] = {}
    for d in devices:
        fams.setdefault(d.dev_type, []).append(d)
    dev, test, blocks = [], [], {}
    for fam in sorted(fams):
        ds = sorted(fams[fam], key=lambda d: (d.L_um, d.W_um, d.tag))
        blocks[fam] = []
        for b0 in range(0, len(ds), block):
            blk = ds[b0:b0 + block]
            k = n_dev_per_block if len(blk) == block else int(round(len(blk) * n_dev_per_block / block))
            perm = rng.permutation(len(blk))
            d_tags = sorted(blk[i].tag for i in perm[:k])
            t_tags = sorted(blk[i].tag for i in perm[k:])
            dev += d_tags
            test += t_tags
            blocks[fam].append({"DEV": d_tags, "TEST": t_tags})
    return {"seed": seed, "block": block, "n_dev_per_block": n_dev_per_block,
            "DEV": sorted(dev), "TEST": sorted(test), "n_dev": len(dev), "n_test": len(test),
            "blocks": blocks}


def json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(type(o))


def _finite(o):
    """Non-finite floats -> None (JSON null), recursively."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (float, np.floating)) and not math.isfinite(float(o)):
        return None
    return o


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_finite(obj), indent=1, default=json_default, allow_nan=False)
                    + "\n")
