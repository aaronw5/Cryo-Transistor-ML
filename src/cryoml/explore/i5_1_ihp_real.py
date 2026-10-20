"""I5-1: the I3-2 bias-only correction on measured IHP SG13G2 transistors.

Pre-registration: ``out/research/explore/I5_1/prereg.json``.

Data: the IHP-Open-PDK MOS measurement folder (``ihp-sg13g2/libs.doc/meas/
MOS``, commit ``5e6d592e``, Apache-2.0; Keysight IC-CAP MDM text files),
downloaded to ``data/external/ihp_sg13g2_open_pdk_meas``.  28 DUTs (8 LV
nMOS, 8 LV pMOS, 6 HV nMOS, 6 HV pMOS) at 233, 300, 343 and 398 K
(header TEMP -40, 27, 70, 125 C), five setups per DUT and temperature:

* ``dc_idvg``: Vg sweep at three Vd values and 5 (nMOS) or 7 (pMOS) Vb values;
* ``dc_idvd`` / ``dc_idvd_vbmin``: Vd sweep at five Vg values, Vb = 0 / Vbmin;
* ``dc_idvd_vth`` / ``dc_idvd_vth_vbmin``: Vd sweep at three near-threshold
  Vg values, Vb = 0 / Vbmin.

Source grounded, ``id`` is the current into the drain terminal (nMOS
positive, pMOS negative).  Files are read as text (``parse_mdm``); nothing in
the download is executed.

Fitted / scored curves (``FIT_SETUPS``): the Vb = 0 curves only, 11 per DUT
and temperature: Id-Vg at the three Vd values (keys ``idvg@d0..d2``, by
|Vd|), Id-Vd at the five ``dc_idvd`` Vg values (``idvd@g0..g4``, by |Vg|)
and at the three ``dc_idvd_vth`` Vg values (``idvd@t0..t2``).  Curve
inclusion and trimming: ``cryoml.pdk2.metric`` (``clean_current``; trim to
the first |I| > 1e-10 A; include iff trimmed mean |I| > 3e-7 A (Id-Vd) or
2e-7 A (Id-Vg)); fixed by the measured current alone.

Simulator (``simulate``): PSP 103.6 (``psp103.osdi``, OpenVAF-Reloaded) on
ngspice-46 as ``cryoml.pdk2.ihp``, generalized to the HV family
(``sg13g2_moshv_parm.lib``, ``cornerMOShv.lib`` mos_tt), a bulk source
``VB``, the header AD/AS/PD/PS values and the measured bias grid; the
deck temperature is the header TEMP.  ``simulate_pdk_subckt`` runs the
PDK's own ``sg13_{lv,hv}_{n,p}mos`` subcircuit through the corner library
(stock reference check).  Currents are returned as the current into the
drain terminal, -i(VD), with the measured sign convention.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import ROOT
from ..metrics import clean_current
from ..parameter_space import ParameterBox
from ..pdk2 import metric as pdk2_metric
from ..pdk2.ihp import (MODELS_DIR, OSDI, _card_value_text, _safe_eval,
                        ngspice46_bin, patch_card)
from .e1_correction import CurveRec, DeviceRec

MEAS_ROOT = (ROOT / "data" / "external" / "ihp_sg13g2_open_pdk_meas" / "ihp-sg13g2"
             / "libs.doc" / "meas" / "MOS")
MEAS_COMMIT = "5e6d592e4002946a4616f798c357f0f3c06cf3b6"
FOLDERS = {"SG13_nmosXm1Y3": ("lv", "nmos"), "SG13_pmosXm1Y3": ("lv", "pmos"),
           "SG13_nmosHVXm1Y3": ("hv", "nmos"), "SG13_pmosHVXm1Y3": ("hv", "pmos")}
SETUPS = ("dc_idvg", "dc_idvd", "dc_idvd_vbmin", "dc_idvd_vth", "dc_idvd_vth_vbmin")
FIT_SETUPS = ("dc_idvg", "dc_idvd", "dc_idvd_vth")
TEMPS_K = (233, 300, 343, 398)
VDD = {"lv": 1.2, "hv": 3.3}               # nominal supply (normalization of g inputs)
CARD_MODEL = "i5psp"
TIMEOUT_S = 120.0
_FILE_RE = re.compile(r"^SG13_(n|p)mos(HV)?~(W\d+u\d+_L\d+u\d+_S\d+_\d+)~"
                      r"(dc_[a-z_]+)~(\d+)K\.mdm$")


# ---------------------------------------------------------------------------
# MDM parsing
# ---------------------------------------------------------------------------
@dataclass
class MdmInput:
    name: str
    smu: str
    compliance: float
    mode: str                  # LIN | LIST | CON
    values: list               # LIN: start, stop, n, step; LIST: values; CON: [value]


@dataclass
class MdmBlock:
    fixed: dict                # ICCAP_VAR name -> value
    columns: list
    data: np.ndarray           # (n_points, n_columns)


@dataclass
class MdmFile:
    inputs: dict
    outputs: list
    values: dict
    blocks: list


def _num(text: str) -> float:
    return float(text)


def parse_mdm(path: Path) -> MdmFile:
    """Read one IC-CAP MDM text file (data only)."""
    lines = Path(path).read_text(errors="replace").splitlines()
    inputs, outputs, values, blocks = {}, [], {}, []
    section = None
    i = 0
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        i += 1
        if s in ("ICCAP_INPUTS", "ICCAP_OUTPUTS", "ICCAP_VALUES"):
            section = s
            continue
        if s == "END_HEADER":
            section = None
            continue
        if s == "BEGIN_DB":
            fixed, columns, rows = {}, [], []
            while i < len(lines):
                t = lines[i].strip()
                i += 1
                if t == "END_DB":
                    break
                if not t:
                    continue
                if t.startswith("ICCAP_VAR"):
                    parts = t.split()
                    fixed[parts[1]] = _num(parts[2])
                elif t.startswith("#"):
                    columns = t[1:].split()
                else:
                    rows.append([_num(x) for x in t.split()])
            blocks.append(MdmBlock(fixed, columns, np.asarray(rows, dtype=np.float64)))
            continue
        if section == "ICCAP_INPUTS" and s:
            p = s.split()
            mode = p[6]
            if mode == "LIN":
                vals = [_num(p[8]), _num(p[9]), int(p[10]), _num(p[11])]
            elif mode == "LIST":
                n = int(p[8])
                vals = [_num(x) for x in p[9:9 + n]]
            elif mode == "CON":
                vals = [_num(p[7])]
            else:
                raise ValueError(f"{path}: unknown input mode {mode}")
            inputs[p[0]] = MdmInput(p[0], p[4], _num(p[5]), mode, vals)
        elif section == "ICCAP_OUTPUTS" and s:
            outputs.append(s.split()[0])
        elif section == "ICCAP_VALUES" and s:
            m = re.match(r'^(\S+)\s+"(.*)"\s*$', s)
            if m:
                values[m.group(1)] = m.group(2).strip()
    return MdmFile(inputs, outputs, values, blocks)


_SI = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "": 1.0}


def si_value(text: str) -> float:
    m = re.match(r"^\s*([-+0-9.eE]+?)([fpnum]?)\s*$", text)
    if not m:
        raise ValueError(f"bad SI value {text!r}")
    return float(m.group(1)) * _SI[m.group(2)]


def dut_geometry(dut: str) -> tuple[float, float]:
    """(W_um, L_um) from a DUT name such as ``W10u0_L0u13_S540_2``."""
    m = re.match(r"^W(\d+)u(\d+)_L(\d+)u(\d+)_", dut)
    if not m:
        raise ValueError(dut)
    return float(f"{int(m.group(1))}.{m.group(2)}"), float(f"{int(m.group(3))}.{m.group(4)}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class MeasCurve:
    setup: str                 # dc_idvg | dc_idvd | ...
    kind: str                  # idvg | idvd
    fixed: float               # Vd (idvg) or Vg (idvd), signed, V
    vb: float                  # bulk bias, V
    sweep: np.ndarray          # swept voltage (Vg or Vd)
    id: np.ndarray             # drain current (into the drain terminal), A
    ig: np.ndarray
    ib: np.ndarray
    is_: np.ndarray
    key: str = ""              # canonical key for the fitted curves ("" otherwise)

    @property
    def vg(self) -> np.ndarray:
        return self.sweep if self.kind == "idvg" else np.full(len(self.sweep), self.fixed)

    @property
    def vd(self) -> np.ndarray:
        return self.sweep if self.kind == "idvd" else np.full(len(self.sweep), self.fixed)

    # ``cryoml.data_io.Curve``-style aliases (used by i2_1_joint / i4_1_baselines)
    Vg = property(lambda self: self.vg)
    Vd = property(lambda self: self.vd)
    Id = property(lambda self: self.id)
    meta = property(lambda self: {})


@dataclass
class MeasDevice:
    tag: str
    cls: str                   # lv | hv
    dev_type: str              # nmos | pmos
    W_um: float
    L_um: float
    dut: str
    geom: dict                 # AD, AS, PD, PS (SI units), NF, M
    temps_C: dict = field(default_factory=dict)      # T_K -> header TEMP (C)
    curves: dict = field(default_factory=dict)       # T_K -> list[MeasCurve]

    @property
    def geometry(self) -> str:
        return f"{self.cls}_W{self.W_um:g}_L{self.L_um:g}"

    @property
    def vdd(self) -> float:
        return VDD[self.cls]


def device_tag(cls: str, dev_type: str, W_um: float, L_um: float) -> str:
    f = lambda x: f"{x:g}".replace(".", "p")
    return f"{cls}_{dev_type}_W{f(W_um)}_L{f(L_um)}"


def _kind(setup: str) -> str:
    return "idvg" if setup == "dc_idvg" else "idvd"


def file_curves(mdm: MdmFile, setup: str) -> list[MeasCurve]:
    """One ``MeasCurve`` per data block."""
    kind = _kind(setup)
    swept = "vg" if kind == "idvg" else "vd"
    fixed_name = "vd" if kind == "idvg" else "vg"
    out = []
    for b in mdm.blocks:
        col = {c: j for j, c in enumerate(b.columns)}
        if b.columns[0] != swept:
            raise ValueError(f"{setup}: first column {b.columns[0]} != {swept}")
        fixed = b.fixed.get(fixed_name)
        vb = b.fixed.get("vb")
        if vb is None:
            vb = mdm.inputs["vb"].values[0]
        if fixed is None:
            raise ValueError(f"{setup}: block without {fixed_name}")
        if abs(b.fixed.get("vs", 0.0)) > 0:
            raise ValueError(f"{setup}: nonzero source bias")
        out.append(MeasCurve(setup, kind, float(fixed), float(vb), b.data[:, 0].copy(),
                             b.data[:, col["id"]].copy(), b.data[:, col["ig"]].copy(),
                             b.data[:, col["ib"]].copy(), b.data[:, col["is"]].copy()))
    return out


def assign_keys(curves: list[MeasCurve]) -> None:
    """Canonical keys of the fitted (Vb = 0) curves, by |fixed bias| order."""
    groups = {"dc_idvg": "idvg@d", "dc_idvd": "idvd@g", "dc_idvd_vth": "idvd@t"}
    for setup, prefix in groups.items():
        sel = [c for c in curves if c.setup == setup and c.vb == 0.0]
        for k, c in enumerate(sorted(sel, key=lambda c: abs(c.fixed))):
            c.key = f"{prefix}{k}"


def discover(root: Path = MEAS_ROOT) -> list[MeasDevice]:
    devices: dict[str, MeasDevice] = {}
    for folder, (cls, dev_type) in FOLDERS.items():
        for path in sorted((root / folder).glob("*.mdm")):
            m = _FILE_RE.match(path.name)
            if not m:
                raise ValueError(f"unexpected file name {path.name}")
            dut, setup, temp_k = m.group(3), m.group(4), int(m.group(5))
            if setup not in SETUPS:
                raise ValueError(f"unexpected setup {setup}")
            W_um, L_um = dut_geometry(dut)
            tag = device_tag(cls, dev_type, W_um, L_um)
            mdm = parse_mdm(path)
            v = mdm.values
            pol = {"nmos": "1", "pmos": "-1"}[dev_type]
            if v.get("TYPE") != pol:
                raise ValueError(f"{path.name}: TYPE {v.get('TYPE')} != {pol}")
            if abs(si_value(v["MAIN.W"]) - W_um * 1e-6) > 1e-3 * W_um * 1e-6 or \
                    abs(si_value(v["MAIN.L"]) - L_um * 1e-6) > 1e-3 * L_um * 1e-6:
                raise ValueError(f"{path.name}: header W/L differ from the DUT name")
            temp_c = float(v["TEMP"])
            if abs(temp_c + 273.15 - temp_k) > 0.5:
                raise ValueError(f"{path.name}: TEMP {temp_c} C vs {temp_k} K")
            geom = {k: si_value(v[f"MAIN.{k}"]) for k in ("AD", "AS", "PD", "PS")}
            geom.update(NF=float(v["MAIN.NF"]), M=float(v["MAIN.M"]))
            dev = devices.setdefault(tag, MeasDevice(tag, cls, dev_type, W_um, L_um, dut, geom))
            if dev.dut != dut or dev.geom != geom:
                raise ValueError(f"{tag}: two DUTs or geometry records")
            dev.temps_C[temp_k] = temp_c
            curves = file_curves(mdm, setup)
            for c in curves:
                c.compliance = {n: mdm.inputs[n].compliance for n in ("vd", "vg", "vb")}
            dev.curves.setdefault(temp_k, []).extend(curves)
    for dev in devices.values():
        for curves in dev.curves.values():
            assign_keys(curves)
    return sorted(devices.values(), key=lambda d: d.tag)


def fit_curves(dev: MeasDevice, temp_k: int) -> list[MeasCurve]:
    """The 11 fitted curves in key order."""
    order = [f"idvg@d{k}" for k in range(3)] + [f"idvd@g{k}" for k in range(5)] + \
        [f"idvd@t{k}" for k in range(3)]
    by_key = {c.key: c for c in dev.curves[temp_k] if c.key}
    missing = [k for k in order if k not in by_key]
    if missing:
        raise KeyError(f"{dev.tag} {temp_k} K: missing {missing}")
    return [by_key[k] for k in order]


def inclusion(curve: MeasCurve) -> dict:
    """Trim start, denominator and inclusion of one fitted curve (pdk2.metric)."""
    start, _, den = pdk2_metric.curve_trim(curve.kind, curve.id)
    inc = bool(np.isfinite(den) and den > pdk2_metric.INCLUDE_MIN[curve.kind])
    return {"start": int(start), "den": float(den), "included": inc}


def save_dataset(devices: list[MeasDevice], npz_path: Path) -> dict:
    """npz of every curve (all setups); returns the index rows."""
    arrays, index = {}, []
    for d_i, dev in enumerate(devices):
        for t in sorted(dev.curves):
            for c_i, c in enumerate(dev.curves[t]):
                p = f"d{d_i}_T{t}_c{c_i}"
                for name in ("sweep", "id", "ig", "ib", "is_"):
                    arrays[f"{p}_{name}"] = getattr(c, name)
                index.append("|".join([dev.tag, str(t), str(c_i), c.setup, f"{c.fixed:.6g}",
                                       f"{c.vb:.6g}", c.key, str(len(c.sweep))]))
    np.savez_compressed(npz_path, **arrays)
    return {"index_columns": INDEX_COLUMNS, "index": index}


INDEX_COLUMNS = "device|T_K|i|setup|fixed_V|vb_V|key|n_points (npz prefix d<device order>_T<T_K>_c<i>)"


def load_dataset(npz_path: Path, manifest: dict,
                 tags: Sequence[str] | None = None) -> list[MeasDevice]:
    """Rebuild ``MeasDevice`` objects from the npz and the manifest; with
    ``tags``, only those DUTs' arrays are read."""
    devices = {}
    keep = None if tags is None else set(tags)
    for row in manifest["devices"]:
        if keep is not None and row["tag"] not in keep:
            continue
        devices[row["tag"]] = MeasDevice(row["tag"], row["cls"], row["dev_type"], row["W_um"],
                                         row["L_um"], row["dut"], row["geom"],
                                         {int(k): v for k, v in row["temps_C"].items()})
    order = {row["tag"]: i for i, row in enumerate(manifest["devices"])}
    with np.load(npz_path, allow_pickle=False) as data:
        for line in manifest["index"]:
            tag, t, c_i, setup, fixed, vb, key, _ = line.split("|")
            if tag not in devices:
                continue
            p = f"d{order[tag]}_T{t}_c{c_i}"
            c = MeasCurve(setup, _kind(setup), float(fixed), float(vb),
                          np.asarray(data[f"{p}_sweep"]), np.asarray(data[f"{p}_id"]),
                          np.asarray(data[f"{p}_ig"]), np.asarray(data[f"{p}_ib"]),
                          np.asarray(data[f"{p}_is_"]), key)
            devices[tag].curves.setdefault(int(t), []).append(c)
    return sorted(devices.values(), key=lambda d: d.tag)


def curve_stats(c: MeasCurve) -> dict:
    """Range, exact zeros, smallest nonzero |id|, and resolution in the low range."""
    a = np.abs(c.id)
    nz = a[a > 0]
    low = np.sort(np.unique(nz[nz < 1e-10]))
    steps = np.diff(low)
    return {"max_abs_id": float(a.max()), "min_nonzero_abs_id": float(nz.min()) if len(nz) else 0.0,
            "n_exact_zero": int(np.sum(c.id == 0.0)),
            "min_step_below_1e-10": float(steps[steps > 0].min()) if np.any(steps > 0) else None,
            "max_sig_digits": int(max(len(f"{x:.6g}".lstrip("-").split("e")[0].replace(".", "")
                                          .lstrip("0")) for x in c.id))}


def off_floor(dev: MeasDevice, temp_k: int) -> float:
    """Median |id| of the Vb = 0, lowest-|Vd| Id-Vg curve at its three most-off
    gate biases (Vg = -0.5..-0.4 V nMOS, +0.5..+0.4 V pMOS)."""
    c = [x for x in fit_curves(dev, temp_k) if x.key == "idvg@d0"][0]
    return float(np.median(np.abs(c.id[:3])))


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def make_split(devices: list[MeasDevice], seed: int) -> dict:
    """LV: the 8 LV geometries are shared by nMOS and pMOS; 3 geometries (both
    polarities) go to TEST.  HV: nMOS and pMOS geometries differ; 2 of 6 per
    polarity go to TEST.  ``numpy.random.default_rng(seed)``, draws in the
    order LV geometries, HV nMOS, HV pMOS (sorted tags)."""
    rng = np.random.default_rng(seed)
    lv = sorted({(d.W_um, d.L_um) for d in devices if d.cls == "lv"})
    for pol in ("nmos", "pmos"):
        if sorted((d.W_um, d.L_um) for d in devices if d.cls == "lv" and d.dev_type == pol) != lv:
            raise RuntimeError("LV nMOS and pMOS geometries differ")
    test_lv = {lv[i] for i in rng.choice(len(lv), size=3, replace=False)}
    test = [d.tag for d in devices if d.cls == "lv" and (d.W_um, d.L_um) in test_lv]
    for pol in ("nmos", "pmos"):
        hv = sorted(d.tag for d in devices if d.cls == "hv" and d.dev_type == pol)
        test += [hv[i] for i in rng.choice(len(hv), size=2, replace=False)]
    test = sorted(test)
    dev = sorted(d.tag for d in devices if d.tag not in test)
    return {"seed": seed, "DEV": dev, "TEST": test, "n_dev": len(dev), "n_test": len(test)}


# ---------------------------------------------------------------------------
# PSP card (LV / HV)
# ---------------------------------------------------------------------------
CORNER_LIBS = {"lv": MODELS_DIR / "cornerMOSlv.lib", "hv": MODELS_DIR / "cornerMOShv.lib"}
PARM_LIBS = {"lv": MODELS_DIR / "sg13g2_moslv_parm.lib", "hv": MODELS_DIR / "sg13g2_moshv_parm.lib"}
SUBCKTS = {(c, p): f"sg13_{c}_{p}" for c in ("lv", "hv") for p in ("nmos", "pmos")}
SUBCKT_PARAMS = {"ng": 1, "m": 1, "wmin": 0.15e-6, "z1": 0.34e-6, "z2": 0.38e-6,
                 "trise": 0, "pre_layout": 1, "SWSOA": 0}


@lru_cache(maxsize=None)
def corner_knobs(cls: str) -> dict[str, float]:
    text = CORNER_LIBS[cls].read_text(errors="replace")
    m = re.search(r"^\.LIB\s+mos_tt\s*$(.*?)^\.ENDL\s+mos_tt\b", text,
                  re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if not m:
        raise RuntimeError(f"mos_tt not found in {CORNER_LIBS[cls]}")
    return {k: float(v) for k, v in re.findall(
        r"^\s*\.param\s+(\w+)\s*=\s*([0-9.eE+\-]+)", m.group(1), re.MULTILINE)}


@lru_cache(maxsize=None)
def model_block(cls: str, dev_type: str) -> str:
    """The PDK ``.model sg13g2_{cls}_{type}_psp`` block renamed to CARD_MODEL."""
    name = f"sg13g2_{cls}_{dev_type}_psp"
    text = PARM_LIBS[cls].read_text(errors="replace")
    m = re.search(rf"^\.model\s+{name}\s+psp103va\b", text, re.MULTILINE | re.IGNORECASE)
    if not m:
        raise RuntimeError(f"{name} not found")
    lines = text[m.start():].splitlines()
    keep = [lines[0]]
    for line in lines[1:]:
        s = line.strip()
        if s and not s.startswith(("+", "*")):
            break
        keep.append(line)
    block = "\n".join(keep).rstrip() + "\n"
    return re.sub(rf"^\.model\s+{name}\b", f".model {CARD_MODEL}", block, count=1,
                  flags=re.MULTILINE | re.IGNORECASE)


def _names(dev: MeasDevice) -> dict[str, float]:
    return {**corner_knobs(dev.cls), **SUBCKT_PARAMS, "w": dev.W_um * 1e-6,
            "l": dev.L_um * 1e-6}


def card_value(dev: MeasDevice, key: str, card: str | None = None) -> float:
    card = card if card is not None else model_block(dev.cls, dev.dev_type)
    return _safe_eval(_card_value_text(card, key), _names(dev))


def stock_card(dev: MeasDevice, names: Sequence[str]) -> str:
    """Model block with ``names`` written as ``%.10g`` numbers of their
    evaluated stock expressions (the rest left as PDK expressions)."""
    block = model_block(dev.cls, dev.dev_type)
    return patch_card(block, {k: card_value(dev, k, block) for k in names})


def stock_params(dev: MeasDevice, names: Sequence[str]) -> dict[str, float]:
    block = model_block(dev.cls, dev.dev_type)
    return {k: card_value(dev, k, block) for k in names}


# ---------------------------------------------------------------------------
# Decks
# ---------------------------------------------------------------------------
def _instance(dev: MeasDevice, drain: str, source: str) -> str:
    g = dev.geom
    return (f"N1 {drain} ngt {source} nb {CARD_MODEL} w={dev.W_um:.6g}u l={dev.L_um:.6g}u "
            f"nf=1 mult=1 as={g['AS']:.6g} ad={g['AD']:.6g} ps={g['PS']:.6g} "
            f"pd={g['PD']:.6g} dta=0 ngcon=2 delvto=0 factuo=1\n")


def _step(v0: float, v1: float, n: int) -> float:
    raw = (v1 - v0) / max(n - 1, 1)
    return math.copysign(math.floor(abs(raw) * 1e12) / 1e12, raw) or 0.05


def sweep_lines(curves: Sequence[MeasCurve], out_dir: Path, prefix: str) -> list[str]:
    lines = []
    for i, c in enumerate(curves):
        fixed_src, swept = ("VD", "VG") if c.kind == "idvg" else ("VG", "VD")
        v = np.asarray(c.sweep, dtype=np.float64)
        lines += [f"alter VB dc = {c.vb:.6g}\n", f"alter {fixed_src} dc = {c.fixed:.6g}\n",
                  f"dc {swept} {v[0]:.12g} {v[-1]:.12g} {_step(v[0], v[-1], len(v)):.12g}\n",
                  f"wrdata {(out_dir / f'{prefix}{i}.dat').as_posix()} -i(VD)\n"]
    return lines


def read_sweeps(out_dir: Path, prefix: str, curves: Sequence[MeasCurve]) -> list[np.ndarray]:
    """Interpolate each sweep onto the measured grid; current into the drain
    terminal (-i(VD)); missing / non-finite / short sweeps give NaN."""
    out = []
    for i, c in enumerate(curves):
        target = np.asarray(c.sweep, dtype=np.float64)
        values = np.full(len(target), np.nan)
        try:
            flat = np.array((out_dir / f"{prefix}{i}.dat").read_text().split(), dtype=np.float64)
        except (OSError, ValueError):
            flat = np.empty(0)
        if flat.size >= 4 and flat.size % 2 == 0:
            table = flat.reshape(-1, 2)
            order = np.argsort(table[:, 0])
            v, i_d = table[order, 0], table[order, 1]
            if np.all(np.isfinite(table)) and v[0] <= target.min() + 1e-6 \
                    and v[-1] >= target.max() - 1e-6:
                values = np.interp(target, v, i_d)
        out.append(values)
    return out


def _header(dev: MeasDevice, card_path: Path, temp_C: float, rs_ext_ohm: float) -> list[str]:
    params = {**SUBCKT_PARAMS, "w": f"{dev.W_um:.6g}u", "l": f"{dev.L_um:.6g}u"}
    drain, source = ("ndi", "nsi") if rs_ext_ohm > 0 else ("nd", "0")
    lines = ["* I5-1 IHP SG13G2 measured-grid DC sweeps\n",
             f".options temp={temp_C:.6g}\n",
             ".param " + " ".join(f"{k}={v}" for k, v in params.items()) + "\n"]
    lines += [f".param {k}={v!r}\n" for k, v in corner_knobs(dev.cls).items()]
    lines += [f'.include "{card_path.as_posix()}"\n', _instance(dev, drain, source)]
    if rs_ext_ohm > 0:
        lines += [f"RDEXT nd ndi {rs_ext_ohm:.10g}\n", f"RSEXT nsi 0 {rs_ext_ohm:.10g}\n"]
    lines += ["VG ngt 0 DC 0\n", "VD nd 0 DC 0\n", "VB nb 0 DC 0\n", ".control\n",
              f"pre_osdi {OSDI.as_posix()}\n", "set filetype=ascii\n"]
    return lines


def _run(deck: Path, cwd: Path, timeout_s: float) -> bool:
    try:
        subprocess.run([ngspice46_bin(), "-b", str(deck)], cwd=cwd, capture_output=True,
                       text=True, timeout=timeout_s, check=False)
        return True
    except subprocess.TimeoutExpired:
        return False


def simulate(dev: MeasDevice, card: str, curves: Sequence[MeasCurve], temp_C: float,
             params: dict[str, float] | None = None, rs_ext_ohm: float = 0.0,
             timeout_s: float = TIMEOUT_S) -> list[np.ndarray]:
    """One ngspice-46 process with every sweep of ``curves`` on the card
    ``card`` patched with ``params``."""
    with tempfile.TemporaryDirectory(prefix="i5_1_") as td_s:
        td = Path(td_s)
        card_path = td / "card.mod"
        card_path.write_text(patch_card(card, params))
        deck = td / "deck.sp"
        deck.write_text("".join(_header(dev, card_path, temp_C, rs_ext_ohm)
                                + sweep_lines(curves, td, "raw_")
                                + ["quit\n", ".endc\n", ".end\n"]))
        if not _run(deck, td, timeout_s):
            return [np.full(len(c.sweep), np.nan) for c in curves]
        return read_sweeps(td, "raw_", curves)


def _library(cls: str) -> str:
    text = CORNER_LIBS[cls].read_text(errors="replace")
    m = re.search(r"^\.LIB\s+mos_tt\s*$(.*?)^\.ENDL\s+mos_tt\b", text,
                  re.MULTILINE | re.DOTALL | re.IGNORECASE)
    return re.sub(r"^(\s*\.include\s+)(\S+)",
                  lambda mm: f'{mm.group(1)}"{(MODELS_DIR / mm.group(2)).as_posix()}"',
                  m.group(1), flags=re.MULTILINE)


def simulate_pdk_subckt(dev: MeasDevice, curves: Sequence[MeasCurve], temp_C: float,
                        timeout_s: float = TIMEOUT_S) -> list[np.ndarray]:
    """``XM1 d g s b sg13_{cls}_{type}`` through the corner library mos_tt
    with the header AD/AS/PD/PS (reference for the top-level card path)."""
    g = dev.geom
    with tempfile.TemporaryDirectory(prefix="i5_1_ref_") as td_s:
        td = Path(td_s)
        lib = td / "mos_tt.lib"
        lib.write_text(_library(dev.cls))
        deck = td / "deck.sp"
        deck.write_text("".join(
            ["* I5-1 PDK reference\n", f".options temp={temp_C:.6g}\n",
             f'.include "{lib.as_posix()}"\n',
             f"XM1 nd ngt 0 nb {SUBCKTS[(dev.cls, dev.dev_type)]} w={dev.W_um:.6g}u "
             f"l={dev.L_um:.6g}u ng=1 m=1 as={g['AS']:.6g} ad={g['AD']:.6g} "
             f"ps={g['PS']:.6g} pd={g['PD']:.6g}\n",
             "VG ngt 0 DC 0\n", "VD nd 0 DC 0\n", "VB nb 0 DC 0\n", ".control\n",
             f"pre_osdi {OSDI.as_posix()}\n",
             f"pre_setcs sourcepath = ( {MODELS_DIR.as_posix()} )\n",
             "set filetype=ascii\n"]
            + sweep_lines(curves, td, "raw_") + ["quit\n", ".endc\n", ".end\n"]))
        if not _run(deck, td, timeout_s):
            return [np.full(len(c.sweep), np.nan) for c in curves]
        return read_sweeps(td, "raw_", curves)


# ---------------------------------------------------------------------------
# Scoring and records
# ---------------------------------------------------------------------------
def score(curves: Sequence[MeasCurve], sims: Sequence[np.ndarray]) -> dict:
    """Strict paper RRMS on the fitted curves (frozen measured inclusion);
    NaN if an included curve is non-finite."""
    per, failed = {}, []
    for c, s in zip(curves, sims):
        inc = inclusion(c)
        if not inc["included"]:
            continue
        sim = np.asarray(s, dtype=np.float64)[inc["start"]:]
        tgt = clean_current(c.id)[inc["start"]:]
        if not np.all(np.isfinite(sim)):
            failed.append(c.key)
            continue
        per[c.key] = math.sqrt(float(np.mean((sim - tgt) ** 2))) / inc["den"]
    rrms = float(np.mean(list(per.values()))) if per and not failed else float("nan")
    return {"rrms": rrms, "n_curves": len(per) + len(failed), "failed": failed,
            "per_curve": per}


def to_record(dev: MeasDevice, temp_k: int, sims: Sequence[np.ndarray],
              params: dict | None = None) -> DeviceRec:
    """E1 ``DeviceRec`` (source ``ihp_meas``; vdd = class nominal supply)."""
    curves = fit_curves(dev, temp_k)
    rec = DeviceRec(dev.tag, dev.dev_type, dev.L_um, dev.W_um, dev.vdd, "ihp_meas",
                    meta={"cls": dev.cls, "T_K": temp_k, "temp_C": dev.temps_C[temp_k],
                          "params": dict(params or {})})
    for c, s in zip(curves, sims):
        inc = inclusion(c)
        rec.curves.append(CurveRec(c.key, c.kind, np.asarray(c.vg, float),
                                   np.asarray(c.vd, float), np.asarray(s, float),
                                   np.asarray(c.id, float), inc["start"], inc["den"],
                                   inc["included"]))
    return rec


# ---------------------------------------------------------------------------
# Fit problem (ngspice_fit.DeviceProblem interface)
# ---------------------------------------------------------------------------
BASE_SET = ("vfbo", "uo", "ctl", "thesato", "axo", "rsw1", "cfl",
            "muew", "themuo", "dphibo", "dphibl", "thesatl")      # pdk2 IHP wide set
BASE_OFFSET = {"vfbo": 0.2, "dphibo": 0.2, "dphibl": 0.2}       # linear +/- (V)
BASE_FACTOR = 2.0                                                # log |v|/2 .. 2|v|
BASE_ZERO_RANGE = {"ctl": (0.0, 0.05)}                           # stock value exactly 0


def base_box(dev: MeasDevice) -> ParameterBox:
    """Starting-fit and refit box around the stock card (12-parameter pdk2
    IHP wide set): VFBO, DPHIBO, DPHIBL linear stock +/- 0.2 V; any other
    parameter with stock value v != 0 log-uniform |v|/2 .. 2|v| with the
    sign of v; v = 0 linear ``BASE_ZERO_RANGE`` (only HV pMOS CTL)."""
    stock = stock_params(dev, BASE_SET)
    bounds, logs = {}, []
    for n in BASE_SET:
        v = float(stock[n])
        if n in BASE_OFFSET:
            bounds[n] = [v - BASE_OFFSET[n], v + BASE_OFFSET[n]]
            logs.append(False)
        elif v == 0.0:
            if n not in BASE_ZERO_RANGE:
                raise ValueError(f"{dev.tag}: {n} = 0 without a range")
            bounds[n] = list(BASE_ZERO_RANGE[n])
            logs.append(False)
        else:
            a, b = v / BASE_FACTOR, v * BASE_FACTOR
            bounds[n] = [min(a, b), max(a, b)]
            logs.append(True)
    return ParameterBox(names=BASE_SET, published=stock, bounds=bounds, log_scale=tuple(logs))


class MeasProblem:
    """``tag``, ``box``, ``device``, ``curves`` (fitted curves), ``layout``,
    ``n_curves``, ``n_sim``; ``simulate(local)``, ``curve_rrms``,
    ``residual_vector``, ``objective`` (as ``pdk2.problem.Pdk2Problem``)."""

    def __init__(self, dev: MeasDevice, temp_k: int, box: ParameterBox,
                 card: str | None = None, rs_ext_ohm: float = 0.0):
        self.dev = dev
        self.device = dev
        self.tag = dev.tag
        self.temp_k = temp_k
        self.temp_C = dev.temps_C[temp_k]
        self.box = box
        self.card = card if card is not None else stock_card(dev, box.names)
        self.rs_ext_ohm = rs_ext_ohm
        self.meas = fit_curves(dev, temp_k)
        self.curves = self.meas
        self.layout = []
        for index, c in enumerate(self.meas):
            inc = inclusion(c)
            if inc["included"]:
                self.layout.append({"index": index, "start": inc["start"], "tag": c.key,
                                    "target": clean_current(c.id)[inc["start"]:],
                                    "denominator": inc["den"]})
        if not self.layout:
            raise RuntimeError(f"{dev.tag} {temp_k} K: no included curves")
        self.n_curves = len(self.layout)
        self.n_sim = 0
        self.bin_index = 0

    def params(self, local) -> dict[str, float]:
        return self.box.local_to_params(np.asarray(local, dtype=np.float64))

    def simulate_params(self, params: dict, rs_ext_ohm: float | None = None) -> list[np.ndarray]:
        self.n_sim += 1
        return simulate(self.dev, self.card, self.meas, self.temp_C, params,
                        self.rs_ext_ohm if rs_ext_ohm is None else rs_ext_ohm)

    def simulate(self, local) -> list[np.ndarray]:
        return self.simulate_params(self.params(local))

    def curve_rrms(self, sims) -> np.ndarray:
        return pdk2_metric.curve_rrms(sims, self.layout)

    def residual_vector(self, sims, weights) -> np.ndarray:
        pieces = []
        scale = 1.0 / math.sqrt(self.n_curves)
        for row, w in zip(self.layout, weights):
            sim = np.asarray(sims[row["index"]], dtype=np.float64)[row["start"]:]
            n = len(row["target"])
            if not np.all(np.isfinite(sim)):
                pieces.append(np.full(n, pdk2_metric.PENALTY * w * scale / math.sqrt(n)))
                continue
            pieces.append((sim - row["target"]) * w * scale / (row["denominator"] * math.sqrt(n)))
        return np.concatenate(pieces)

    def objective(self, local) -> float:
        return float(np.mean(self.curve_rrms(self.simulate(local))))


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------
def aggregate(values: dict) -> dict:
    """all-device mean (NaN if any failed), combined (mean of nMOS / pMOS
    family means), median, failed count."""
    v = np.array(list(values.values()), dtype=float)
    finite = v[np.isfinite(v)]
    fam = {f: [x for t, x in values.items() if f"_{f}_" in t] for f in ("nmos", "pmos")}
    out = {"n": int(len(v)), "n_failed": int(np.sum(~np.isfinite(v))),
           "all_device_mean": float(np.mean(v)) if len(v) and np.all(np.isfinite(v)) else float("nan"),
           "valid_only_mean": float(np.mean(finite)) if len(finite) else float("nan"),
           "median": float(np.median(finite)) if len(finite) else float("nan")}
    means = [float(np.mean(fam[f])) for f in ("nmos", "pmos") if fam[f]]
    out["combined"] = float(np.mean(means)) if len(means) == 2 else float("nan")
    return out


def json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=json_default) + "\n")


# ---------------------------------------------------------------------------
# Correction in the loop, baselines
# ---------------------------------------------------------------------------
def corrected_problem(prob: "MeasProblem", corr):
    """``i2_1_joint.CorrectedProblem`` over ``prob`` with the position keys
    of the fitted curves (``simulate`` returns I_card * exp(g))."""
    from .i2_1_joint import CorrectedProblem
    dev = prob.dev
    cp = CorrectedProblem(prob, corr, dev.dev_type, dev.L_um, dev.W_um, dev.vdd, "ihp_meas")
    cp._keys = [c.key for c in prob.curves]
    return cp


def b1_universe() -> list[str]:
    from .i4_1_baselines import PSP_B1_CANDIDATES
    return [n for n in PSP_B1_CANDIDATES if n not in BASE_SET]


def b1_extras(dev: MeasDevice, names: Sequence[str] | None = None) -> dict:
    """``i4_1_baselines.extend_box`` extras at the stock card values (I4-1
    box rule without zero ranges: nonzero v -> log |v|/2 .. 2|v|, 0 -> inactive)."""
    from .i4_1_baselines import b1_extra_box
    universe = b1_universe() if names is None else [n for n in b1_universe() if n in set(names)]
    block = model_block(dev.cls, dev.dev_type)
    values = {n: card_value(dev, n, block) for n in universe}
    return b1_extra_box(universe, values, None)


def rext_max(dev: MeasDevice, start: dict) -> float:
    """B2 resistor upper bound: 4 * RSW1 (ohm um, per side) / W_um."""
    from .i4_1_baselines import B2_REXT_FACTOR
    return B2_REXT_FACTOR * float(start["rsw1"]) / dev.W_um
