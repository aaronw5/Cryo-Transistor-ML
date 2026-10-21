"""I10-1: predict a device's 4 K I-V from its room-temperature (RT) data.

Pre-registration: ``out/research/explore/I10_1/prereg.json``.

DEV data (this iteration):

- NIST mds2-2997 ETest tile, ``nfet_g5v0d10v5`` / ``pfet_g5v0d10v5``, 4 K files
  paired by module/pad ID with the RT files of
  ``google/skywater-pdk-sky130-raw-data`` (``i10_1_etest_rt``).  Only the
  files of ``DEV_TYPES`` are parsed (``discover_types``); the 1.8 V core
  files (TEST) are not read.
- TU Delft 40 nm (I6-1 DEV devices and stored starting cards; handled in the
  driver with ``cryoml.explore.i6_1_tudelft``).

This module holds the parts that do not depend on the dataset:

- open_pdks sky130 ``tt`` card handling for any sky130 FET type (the I9-1
  text helpers ``normalize_block`` / ``patch_block`` generalized from
  nfet_01v8 / pfet_01v8 to ``TYPE_INFO``), bins, deck, showmod readback;
- the RT voltage restriction (``restrict``), RT descriptors;
- the parameter shift in ``ParameterBox`` local coordinates
  (``shift_stat``, ``apply_shift``), the ridge regression with inner
  leave-one-out lambda selection (``RidgeShift``);
- the paired geometric-mean ratio with a confidence level argument
  (``gmr_ci``; equals ``i3_2_bias.paired_ratio`` at level 0.95).
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import ROOT
from . import i8_1_nist as base
from . import i9_1_nist_rt as rt9
from .i5_1_ihp_real import parse_mdm

OUT = ROOT / "out" / "research" / "explore" / "I10_1"
DEV_TYPES = ("nfet_g5v0d10v5", "pfet_g5v0d10v5")
CORE_TYPES = ("nfet_01v8", "nfet_01v8_lvt", "pfet_01v8", "pfet_01v8_lvt", "pfet_01v8_hvt")
TNOM_C = 30.0
DECK_RT_C = TNOM_C                     # D1: RT and 4 K cards fitted at the cards' tnom
DECK_4K_C = 4.0 - 273.15               # A0: BSIM4 temperature equations at 4 K
TIMEOUT_S = 60.0
N_TIMEOUT = 0
VMAX_4K = {"nfet_g5v0d10v5": 3.3, "pfet_g5v0d10v5": 3.3}   # largest |V| of the 4 K sweeps
VDD = {**{t: 3.3 for t in DEV_TYPES}, **{t: 1.8 for t in CORE_TYPES}}
V_EPS = 1e-6
PARAMS7 = ("vth0", "u0", "nfactor", "vsat", "delta", "rdsw", "eta0")


# ---------------------------------------------------------------------------
# Card files
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TypeInfo:
    dev_type: str
    polarity: str
    pm3: str                    # file with the subckt and the bins
    corner: str | None          # tt.corner file (.param block, then .include pm3) or None
    mismatch: str


def _info(t: str, corner: bool) -> TypeInfo:
    pol = "pmos" if t.startswith("pfet") else "nmos"
    return TypeInfo(t, pol, f"sky130_fd_pr__{t}__tt.pm3.spice",
                    f"sky130_fd_pr__{t}__tt.corner.spice" if corner else None,
                    f"sky130_fd_pr__{t}__mismatch.corner.spice")


# corner flag = what open_pdks libs.tech/ngspice/corners/tt.spice includes
TYPE_INFO = {t: _info(t, c) for t, c in (
    ("nfet_01v8", False), ("nfet_01v8_lvt", True), ("pfet_01v8", True),
    ("pfet_01v8_lvt", True), ("pfet_01v8_hvt", False),
    ("nfet_g5v0d10v5", True), ("pfet_g5v0d10v5", True))}


def subckt(dev_type: str) -> str:
    return f"sky130_fd_pr__{dev_type}"


def model(dev_type: str) -> str:
    return subckt(dev_type) + "__model"


def card_sources(dev_type: str) -> dict[str, Path]:
    d = rt9._pdk_spice_dir()
    ng = d.parents[2] / "libs.tech" / "ngspice"
    ti = TYPE_INFO[dev_type]
    out = {"pm3": d / ti.pm3, "mismatch": d / ti.mismatch, "lod": ng / "parameters" / "lod.spice",
           "official_lib": ng / "sky130.lib.spice", "official_tt": ng / "corners" / "tt.spice"}
    if ti.corner:
        out["corner"] = d / ti.corner
    return out


def _bin_hdr(dev_type: str) -> re.Pattern:
    return re.compile(rf"^\.model\s+{re.escape(model(dev_type))}\.(\d+)\s", re.IGNORECASE | re.MULTILINE)


def bin_spans(text: str, dev_type: str) -> list[tuple[int, int, int]]:
    hits = list(_bin_hdr(dev_type).finditer(text))
    return [(int(h.group(1)), h.start(), hits[i + 1].start() if i + 1 < len(hits) else len(text))
            for i, h in enumerate(hits)]


def split_card(text: str, dev_type: str, corner_text: str = "") -> dict:
    """Parsed card text (as ``i9_1_nist_rt.card_parts``) for ``dev_type``."""
    spans = bin_spans(text, dev_type)
    if not spans:
        raise RuntimeError(f"no bins for {dev_type}")
    prefix = text[:spans[0][1]]
    blocks, geometry, unparsed, suffix = {}, {}, 0, ""
    for idx, a, b in spans:
        blk = text[a:b]
        m = re.search(r"^\.ends\b", blk, re.MULTILINE)
        if m:
            suffix, blk = blk[m.start():], blk[:m.start()]
        norm, kept = rt9.normalize_block(blk)
        if idx in blocks:
            raise RuntimeError(f"{dev_type}: bin {idx} twice")
        blocks[idx], geometry[idx] = norm, rt9.block_geometry(norm)
        unparsed += kept
        if len(geometry[idx]) != 4:
            raise RuntimeError(f"{dev_type} bin {idx}: geometry {geometry[idx]}")
    if not suffix.strip():
        raise RuntimeError(f"{dev_type}: no .ends after the last bin")
    pre = re.sub(r'^\.include\s+"[^"]*"\s*$', "", corner_text, flags=re.MULTILINE) if corner_text else ""
    return {"prefix": prefix, "blocks": blocks, "suffix": suffix, "pre_param": pre,
            "geometry": geometry, "n_unparsed": unparsed}


@lru_cache(maxsize=None)
def card_parts(dev_type: str) -> dict:
    src = card_sources(dev_type)
    corner = src["corner"].read_text(errors="replace") if "corner" in src else ""
    return split_card(src["pm3"].read_text(errors="replace"), dev_type, corner)


def force_bin(text: str, dev_type: str, bin_index: int) -> str:
    out, n = re.subn(rf"\b{re.escape(model(dev_type))}\b(\s+l\s*=)",
                     f"{model(dev_type)}.{bin_index}\\1", text, count=1)
    if n == 0:
        raise ValueError(f"m-instance for {dev_type} not found")
    return out


def card_text(dev_type: str, bin_index: int, params: dict | None = None, all_bins: bool = False) -> str:
    p = card_parts(dev_type)
    if all_bins:
        blocks = [rt9.patch_block(b, params) if (params and i == bin_index) else b
                  for i, b in sorted(p["blocks"].items())]
        return p["pre_param"] + p["prefix"] + "".join(blocks) + p["suffix"]
    if bin_index not in p["blocks"]:
        raise KeyError(f"{dev_type} bin {bin_index}")
    blk = rt9.patch_block(p["blocks"][bin_index], params) if params else p["blocks"][bin_index]
    return p["pre_param"] + force_bin(p["prefix"], dev_type, bin_index) + blk + p["suffix"]


def write_lib(td: Path, dev_type: str, bin_index: int, params: dict | None = None,
              all_bins: bool = False, card: str | None = None) -> Path:
    src = card_sources(dev_type)
    path = td / f"{subckt(dev_type)}__tt_i10.spice"
    path.write_text(card if card is not None else card_text(dev_type, bin_index, params, all_bins))
    lib = td / "lib_i10.spice"
    lib.write_text("* I10-1 standalone sky130 tt library (open_pdks cards)\n.lib tt_i10\n"
                   ".param mc_mm_switch=0\n.param mc_pr_switch=0\n"
                   f'.include "{src["lod"].as_posix()}"\n.include "{path.as_posix()}"\n'
                   f'.include "{src["mismatch"].as_posix()}"\n.endl tt_i10\n')
    return lib


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------
def bin_boxes(dev_type: str) -> dict[int, dict[str, float]]:
    return {i: {k: v * 1e6 for k, v in g.items()} for i, g in card_parts(dev_type)["geometry"].items()}


def bin_candidates(dev_type: str, W_um: float, L_um: float, eps: float = 1e-6) -> list[int]:
    return sorted(i for i, g in bin_boxes(dev_type).items()
                  if g["lmin"] - eps <= L_um <= g["lmax"] + eps and g["wmin"] - eps <= W_um <= g["wmax"] + eps)


def instance_line(dev_type: str, W_um: float, L_um: float, rs_ext_ohm: float = 0.0) -> str:
    """I8-1 / I9-1 instance form (mult = m = 1) with the subckt of ``dev_type``."""
    drain, source = ("ndi", "nsi") if rs_ext_ohm > 0 else ("nd", "0")
    ext = (f"RDEXT nd ndi {rs_ext_ohm:.10g}\nRSEXT nsi 0 {rs_ext_ohm:.10g}\n" if rs_ext_ohm > 0 else "")
    area, perim, sq = W_um * 0.29, 2 * (W_um + 0.29), 0.29 / W_um
    return (f"Xm1 {drain} ng {source} 0 {subckt(dev_type)} l={L_um:.6g}u w={W_um:.6g}u nf=1 "
            f"ad={area:.6g}p as={area:.6g}p pd={perim:.6g}u ps={perim:.6g}u "
            f"nrd={sq:.6g} nrs={sq:.6g} sa=0 sb=0 sd=0 mult=1 m=1\n") + ext


def _ngspice(deck: Path, td: Path, timeout_s: float) -> tuple[bool, str]:
    global N_TIMEOUT
    try:
        proc = subprocess.run([base._sp()._NGSPICE_BIN, "-b", str(deck)], cwd=str(td),
                              capture_output=True, timeout=timeout_s, text=True, check=False)
        log = proc.stdout + proc.stderr
        return "could not find a valid modelname" in log.lower(), log
    except subprocess.TimeoutExpired:
        N_TIMEOUT += 1
        return True, "TIMEOUT"


def native_bin(dev_type: str, W_um: float, L_um: float) -> int | None:
    """Bin ngspice selects with the full card (eta0 sentinel readback when
    several closed boxes contain the geometry)."""
    cands = bin_candidates(dev_type, W_um, L_um)
    if len(cands) <= 1:
        return cands[0] if cands else None
    sentinels = {i: 0.101 + 0.0013 * k for k, i in enumerate(cands)}
    p = card_parts(dev_type)
    blocks = [rt9.patch_block(b, {"eta0": sentinels[i]}) if i in sentinels else b
              for i, b in sorted(p["blocks"].items())]
    pol = TYPE_INFO[dev_type].polarity
    sign = -1 if pol == "pmos" else 1
    with tempfile.TemporaryDirectory(prefix="i10_1_native_") as td_s:
        td = Path(td_s)
        lib = write_lib(td, dev_type, 0, card=p["pre_param"] + p["prefix"] + "".join(blocks) + p["suffix"])
        deck = td / "d.sp"
        deck.write_text(f"* native bin\n.options scale=1.0\n.options temp={TNOM_C:.4f}\n"
                        f'.lib "{lib.as_posix()}" tt_i10\n' + instance_line(dev_type, W_um, L_um)
                        + f"VG ng 0 DC {sign}\nVD nd 0 DC {sign}\n.op\n.control\nrun\n"
                        f"showmod m.xm1.m{subckt(dev_type)} : eta0\nquit\n.endc\n.end\n")
        _, out = _ngspice(deck, td, 120)
    m = re.search(r"^\s*eta0\s+([0-9eE.+\-]+)\s*$", out, re.MULTILINE)
    if not m:
        return None
    got = float(m.group(1))
    best = min(sentinels, key=lambda i: abs(sentinels[i] - got))
    return best if abs(sentinels[best] - got) < 5e-5 else None


def bin_choice(dev_type: str, W_um: float, L_um: float) -> dict:
    """Natively selected bin if the geometry lies in a bin; else the bin
    nearest in (ln L, ln W) (I8-1 / I9-1 bin stretch)."""
    boxes = bin_boxes(dev_type)
    cands = bin_candidates(dev_type, W_um, L_um)
    if cands:
        b = native_bin(dev_type, W_um, L_um)
        if b is None or b not in cands:
            raise RuntimeError(f"{dev_type} W{W_um} L{L_um}: native bin lookup failed ({b}, {cands})")
        covered = True
    else:
        b = min(boxes, key=lambda i: math.hypot(*map(math.log, base.stretch(boxes[i], W_um, L_um))))
        covered = False
    sl, sw = base.stretch(boxes[b], W_um, L_um)
    return {"bin": int(b), "covered": covered, "candidates": cands, "stretch_L": sl, "stretch_W": sw,
            "log_distance": math.hypot(math.log(sl), math.log(sw)), "bin_box_um": boxes[b]}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(dev_type: str, W_um: float, L_um: float, bin_index: int, curves,
             params: dict | None = None, temp_c: float = DECK_RT_C, rs_ext_ohm: float = 0.0,
             timeout_s: float = TIMEOUT_S, all_bins: bool = False, return_log: bool = False):
    """All sweeps of ``curves`` in one ngspice process at ``temp_c``; pMOS
    currents negated (magnitude convention)."""
    sp = base._sp()
    curves = list(curves)
    pol = TYPE_INFO[dev_type].polarity
    with tempfile.TemporaryDirectory(prefix="i10_1_sim_") as td_s:
        td = Path(td_s)
        lib = write_lib(td, dev_type, bin_index, params, all_bins)
        lines = ["* I10-1 measured-grid DC sweeps (open_pdks sky130 tt card)\n", ".options scale=1.0\n",
                 f".options temp={temp_c:.4f}\n", f'.lib "{lib.as_posix()}" tt_i10\n',
                 instance_line(dev_type, W_um, L_um, rs_ext_ohm), "VG ng 0 DC 0\n", "VD nd 0 DC 0\n",
                 ".control\n", "set filetype=ascii\n"]
        lines += sp._sweep_lines(curves, td, "raw_")
        lines += ["quit\n", ".endc\n", ".end\n"]
        deck = td / "deck.sp"
        deck.write_text("".join(lines))
        fatal, log = _ngspice(deck, td, timeout_s)
        sims = sp._read_curves(td, "raw_", curves, pol, fatal)
    return (sims, log) if return_log else sims


def simulate_official(dev_type: str, W_um: float, L_um: float, curves, temp_c: float = DECK_RT_C,
                      timeout_s: float = 300.0):
    """Unmodified open_pdks ``sky130.lib.spice tt`` (``.option scale=1.0u``)."""
    sp = base._sp()
    curves = list(curves)
    lib = card_sources(dev_type)["official_lib"]
    area, perim, sq = W_um * 0.29, 2 * (W_um + 0.29), 0.29 / W_um
    inst = (f"Xm1 nd ng 0 0 {subckt(dev_type)} l={L_um:.6g} w={W_um:.6g} nf=1 ad={area:.6g} "
            f"as={area:.6g} pd={perim:.6g} ps={perim:.6g} nrd={sq:.6g} nrs={sq:.6g} sa=0 sb=0 sd=0 "
            f"mult=1 m=1\n")
    with tempfile.TemporaryDirectory(prefix="i10_1_official_") as td_s:
        td = Path(td_s)
        lines = ["* I10-1 reference deck, open_pdks sky130.lib.spice tt\n", f".options temp={temp_c:.4f}\n",
                 f'.lib "{lib.as_posix()}" tt\n', inst, "VG ng 0 DC 0\n", "VD nd 0 DC 0\n", ".control\n",
                 "set filetype=ascii\n"]
        lines += sp._sweep_lines(curves, td, "raw_")
        lines += ["quit\n", ".endc\n", ".end\n"]
        deck = td / "deck.sp"
        deck.write_text("".join(lines))
        fatal, _ = _ngspice(deck, td, timeout_s)
        return sp._read_curves(td, "raw_", curves, TYPE_INFO[dev_type].polarity, fatal)


def read_params(dev_type: str, W_um: float, L_um: float, bin_index: int, names: Sequence[str],
                required: bool = True) -> dict[str, float]:
    """showmod readback (model parameters of the explicit bin, deck 30 C)."""
    sign = -1 if TYPE_INFO[dev_type].polarity == "pmos" else 1
    vals: dict[str, float] = {}
    names = list(names)
    for i0 in range(0, len(names), 40):
        chunk = names[i0:i0 + 40]
        with tempfile.TemporaryDirectory(prefix="i10_1_showmod_") as td_s:
            td = Path(td_s)
            lib = write_lib(td, dev_type, bin_index, None)
            deck = td / "d.sp"
            deck.write_text(f'* showmod readback\n.options scale=1.0\n.lib "{lib.as_posix()}" tt_i10\n'
                            f".options temp={TNOM_C:.4f}\n" + instance_line(dev_type, W_um, L_um)
                            + f"VG ng 0 DC {sign}\nVD nd 0 DC {sign}\n.op\n.control\nrun\n"
                            f"showmod m.xm1.m{subckt(dev_type)} : {' '.join(chunk)}\nquit\n.endc\n.end\n")
            _, out = _ngspice(deck, td, 30)
        for p in chunk:
            mm = re.search(rf"^\s*{p}\s+([0-9eE.+\-]+)\s*$", out, re.MULTILINE)
            if mm:
                vals[p] = float(mm.group(1))
    missing = [p for p in names if p not in vals]
    if missing and required:
        raise RuntimeError(f"showmod readback missing {missing} ({dev_type} bin {bin_index})")
    return vals


class MeasProblemI10(base.MeasProblem):
    """I8-1 ``MeasProblem`` (measured-only inclusion, polish interface) with
    the card of ``dev.dev_type`` and the deck temperature ``temp_c``."""

    def __init__(self, dev, binfo: dict, stock: dict, box, temp_c: float):
        super().__init__(dev, binfo, stock, box)
        self.temp_c = float(temp_c)

    def simulate_params(self, params: dict | None, rs_ext_ohm: float = 0.0) -> list[np.ndarray]:
        self.n_sim += 1
        return simulate(self.dev.dev_type, self.dev.W_um, self.dev.L_um, self.bin_index, self.curves,
                        base.changed_params(params, self.stock), self.temp_c, rs_ext_ohm)


def to_record(dev, binfo: dict, sims, params: dict | None = None, vdd: float | None = None,
              extra_meta: dict | None = None):
    """E1 ``DeviceRec`` (source ``nist4k``) with the frozen measured inclusion
    and VDD of the device type (``VDD``)."""
    rec = base.to_record(dev, binfo, sims, params)
    rec.vdd = float(VDD[dev.dev_type] if vdd is None else vdd)
    rec.meta.update(extra_meta or {})
    return rec


# ---------------------------------------------------------------------------
# Data: DEV-type discovery (TEST files are not parsed), RT restriction
# ---------------------------------------------------------------------------
def discover_types(types: Sequence[str], root: Path = base.ETEST_ROOT) -> list:
    """``i8_1_nist.discover_etest`` limited to ``types``: file names of other
    types are matched by name only and their files are not opened."""
    found: dict[tuple, dict] = {}
    for path in sorted((root / "Sky130-4k").iterdir()):
        m = base._ETEST_RE.match(path.name)
        if not m:
            raise ValueError(f"unexpected file {path.name}")
        if m.group(1) not in types:
            continue
        W, L = (float(m.group(i).replace("p", ".")) for i in (2, 3))
        key = (m.group(1), W, L, int(m.group(4)), int(m.group(5)), m.group(6), m.group(7))
        found.setdefault(key, {})[m.group(8)] = path
    devices = []
    for (dev_type, W, L, mult, module, pa, pb), by_kind in sorted(found.items()):
        pol = "pmos" if dev_type.startswith("pfet") else "nmos"
        dev = base.NistDevice(f"etest_{base.device_tag(dev_type, W, L, module, int(pa))}", "ETest",
                              dev_type, pol, module, int(pa), W, L)
        if mult != 1:
            dev.flags.append(f"m={mult}")
        for name, kind in (("IDVG", "idvg"), ("IDVD", "idvd")):
            if name not in by_kind:
                dev.flags.append(f"missing {name}")
                continue
            mdm = parse_mdm(by_kind[name])
            dev.files[name] = by_kind[name].relative_to(root).as_posix()
            dev.inputs[name] = base._inputs_row(mdm)
            dev.curves.extend(base.file_curves(mdm, kind))
        base.assign_keys(dev.curves)
        devices.append(dev)
    return devices


def restrict(dev, vmax: float):
    """Copy of ``dev`` with the Vb = 0 curves whose fixed bias satisfies
    |V| <= vmax, each truncated to the sweep points with |V| <= vmax; keys
    reassigned (``assign_keys``).  Curves at Vb != 0 are dropped."""
    out = base.NistDevice(dev.tag, dev.folder, dev.dev_type, dev.polarity, dev.module, dev.position,
                          dev.W_um, dev.L_um, dict(dev.files), dict(dev.inputs), [], list(dev.flags))
    for c in dev.curves:
        if c.vb != 0.0 or abs(c.fixed) > vmax + V_EPS:
            continue
        keep = np.abs(np.asarray(c.sweep, float)) <= vmax + V_EPS
        if keep.sum() < 2:
            continue
        out.curves.append(base.NistCurve(c.kind, c.fixed, c.vb, np.asarray(c.sweep)[keep].copy(),
                                         np.asarray(c.id)[keep].copy(),
                                         np.asarray(c.aux)[keep].copy() if len(c.aux) == len(c.sweep)
                                         else np.zeros(0)))
    base.assign_keys(out.curves)
    return out


# ---------------------------------------------------------------------------
# RT descriptors (features of arm A3)
# ---------------------------------------------------------------------------
DESCRIPTORS = ("vth_cc_V", "log10_ion_lin", "log10_ion_sat")
I_CC_A = 1e-7                 # constant-current threshold criterion x W/L
I_CC_MIN_A = 1e-8             # floor of the criterion (RT measurement floor near 1e-9 A)


def descriptors(curves, W_um: float, L_um: float) -> dict[str, float]:
    """From pipeline curves (``data_io.Curve``, attributes Vg / Id) or I6-1
    ``MeasCurve`` (vg / id); magnitudes are used:
    constant-current threshold |Vg| at the smallest-|Vd| Id-Vg curve
    (|Id| = max(1e-7 W/L, 1e-8) A, log interpolation; NaN if not crossed),
    log10 |Id| at the largest |Vg| of that curve, log10 |Id| at the last point
    of the largest-|Vg| Id-Vd curve."""
    idvg = [c for c in curves if c.kind == "idvg"]
    idvd = [c for c in curves if c.kind == "idvd"]
    out = {k: float("nan") for k in DESCRIPTORS}
    vg_of = lambda c: getattr(c, "Vg", None) if hasattr(c, "Vg") else c.vg      # noqa: E731
    id_of = lambda c: getattr(c, "Id", None) if hasattr(c, "Id") else c.id      # noqa: E731
    if idvg:
        c = min(idvg, key=lambda c: abs(c.fixed))
        vg, i = np.abs(np.asarray(vg_of(c), float)), np.abs(np.asarray(id_of(c), float))
        o = np.argsort(vg)
        vg, i = vg[o], i[o]
        ic = max(I_CC_A * W_um / L_um, I_CC_MIN_A)
        for k in range(len(i) - 1):
            if i[k] < ic <= i[k + 1] and i[k] > 0:
                f = (math.log10(ic) - math.log10(i[k])) / (math.log10(i[k + 1]) - math.log10(i[k]))
                out["vth_cc_V"] = float(vg[k] + f * (vg[k + 1] - vg[k]))
                break
        if i[-1] > 0:
            out["log10_ion_lin"] = math.log10(i[-1])
    if idvd:
        c = max(idvd, key=lambda c: abs(c.fixed))
        last = abs(float(np.asarray(id_of(c), float)[-1]))
        if last > 0:
            out["log10_ion_sat"] = math.log10(last)
    return out


# ---------------------------------------------------------------------------
# Shift in local coordinates
# ---------------------------------------------------------------------------
def local(box, params: dict) -> np.ndarray:
    """ParameterBox local coordinates (inactive coordinates 0)."""
    z = np.asarray(box.params_to_local(params), float)
    z[~np.asarray(box.active, bool)] = 0.0
    return z


def shift_stat(dz: np.ndarray, stat: str = "median") -> np.ndarray:
    """Per-coordinate median (or mean) of the rows of ``dz`` (n x p)."""
    dz = np.asarray(dz, float)
    if dz.ndim != 2 or dz.shape[0] == 0:
        raise ValueError("shift_stat needs at least one training row")
    return np.median(dz, axis=0) if stat == "median" else np.mean(dz, axis=0)


def apply_shift(box, z_rt: np.ndarray, dz: np.ndarray, subset: Sequence[str] | None = None) -> dict:
    """Card ``z_rt + dz`` (clipped to the box, inactive coordinates kept);
    with ``subset`` only those parameter names are shifted."""
    dz = np.asarray(dz, float).copy()
    if subset is not None:
        keep = np.array([n in set(subset) for n in box.names])
        dz[~keep] = 0.0
    dz[~np.asarray(box.active, bool)] = 0.0
    return box.local_to_params(np.clip(np.asarray(z_rt, float) + dz, -1.0, 1.0))


LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)      # x n_train; plus infinity (= the median shift)


class RidgeShift:
    """Arm A3: dz_hat(x) = median(dz) + B^T s(x), s(x) = standardized
    features (training mean / SD; columns with SD < 1e-12 dropped; NaN
    features imputed with the training median), B ridge slopes of
    (dz - median) on s(X) (the column means of s(X) are 0, so B is the ridge
    slope with an unpenalized intercept; the intercept is not used, so
    lambda -> infinity gives the median shift, arm A1).  lambda per target
    column from ``LAMBDA_GRID`` x n and infinity by exact inner leave-one-out
    squared error (ties: the larger lambda)."""

    def __init__(self, grid: Sequence[float] = LAMBDA_GRID):
        self.grid = tuple(grid)

    @staticmethod
    def _prep(X: np.ndarray):
        X = np.asarray(X, float)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xf = np.where(np.isfinite(X), X, med)
        mu, sd = Xf.mean(axis=0), Xf.std(axis=0)
        keep = sd > 1e-12
        return med, mu, np.where(keep, sd, 1.0), keep

    @staticmethod
    def _scale(X, med, mu, sd, keep):
        X = np.asarray(X, float)
        X = np.where(np.isfinite(X), X, med)
        return ((X - mu) / sd)[..., keep]

    def _fit_slopes(self, X, Y, lams):
        """Slopes for every lambda in ``lams`` (list of (p x q) arrays) and the
        scaling of X; Y centered by its median."""
        med, mu, sd, keep = self._prep(X)
        S = self._scale(X, med, mu, sd, keep)
        R = Y - np.median(Y, axis=0)
        Rc = R - R.mean(axis=0)
        G = S.T @ S
        out = []
        for lam in lams:
            if not np.isfinite(lam) or S.shape[1] == 0:
                out.append(np.zeros((S.shape[1], Y.shape[1])))
            else:
                out.append(np.linalg.solve(G + lam * np.eye(G.shape[0]), S.T @ Rc))
        return out, (med, mu, sd, keep), np.median(Y, axis=0)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RidgeShift":
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        n, q = Y.shape
        lams = [g * n for g in self.grid] + [float("inf")]
        if n >= 3:
            err = np.zeros((len(lams), q))
            for i in range(n):
                tr = np.arange(n) != i
                slopes, sc, med = self._fit_slopes(X[tr], Y[tr], [g * (n - 1) for g in self.grid]
                                                   + [float("inf")])
                s_i = self._scale(X[i:i + 1], *sc)
                for k, B in enumerate(slopes):
                    err[k] += (Y[i] - (med + (s_i @ B)[0])) ** 2
            best = np.zeros(q, int)
            for j in range(q):
                order = np.argsort(err[:, j], kind="stable")
                m = err[order[0], j]
                ties = [k for k in range(len(lams)) if err[k, j] <= m * (1 + 1e-12) + 1e-300]
                best[j] = max(ties)              # larger lambda on ties (list ascending, inf last)
        else:
            best = np.full(q, len(lams) - 1)
        slopes, self.scaling, self.median = self._fit_slopes(X, Y, lams)
        self.B = np.column_stack([slopes[best[j]][:, j] for j in range(q)]) if q else np.zeros((0, 0))
        self.lam_index = best
        self.lams = lams
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        s = self._scale(np.atleast_2d(np.asarray(x, float)), *self.scaling)
        return self.median + (s @ self.B)[0]

    def summary(self, names: Sequence[str]) -> dict:
        """lambda / n per target column (None = infinity, i.e. the median shift)."""
        per = {n: (self.grid[k] if k < len(self.grid) else None) for n, k in zip(names, self.lam_index)}
        return {"lambda_over_n": per, "n_finite_lambda": int(sum(v is not None for v in per.values()))}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def gmr_ci(method: dict, control: dict, clusters: dict | None = None, level: float = 0.95,
           n_boot: int = 10000, seed: int = 0) -> dict:
    """``i3_2_bias.paired_ratio`` with a two-sided percentile CI at ``level``."""
    tags = sorted(set(method) & set(control))
    a = np.array([method[t] for t in tags], float)
    b = np.array([control[t] for t in tags], float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    lr = np.log(a[ok] / b[ok])
    used = [t for t, k in zip(tags, ok) if k]
    out = {"n": len(tags), "n_used": int(ok.sum()), "n_method_failed": int(np.sum(~np.isfinite(a))),
           "n_control_failed": int(np.sum(~np.isfinite(b))), "wins": int(np.sum(a[ok] < b[ok])),
           "ties": int(np.sum(a[ok] == b[ok])), "level": level}
    if not len(lr):
        return {**out, "ratio": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    out["ratio"] = float(np.exp(lr.mean()))
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(lr), size=(n_boot, len(lr)))
        boots = lr[idx].mean(axis=1)
        out["resample_unit"] = "device"
    else:
        names = sorted({clusters[t] for t in used})
        members = [np.array([i for i, t in enumerate(used) if clusters[t] == c]) for c in names]
        sums = np.array([lr[mm].sum() for mm in members])
        sizes = np.array([len(mm) for mm in members], float)
        pick = rng.integers(0, len(names), size=(n_boot, len(names)))
        boots = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
        out["resample_unit"] = f"cluster ({len(names)})"
    q = 100 * (1 - level) / 2
    lo, hi = np.percentile(boots, [q, 100 - q])
    out["ci_low"], out["ci_high"], out["n_boot"] = float(np.exp(lo)), float(np.exp(hi)), int(n_boot)
    return out


def ci_label(c: dict) -> str:
    lo, hi = c.get("ci_low", float("nan")), c.get("ci_high", float("nan"))
    if hi < 1:
        return "lower"
    if lo > 1:
        return "higher"
    return "inconclusive" if np.isfinite(lo) and np.isfinite(hi) else "no pairs"
