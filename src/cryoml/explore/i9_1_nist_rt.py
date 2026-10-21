"""I9-1: NIST mds2-2997 MPW-5 sky130 4 K data, families nfet_01v8 / pfet_01v8,
with the open sky130 room-temperature BSIM4 cards (tt corner) as stock cards.

Data parsing, curve conversion, inclusion, records, box, aggregates and the
prediction guard are the I8-1 functions (``cryoml.explore.i8_1_nist``).  This
module adds the card side: the sky130 ``tt`` model files of the Volare
open_pdks build already used by the canonical setup
(``spice_pdk.PDK_SPICE_DIR``, open_pdks a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b):

- nMOS ``sky130_fd_pr__nfet_01v8__tt.pm3.spice`` (included directly by the
  open_pdks ngspice ``corners/tt.spice``),
- pMOS ``sky130_fd_pr__pfet_01v8__tt.corner.spice`` (a ``.param`` block, then
  ``.include`` of ``sky130_fd_pr__pfet_01v8__tt.pm3.spice``),
- ``sky130_fd_pr__{n,p}fet_01v8__mismatch.corner.spice`` and
  ``libs.tech/ngspice/parameters/lod.spice`` for the parameters the cards
  reference.

Card text handling: in these files one ``+`` line can hold several
assignments (``+ vth0 = {..} lvth0 = .. wvth0 = .. pvth0 = ..``).  Each bin's
``.model`` block is rewritten with one assignment per line
(``normalize_block``) so that a parameter can be read and patched without
touching its ``l``/``w``/``p`` binning terms.  Each simulation uses a card
holding the subckt with its m-instance pointing at one explicit bin
(``force_bin``) and that bin's ``.model`` block only (``card_text``).
``scripts/research/explore/i9_1_nist_rt.py --mode smoke`` checks that this
card reproduces the unmodified open_pdks ``sky130.lib.spice`` ``tt`` library.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import ROOT
from . import i8_1_nist as base

OUT = ROOT / "out" / "research" / "explore" / "I9_1"
CHOSEN = ("nthin", "pthin")                  # nfet_01v8, pfet_01v8 (i8_1_nist.FOLDERS)
VDD = base.VDD
SOURCE = base.SOURCE                         # e1 DeviceRec source tag (non-sky130 scoring branch)
TIMEOUT_S = 60.0
N_TIMEOUT = 0

TNOM_C = 30.0                                # '+ tnom = 30.0' in every bin of both cards
DECK_TEMP_C = TNOM_C                         # draft decision D3 (card at its tnom)

SUBCKT = {"nmos": "sky130_fd_pr__nfet_01v8", "pmos": "sky130_fd_pr__pfet_01v8"}
MODEL = {p: s + "__model" for p, s in SUBCKT.items()}
OPEN_PDKS_COMMIT = "a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b"


def _pdk_spice_dir() -> Path:
    from .. import spice_pdk
    return spice_pdk.PDK_SPICE_DIR


def card_sources() -> dict[str, Path]:
    d = _pdk_spice_dir()
    ng = d.parents[2] / "libs.tech" / "ngspice"
    return {"nmos_pm3": d / "sky130_fd_pr__nfet_01v8__tt.pm3.spice",
            "pmos_pm3": d / "sky130_fd_pr__pfet_01v8__tt.pm3.spice",
            "pmos_corner": d / "sky130_fd_pr__pfet_01v8__tt.corner.spice",
            "nmos_mismatch": d / "sky130_fd_pr__nfet_01v8__mismatch.corner.spice",
            "pmos_mismatch": d / "sky130_fd_pr__pfet_01v8__mismatch.corner.spice",
            "lod": ng / "parameters" / "lod.spice",
            "official_lib": ng / "sky130.lib.spice",
            "official_tt": ng / "corners" / "tt.spice"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Card text
# ---------------------------------------------------------------------------
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_assignments(body: str) -> list[tuple[str, str]] | None:
    """``name = value`` pairs of one continuation-line body (text after '+').
    A value is a ``{...}`` expression (nested braces allowed), a quoted
    ``'...'`` expression, or a token without whitespace.  Returns None when
    the body is not a sequence of such pairs (the line is then kept as is)."""
    out, i, n = [], 0, len(body)
    while True:
        while i < n and body[i].isspace():
            i += 1
        if i >= n:
            return out
        m = _NAME_RE.match(body, i)
        if not m:
            return None
        name, i = m.group(0), m.end()
        while i < n and body[i].isspace():
            i += 1
        if i >= n or body[i] != "=":
            return None
        i += 1
        while i < n and body[i].isspace():
            i += 1
        if i >= n:
            return None
        if body[i] == "{":
            depth, j = 0, i
            while j < n:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if depth != 0:
                return None
            value, i = body[i:j + 1], j + 1
        elif body[i] == "'":
            j = body.find("'", i + 1)
            if j < 0:
                return None
            value, i = body[i:j + 1], j + 1
        else:
            j = i
            while j < n and not body[j].isspace():
                j += 1
            value, i = body[i:j], j
            if "=" in value:
                return None
        out.append((name, value))


def normalize_block(block: str) -> tuple[str, int]:
    """One assignment per '+' line; returns (text, number of lines that were
    not parsed and kept unchanged)."""
    lines, kept = [], 0
    for line in block.split("\n"):
        if line.startswith("+"):
            pairs = parse_assignments(line[1:])
            if pairs is None:
                kept += 1
                lines.append(line)
                continue
            if len(pairs) > 1:
                lines.extend(f"+ {k} = {v}" for k, v in pairs)
                continue
        lines.append(line)
    return "\n".join(lines), kept


def _bin_hdr(polarity: str) -> re.Pattern:
    return re.compile(rf"^\.model\s+{re.escape(MODEL[polarity])}\.(\d+)\s", re.IGNORECASE | re.MULTILINE)


def bin_spans(text: str, polarity: str) -> list[tuple[int, int, int]]:
    hits = list(_bin_hdr(polarity).finditer(text))
    return [(int(h.group(1)), h.start(), hits[i + 1].start() if i + 1 < len(hits) else len(text))
            for i, h in enumerate(hits)]


def block_geometry(block: str) -> dict[str, float]:
    """lmin, lmax, wmin, wmax (m) of one normalized bin block."""
    out = {}
    for key in ("lmin", "lmax", "wmin", "wmax"):
        m = re.search(rf"^\+\s*{key}\s*=\s*([0-9.eE+\-]+)\s*$", block, re.IGNORECASE | re.MULTILINE)
        if m:
            out[key] = float(m.group(1))
    return out


@lru_cache(maxsize=None)
def card_parts(polarity: str) -> dict:
    """Parsed card of one polarity: ``prefix`` (header, subckt(s) up to the
    first bin), ``blocks`` {bin: normalized .model text}, ``suffix``
    (``.ends`` and after), ``pre_param`` (pMOS: the tt.corner .param block),
    ``geometry`` {bin: {lmin, lmax, wmin, wmax} m}, ``n_unparsed``."""
    src = card_sources()
    text = src[f"{polarity}_pm3"].read_text(errors="replace")
    spans = bin_spans(text, polarity)
    if not spans:
        raise RuntimeError(f"no bins in {src[f'{polarity}_pm3']}")
    prefix = text[:spans[0][1]]
    blocks, geometry, unparsed = {}, {}, 0
    suffix = ""
    for idx, a, b in spans:
        blk = text[a:b]
        m = re.search(r"^\.ends\b", blk, re.MULTILINE)
        if m:
            suffix, blk = blk[m.start():], blk[:m.start()]
        norm, kept = normalize_block(blk)
        if idx in blocks:
            raise RuntimeError(f"{polarity}: bin {idx} twice")
        blocks[idx], geometry[idx] = norm, block_geometry(norm)
        unparsed += kept
        if len(geometry[idx]) != 4:
            raise RuntimeError(f"{polarity} bin {idx}: geometry {geometry[idx]}")
    if not suffix.strip():
        raise RuntimeError(f"{polarity}: no .ends after the last bin")
    pre = ""
    if polarity == "pmos":
        ctext = src["pmos_corner"].read_text(errors="replace")
        pre = re.sub(r'^\.include\s+"[^"]*"\s*$', "", ctext, flags=re.MULTILINE)
    return {"prefix": prefix, "blocks": blocks, "suffix": suffix, "pre_param": pre,
            "geometry": geometry, "n_unparsed": unparsed}


def force_bin(text: str, polarity: str, bin_index: int) -> str:
    """The subckt's m-instance references ``<model>.<bin>`` (no automatic
    L/W bin selection)."""
    out, n = re.subn(rf"\b{re.escape(MODEL[polarity])}\b(\s+l\s*=)",
                     f"{MODEL[polarity]}.{bin_index}\\1", text, count=1)
    if n == 0:
        raise ValueError(f"m-instance for {polarity} not found")
    return out


def patch_block(block: str, params: dict) -> str:
    """Replace ``+ name = value`` lines of a normalized block (one assignment
    per line, so the l/w/p binning terms stay); a name not present is added
    after the .model line."""
    for k, v in (params or {}).items():
        text_v = v if isinstance(v, str) else f"{float(v):.10g}"
        pat = re.compile(rf"^(\+\s*{re.escape(k)}\s*=\s*)[^\n]*$", re.IGNORECASE | re.MULTILINE)
        block, n = pat.subn(lambda m: m.group(1) + text_v, block, count=1)
        if n == 0:
            hdr_end = block.find("\n")
            block = block[:hdr_end + 1] + f"+ {k} = {text_v}\n" + block[hdr_end + 1:]
    return block


def card_text(polarity: str, bin_index: int | None, params: dict | None = None,
              all_bins: bool = False) -> str:
    """Card file text.  Default: subckt forced to ``bin_index`` and that bin's
    block only (``params`` patched into it).  ``all_bins``: every normalized
    block and the unforced subckt (native ngspice bin selection; ``params``
    are then patched into ``bin_index``)."""
    p = card_parts(polarity)
    if all_bins:
        blocks = [patch_block(b, params) if (params and i == bin_index) else b
                  for i, b in sorted(p["blocks"].items())]
        return p["pre_param"] + p["prefix"] + "".join(blocks) + p["suffix"]
    if bin_index not in p["blocks"]:
        raise KeyError(f"{polarity} bin {bin_index}")
    blk = patch_block(p["blocks"][bin_index], params) if params else p["blocks"][bin_index]
    return p["pre_param"] + force_bin(p["prefix"], polarity, bin_index) + blk + p["suffix"]


def write_lib(td: Path, polarity: str, bin_index: int | None, params: dict | None = None,
              all_bins: bool = False) -> Path:
    src = card_sources()
    card = td / f"{SUBCKT[polarity]}__tt_rt.spice"
    card.write_text(card_text(polarity, bin_index, params, all_bins))
    lib = td / "lib_rt.spice"
    lib.write_text(
        "* I9-1 standalone sky130 tt library (open_pdks room-temperature cards)\n"
        ".lib tt_rt\n.param mc_mm_switch=0\n.param mc_pr_switch=0\n"
        f'.include "{src["lod"].as_posix()}"\n'
        f'.include "{card.as_posix()}"\n'
        f'.include "{src[f"{polarity}_mismatch"].as_posix()}"\n'
        ".endl tt_rt\n")
    return lib


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------
def bin_boxes(polarity: str) -> dict[int, dict[str, float]]:
    """{bin: {lmin, lmax, wmin, wmax} in um}."""
    return {i: {k: v * 1e6 for k, v in g.items()} for i, g in card_parts(polarity)["geometry"].items()}


def bin_candidates(polarity: str, W_um: float, L_um: float, eps: float = 1e-6) -> list[int]:
    return sorted(i for i, g in bin_boxes(polarity).items()
                  if g["lmin"] - eps <= L_um <= g["lmax"] + eps and g["wmin"] - eps <= W_um <= g["wmax"] + eps)


def native_bin(polarity: str, W_um: float, L_um: float, temp_c: float = DECK_TEMP_C) -> int | None:
    """Bin ngspice selects for this geometry with the full card: a unique
    closed-box candidate, or, for several candidates (shared box edges), the
    candidate whose eta0 sentinel the showmod readback returns."""
    cands = bin_candidates(polarity, W_um, L_um)
    if len(cands) <= 1:
        return cands[0] if cands else None
    sentinels = {i: 0.101 + 0.0013 * k for k, i in enumerate(cands)}
    p = card_parts(polarity)
    blocks = [patch_block(b, {"eta0": sentinels[i]}) if i in sentinels else b
              for i, b in sorted(p["blocks"].items())]
    sign = -1 if polarity == "pmos" else 1
    with tempfile.TemporaryDirectory(prefix="i9_1_native_") as td_s:
        td = Path(td_s)
        lib = write_lib(td, polarity, None, all_bins=True)
        (td / f"{SUBCKT[polarity]}__tt_rt.spice").write_text(
            p["pre_param"] + p["prefix"] + "".join(blocks) + p["suffix"])
        deck = td / "d.sp"
        deck.write_text(f'* native bin\n.options scale=1.0\n.options temp={temp_c:.4f}\n'
                        f'.lib "{lib.as_posix()}" tt_rt\n' + instance_line(polarity, W_um, L_um)
                        + f"VG ng 0 DC {sign}\nVD nd 0 DC {sign}\n.op\n.control\nrun\n"
                        f"showmod m.xm1.m{SUBCKT[polarity]} : eta0\nquit\n.endc\n.end\n")
        out = subprocess.run([base._sp()._NGSPICE_BIN, "-b", str(deck)], capture_output=True,
                             text=True, timeout=120, cwd=str(td)).stdout
    m = re.search(r"^\s*eta0\s+([0-9eE.+\-]+)\s*$", out, re.MULTILINE)
    if not m:
        return None
    got = float(m.group(1))
    best = min(sentinels, key=lambda i: abs(sentinels[i] - got))
    return best if abs(sentinels[best] - got) < 5e-5 else None


def stretch(box: dict, W_um: float, L_um: float) -> tuple[float, float]:
    return base.stretch(box, W_um, L_um)


def bin_choice(polarity: str, W_um: float, L_um: float) -> dict:
    """Explicit bin: the natively selected bin if the geometry lies in a bin,
    else the bin nearest in (ln L, ln W)."""
    boxes = bin_boxes(polarity)
    cands = bin_candidates(polarity, W_um, L_um)
    if cands:
        b = native_bin(polarity, W_um, L_um)
        if b is None or b not in cands:
            raise RuntimeError(f"{polarity} W{W_um} L{L_um}: native bin lookup failed ({b}, {cands})")
        covered = True
    else:
        b = min(boxes, key=lambda i: math.hypot(*map(math.log, stretch(boxes[i], W_um, L_um))))
        covered = False
    sl, sw = stretch(boxes[b], W_um, L_um)
    return {"bin": int(b), "covered": covered, "candidates": cands, "stretch_L": sl,
            "stretch_W": sw, "log_distance": math.hypot(math.log(sl), math.log(sw)),
            "bin_box_um": boxes[b]}


def params_in_bin_text(polarity: str, bin_index: int, names: Sequence[str]) -> list[str]:
    blk = card_parts(polarity)["blocks"][bin_index]
    return [n for n in names if re.search(rf"^\+\s*{re.escape(n)}\s*=", blk, re.IGNORECASE | re.MULTILINE)]


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------
def instance_line(polarity: str, W_um: float, L_um: float, rs_ext_ohm: float = 0.0) -> str:
    """I8-1 instance form (mult = m = 1 for every device) with the
    room-temperature subckt."""
    drain, source = ("ndi", "nsi") if rs_ext_ohm > 0 else ("nd", "0")
    ext = (f"RDEXT nd ndi {rs_ext_ohm:.10g}\nRSEXT nsi 0 {rs_ext_ohm:.10g}\n"
           if rs_ext_ohm > 0 else "")
    area, perim, sq = W_um * 0.29, 2 * (W_um + 0.29), 0.29 / W_um
    return (f"Xm1 {drain} ng {source} 0 {SUBCKT[polarity]} l={L_um:.6g}u w={W_um:.6g}u nf=1 "
            f"ad={area:.6g}p as={area:.6g}p pd={perim:.6g}u ps={perim:.6g}u "
            f"nrd={sq:.6g} nrs={sq:.6g} sa=0 sb=0 sd=0 mult=1 m=1\n") + ext


def _run(deck: Path, td: Path, timeout_s: float) -> tuple[bool, str]:
    global N_TIMEOUT
    try:
        proc = subprocess.run([base._sp()._NGSPICE_BIN, "-b", str(deck)], cwd=str(td),
                              capture_output=True, timeout=timeout_s, text=True, check=False)
        log = proc.stdout + proc.stderr
        return "could not find a valid modelname" in log.lower(), log
    except subprocess.TimeoutExpired:
        N_TIMEOUT += 1
        return True, "TIMEOUT"


def simulate(polarity: str, W_um: float, L_um: float, bin_index: int, curves,
             params: dict | None = None, rs_ext_ohm: float = 0.0,
             timeout_s: float = TIMEOUT_S, temp_c: float | None = None,
             all_bins: bool = False, return_log: bool = False):
    """All sweeps of ``curves`` in one ngspice process at ``temp_c`` (default
    DECK_TEMP_C); pMOS currents negated (magnitude convention)."""
    sp = base._sp()
    curves = list(curves)
    t = DECK_TEMP_C if temp_c is None else float(temp_c)
    with tempfile.TemporaryDirectory(prefix="i9_1_sim_") as td_s:
        td = Path(td_s)
        lib = write_lib(td, polarity, bin_index, params, all_bins)
        lines = ["* I9-1 NIST 4 K measured-grid DC sweeps (room-temperature sky130 card)\n",
                 ".options scale=1.0\n", f".options temp={t:.4f}\n", f'.lib "{lib.as_posix()}" tt_rt\n',
                 instance_line(polarity, W_um, L_um, rs_ext_ohm), "VG ng 0 DC 0\n", "VD nd 0 DC 0\n",
                 ".control\n", "set filetype=ascii\n"]
        lines += sp._sweep_lines(curves, td, "raw_")
        lines += ["quit\n", ".endc\n", ".end\n"]
        deck = td / "deck.sp"
        deck.write_text("".join(lines))
        fatal, log = _run(deck, td, timeout_s)
        sims = sp._read_curves(td, "raw_", curves, polarity, fatal)
    return (sims, log) if return_log else sims


def simulate_official(polarity: str, W_um: float, L_um: float, curves, temp_c: float = DECK_TEMP_C,
                      timeout_s: float = 300.0):
    """Reference: the unmodified open_pdks ``sky130.lib.spice`` ``tt`` library
    (its ``.option scale=1.0u``: instance l, w in um, ad/as in um^2, pd/ps in
    um), native bin selection."""
    sp = base._sp()
    curves = list(curves)
    lib = card_sources()["official_lib"]
    area, perim, sq = W_um * 0.29, 2 * (W_um + 0.29), 0.29 / W_um
    inst = (f"Xm1 nd ng 0 0 {SUBCKT[polarity]} l={L_um:.6g} w={W_um:.6g} nf=1 ad={area:.6g} "
            f"as={area:.6g} pd={perim:.6g} ps={perim:.6g} nrd={sq:.6g} nrs={sq:.6g} sa=0 sb=0 sd=0 "
            f"mult=1 m=1\n")
    with tempfile.TemporaryDirectory(prefix="i9_1_official_") as td_s:
        td = Path(td_s)
        lines = ["* I9-1 reference deck, open_pdks sky130.lib.spice tt\n",
                 f".options temp={temp_c:.4f}\n", f'.lib "{lib.as_posix()}" tt\n', inst,
                 "VG ng 0 DC 0\n", "VD nd 0 DC 0\n", ".control\n", "set filetype=ascii\n"]
        lines += sp._sweep_lines(curves, td, "raw_")
        lines += ["quit\n", ".endc\n", ".end\n"]
        deck = td / "deck.sp"
        deck.write_text("".join(lines))
        fatal, log = _run(deck, td, timeout_s)
        return sp._read_curves(td, "raw_", curves, polarity, fatal)


def read_params(polarity: str, W_um: float, L_um: float, bin_index: int,
                names: Sequence[str], required: bool = True,
                temp_c: float = DECK_TEMP_C) -> dict[str, float]:
    """showmod readback of model parameters of the explicit bin."""
    sign = -1 if polarity == "pmos" else 1
    vals: dict[str, float] = {}
    names = list(names)
    for i0 in range(0, len(names), 40):
        chunk = names[i0:i0 + 40]
        with tempfile.TemporaryDirectory(prefix="i9_1_showmod_") as td_s:
            td = Path(td_s)
            lib = write_lib(td, polarity, bin_index, None)
            deck = td / "d.sp"
            deck.write_text(
                f'* showmod readback\n.options scale=1.0\n.lib "{lib.as_posix()}" tt_rt\n'
                f".options temp={temp_c:.4f}\n" + instance_line(polarity, W_um, L_um)
                + f"VG ng 0 DC {sign}\nVD nd 0 DC {sign}\n.op\n.control\nrun\n"
                f"showmod m.xm1.m{SUBCKT[polarity]} : {' '.join(chunk)}\nquit\n.endc\n.end\n")
            out = subprocess.run([base._sp()._NGSPICE_BIN, "-b", str(deck)], capture_output=True,
                                 text=True, timeout=30, cwd=str(td)).stdout
        for p in chunk:
            mm = re.search(rf"^\s*{p}\s+([0-9eE.+\-]+)\s*$", out, re.MULTILINE)
            if mm:
                vals[p] = float(mm.group(1))
    missing = [p for p in names if p not in vals]
    if missing and required:
        raise RuntimeError(f"showmod readback missing {missing} ({polarity} bin {bin_index})")
    return vals


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------
class MeasProblemRT(base.MeasProblem):
    """I8-1 ``MeasProblem`` with the room-temperature card deck."""

    def simulate_params(self, params: dict | None, rs_ext_ohm: float = 0.0) -> list[np.ndarray]:
        self.n_sim += 1
        return simulate(self.dev.polarity, self.dev.W_um, self.dev.L_um, self.bin_index,
                        self.curves, base.changed_params(params, self.stock), rs_ext_ohm)


# ---------------------------------------------------------------------------
# Descriptive check (smoke): subthreshold swing
# ---------------------------------------------------------------------------
def min_swing_mV_dec(vg: np.ndarray, i: np.ndarray, lo: float = 1e-10, hi: float = 1e-7) -> float:
    """Smallest |dVg / dlog10 |I|| (mV/decade) over consecutive points with
    lo <= |I| <= hi (both ends), from a measured or simulated Id-Vg curve;
    NaN if fewer than 2 such points.  Descriptive only."""
    vg = np.abs(np.asarray(vg, float))
    a = np.abs(np.asarray(i, float))
    ok = np.isfinite(a) & (a >= lo) & (a <= hi)
    best = float("nan")
    for k in range(len(a) - 1):
        if ok[k] and ok[k + 1] and a[k + 1] > a[k]:
            s = 1e3 * (vg[k + 1] - vg[k]) / (math.log10(a[k + 1]) - math.log10(a[k]))
            best = s if not np.isfinite(best) else min(best, s)
    return best


TC_ZERO = ("kt1", "kt1l", "kt2", "ute", "ua1", "ub1", "uc1", "at", "prt")   # i6_1_tudelft.TC_ZERO
