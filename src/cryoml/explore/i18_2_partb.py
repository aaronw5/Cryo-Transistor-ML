"""I18-2 Part B: deployment checks (I17-1 rules) of the full-data g v2 on the
TU Delft BSIM4 card (CMOSedu 50 nm N1 / P1, I6-1), ngspice-41 OSDI build,
4.13 K and 295.0 K.

Pre-registration: ``out/research/explore/I18_2/prereg.md`` / ``prereg.json``.

Element (I14-1 construction on a top-level ``.model`` card): card instance
(model ``i6card``), channel-current replica (model ``i6card_ch`` = the card
with ``REPLICA``) driven by unit-gain VCVS copies of V(d), V(g), V(s), V(b),
a VCCS of -gmin between the replica drain and body copies, and the OSDI
module (``i12_1_symm.to_verilog_a_v2``) adding (exp(g) - 1) I(sense) from d
to s.  Both models carry the b4check.c A1/A2 clamp (state C).

Pure helpers: grids and op points scaled to VDD 1.1 V, card / replica text,
netlists, control blocks, the Part B decision (``i12_1_symm.decide_deploy``
with the I17-1 V(d,s) = 0 value and Q3 not run).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np

from . import i12_1_symm as v2
from .i14_1_chan import GMIN, M_DEV, MODELS, NONCHANNEL_OUTPUTS, REPLICA_CHAN, corr_models  # noqa: F401
from .i18_2_tud import VDD, clamp_params

STEP = 0.025
GRID_VGS = np.round(np.arange(45) * STEP, 6)                 # 0 .. 1.1
GRID_VDS = np.round(-1.1 + np.arange(89) * STEP, 6)          # -1.1 .. 1.1
GM_VDS = (0.05, 0.55, 1.1)
GDS_VGS = (0.35, 0.525, 0.725, 0.9, 1.1)
DENSE_STEP = 0.001
CARD_MODEL = "i6card"
REP_MODEL = "i6card_ch"
REPLICA = {**v2.REPLICA_ZERO, **REPLICA_CHAN}
VT_CRIT_PER_SQ = 1e-7       # A x W / L (I11-1 rule; m = 1 here)
Q3_STATUS = "not run (sky130 circuits at 1.8 V; TU Delft card 1 V, measured range 1.1 V)"


# ---------------------------------------------------------------------------
# card text
# ---------------------------------------------------------------------------
def card_text(dev_type: str, params: dict, replica: bool = False) -> str:
    """The I6-1 model block with ``params`` and the b4check.c clamp written
    (state C); ``replica``: also the REPLICA values and model ``i6card_ch``."""
    from .i6_1_tudelft import CARD_MODEL as M6, model_block, patch_card
    p, _ = clamp_params(params)
    if replica:
        p = {**p, **REPLICA}
    text = patch_card(model_block(dev_type), p)
    if replica:
        text, n = re.subn(rf"^\.model\s+{M6}\b", f".model {REP_MODEL}", text, count=1, flags=re.M | re.I)
        if n != 1:
            raise ValueError("model header not found")
    return text


def replica_ok(text: str) -> bool:
    """Every REPLICA value appears once in the replica text with that value."""
    for k, v in REPLICA.items():
        mm = re.findall(rf"^\+\s*{k}\s*=\s*([^\s]+)", text, re.I | re.M)
        if len(mm) != 1 or float(mm[0]) != float(v):
            return False
    return bool(re.search(rf"^\.model\s+{REP_MODEL}\s", text, re.M | re.I))


# ---------------------------------------------------------------------------
# netlists
# ---------------------------------------------------------------------------
MODES = ("card", "card_rep", "elem")


def instance(name: str, dev, d: str, g: str, s: str, b: str, mode: str) -> str:
    """Transistor ``name`` (an I6-1 MeasDevice: dev_type, W_um, L_um) at (d, g, s, b)."""
    if mode not in MODES:
        raise ValueError(mode)
    geo = f"w={dev.W_um:.6g}u l={dev.L_um:.6g}u nf=1"
    out = f"M{name} {d} {g} {s} {b} {CARD_MODEL} {geo}\n"
    if mode == "card":
        return out
    p = f"{name}_"
    out += (f"M{name}r {p}ri {p}rg {p}rs {p}rb {REP_MODEL} {geo}\n"
            f"E{name}d {p}rx 0 {d} 0 1\nE{name}g {p}rg 0 {g} 0 1\n"
            f"E{name}s {p}rs 0 {s} 0 1\nE{name}b {p}rb 0 {b} 0 1\n"
            f"G{name}c {p}ri {p}rb {p}ri {p}rb {-M_DEV * GMIN:.17g}\n")
    if mode == "card_rep":
        return out + f"V{name}R {p}rx {p}ri DC 0 AC 0\n"
    nmod, pmod = MODELS[v2.VA_MODULE_V2]
    return out + f"N{name} {d} {g} {s} {p}rx {p}ri {pmod if dev.dev_type == 'pmos' else nmod}\n"


def device_body(dev, mode: str, swap: bool = False) -> str:
    src = "VG ng 0 DC 0 AC 0\nVD nd 0 DC 0 AC 0\n"
    if swap:
        return instance("m1", dev, "nz", "ng", "nd", "0", mode) + "VZ nz 0 DC 0\n" + src
    return instance("m1", dev, "nd", "ng", "0", "0", mode) + src


def deck(title: str, temp_K: float, card: Path, rep: Path | None, body: str, control: Sequence[str],
         osdi: Path | None, options: str = "", numdgt: int = 17) -> str:
    if "gmin" in options.lower():
        raise ValueError("the gmin compensation assumes the ngspice default gmin")
    lines = [f"* {title}\n", f".options temp={temp_K - 273.15:.6g}\n"]
    if options:
        lines.append(f".options {options}\n")
    lines.append(f'.include "{Path(card).as_posix()}"\n')
    if rep is not None:
        lines.append(f'.include "{Path(rep).as_posix()}"\n')
    lines.append(body if body.endswith("\n") else body + "\n")
    if osdi is not None:
        lines.append(corr_models(v2.VA_MODULE_V2))
    lines.append(".control\n")
    if osdi is not None:
        lines.append(f"pre_osdi {Path(osdi).as_posix()}\n")
    lines += ["set filetype=ascii\n", f"set numdgt={int(numdgt)}\n"]
    lines += [c if c.endswith("\n") else c + "\n" for c in control]
    lines += ["quit\n", ".endc\n", ".end\n"]
    return "".join(lines)


def grid_control_serp(dev_type: str, out: Path, sense: bool) -> list[str]:
    """Serpentine Q1 grid (I15-1): V(g) line k ascending V(d) for even k,
    descending for odd k; rows appended to ``out``."""
    sg = -1.0 if dev_type == "pmos" else 1.0
    vecs = "-i(VD)" + (" i(Vm1R)" if sense else "")
    lines = ["set appendwrite"]
    for k, vg in enumerate(GRID_VGS):
        lo, hi = GRID_VDS[0], GRID_VDS[-1]
        a, b, st = (lo, hi, STEP) if k % 2 == 0 else (hi, lo, -STEP)
        lines += [f"alter VG dc = {sg * vg:.6g}", f"dc VD {a:.6g} {b:.6g} {st:.6g}",
                  f"wrdata {Path(out).as_posix()} {vecs}"]
    return lines


def grid_rows(rows: np.ndarray) -> np.ndarray:
    """Serpentine rows -> (len(GRID_VGS), len(GRID_VDS), ncols - 1) nested
    order, V(d) ascending; NaN-filled array when the rows do not match."""
    nv, nd = len(GRID_VGS), len(GRID_VDS)
    rows = np.asarray(rows, dtype=np.float64)
    nc = rows.shape[1] if rows.ndim == 2 else 2
    bad = np.full((nv, nd, nc - 1), np.nan)
    if rows.ndim != 2 or rows.shape[0] != nv * nd:
        return bad
    out = np.empty((nv, nd, nc - 1))
    for k in range(nv):
        blk = rows[k * nd:(k + 1) * nd]
        blk = blk[np.argsort(blk[:, 0], kind="stable")]
        if not np.allclose(blk[:, 0], GRID_VDS, rtol=0.0, atol=1e-9):
            return bad
        out[k] = blk[:, 1:]
    return out


def op_points(dev_type: str) -> np.ndarray:
    """gm sets (|V(d,s)| in GM_VDS, |V(g,s)| on GRID_VGS), then gds sets
    (|V(g,s)| in GDS_VGS, |V(d,s)| on GRID_VGS); signed."""
    s = -1.0 if dev_type == "pmos" else 1.0
    pts = [(s * vg, s * vd) for vd in GM_VDS for vg in GRID_VGS]
    pts += [(s * vg, s * vd) for vg in GDS_VGS for vd in GRID_VGS]
    return np.array(pts, dtype=np.float64)


def dense_control(dev_type: str, prefix: Path) -> list[str]:
    s = -1.0 if dev_type == "pmos" else 1.0
    lines = []
    for k, vd in enumerate(GM_VDS):
        lines += [f"alter VD dc = {s * vd:.6g}", f"dc VG 0 {s * VDD:.6g} {s * DENSE_STEP:.6g}",
                  f"wrdata {Path(prefix).as_posix()}_gm{k}.dat -i(VD)"]
    lines.append("alter VG dc = 0")
    for k, vg in enumerate(GDS_VGS):
        lines += [f"alter VG dc = {s * vg:.6g}", f"dc VD 0 {s * VDD:.6g} {s * DENSE_STEP:.6g}",
                  f"wrdata {Path(prefix).as_posix()}_gds{k}.dat -i(VD)"]
    return lines


def nonchannel_control(points: np.ndarray) -> list[str]:
    """Replica pre-check: op at each point, print the replica's BSIM4
    non-channel outputs (instance mm1r)."""
    lines = []
    for k, (vg, vd) in enumerate(points):
        lines += [f"alter VG dc = {vg:.6g}", f"alter VD dc = {vd:.6g}", "op"]
        lines += [f"print @mm1r[{o}]" for o in NONCHANNEL_OUTPUTS]
        lines.append("destroy all")
    return lines


_NC = re.compile(r"^\s*@mm1r\[(\w+)\]\s*=\s*([-+0-9.eEnaNinf]+)", re.I)


def parse_nonchannel(text: str) -> dict:
    """max |value| per output name over the printed op points."""
    out: dict = {}
    for line in text.splitlines():
        mt = _NC.match(line)
        if mt:
            try:
                v = abs(float(mt.group(2)))
            except ValueError:
                v = float("inf")
            k = mt.group(1).lower()
            out[k] = max(out.get(k, 0.0), v)
    return out


# ---------------------------------------------------------------------------
# decision (pure, pre-registered)
# ---------------------------------------------------------------------------
def decide_partb(values: dict) -> dict:
    """``i12_1_symm.decide_deploy`` with the I17-1 V(d,s) = 0 value (tight
    forward grid only) in ``values['q1_vds0_max']`` and Q3 not evaluated
    (q3_fail = 0, status recorded)."""
    vals = {**values, "q3_fail": 0}
    dec = v2.decide_deploy(vals)
    return {**dec, "q3": Q3_STATUS, "label_full": f"{dec['deploy']} (Q3 not run)"}


def decide_validity_b(v: dict) -> dict:
    fails = []
    if v["m2_failed"] != 0 or v["m2_n"] != 45:
        fails.append("VB 45 M2 refits, 0 failed")
    if not v["unit_tests_pass"]:
        fails.append("VB unit tests")
    if v["n_rows"] != 45:
        fails.append("VB 45 Part B rows")
    return {"valid": not fails, "fails": fails}
