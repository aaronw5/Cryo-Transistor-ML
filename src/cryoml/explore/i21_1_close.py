"""I21-1: open items that need no user decision.

Pre-registration: ``out/research/explore/I21_1/prereg.md`` / ``prereg.json``.

Part A: Part B deployment checks (I17-1 rules) of the I20-1 full-data g v2
models (lambda 0 V) on the two NIST DEV datasets: NIST lvt (sky130 77 K lvt
library, explicit bin, 77 K) and NIST RT card (open_pdks tt nfet_01v8 /
pfet_01v8 card reduced to one explicit bin, 30 C).  Element: the I14-1
construction (card instance, channel-current replica ``<subckt>_ch`` driven
by VCVS copies, -gmin VCCS, OSDI module).  Q3: the sky130 circuit decks at
1.8 V on the NIST DEV devices nearest to the sky130 circuit geometries.

Part B: TU Delft v1-recipe S3_D on the I18-2 state-C records (report only).

Part C: selected g v2 recipe on IHP SG13G2 233 / 343 / 398 K (I17-2 setup).

This module holds the pure helpers (grids, op points, card / replica /
library text, netlists, the circuit-device rule, the decision tables).  No
simulation and no training here.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import i12_1_symm as v2
from .i14_1_chan import GMIN, M_DEV, MODELS, REPLICA_CHAN, corr_models  # noqa: F401

DATASETS = ("nist4", "nistrt")
VDD = 1.8                                                 # V, NIST supply and circuit supply
STEP = 0.025
GRID_VGS = np.round(np.arange(73) * STEP, 6)              # 0 .. 1.8
GRID_VDS = np.round(-1.8 + np.arange(145) * STEP, 6)      # -1.8 .. 1.8
GM_VDS = (0.05, 0.9, 1.8)
GDS_VGS = (0.6, 0.9, 1.2, 1.5, 1.8)
DENSE_STEP = 0.001
VT_CRIT_PER_SQ = 1e-7                                     # A x W / L (I11-1 rule)
REPLICA = {**v2.REPLICA_ZERO, **REPLICA_CHAN}
SUFFIX = "_ch"
DECK_TEMP_C = {"nist4": 77.0 - 273.15, "nistrt": 30.0}
LIB_SECTION = {"nist4": "tt_77k", "nistrt": "tt_rt"}
EXPECTED_N = {"nist4": 53, "nistrt": 87}
SUBCKT = {"nist4": {"nmos": "sky130_fd_pr__nfet_01v8_lvt", "pmos": "sky130_fd_pr__pfet_01v8_lvt"},
          "nistrt": {"nmos": "sky130_fd_pr__nfet_01v8", "pmos": "sky130_fd_pr__pfet_01v8"}}
MODEL = {ds: {p: s + "__model" for p, s in d.items()} for ds, d in SUBCKT.items()}

# sky130 circuit devices of I14-1 / I17-1 (role -> polarity, L um, W um)
CIRCUIT_ROLES = {"nmos_L0p15_W1p6": ("nmos", 0.15, 1.6), "pmos_L0p35_W1p6": ("pmos", 0.35, 1.6),
                 "nmos_L1_W1p6": ("nmos", 1.0, 1.6), "pmos_L2_W5": ("pmos", 2.0, 5.0)}
# name, kind, roles, settings (I12-1 / I14-1 CIRCUITS with the sky130 tags replaced by roles)
CIRCUITS = (
    ("inv_A_vtc", "vtc", ("nmos_L0p15_W1p6", "pmos_L0p35_W1p6"), {}),
    ("inv_B_vtc", "vtc", ("nmos_L1_W1p6", "pmos_L2_W5"), {}),
    ("ro5_A", "ring", ("nmos_L0p15_W1p6", "pmos_L0p35_W1p6"), {"tstep": 1e-12, "tstop": 10e-9}),
    ("ro5_B", "ring", ("nmos_L1_W1p6", "pmos_L2_W5"), {"tstep": 5e-12, "tstop": 50e-9}),
    ("mirror_n", "mirror", ("nmos_L1_W1p6",), {"iref": 10e-6}),
    ("mirror_p", "mirror", ("pmos_L2_W5",), {"iref": 10e-6}),
)
Q3_ARMS = ("published", "C0_refit", "M2_card", "M2_full")
# pre-registered choice (prereg.json part_A.q3.devices), recomputed by ``circuit_devices``
PREREG_CIRCUIT_DEVICES = {
    "nist4": {"nmos_L0p15_W1p6": "nfet_01v8_lvt_W1p65_L0p18_m23p3", "pmos_L0p35_W1p6": "pfet_01v8_lvt_W1_L0p35_m28p6",
              "nmos_L1_W1p6": "nfet_01v8_lvt_W1p65_L1_m23p6", "pmos_L2_W5": "pfet_01v8_lvt_W5_L1p5_m30p6"},
    "nistrt": {"nmos_L0p15_W1p6": "nfet_01v8_W1p68_L0p15_m6p7", "pmos_L0p35_W1p6": "pfet_01v8_W1p68_L0p5_m16p1",
               "nmos_L1_W1p6": "nfet_01v8_W1p68_L1_m7p2", "pmos_L2_W5": "pfet_01v8_W5_L2_m18p5"}}


# ---------------------------------------------------------------------------
# circuit devices (pre-registered rule)
# ---------------------------------------------------------------------------
def circuit_devices(geoms: Mapping[str, tuple[str, float, float]]) -> dict:
    """Role -> tag of the device of the same polarity with the smallest
    Euclidean distance in (ln L, ln W) to the role geometry; ties -> first
    tag in sorted order.  ``geoms``: tag -> (polarity, L um, W um)."""
    out = {}
    for role, (pol, L, W) in CIRCUIT_ROLES.items():
        cands = sorted((math.hypot(math.log(l / L), math.log(w / W)), t)
                       for t, (p, l, w) in geoms.items() if p == pol)
        if not cands:
            raise ValueError(f"no {pol} device")
        out[role] = cands[0][1]
    return out


# ---------------------------------------------------------------------------
# replica text
# ---------------------------------------------------------------------------
def replica_rt(card: str, polarity: str) -> str:
    """Channel-current replica of a one-bin room-temperature card text
    (``i9_1_nist_rt.card_text``): the ``.subckt <subckt> ... .ends <subckt>``
    block, subcircuit and model renamed with ``SUFFIX``, ``REPLICA`` written
    into its bin block (one assignment per line, l / w / p terms kept unless
    listed in ``REPLICA``)."""
    from .i9_1_nist_rt import patch_block
    sub, mod = SUBCKT["nistrt"][polarity], MODEL["nistrt"][polarity]
    head = re.compile(rf"^\.subckt\s+{re.escape(sub)}\s", re.M | re.I)
    tail = re.compile(rf"^\.ends\s+{re.escape(sub)}\s*$", re.M | re.I)
    h, t = list(head.finditer(card)), list(tail.finditer(card))
    if len(h) != 1 or len(t) != 1 or t[0].start() < h[0].start():
        raise ValueError("card text does not hold exactly one subcircuit block")
    blk = card[h[0].start():t[0].end()]
    hdr = re.compile(rf"^\.model\s+{re.escape(mod)}\.\d+\s", re.M | re.I)
    hits = list(hdr.finditer(blk))
    if len(hits) != 1:
        raise ValueError(f"expected one bin block, found {len(hits)}")
    end = re.search(r"^\.ends\b", blk[hits[0].start():], re.M)
    a, b = hits[0].start(), hits[0].start() + end.start()
    blk = blk[:a] + patch_block(blk[a:b], REPLICA) + blk[b:]
    blk = re.sub(rf"^(\.subckt\s+){re.escape(sub)}(\s)", rf"\g<1>{sub}{SUFFIX}\g<2>", blk, count=1, flags=re.M | re.I)
    blk = re.sub(rf"^(\.ends\s+){re.escape(sub)}(\s*)$", rf"\g<1>{sub}{SUFFIX}\g<2>", blk, count=1, flags=re.M | re.I)
    blk = re.sub(rf"\b{re.escape(mod)}\b", f"{mod}{SUFFIX}", blk)
    return "* channel-current replica (I21-1, I14-1 construction)\n" + blk + "\n"


def replica_values_ok(text: str, ds: str, polarity: str) -> bool:
    """Every model block of ``<model>_ch`` in ``text`` carries every REPLICA
    value exactly once, and the replica subcircuit header is present."""
    mod = MODEL[ds][polarity] + SUFFIX
    hdr = re.compile(rf"^\.model\s+{re.escape(mod)}\.(\d+)\s", re.I | re.M)
    hits = list(hdr.finditer(text))
    if not hits:
        return False
    for i, h in enumerate(hits):
        span = text[h.start(): hits[i + 1].start() if i + 1 < len(hits) else len(text)]
        for k, v in REPLICA.items():
            mm = re.findall(rf"^\+\s*{k}\s*=\s*([^\s]+)", span, re.I | re.M)
            if len(mm) != 1 or float(mm[0]) != float(v):
                return False
    return bool(re.search(rf"^\.subckt\s+{re.escape(SUBCKT[ds][polarity] + SUFFIX)}\s", text, re.M | re.I))


# ---------------------------------------------------------------------------
# library (writes files)
# ---------------------------------------------------------------------------
def card_params(ds: str, polarity: str, bin_index: int, W_um: float, L_um: float,
                params: dict | None, stock: dict) -> dict | None:
    """Parameters written into the bin, as the I20-1 state-C decks:
    ``changed_params`` vs the stock readback, then the b4check.c clamp."""
    from . import i8_1_nist as b8
    from . import i9_1_nist_rt as b9
    ch = b8.changed_params(params, stock) if params is not None else None
    if ds == "nist4":
        return b8.bsim4_clamp_params(polarity, bin_index, ch)[0] or ch
    return b9.bsim4_clamp_params(polarity, bin_index, W_um, L_um, ch)[0] or ch


def write_library(td: Path, ds: str, cards: Mapping[str, tuple[int, dict | None]], replica: bool) -> Path:
    """Library with, per polarity in ``cards`` (polarity -> (bin, written
    params or None)), the explicit-bin card and, with ``replica``, its
    channel-current replica."""
    from . import i8_1_nist as b8
    from . import i9_1_nist_rt as b9
    td = Path(td)
    if ds == "nist4":
        from .i14_1_chan import replica_corner_chan
        sp = b8._sp()
        lib = sp.LIB_FILE.read_text()
        adds = []
        for pol, (bi, p) in cards.items():
            text = b8.corner_text(pol, bi, p)
            corner = td / sp._CORNER_BASENAME[pol]
            corner.write_text(text)
            base = (sp.PDK77K_DIR / sp._CORNER_BASENAME[pol]).as_posix()
            if lib.count(base) != 1:
                raise ValueError("library does not reference the corner file exactly once")
            lib = lib.replace(base, corner.as_posix())
            if replica:
                rep = replica_corner_chan(text, pol)
                if not replica_values_ok(rep, ds, pol):
                    raise RuntimeError(f"replica of {pol} lacks a REPLICA value")
                rp = td / f"replica_{pol}.spice"
                rp.write_text(rep)
                adds.append(f'.include "{rp.as_posix()}"\n')
        if replica:
            if lib.count(".endl") != 1:
                raise ValueError("library text does not have exactly one .endl")
            lib = lib.replace(".endl", "".join(adds) + ".endl")
        path = td / "lib77k.spice"
        path.write_text(lib)
        return path
    src = b9.card_sources()
    lines = ["* I21-1 standalone sky130 tt library (open_pdks room-temperature cards)\n", ".lib tt_rt\n",
             ".param mc_mm_switch=0\n", ".param mc_pr_switch=0\n", f'.include "{src["lod"].as_posix()}"\n']
    for pol, (bi, p) in cards.items():
        text = b9.card_text(pol, bi, p)
        card = td / f"{b9.SUBCKT[pol]}__tt_rt.spice"
        card.write_text(text)
        lines += [f'.include "{card.as_posix()}"\n', f'.include "{src[f"{pol}_mismatch"].as_posix()}"\n']
        if replica:
            rep = replica_rt(text, pol)
            if not replica_values_ok(rep, ds, pol):
                raise RuntimeError(f"replica of {pol} lacks a REPLICA value")
            rp = td / f"replica_{pol}.spice"
            rp.write_text(rep)
            lines.append(f'.include "{rp.as_posix()}"\n')
    lines.append(".endl tt_rt\n")
    path = td / "lib_rt.spice"
    path.write_text("".join(lines))
    return path


# ---------------------------------------------------------------------------
# netlists
# ---------------------------------------------------------------------------
MODES = ("card", "card_rep", "elem")


def inst_body(ds: str, polarity: str, W_um: float, L_um: float) -> str:
    """I8-1 / I9-1 instance line without 'Xm1 nd ng 0 0 '."""
    from . import i8_1_nist as b8
    from . import i9_1_nist_rt as b9
    line = (b8 if ds == "nist4" else b9).instance_line(polarity, W_um, L_um)
    head = "Xm1 nd ng 0 0 "
    if not line.startswith(head) or line.count("\n") != 1:
        raise ValueError("unexpected instance line")
    body = line[len(head):]
    if not body.startswith(SUBCKT[ds][polarity] + " "):
        raise ValueError("instance body does not start with the subcircuit name")
    return body


def instance(name: str, dev, d: str, g: str, s: str, b: str, mode: str, ds: str) -> str:
    """Transistor ``name`` (``dev``: polarity, W_um, L_um) at (d, g, s, b).
    card: the card instance; card_rep: + replica, VCVS copies, -gmin VCCS
    and a 0 V sense source V<name>R; elem: the OSDI module in place of the
    sense source."""
    if mode not in MODES:
        raise ValueError(mode)
    sub = SUBCKT[ds][dev.polarity]
    body = inst_body(ds, dev.polarity, dev.W_um, dev.L_um)
    out = f"X{name} {d} {g} {s} {b} {body}"
    if mode == "card":
        return out
    p = f"{name}_"
    out += f"X{name}r {p}ri {p}rg {p}rs {p}rb {sub}{SUFFIX}{body[len(sub):]}"
    out += (f"E{name}d {p}rx 0 {d} 0 1\nE{name}g {p}rg 0 {g} 0 1\n"
            f"E{name}s {p}rs 0 {s} 0 1\nE{name}b {p}rb 0 {b} 0 1\n"
            f"G{name}c {p}ri {p}rb {p}ri {p}rb {-M_DEV * GMIN:.17g}\n")
    if mode == "card_rep":
        return out + f"V{name}R {p}rx {p}ri DC 0 AC 0\n"
    nmod, pmod = MODELS[v2.VA_MODULE_V2]
    return out + f"N{name} {d} {g} {s} {p}rx {p}ri {pmod if dev.polarity == 'pmos' else nmod}\n"


def device_body(dev, mode: str, ds: str, swap: bool = False) -> str:
    src = "VG ng 0 DC 0 AC 0\nVD nd 0 DC 0 AC 0\n"
    if swap:
        return instance("m1", dev, "nz", "ng", "nd", "0", mode, ds) + "VZ nz 0 DC 0\n" + src
    return instance("m1", dev, "nd", "ng", "0", "0", mode, ds) + src


def deck(title: str, ds: str, lib: Path, body: str, control: Sequence[str], osdi: Path | None,
         options: str = "", numdgt: int = 17) -> str:
    if "gmin" in options.lower():
        raise ValueError("the gmin compensation assumes the ngspice default gmin")
    lines = [f"* {title}\n", ".options scale=1.0\n", f".options temp={DECK_TEMP_C[ds]:.4f}\n"]
    if options:
        lines.append(f".options {options}\n")
    lines.append(f'.lib "{Path(lib).as_posix()}" {LIB_SECTION[ds]}\n')
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


def grid_control_serp(polarity: str, out: Path, sense: bool) -> list[str]:
    """Serpentine Q1 grid (I15-1): V(g) line k ascending V(d) for even k,
    descending for odd k; rows appended to ``out``."""
    sg = -1.0 if polarity == "pmos" else 1.0
    vecs = "-i(VD)" + (" i(Vm1R)" if sense else "")
    lines = ["set appendwrite"]
    for k, vg in enumerate(GRID_VGS):
        lo, hi = GRID_VDS[0], GRID_VDS[-1]
        a, b, st = (lo, hi, STEP) if k % 2 == 0 else (hi, lo, -STEP)
        lines += [f"alter VG dc = {sg * vg:.6g}", f"dc VD {a:.6g} {b:.6g} {st:.6g}",
                  f"wrdata {Path(out).as_posix()} {vecs}"]
    return lines


def grid_rows(rows: np.ndarray) -> np.ndarray:
    """Serpentine rows -> (len(GRID_VGS), len(GRID_VDS), ncols - 1), V(d)
    ascending; NaN-filled when the rows do not match."""
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


def op_points(polarity: str) -> np.ndarray:
    """gm sets (|V(d,s)| in GM_VDS, |V(g,s)| on GRID_VGS), then gds sets
    (|V(g,s)| in GDS_VGS, |V(d,s)| on GRID_VGS); signed."""
    s = -1.0 if polarity == "pmos" else 1.0
    pts = [(s * vg, s * vd) for vd in GM_VDS for vg in GRID_VGS]
    pts += [(s * vg, s * vd) for vg in GDS_VGS for vd in GRID_VGS]
    return np.array(pts, dtype=np.float64)


def dense_control(polarity: str, prefix: Path) -> list[str]:
    s = -1.0 if polarity == "pmos" else 1.0
    lines = []
    for k, vd in enumerate(GM_VDS):
        lines += [f"alter VD dc = {s * vd:.6g}", f"dc VG 0 {s * VDD:.6g} {s * DENSE_STEP:.6g}",
                  f"wrdata {Path(prefix).as_posix()}_gm{k}.dat -i(VD)"]
    lines.append("alter VG dc = 0")
    for k, vg in enumerate(GDS_VGS):
        lines += [f"alter VG dc = {s * vg:.6g}", f"dc VD 0 {s * VDD:.6g} {s * DENSE_STEP:.6g}",
                  f"wrdata {Path(prefix).as_posix()}_gds{k}.dat -i(VD)"]
    return lines


NONCHANNEL_OUTPUTS = ("isub", "igidl", "igisl", "igs", "igd", "igb", "igcs", "igcd")


def replica_bsim_name(ds: str, polarity: str) -> str:
    """ngspice name of the replica's BSIM4 instance (instance Xm1r, m-line
    ``m<subckt>`` inside the replica subcircuit)."""
    return f"m.xm1r.m{SUBCKT[ds][polarity]}"


def nonchannel_control(points: np.ndarray, ds: str, polarity: str) -> list[str]:
    nm = replica_bsim_name(ds, polarity)
    lines = []
    for vg, vd in points:
        lines += [f"alter VG dc = {vg:.6g}", f"alter VD dc = {vd:.6g}", "op"]
        lines += [f"print @{nm}[{o}]" for o in NONCHANNEL_OUTPUTS]
        lines.append("destroy all")
    return lines


_NC = re.compile(r"^\s*@m\.xm1r\.\w+\[(\w+)\]\s*=\s*([-+0-9.eEnaNinf]+)", re.I)


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


def inverter_body(n, p, mode: str, vin: str, vout: str, name: str, ds: str) -> str:
    return (instance(f"{name}n", n, vout, vin, "0", "0", mode, ds)
            + instance(f"{name}p", p, vout, vin, "vdd", "vdd", mode, ds))


def vtc_body(n, p, mode: str, ds: str) -> str:
    return inverter_body(n, p, mode, "nin", "nout", "i1", ds) + f"VDD vdd 0 DC {VDD}\nVIN nin 0 DC 0\n"


def ring_body(n, p, mode: str, ds: str, stages: int = 5) -> str:
    if stages % 2 == 0:
        raise ValueError("ring oscillator needs an odd number of stages")
    nodes = [f"o{k + 1}" for k in range(stages)]
    body = "".join(inverter_body(n, p, mode, nodes[k - 1], nodes[k], f"s{k + 1}", ds) for k in range(stages))
    ics = " ".join(f"v({nd})={VDD if k % 2 == 0 else 0.0}" for k, nd in enumerate(nodes))
    return body + f"VDD vdd 0 DC {VDD}\n.ic {ics}\n"


def mirror_body(dev, mode: str, iref: float, ds: str) -> str:
    if dev.polarity == "nmos":
        return (instance("m1", dev, "nref", "nref", "0", "0", mode, ds)
                + instance("m2", dev, "nout", "nref", "0", "0", mode, ds)
                + f"IREF 0 nref DC {iref:.6g}\nVOUT nout 0 DC 0\n")
    return (instance("m1", dev, "nref", "nref", "vdd", "vdd", mode, ds)
            + instance("m2", dev, "nout", "nref", "vdd", "vdd", mode, ds)
            + f"VDD vdd 0 DC {VDD}\nIREF nref 0 DC {iref:.6g}\nVOUT nout 0 DC 0\n")


# ---------------------------------------------------------------------------
# decisions (pure, pre-registered)
# ---------------------------------------------------------------------------
def decide_part_a(values: dict) -> dict:
    """``i12_1_symm.decide_deploy`` with the I17-1 V(d,s) = 0 value (raw
    element current, tight forward grid) and the Q3 count."""
    assert v2.THR["q1_vds0_abs"] == 1e-12
    return v2.decide_deploy(values)


def validity_part_a(v: dict, ds: str) -> dict:
    n = EXPECTED_N[ds]
    fails = []
    if not (v["va1_max_diff"] <= 1e-9):
        fails.append("VA1 card-only deck reproduces the M2 Python rows")
    if v["m2_failed"] != 0 or v["m2_n"] != n:
        fails.append(f"VA2 {n} M2 refits, 0 failed")
    if not v["unit_tests_pass"]:
        fails.append("VA3 unit tests")
    if v["n_rows"] != n or v["n_q3"] != len(CIRCUITS) * len(Q3_ARMS):
        fails.append(f"VA4 {n} Part B rows, 24 Q3 runs")
    if not v["dev_ok"]:
        fails.append("VA5 DEV only")
    return {"valid": not fails, "fails": fails}


def part_b_report(s3_c: float, s3_l: float) -> dict:
    """Part B (report only): state-C S3_D, its decision word and whether its
    sign differs from the stored state-L value."""
    from .i19_1_dataset_gate import apply_gate
    word = lambda s: "apply" if apply_gate(s) else "C0_refit"  # noqa: E731
    return {"S3_D_state_C": s3_c, "S3_D_state_L": s3_l, "decision_state_C": word(s3_c),
            "decision_state_L": word(s3_l), "sign_differs": bool(np.sign(s3_c) != np.sign(s3_l))}


def validity_part_b(v: dict) -> dict:
    fails = []
    if not (v["vb1_max_diff"] <= 1e-9):
        fails.append("VB1 state-L recompute reproduces I19-1 values")
    if not v["dev_ok"]:
        fails.append("VB2 DEV only")
    if v["n_finite"] != 45:
        fails.append("VB3 45 finite per-record values")
    return {"valid": not fails, "fails": fails}


def validity_part_c(v: dict) -> dict:
    """VC1 reruns reproduce I5-1 rows (<= 1e-9, 18 each); VC2 unit tests;
    VC3 18 rows per arm, inner complete; VC4 grids; VC5 DEV only."""
    fails = []
    if not (v["vc1_c0_max"] <= 1e-9 and v["vc1_mb_max"] <= 1e-9 and v["vc1_n"] == 18):
        fails.append("VC1 C0_refit / M_bias v1 reruns reproduce I5-1 rows")
    if not v["unit_tests_pass"]:
        fails.append("VC2 unit tests")
    if not (all(n == 18 for n in v["rows"].values()) and v["inner_complete"]):
        fails.append("VC3 18 rows per arm / inner selection complete")
    if not v["grids_ok"]:
        fails.append("VC4 penalty grids")
    if not v["dev_ok"]:
        fails.append("VC5 DEV only")
    return {"valid": not fails, "fails": fails}
