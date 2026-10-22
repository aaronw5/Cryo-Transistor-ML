"""I14-1: deployment test (Part B) of the bias-only correction g v2 with a
channel-current replica, one BSIM4 model-check state, and the corrected Q2
reference; M_bias v1 (full-data g) as a reference arm.

Pre-registration: ``out/research/explore/I14_1/prereg.md`` / ``prereg.json``.

Changes from the I12-1 element (``i12_1_symm``):

1. Clamp state.  Every deck writes the ngspice-41 ``b4check.c`` A1/A2 clamp
   into the card (``spice_pdk.bsim4_clamp_params``; ``patched_library``
   default "clamped"), so card-only, card + replica and element decks use one
   model-check state in every analysis.
2. Channel current only.  The sensed replica is the card with every
   non-channel DC current removed (``REPLICA_CHAN``): no intrinsic charge and
   zero overlap / fringing / junction capacitances (as I12-1), impact
   ionization off (alpha0 = alpha1 = beta0 = 0), GIDL / GISL off
   (agidl = agisl = 0), junction saturation currents zero (jss = jsd = jsws =
   jswd = jswgs = jswgd = 0) and gate tunneling off (igcmod = igbmod = 0).
   With zero saturation current BSIM4 keeps a junction conductance gmin
   between the internal drain and body nodes (``b4ld.c``: gbd = CKTgmin,
   cbd = gbd vbd); a VCCS of value -M_DEV gmin between the replica drain and
   body copies cancels it in the sensed current (M_DEV = 1, the BSIM4
   instance multiplier in these decks; gmin = 1e-12 S, the ngspice default;
   no deck sets gmin).  The sensed
   current is then the replica's channel current.
3. Q2 reference: Re Y_elem = Re Y_card + (exp(g) - 1) Re Y_rep +
   I_rep exp(g) dg/dV, with Y_card, Y_rep, I_rep from one card + replica run
   (same clamp state); Im Y_elem = Im Y_card.

The module text for g v2 is the I12-1 text (``i12_1_symm.to_verilog_a_v2``):
I(d, s) <+ (exp(g) - 1) I(sense).  The M_bias v1 reference arm uses the same
element with the v1 inputs (is_pmos, |V(g,s)| / VDD, |V(d,s)| / VDD).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from . import i12_1_symm as v2

GMIN = 1e-12                     # S, ngspice default (no deck sets gmin)
# BSIM4 instance multiplier m inside the subcircuit: 1 for every device of these
# decks (the X-line 'mult=.. m=..' does not reach the BSIM4 instance; checked by
# @m.xm1.m<sub>[m] = 1 in the repcheck stage), so the junction gmin conductance
# of the replica is 1 x gmin.
M_DEV = 1
VA_MODULE_V1 = "i14_1_bias_corr_v1"
MODELS = {v2.VA_MODULE_V2: ("corr2n", "corr2p"), VA_MODULE_V1: ("corr1n", "corr1p")}

# Replica: every bin of the replica subcircuit carries these values.
REPLICA_CHAN = {**v2.REPLICA_ZERO,
                "alpha0": 0, "alpha1": 0, "beta0": 0,       # impact ionization (Isub),
                "lalpha0": 0, "walpha0": 0, "palpha0": 0,    # with the size-dependent terms
                "lalpha1": 0, "walpha1": 0, "palpha1": 0,    # (b4temp.c: alpha0 = alpha0 +
                "lbeta0": 0, "wbeta0": 0, "pbeta0": 0,       #  lalpha0/L + walpha0/W + ...)
                "agidl": 0, "agisl": 0,                      # GIDL / GISL
                "lagidl": 0, "wagidl": 0, "pagidl": 0,
                "lagisl": 0, "wagisl": 0, "pagisl": 0,
                "jss": 0, "jsd": 0, "jsws": 0, "jswd": 0,    # junction saturation currents
                "jswgs": 0, "jswgd": 0,
                "igcmod": 0, "igbmod": 0}                    # gate tunneling
REPLICA_SUFFIX = "_ch"

# BSIM4 instance outputs of the replica that must be exactly zero
NONCHANNEL_OUTPUTS = ("isub", "igidl", "igisl", "igs", "igd", "igb", "igcs", "igcd")


# ---------------------------------------------------------------------------
# Replica library (text)
# ---------------------------------------------------------------------------
def replica_corner_chan(text: str, dev_type: str) -> str:
    """Channel-current replica of the corner text's subcircuit: the block
    ``.subckt <name> ... .ends <name>`` (with the bin patches and the clamp
    already in ``text``), renamed ``<name>_ch`` with model ``<model>_ch``,
    and ``REPLICA_CHAN`` written into every bin."""
    from .. import spice_pdk as sp
    sub, mod = sp._SUBCKT[dev_type], sp._MODEL[dev_type]
    head, tail = f".subckt {sub} ", f".ends {sub}"
    if text.count(head) != 1 or text.count(tail) != 1:
        raise ValueError("corner text does not hold exactly one subcircuit block")
    a = text.index(head)
    b = text.index(tail) + len(tail)
    blk = text[a:b]
    bins = [idx for idx, _, _ in sp._bin_spans(blk, dev_type)]
    if not bins:
        raise ValueError("no model bins in the subcircuit block")
    for idx in bins:
        blk = sp._patch_params_in_bin(blk, dev_type, idx, REPLICA_CHAN)
    blk = blk.replace(head, f".subckt {sub}{REPLICA_SUFFIX} ", 1)
    blk = blk.replace(tail, f".ends {sub}{REPLICA_SUFFIX}", 1)
    blk = re.sub(rf"\b{re.escape(mod)}\b", f"{mod}{REPLICA_SUFFIX}", blk)
    return ("* channel-current replica (I14-1): no charge, no Isub, no GIDL/GISL, "
            "no junction saturation current, no gate tunneling\n" + blk + "\n")


def replica_bins_ok(rep_text: str, dev_type: str) -> bool:
    """Every bin of a replica block carries every REPLICA_CHAN value once."""
    from .. import spice_pdk as sp
    mod = sp._MODEL[dev_type] + REPLICA_SUFFIX
    hdr = re.compile(rf"^\.model\s+{re.escape(mod)}\.(\d+)\s", re.I | re.M)
    hits = list(hdr.finditer(rep_text))
    if not hits:
        return False
    for i, h in enumerate(hits):
        span = rep_text[h.start(): hits[i + 1].start() if i + 1 < len(hits) else len(rep_text)]
        for k, v in REPLICA_CHAN.items():
            mm = re.findall(rf"^\+\s*{k}\s*=\s*([^\s]+)", span, re.I | re.M)
            if len(mm) != 1 or float(mm[0]) != float(v):
                return False
    return True


def patched_library(td: Path, cards: dict) -> Path:
    """``i11_1_deploy.patched_library`` (clamp written into every patched
    card) plus, for both polarities, the channel-current replica of the
    (patched or base) corner text, included in the tt_77k section."""
    from .. import spice_pdk as sp
    from .i11_1_deploy import patched_library as base_library
    td = Path(td)
    lib = base_library(td, cards, "clamped")
    text = lib.read_text()
    adds = []
    for dev_type in ("nmos", "pmos"):
        base = sp.PDK77K_DIR / sp._CORNER_BASENAME[dev_type]
        patched = td / sp._CORNER_BASENAME[dev_type]
        src = patched if patched.exists() and patched.as_posix() in text else base
        rep = replica_corner_chan(src.read_text(errors="replace"), dev_type)
        if not replica_bins_ok(rep, dev_type):
            raise RuntimeError(f"replica bins of {dev_type} lack a REPLICA_CHAN value")
        path = td / f"replica_ch_{dev_type}.spice"
        path.write_text(rep)
        adds.append(f'.include "{path.as_posix()}"\n')
    if text.count(".endl") != 1:
        raise ValueError("library text does not have exactly one .endl")
    lib.write_text(text.replace(".endl", "".join(adds) + ".endl"))
    return lib


# ---------------------------------------------------------------------------
# Netlist (text)
# ---------------------------------------------------------------------------
INSTANCE_MODES = ("card", "card_rep", "elem")


def instance(name: str, dev, d: str, g: str, s: str, b: str, mode: str,
             module: str = v2.VA_MODULE_V2) -> str:
    """Transistor ``name`` (an ``i11_1_deploy.Dev``) at (d, g, s, b).

    mode "card": the card instance only.  "card_rep": the card, the
    channel-current replica at (ri, rg, rs, rb) driven by unit-gain VCVS
    copies of V(d), V(g), V(s), V(b), the gmin compensation VCCS and a 0 V
    source V<name>R from rx to ri (i(V<name>R) = sensed replica current).
    "elem": as card_rep with the OSDI module (model of ``module``) in place
    of the 0 V source.  The card instance is the same line in every mode."""
    from .. import spice_pdk as sp
    if mode not in INSTANCE_MODES:
        raise ValueError(mode)
    subckt = sp._SUBCKT[dev.dev_type]
    body = v2.inst_body(dev.dev_type, dev.L_um, dev.W_um)
    if not body.startswith(subckt + " "):
        raise ValueError("instance body does not start with the subcircuit name")
    out = f"X{name} {d} {g} {s} {b} {body}"
    if not out.endswith("\n"):
        out += "\n"
    if mode == "card":
        return out
    p = f"{name}_"
    out += f"X{name}r {p}ri {p}rg {p}rs {p}rb {subckt}{REPLICA_SUFFIX}{body[len(subckt):]}"
    if not out.endswith("\n"):
        out += "\n"
    out += (f"E{name}d {p}rx 0 {d} 0 1\nE{name}g {p}rg 0 {g} 0 1\n"
            f"E{name}s {p}rs 0 {s} 0 1\nE{name}b {p}rb 0 {b} 0 1\n"
            f"G{name}c {p}ri {p}rb {p}ri {p}rb {-M_DEV * GMIN:.17g}\n")
    if mode == "card_rep":
        return out + f"V{name}R {p}rx {p}ri DC 0 AC 0\n"
    nmod, pmod = MODELS[module]
    model = pmod if dev.dev_type == "pmos" else nmod
    return out + f"N{name} {d} {g} {s} {p}rx {p}ri {model}\n"


def corr_models(module: str) -> str:
    nmod, pmod = MODELS[module]
    return "".join(f".model {m} {module} is_pmos={p} RS=0\n" for m, p in ((nmod, 0), (pmod, 1)))


def deck(title: str, lib: Path, body: str, control: Sequence[str], osdi: Path | None,
         module: str = v2.VA_MODULE_V2, options: str = "", numdgt: int = 17) -> str:
    """Deck: scale, 77 K, the library, ``body``, the module's models when
    ``osdi`` is given, the control block (``numdgt`` digits in wrdata)."""
    from .i11_1_deploy import TEMP_K
    if "gmin" in options.lower():
        raise ValueError("the gmin compensation assumes the ngspice default gmin")
    lines = [f"* {title}\n", ".options scale=1.0\n", f".options temp={TEMP_K - 273.15:.4f}\n"]
    if options:
        lines.append(f".options {options}\n")
    lines.append(f'.lib "{Path(lib).as_posix()}" tt_77k\n')
    lines.append(body if body.endswith("\n") else body + "\n")
    if osdi is not None:
        lines.append(corr_models(module))
    lines.append(".control\n")
    if osdi is not None:
        lines.append(f"pre_osdi {Path(osdi).as_posix()}\n")
    lines += ["set filetype=ascii\n", f"set numdgt={int(numdgt)}\n"]
    lines += [c if c.endswith("\n") else c + "\n" for c in control]
    lines += ["quit\n", ".endc\n", ".end\n"]
    return "".join(lines)


def device_body(dev, mode: str, swap: bool = False, module: str = v2.VA_MODULE_V2) -> str:
    """Forward bench: transistor at (nd, ng, 0, 0).  Exchanged bench: card
    drain at nz (VZ 0 V to ground), card source at nd, body at 0."""
    src = "VG ng 0 DC 0 AC 0\nVD nd 0 DC 0 AC 0\n"
    if swap:
        return instance("m1", dev, "nz", "ng", "nd", "0", mode, module) + "VZ nz 0 DC 0\n" + src
    return instance("m1", dev, "nd", "ng", "0", "0", mode, module) + src


def grid_control(dev, out: Path, swap: bool, sense: bool) -> list[str]:
    """Nested DC sweep (VD inner over GRID_VDS, VG outer 0 -> +-1.85); writes
    -i(VD) and, with ``sense``, i(Vm1R) (sensed replica current)."""
    from .i11_1_deploy import GRID_VDS, GRID_VGS, STEP
    sg = -1.0 if dev.dev_type == "pmos" else 1.0
    vecs = "-i(VD)" + (" i(Vm1R)" if sense else "")
    return [f"dc VD {GRID_VDS[0]:.6g} {GRID_VDS[-1]:.6g} {STEP:.6g} "
            f"VG 0 {sg * GRID_VGS[-1]:.6g} {sg * STEP:.6g}",
            f"wrdata {Path(out).as_posix()} {vecs}"]


def ac_control(points: np.ndarray, excite: str, sense: bool) -> list[str]:
    """Per op point: op, then AC at AC_FREQ with unit excitation on
    ``excite``.  Prints q<k>i (DC -i(VD)), q<k>r / q<k>m (Re / Im AC -i(VD)),
    q<k>g (Im AC -i(VG)); with ``sense`` also q<k>s (DC i(Vm1R)) and q<k>a /
    q<k>b (Re / Im AC i(Vm1R)).  Parsed by ``parse_ac``."""
    from .i11_1_deploy import AC_FREQ
    if excite not in ("VG", "VD"):
        raise ValueError(excite)
    lines = [f"alter {excite} ac = 1"]
    for k, (vg, vd) in enumerate(points):
        lines += [f"alter VG dc = {vg:.6g}", f"alter VD dc = {vd:.6g}", "op",
                  f"let q{k}i = -i(VD)", f"print q{k}i"]
        if sense:
            lines += [f"let q{k}s = i(Vm1R)", f"print q{k}s"]
        lines += [f"ac lin 1 {AC_FREQ:g} {AC_FREQ:g}",
                  f"let q{k}r = real(-i(VD))", f"let q{k}m = imag(-i(VD))",
                  f"let q{k}g = imag(-i(VG))", f"print q{k}r q{k}m q{k}g"]
        if sense:
            lines += [f"let q{k}a = real(i(Vm1R))", f"let q{k}b = imag(i(Vm1R))",
                      f"print q{k}a q{k}b"]
        lines.append("destroy all")
    return lines


_PRINT = re.compile(r"^\s*q(\d+)([irmgsab])\s*=\s*([-+0-9.eEnaNinf]+)")
_AC_KEYS = {"i": "i0", "r": "yr", "m": "ym", "g": "yg", "s": "s0", "a": "sr", "b": "sm"}


def parse_ac(text: str, n: int) -> dict[str, np.ndarray]:
    """``ac_control`` output -> arrays i0, yr, ym, yg, s0, sr, sm (length n,
    NaN where missing)."""
    out = {v: np.full(n, np.nan) for v in _AC_KEYS.values()}
    for line in text.splitlines():
        mt = _PRINT.match(line)
        if mt:
            k = int(mt.group(1))
            if k < n:
                try:
                    out[_AC_KEYS[mt.group(2)]][k] = float(mt.group(3))
                except ValueError:
                    pass
    return out


def insert_element(deck_text: str, osdi: Path | str, dev, module: str = v2.VA_MODULE_V2) -> str:
    """Measured-curve batch deck (``spice_pdk._batch_deck``): the instance
    'Xm1 nd ng 0 0 ...' gets the replica, VCVS, gmin compensation and the
    module; models and ``pre_osdi`` added."""
    lines = deck_text.splitlines(keepends=True)
    inst = [i for i, ln in enumerate(lines) if ln.startswith("Xm1 nd ng 0 0 ")]
    ctrl = [i for i, ln in enumerate(lines) if ln.strip() == ".control"]
    if len(inst) != 1 or len(ctrl) != 1:
        raise ValueError("deck does not have exactly one Xm1 instance and one .control")
    lines[inst[0]] = instance("m1", dev, "nd", "ng", "0", "0", "elem", module) + corr_models(module)
    out = "".join(lines)
    return out.replace(".control\n", f".control\npre_osdi {Path(osdi).as_posix()}\n", 1)


def inverter_body(n, p, mode: str, vin: str, vout: str, name: str, module: str) -> str:
    return (instance(f"{name}n", n, vout, vin, "0", "0", mode, module)
            + instance(f"{name}p", p, vout, vin, "vdd", "vdd", mode, module))


def vtc_body(n, p, mode: str, module: str) -> str:
    from .i11_1_deploy import VDD_CIRCUIT
    return inverter_body(n, p, mode, "nin", "nout", "i1", module) + f"VDD vdd 0 DC {VDD_CIRCUIT}\nVIN nin 0 DC 0\n"


def ring_body(n, p, mode: str, module: str, stages: int = 5) -> str:
    from .i11_1_deploy import VDD_CIRCUIT
    if stages % 2 == 0:
        raise ValueError("ring oscillator needs an odd number of stages")
    nodes = [f"o{k + 1}" for k in range(stages)]
    body = "".join(inverter_body(n, p, mode, nodes[k - 1], nodes[k], f"s{k + 1}", module)
                   for k in range(stages))
    ics = " ".join(f"v({nd})={VDD_CIRCUIT if k % 2 == 0 else 0.0}" for k, nd in enumerate(nodes))
    return body + f"VDD vdd 0 DC {VDD_CIRCUIT}\n.ic {ics}\n"


def mirror_body(dev, mode: str, iref: float, module: str) -> str:
    from .i11_1_deploy import VDD_CIRCUIT
    if dev.dev_type == "nmos":
        return (instance("m1", dev, "nref", "nref", "0", "0", mode, module)
                + instance("m2", dev, "nout", "nref", "0", "0", mode, module)
                + f"IREF 0 nref DC {iref:.6g}\nVOUT nout 0 DC 0\n")
    return (instance("m1", dev, "nref", "nref", "vdd", "vdd", mode, module)
            + instance("m2", dev, "nout", "nref", "vdd", "vdd", mode, module)
            + f"VDD vdd 0 DC {VDD_CIRCUIT}\nIREF nref 0 DC {iref:.6g}\nVOUT nout 0 DC 0\n")


# ---------------------------------------------------------------------------
# Corrections and Verilog-A
# ---------------------------------------------------------------------------
def v1_as_sym(corr) -> v2.SymCorrection:
    """M_bias v1 (``i3_2_bias.BiasCorrection`` on E1 columns 0-2) as a
    ``SymCorrection`` with the v1 inputs (is_pmos, |Vgs| / VDD, |Vds| / VDD):
    same mean / std and networks."""
    if tuple(getattr(corr, "cols", ())) != (0, 1, 2):
        raise ValueError("expected an M_bias v1 correction on E1 columns 0-2")
    return v2.SymCorrection(corr.arm, np.asarray(corr.mu, float), np.asarray(corr.sd, float),
                            nets=corr.nets, inputs="abs_v1", eps=v2.EPS, vdd=v2.VDD_NORM)


def to_verilog_a(corr: v2.SymCorrection, module: str | None = None) -> str:
    """sym_v2: the I12-1 text (``to_verilog_a_v2``).  abs_v1: the same module
    structure with x1 = abs(V(g,s)) / VDD, x2 = abs(V(d,s)) / VDD."""
    if corr.inputs == "sym_v2":
        return v2.to_verilog_a_v2(corr, module or v2.VA_MODULE_V2)
    if corr.inputs != "abs_v1":
        raise ValueError(corr.inputs)
    module = module or VA_MODULE_V1
    fmt = lambda v: f"{float(v):.17e}"    # noqa: E731
    hidden = max(w.shape[1] for layers in corr.nets for w, _ in layers[:-1])
    lines = [
        "`include \"disciplines.vams\"",
        "`include \"constants.vams\"",
        f"// I14-1 reference arm: M_bias v1 ({corr.arm}), {len(corr.nets)}-member "
        f"3-{hidden}-...-1 tanh MLP ensemble.",
        "// I(d->s) added = (exp(g) - 1) * I(sense); I(sense) = channel current of a",
        "// charge-free replica of the card without non-channel currents (static).",
        f"// Inputs (v1, not d/s symmetric): x1 = abs(V(g,s)) / VDD, x2 = abs(V(d,s)) / VDD, VDD = {corr.vdd:g} V.",
        f"module {module}(d, g, s, rx, ri);",
        "  inout d, g, s, rx, ri;",
        "  electrical d, g, s, rx, ri;",
        "  branch (rx, ri) sense;   // I(sense) = replica channel current",
        "  parameter real is_pmos = 0;  // 0 nMOS, 1 pMOS",
        "  parameter real RS = 0.0;   // ohm; sense branch V = RS * I (0: zero-volt source)",
        "  real ib, gsum, acc, vgs, vds, x0, x1, x2;",
        "  real " + ", ".join(f"h{i}" for i in range(hidden)) + ";",
        "  real " + ", ".join(f"k{i}" for i in range(hidden)) + ";",
        "  analog begin",
        v2.SENSE,
        "    ib = I(sense);",
        "    vgs = V(g, s);",
        "    vds = V(d, s);",
        f"    x0 = (is_pmos - {fmt(corr.mu[0])}) / {fmt(corr.sd[0])};",
        f"    x1 = (abs(vgs) / {fmt(corr.vdd)} - {fmt(corr.mu[1])}) / {fmt(corr.sd[1])};",
        f"    x2 = (abs(vds) / {fmt(corr.vdd)} - {fmt(corr.mu[2])}) / {fmt(corr.sd[2])};",
        "    gsum = 0.0;",
    ]
    xs = ["x0", "x1", "x2"]
    for m_i, layers in enumerate(corr.nets):
        lines.append(f"    // ensemble member {m_i}")
        inputs = xs
        for l_i, (w, b) in enumerate(layers):
            last = l_i == len(layers) - 1
            outs = []
            for j in range(w.shape[1]):
                terms = " + ".join(f"{fmt(w[i, j])} * {inputs[i]}" for i in range(w.shape[0]))
                name = "acc" if last else (f"h{j}" if l_i % 2 == 0 else f"k{j}")
                expr = f"{fmt(b[j])} + {terms}"
                lines.append(f"    {name} = {expr if last else f'tanh({expr})'};")
                outs.append(name)
            inputs = outs
        lines.append(f"    gsum = gsum + {v2.OUT_SCALE:g} * tanh(acc);")
    lines += [f"    I(d, s) <+ (exp(gsum / {len(corr.nets)}) - 1.0) * ib;", "  end", "endmodule", ""]
    return "\n".join(lines) + "\n"


def eval_verilog_a(text: str, is_pmos: float, vgs: float, vds: float) -> float:
    """g from the assignment lines of a module text (v2 or v1 form) at
    V(g,s) = vgs, V(d,s) = vds (V(g,d) = vgs - vds)."""
    body = text.split("ib = I(sense);", 1)[1].split("I(d, s) <+", 1)[0]
    env = {"tanh": math.tanh, "sqrt": math.sqrt, "exp": math.exp, "abs": abs,
           "is_pmos": float(is_pmos), "VGS": float(vgs), "VGD": float(vgs) - float(vds),
           "VDS": float(vds)}
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        name, expr = line.rstrip(";").split(" = ", 1)
        expr = (expr.replace("V(g, s)", "VGS").replace("V(g, d)", "VGD")
                .replace("V(d, s)", "VDS"))
        env[name.strip()] = eval(expr, {"__builtins__": {}}, env)  # noqa: S307
    n = int(text.split("I(d, s) <+ (exp(gsum / ", 1)[1].split(")", 1)[0])
    return env["gsum"] / n


def module_name(corr: v2.SymCorrection) -> str:
    return v2.VA_MODULE_V2 if corr.inputs == "sym_v2" else VA_MODULE_V1


# ---------------------------------------------------------------------------
# Python reference model of the element (pure)
# ---------------------------------------------------------------------------
def current_fwd(i_card, i_rep, g) -> np.ndarray:
    """Forward bench (module d = card drain = replica drain copy):
    -i(VD) = I_card + (exp(g) - 1) I_rep."""
    return np.asarray(i_card, float) + np.expm1(np.asarray(g, float)) * np.asarray(i_rep, float)


def current_swap(i_card_nd, i_rep, g) -> np.ndarray:
    """Exchanged bench (module d = card drain at nz, module s = card source
    at nd): the element current (exp(g) - 1) I_rep flows from nz to nd, so
    -i(VD) = I_card(nd) - (exp(g) - 1) I_rep."""
    return np.asarray(i_card_nd, float) - np.expm1(np.asarray(g, float)) * np.asarray(i_rep, float)


def admittance(y_card, y_rep, i_rep, g, dg_dv) -> np.ndarray:
    """Element-identity reference for one excitation:
    Re = Re Y_card + (exp(g) - 1) Re Y_rep + I_rep exp(g) dg/dV,
    Im = Im Y_card (the element adds no charge)."""
    eg = np.exp(np.asarray(g, float))
    yc = np.asarray(y_card)
    re_ = (yc.real + (eg - 1.0) * np.real(np.asarray(y_rep))
           + np.asarray(i_rep, float) * eg * np.asarray(dg_dv, float))
    return re_ + 1j * yc.imag


# ---------------------------------------------------------------------------
# Replica pre-check statistics (pure)
# ---------------------------------------------------------------------------
def drain_channel_current(id_out, vds, is_pmos: bool) -> np.ndarray:
    """Channel current entering the drain terminal from the BSIM4 output
    ``id`` (= m x cd, mode-normalized): type x mode x id with mode =
    sign(type x V(d,s)) (+1 at V(d,s) = 0)."""
    t = -1.0 if is_pmos else 1.0
    mode = np.where(t * np.asarray(vds, float) >= 0.0, 1.0, -1.0)
    return t * mode * np.asarray(id_out, float)


def rel_stats(a, b, mask, floor: float = 1e-300) -> dict:
    """max |a - b| / |b| and fractions <= 1e-12 / 1e-9 / 1e-6 over ``mask``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.asarray(mask, bool) & np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return {"n": 0, "max_rel": 0.0, "max_abs": 0.0}
    d = np.abs(a - b)[mask]
    r = d / np.maximum(np.abs(b[mask]), floor)
    return {"n": int(mask.sum()), "max_rel": float(np.max(r)), "max_abs": float(np.max(d)),
            "frac_le_1e-12": float(np.mean(r <= 1e-12)), "frac_le_1e-9": float(np.mean(r <= 1e-9)),
            "frac_le_1e-6": float(np.mean(r <= 1e-6))}


# ---------------------------------------------------------------------------
# Decision (pure, pre-registered)
# ---------------------------------------------------------------------------
THR_VALID = {"v1_rrms": 1e-9}


def decide_validity(v: dict) -> dict:
    """V1 card-only library deck reproduces the I13-1 Python rows
    (max |diff| <= 1e-9, v2 and v1 arms, 18 devices each); V2 the
    simulate_pdk_multi / sky130_misspec / simulate_sky130 / library clamp
    unit tests pass; V3 18 element-deck accuracy rows per arm and 0 failed
    M2 refits per arm."""
    fails = []
    if not (v["v1_max_rrms_diff"] <= THR_VALID["v1_rrms"]):
        fails.append("V1 card-only deck reproduces the I13-1 rows")
    if not v["v2_unit_tests_pass"]:
        fails.append("V2 clamp unit tests")
    if v["v3_n_rows_min"] != 18 or v["v3_m2_failed"] != 0:
        fails.append("V3 18 rows / 0 failed M2 refits")
    return {"valid": not fails, "fails": fails}


def decide_outcome(validity: dict, acc: dict, dep: dict) -> dict:
    """I12-1 outcome table with A1 / A2 from the I13-1 state-C rows."""
    return v2.decide_outcome(validity, acc, dep)
